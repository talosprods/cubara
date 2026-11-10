//! The order streamed geometry reaches the arena in.
//!
//! The caller hands over node updates in batches ([`Renderer::apply_node_updates`]
//! (crate::Renderer::apply_node_updates)): meshes to upload and nodes to drop.
//! Uploads are paced -- a whole ring's worth at once would spike the frame --
//! so they wait in a queue. **Drops wait in the same queue, behind the uploads
//! handed over with them.**
//!
//! They used to happen at once. The caller drops a node in the same batch as
//! the meshes that replace it, precisely so the two swap on one frame (#262) --
//! but with the drop immediate and the replacements queued, a batch of a few
//! hundred left the ground they covered undrawn for as many frames as the
//! queue took to reach them. Measured by `--bench flight` as the ground under
//! the camera vanishing for a moment every time it crossed a chunk layer.
//!
//! Now a drop can at worst come a frame or two *late*, while the last of its
//! replacements upload: the old node and some of the new drawn together, which
//! is two nearly identical surfaces rather than none.
//!
//! How much goes up each frame is [`upload_budget`].
//!
//! ## What this queue guarantees (#390)
//!
//! A batch's Removes are the nodes a batch's Inserts replace -- a swap (#262).
//! **A swap's Remove applies only once every Insert in its batch has been
//! placed, whatever order the queue hands steps out in.** Each [`push`](Uploads::push)
//! records, for a batch with at least one Insert, how many of that batch's
//! Inserts are still unresolved; a Remove from that batch carries the batch
//! (`after`) and is withheld -- left exactly where it is, not reordered or
//! dropped -- until that count reaches zero. A Remove from a batch with no
//! Inserts (a node leaving with nothing replacing it) has no dependency and
//! is always ready.
//!
//! The reason this matters: a full arena (`ChunkArena::insert` returning
//! `false`) used to [`requeue_insert`](Uploads::requeue_insert) the failed Insert to the
//! *back* of the queue and stop draining for the frame -- one attempt per
//! frame while the arena stayed full, and the requeued Insert could end up
//! queued *behind* its own batch's Remove, which [`pop`](Uploads::pop) would
//! then hand out and apply with the Insert still missing: the swapped-in
//! child's footprint went undrawn until its retry finally landed. [`drain`]
//! fixes both: a failed Insert no longer stalls the frame -- every remaining
//! step that is not an Insert and is ready still lands, within budget, and
//! the failed Insert gets one retry if that freed anything -- and the
//! dependency above means the requeued Insert can never again be overtaken by
//! its own batch's Remove, however far back in the queue it now sits.

use std::collections::{HashMap, VecDeque};

use crate::arena::NodeId;

/// The [`push`](Uploads::push) call that queued a step, used only to tell a
/// swap's Remove apart from its batch's Inserts ([`ArenaOp::Remove`]'s
/// `after`) -- opaque outside this module, never compared to anything but
/// another `Batch` handed out by the same queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Batch(u64);

/// One thing for the arena to do.
#[derive(Debug, PartialEq)]
enum ArenaOp<T> {
    /// Upload (or replace) this node's geometry.
    Insert(NodeId, T),
    /// Stop drawing this node. `after` is the batch of Inserts this Remove
    /// must not apply before (`None` when nothing replaces the node) --
    /// decision 1 of issue #390.
    Remove { id: NodeId, after: Option<Batch> },
}

fn op_id<T>(op: &ArenaOp<T>) -> NodeId {
    match op {
        ArenaOp::Insert(id, _) => *id,
        ArenaOp::Remove { id, .. } => *id,
    }
}

/// What [`Uploads::drain`] applies queued steps against: the GPU arena in
/// `render.rs`, or a fake in this module's own tests. One value with both
/// operations, rather than two separate closures, so `drain` only ever needs
/// one mutable borrow of it at a time.
pub(crate) trait Sink<T> {
    /// Try to place `item`'s geometry under `id`. `Err` hands `item` back
    /// whole so [`Uploads::drain`] can requeue it unchanged.
    fn try_insert(&mut self, id: NodeId, item: T) -> Result<(), T>;
    /// Stop drawing `id`. Infallible -- freeing a slot cannot run out of
    /// room the way taking one can.
    fn remove(&mut self, id: NodeId);
}

/// Node updates waiting for their turn.
pub(crate) struct Uploads<T> {
    /// Each queued step with the generation it was pushed under and the
    /// batch that pushed it (used only to resolve an Insert's pending count --
    /// see [`pending_inserts`](Self::pending_inserts); a Remove's own
    /// dependency lives in the op itself).
    queue: VecDeque<(u64, Batch, ArenaOp<T>)>,
    /// The generation of the newest step queued for each node: only that step
    /// still applies when its turn comes, whatever was queued before it.
    /// Looked up by id only, never iterated, so its order cannot reach a
    /// result.
    latest: HashMap<NodeId, u64>,
    /// The generation the next pushed step gets.
    next_generation: u64,
    /// The batch the next [`push`](Self::push) call gets.
    next_batch: u64,
    /// For every batch that pushed at least one Insert and isn't fully
    /// resolved yet: how many of its Inserts are still outstanding (queued,
    /// or just handed out and not yet known to have failed -- see
    /// [`take`](Self::take)/[`requeue_insert`](Self::requeue_insert)). Absence means zero --
    /// the entry is removed as soon as it would hit zero, so a batch with no
    /// Inserts never appears here at all, and `after: Some(batch)` is ready
    /// exactly when `batch` is absent.
    pending_inserts: HashMap<Batch, u32>,
    /// The batch of an Insert [`take`](Self::take)/[`pop_id`](Self::pop_id)
    /// most recently handed out for a node, kept only until
    /// [`requeue_insert`](Self::requeue_insert) (it failed -- undo the optimistic resolve)
    /// or [`drain`](Self::drain) confirms it landed (forget it).
    in_flight_batch: HashMap<NodeId, Batch>,
}

impl<T> Uploads<T> {
    pub(crate) fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            latest: HashMap::new(),
            next_generation: 0,
            next_batch: 0,
            pending_inserts: HashMap::new(),
            in_flight_batch: HashMap::new(),
        }
    }

    fn enqueue(&mut self, op: ArenaOp<T>, batch: Batch) {
        let id = op_id(&op);
        let generation = self.next_generation;
        self.next_generation += 1;
        self.latest.insert(id, generation);
        self.queue.push_back((generation, batch, op));
    }

    /// Queue one batch: `meshed` first, then the drops that must not happen
    /// before them. When `meshed` is non-empty this is a swap (#262): every
    /// drop in `to_unload` gets `after` set to this batch, so none of them
    /// can apply until every one of `meshed`'s Inserts has been placed
    /// (#390) -- regardless of what order the queue later hands steps out in.
    pub(crate) fn push(
        &mut self,
        to_unload: impl IntoIterator<Item = NodeId>,
        meshed: impl IntoIterator<Item = (NodeId, T)>,
    ) {
        let batch = Batch(self.next_batch);
        self.next_batch += 1;
        let mut insert_count = 0u32;
        for (id, item) in meshed {
            self.enqueue(ArenaOp::Insert(id, item), batch);
            insert_count += 1;
        }
        let after = if insert_count > 0 {
            self.pending_inserts.insert(batch, insert_count);
            Some(batch)
        } else {
            None
        };
        for id in to_unload {
            self.enqueue(ArenaOp::Remove { id, after }, batch);
        }
    }

    /// Whether a Remove carrying this dependency may apply now.
    fn remove_ready(&self, after: Option<Batch>) -> bool {
        match after {
            None => true,
            Some(batch) => !self.pending_inserts.contains_key(&batch),
        }
    }

    /// One of `batch`'s Inserts is resolved -- either placed, handed out and
    /// about to be attempted, or permanently discarded as stale -- so it no
    /// longer blocks that batch's Remove(s).
    fn resolve_insert(&mut self, batch: Batch) {
        if let Some(count) = self.pending_inserts.get_mut(&batch) {
            *count -= 1;
            if *count == 0 {
                self.pending_inserts.remove(&batch);
            }
        }
    }

    /// Scan the queue from the front for the next step that is both the
    /// newest queued for its node ([`pop`](Self::pop)'s long-standing
    /// newest-wins rule -- a stale entry found along the way is discarded
    /// here for good, which is also a resolution for a stale Insert) *and*
    /// ready to apply: an Insert is ready only when `allow_insert`, a Remove
    /// is ready per [`remove_ready`](Self::remove_ready). A not-ready Remove,
    /// or (when `!allow_insert`) any Insert, is left exactly where it is --
    /// this is what lets a later, already-satisfied step be taken without
    /// reordering anything the queue hasn't resolved yet.
    ///
    /// An Insert this returns has its batch optimistically resolved (so a
    /// sibling Remove queued behind it can become ready the moment every
    /// other Insert in the batch is also handed out, without waiting for
    /// confirmation that any of them actually landed) and is recorded in
    /// [`in_flight_batch`](Self::in_flight_batch) until
    /// [`requeue_insert`](Self::requeue_insert) undoes that or the caller confirms it
    /// landed.
    fn take(&mut self, allow_insert: bool) -> Option<ArenaOp<T>> {
        let mut i = 0;
        while i < self.queue.len() {
            let (generation, id, is_insert, after) = {
                let (generation, _batch, op) = &self.queue[i];
                let id = op_id(op);
                match op {
                    ArenaOp::Insert(..) => (*generation, id, true, None),
                    ArenaOp::Remove { after, .. } => (*generation, id, false, *after),
                }
            };
            if self.latest.get(&id) != Some(&generation) {
                // Superseded by a newer step for this node -- gone for good.
                let (_, batch, op) = self.queue.remove(i).expect("i < queue.len()");
                if matches!(op, ArenaOp::Insert(..)) {
                    self.resolve_insert(batch);
                }
                continue; // `remove` shifted the next entry into index `i`.
            }
            let ready = if is_insert {
                allow_insert
            } else {
                self.remove_ready(after)
            };
            if !ready {
                i += 1;
                continue;
            }
            let (_, batch, op) = self.queue.remove(i).expect("i < queue.len()");
            self.latest.remove(&id);
            if matches!(op, ArenaOp::Insert(..)) {
                self.resolve_insert(batch);
                self.in_flight_batch.insert(id, batch);
            }
            return Some(op);
        }
        None
    }

    /// The next step still wanted, in order, skipping over (without
    /// reordering or dropping) a swap Remove whose batch isn't fully placed
    /// yet. **The newest step queued for a node wins**; every older one for
    /// it is skipped, however many steps for other nodes came between. So an
    /// upload for a node dropped since is skipped, and so is a drop for a
    /// node handed over again since (its new geometry replaces the old in
    /// place, with no gap) -- and an upload queued before a drop and a
    /// re-upload cannot sneak through when the re-upload makes the node
    /// wanted again.
    fn pop(&mut self) -> Option<ArenaOp<T>> {
        self.take(true)
    }

    /// The next ready Remove anywhere in the queue, leaving every Insert and
    /// every not-yet-ready Remove untouched -- the recovery pass [`drain`]
    /// runs after a failed Insert (#390 decision 3): steps that free space
    /// still land, but no Insert behind the failed one is attempted this
    /// call.
    fn pop_ready_remove(&mut self) -> Option<NodeId> {
        match self.take(false) {
            Some(ArenaOp::Remove { id, .. }) => Some(id),
            Some(ArenaOp::Insert(..)) => unreachable!("take(false) never returns an Insert"),
            None => None,
        }
    }

    /// Pop the queued step for `target` specifically, if it is still the
    /// newest queued for it -- used only to retry, in the same [`drain`]
    /// call, the Insert a [`requeue_insert`](Self::requeue_insert) just sent to the back.
    fn pop_id(&mut self, target: NodeId) -> Option<ArenaOp<T>> {
        let i = self
            .queue
            .iter()
            .position(|(_, _, op)| op_id(op) == target)?;
        let generation = self.queue[i].0;
        if self.latest.get(&target) != Some(&generation) {
            return None;
        }
        let (_, batch, op) = self.queue.remove(i).expect("i came from position()");
        self.latest.remove(&target);
        if matches!(op, ArenaOp::Insert(..)) {
            self.resolve_insert(batch);
            self.in_flight_batch.insert(target, batch);
        }
        Some(op)
    }

    /// Put an Insert [`take`](Self::take)/[`pop_id`](Self::pop_id) handed out
    /// but could not carry out back at the end of the queue, to be tried
    /// again. It is still the newest step for its node unless a newer one
    /// was pushed meanwhile, in which case it is superseded and dropped
    /// here, as `take` would have. Its batch gets its pending count back --
    /// `take`/`pop_id` resolved it optimistically, and this undoes that
    /// since the Insert didn't land.
    ///
    /// Insert-only, not `ArenaOp<T>`-generic: a Remove is never requeued --
    /// freeing an arena slot cannot fail the way taking one can, so nothing
    /// in this crate ever calls this with one, and a Remove branch here
    /// would be dead code no test could ever exercise honestly.
    fn requeue_insert(&mut self, id: NodeId, item: T) {
        if self.latest.contains_key(&id) {
            // Superseded since it was handed out -- the newer step already
            // queued carries whatever bookkeeping matters now; this one is
            // genuinely gone.
            self.in_flight_batch.remove(&id);
            return;
        }
        let batch = self.in_flight_batch.remove(&id).expect(
            "every Insert take()/pop_id() hands out is recorded in in_flight_batch until \
             requeue_insert or drain confirms it landed",
        );
        *self.pending_inserts.entry(batch).or_insert(0) += 1;
        self.enqueue(ArenaOp::Insert(id, item), batch);
    }

    /// Apply queued steps against `sink` -- the GPU arena, wrapped by the
    /// caller so `drain` can touch it through one value (two separate
    /// closures would each need their own exclusive borrow of the same
    /// arena/trees to call both `try_insert` and `remove`, which don't run at
    /// the same time but the borrow checker can't see that). `min_inserts`
    /// is the fewest Inserts to place before `over_budget` may end the call
    /// (mirrors the old loop's `MIN_UPLOADS_PER_FRAME`, so a frame never
    /// uploads less than it used to); `over_budget` is checked after every
    /// placed Insert, and before every step of the recovery pass below.
    ///
    /// **A full arena no longer stalls the whole queue (#390).** When
    /// `sink.try_insert` refuses a node, it is requeued (what
    /// [`requeue_insert`](Self::requeue_insert) always did) and, instead of
    /// stopping the frame outright, every
    /// remaining step that is *not* an Insert and whose dependency is
    /// already met -- `Remove`s with no dependency, and swap `Remove`s whose
    /// batch is now fully placed -- is still applied, within whatever budget
    /// is left. A swap's Remove still never applies before its batch's
    /// Inserts are all placed (the #262 rule), so the dependency check is
    /// what makes this safe rather than a second source of the hole this
    /// closes: a requeued child Insert can no longer be overtaken by its own
    /// batch's Remove, however far back in the queue it now sits. If that
    /// recovery pass applied anything, the failed Insert is retried once,
    /// since a Remove may have freed exactly the room it needed. Either way,
    /// this call then stops: Inserts behind the failed one are never
    /// attempted here -- they would fail the same way and double the work
    /// for nothing.
    pub(crate) fn drain(
        &mut self,
        min_inserts: usize,
        mut over_budget: impl FnMut() -> bool,
        sink: &mut impl Sink<T>,
    ) {
        let mut placed = 0usize;
        while let Some(op) = self.pop() {
            match op {
                ArenaOp::Remove { id, .. } => sink.remove(id),
                ArenaOp::Insert(id, item) => match sink.try_insert(id, item) {
                    Ok(()) => {
                        self.in_flight_batch.remove(&id);
                        placed += 1;
                        if placed >= min_inserts && over_budget() {
                            return;
                        }
                    }
                    Err(item) => {
                        self.requeue_insert(id, item);
                        let mut freed_any = false;
                        while !over_budget() {
                            match self.pop_ready_remove() {
                                Some(freed_id) => {
                                    sink.remove(freed_id);
                                    freed_any = true;
                                }
                                None => break,
                            }
                        }
                        if freed_any && !over_budget() {
                            if let Some(ArenaOp::Insert(retry_id, retry_item)) = self.pop_id(id) {
                                match sink.try_insert(retry_id, retry_item) {
                                    Ok(()) => {
                                        self.in_flight_batch.remove(&retry_id);
                                    }
                                    Err(retry_item) => {
                                        self.requeue_insert(retry_id, retry_item);
                                    }
                                }
                            }
                        }
                        return;
                    }
                },
            }
        }
    }

    /// Forget everything waiting: what is queued describes a world that is
    /// gone.
    pub(crate) fn clear(&mut self) {
        self.queue.clear();
        self.latest.clear();
        self.pending_inserts.clear();
        self.in_flight_batch.clear();
    }

    /// Steps still waiting.
    pub(crate) fn len(&self) -> usize {
        self.queue.len()
    }
}

/// How long uploads may take in a frame that followed one of `last_frame`:
/// [`UPLOAD_SHARE`] of it, and never more than of a 30 FPS frame -- a frame
/// that hitched must not make the next one hitch too.
///
/// **A share of the frame, not a number of nodes.** It was 32 nodes a frame,
/// chosen at a thousand frames a second -- 32,000 a second. At a 60 Hz
/// display's pace the same 32 was under 2,000 a second, and dropping one
/// chunk layer while flying hands over about 2,400 at once: over a second of
/// ground missing its detail (`--bench flight`). Uploading is cheap -- about
/// 1,000 nodes in 3.6 ms on an M3 -- so a quarter of a 60 Hz frame clears such
/// a batch in a handful of frames, and a quarter of a 1 ms frame still spikes
/// nothing.
pub(crate) fn upload_budget(last_frame: std::time::Duration) -> std::time::Duration {
    last_frame
        .min(std::time::Duration::from_micros(33_333))
        .mul_f64(UPLOAD_SHARE)
}

/// The share of a frame uploads may take.
const UPLOAD_SHARE: f64 = 0.25;

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn id(n: i32) -> NodeId {
        NodeId {
            level: 0,
            pos: [n, 0, 0],
        }
    }

    fn drain_pop<T>(q: &mut Uploads<T>) -> Vec<ArenaOp<T>> {
        std::iter::from_fn(|| q.pop()).collect()
    }

    /// A fake GPU arena for [`Uploads::drain`]: accepts `capacity` Inserts,
    /// then refuses every one after. Records what was applied, in order, so
    /// a test can see exactly what one `drain` call did.
    #[derive(Default)]
    struct FakeArena {
        capacity: usize,
        inserted: Vec<NodeId>,
        removed: Vec<NodeId>,
    }

    impl FakeArena {
        fn new(capacity: usize) -> Self {
            Self {
                capacity,
                ..Default::default()
            }
        }
    }

    impl Sink<i32> for FakeArena {
        fn try_insert(&mut self, id: NodeId, item: i32) -> Result<(), i32> {
            if self.capacity > 0 {
                self.capacity -= 1;
                self.inserted.push(id);
                Ok(())
            } else {
                Err(item)
            }
        }

        fn remove(&mut self, id: NodeId) {
            self.removed.push(id);
        }
    }

    /// Drive one [`Uploads::drain`] call against `arena`, with no time
    /// budget at all (`over_budget` always false) unless `min_inserts` ends
    /// it first -- these tests care about ordering and dependencies, not
    /// pacing, which [`upload_budget`] already has its own tests for.
    fn drain_unbounded(q: &mut Uploads<i32>, arena: &mut FakeArena) {
        q.drain(usize::MAX, || false, arena);
    }

    #[test]
    fn a_node_is_dropped_only_after_what_replaces_it_is_uploaded() {
        let mut q = Uploads::new();
        // A parent (9) replaced by its children (1..=3).
        q.push([id(9)], (1..=3).map(|n| (id(n), n)));
        assert_eq!(
            drain_pop(&mut q),
            vec![
                ArenaOp::Insert(id(1), 1),
                ArenaOp::Insert(id(2), 2),
                ArenaOp::Insert(id(3), 3),
                ArenaOp::Remove {
                    id: id(9),
                    after: Some(Batch(0))
                },
            ],
            "the parent left before its children arrived"
        );
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn a_step_a_later_batch_overruled_is_skipped() {
        let mut q = Uploads::new();
        q.push([], [(id(1), 1)]);
        q.push([id(1)], []);
        assert_eq!(
            drain_pop(&mut q),
            vec![ArenaOp::Remove {
                id: id(1),
                after: None
            }],
            "uploaded a node already dropped"
        );
        q.push([id(2)], []);
        q.push([], [(id(2), 2)]);
        assert_eq!(
            drain_pop(&mut q),
            vec![ArenaOp::Insert(id(2), 2)],
            "dropped a node handed over again -- a gap before its new geometry"
        );
    }

    #[test]
    fn the_newest_step_for_a_node_wins_even_when_it_makes_the_node_wanted_again() {
        let mut q = Uploads::new();
        q.push([], [(id(1), 10)]);
        q.push([id(1)], []);
        q.push([], [(id(1), 11)]);
        assert_eq!(
            drain_pop(&mut q),
            vec![ArenaOp::Insert(id(1), 11)],
            "stale geometry uploaded before the newer"
        );
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn a_drop_still_waits_behind_the_uploads_queued_before_it() {
        let mut q = Uploads::new();
        q.push([], [(id(1), 1), (id(2), 2)]);
        q.push([id(9)], [(id(3), 3)]);
        assert_eq!(
            drain_pop(&mut q),
            vec![
                ArenaOp::Insert(id(1), 1),
                ArenaOp::Insert(id(2), 2),
                ArenaOp::Insert(id(3), 3),
                ArenaOp::Remove {
                    id: id(9),
                    after: Some(Batch(1))
                },
            ]
        );
    }

    #[test]
    fn clear_leaves_nothing_and_later_pushes_behave_normally() {
        let mut q = Uploads::new();
        q.push([id(9)], [(id(1), 1)]);
        q.clear();
        assert_eq!(q.len(), 0);
        assert_eq!(drain_pop(&mut q), vec![], "a step survived the clear");
        q.push([], [(id(1), 2)]);
        q.push([id(2)], []);
        assert_eq!(
            drain_pop(&mut q),
            vec![
                ArenaOp::Insert(id(1), 2),
                ArenaOp::Remove {
                    id: id(2),
                    after: None
                }
            ]
        );
    }

    #[test]
    fn a_requeued_step_goes_to_the_back_unless_something_newer_replaced_it() {
        let mut q = Uploads::new();
        q.push([], [(id(1), 1), (id(2), 2), (id(3), 3)]);
        let ArenaOp::Insert(first_id, first_item) = q.pop().unwrap() else {
            panic!("expected an Insert")
        };
        q.requeue_insert(first_id, first_item);
        let second = q.pop().unwrap();
        assert_eq!(second, ArenaOp::Insert(id(2), 2));
        // A newer step for node 2 supersedes the one being requeued.
        q.push([id(2)], []);
        let ArenaOp::Insert(second_id, second_item) = second else {
            panic!("expected an Insert")
        };
        q.requeue_insert(second_id, second_item);
        assert_eq!(
            drain_pop(&mut q),
            vec![
                ArenaOp::Insert(id(3), 3),
                ArenaOp::Insert(id(1), 1),
                ArenaOp::Remove {
                    id: id(2),
                    after: None
                },
            ],
            "requeued steps must go behind the rest, and a superseded one must not return"
        );
    }

    #[test]
    fn uploads_get_a_quarter_of_the_frame_at_any_frame_rate() {
        assert_eq!(
            upload_budget(Duration::from_micros(16_667)),
            Duration::from_nanos(4_166_750)
        );
        assert_eq!(
            upload_budget(Duration::from_millis(1)),
            Duration::from_micros(250)
        );
        assert_eq!(
            upload_budget(Duration::from_millis(500)),
            Duration::from_micros(33_333).mul_f64(0.25),
            "a hitch handed on to the next frame"
        );
    }

    // -- #390: a full arena still drains, and never opens the #262 hole. --

    #[test]
    fn a_full_arena_still_applies_the_removes_behind_the_failed_insert() {
        // Insert A (fails), Remove X (no dependency), Insert B -- three
        // separate pushes so they land in the queue in exactly this order.
        let mut q = Uploads::new();
        q.push([], [(id(1) /* A */, 1)]);
        q.push([id(9) /* X */], []);
        q.push([], [(id(2) /* B */, 2)]);

        // The arena is full: A fails, and stays failed (capacity 0) for this
        // whole call -- nothing frees up, so B is never even attempted.
        let mut arena = FakeArena::new(0);
        drain_unbounded(&mut q, &mut arena);

        assert_eq!(
            arena.removed,
            vec![id(9)],
            "X does not depend on A and must land in the same drain call, \
             not wait for a frame where A happens to succeed first"
        );
        assert!(arena.inserted.is_empty(), "nothing fits yet");
        // B (behind the failed A) is left alone, per decision 3.
        assert_eq!(q.len(), 2, "A (requeued) and B are both still queued");
    }

    /// Like [`FakeArena`], except a Remove frees exactly one slot's worth of
    /// room -- standing in for whatever space a real Remove frees in the GPU
    /// arena -- so a retry right after it can be told apart from one that
    /// just happens to succeed on its own.
    #[derive(Default)]
    struct ArenaThatFreesRoomOnRemove {
        capacity: usize,
        inserted: Vec<NodeId>,
        removed: Vec<NodeId>,
    }

    impl Sink<i32> for ArenaThatFreesRoomOnRemove {
        fn try_insert(&mut self, id: NodeId, item: i32) -> Result<(), i32> {
            if self.capacity > 0 {
                self.capacity -= 1;
                self.inserted.push(id);
                Ok(())
            } else {
                Err(item)
            }
        }

        fn remove(&mut self, id: NodeId) {
            self.removed.push(id);
            self.capacity += 1;
        }
    }

    #[test]
    fn a_retry_lands_when_a_remove_freed_space() {
        let mut q = Uploads::new();
        q.push([], [(id(1) /* A */, 1)]);
        q.push([id(9) /* X */], []);

        // The arena starts full: A fails on the first attempt. Removing X
        // frees exactly the room A needs, so the same `drain` call must
        // retry A and place it.
        let mut arena = ArenaThatFreesRoomOnRemove::default();
        q.drain(usize::MAX, || false, &mut arena);

        assert_eq!(arena.removed, vec![id(9)]);
        assert_eq!(
            arena.inserted,
            vec![id(1)],
            "A must be retried and placed in the same drain call that freed its room"
        );
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn a_swap_remove_never_overtakes_its_batch_s_inserts() {
        // One batch: Insert C1, Insert C2, Remove P -- a parent P replaced by
        // children C1/C2 (#262).
        let mut q = Uploads::new();
        q.push([id(9) /* P */], [(id(1) /* C1 */, 1), (id(2) /* C2 */, 2)]);

        // Frame 1: room for exactly one Insert -- C1 lands, C2 fails. P must
        // not apply: its batch still has an unplaced Insert.
        let mut arena = FakeArena::new(1);
        drain_unbounded(&mut q, &mut arena);
        assert_eq!(arena.inserted, vec![id(1)]);
        assert!(
            arena.removed.is_empty(),
            "P applied while C2 (its sibling) was still missing -- the #262 hole"
        );

        // Frame 2: room freed up (C2 would now fit), but C2's failed retry
        // sits behind P in raw queue order -- without the dependency, P
        // would be handed out and applied before C2 is even reached.
        let mut arena = FakeArena::new(1);
        drain_unbounded(&mut q, &mut arena);
        assert_eq!(
            arena.inserted,
            vec![id(2)],
            "C2 must land before P is allowed to apply"
        );
        assert_eq!(
            arena.removed,
            vec![id(9)],
            "once both children are placed, P may finally leave"
        );
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn a_superseded_swap_remove_does_not_free_its_batch_early() {
        // One batch: Insert C1 (the only real replacement) plus two swap
        // Removes, P1 and P2, both depending on it.
        let mut q = Uploads::new();
        q.push([id(8) /* P1 */, id(9) /* P2 */], [(id(1) /* C1 */, 1)]);
        // P1 is handed over again before its queued Remove ever gets a turn
        // -- a legitimate case the module already handles for a plain
        // Remove ("a drop for a node handed over again"). The old Remove
        // for P1 is now stale and must be discarded for good *without*
        // resolving the batch it happened to carry -- that bookkeeping
        // belongs to C1, which this has nothing to do with.
        q.push([], [(id(8), 99)]);

        // Scanning for a ready Remove must discard the now-stale P1 along
        // the way (same as always), but P2 -- the swap Remove that *does*
        // still depend on this batch -- must not become ready from that:
        // C1 (the batch's only Insert) has not been placed.
        assert_eq!(
            q.pop_ready_remove(),
            None,
            "P2 became ready before C1 (its batch's only Insert) landed -- discarding the \
             superseded P1 must not resolve the batch early"
        );
    }
}
