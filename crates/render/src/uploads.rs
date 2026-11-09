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

use std::collections::{HashMap, VecDeque};

use crate::arena::NodeId;

/// One thing for the arena to do.
#[derive(Debug, PartialEq)]
pub(crate) enum ArenaOp<T> {
    /// Upload (or replace) this node's geometry.
    Insert(NodeId, T),
    /// Stop drawing this node.
    Remove(NodeId),
}

/// Node updates waiting for their turn.
pub(crate) struct Uploads<T> {
    /// Each queued step with the generation it was pushed under.
    queue: VecDeque<(u64, ArenaOp<T>)>,
    /// The generation of the newest step queued for each node: only that step
    /// still applies when its turn comes, whatever was queued before it.
    /// Looked up by id only, never iterated, so its order cannot reach a
    /// result.
    latest: HashMap<NodeId, u64>,
    /// The generation the next pushed step gets.
    next_generation: u64,
}

impl<T> Uploads<T> {
    pub(crate) fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            latest: HashMap::new(),
            next_generation: 0,
        }
    }

    fn enqueue(&mut self, op: ArenaOp<T>) {
        let (ArenaOp::Insert(id, _) | ArenaOp::Remove(id)) = &op;
        let generation = self.next_generation;
        self.next_generation += 1;
        self.latest.insert(*id, generation);
        self.queue.push_back((generation, op));
    }

    /// Queue one batch: `meshed` first, then the drops that must not happen
    /// before them.
    pub(crate) fn push(
        &mut self,
        to_unload: impl IntoIterator<Item = NodeId>,
        meshed: impl IntoIterator<Item = (NodeId, T)>,
    ) {
        for (id, item) in meshed {
            self.enqueue(ArenaOp::Insert(id, item));
        }
        for id in to_unload {
            self.enqueue(ArenaOp::Remove(id));
        }
    }

    /// The next step still wanted, in order. **The newest step queued for a
    /// node wins**; every older one for it is skipped, however many steps
    /// for other nodes came between. So an upload for a node dropped since is
    /// skipped, and so is a drop for a node handed over again since (its new
    /// geometry replaces the old in place, with no gap) -- and an upload
    /// queued before a drop and a re-upload cannot sneak through when the
    /// re-upload makes the node wanted again.
    pub(crate) fn pop(&mut self) -> Option<ArenaOp<T>> {
        while let Some((generation, op)) = self.queue.pop_front() {
            let (ArenaOp::Insert(id, _) | ArenaOp::Remove(id)) = &op;
            if self.latest.get(id) == Some(&generation) {
                self.latest.remove(id);
                return Some(op);
            }
        }
        None
    }

    /// Put a step [`pop`](Self::pop) returned but could not carry out back at
    /// the end of the queue, to be tried again. It is still the newest step
    /// for its node unless a newer one was pushed meanwhile, in which case it
    /// is superseded and dropped here, as `pop` would have.
    pub(crate) fn requeue(&mut self, op: ArenaOp<T>) {
        let (ArenaOp::Insert(id, _) | ArenaOp::Remove(id)) = &op;
        if !self.latest.contains_key(id) {
            self.enqueue(op);
        }
    }

    /// Forget everything waiting: what is queued describes a world that is
    /// gone.
    pub(crate) fn clear(&mut self) {
        self.queue.clear();
        self.latest.clear();
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

    fn drain<T>(q: &mut Uploads<T>) -> Vec<ArenaOp<T>> {
        std::iter::from_fn(|| q.pop()).collect()
    }

    #[test]
    fn a_node_is_dropped_only_after_what_replaces_it_is_uploaded() {
        let mut q = Uploads::new();
        // A parent (9) replaced by its children (1..=3).
        q.push([id(9)], (1..=3).map(|n| (id(n), n)));
        assert_eq!(
            drain(&mut q),
            vec![
                ArenaOp::Insert(id(1), 1),
                ArenaOp::Insert(id(2), 2),
                ArenaOp::Insert(id(3), 3),
                ArenaOp::Remove(id(9)),
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
            drain(&mut q),
            vec![ArenaOp::Remove(id(1))],
            "uploaded a node already dropped"
        );
        q.push([id(2)], []);
        q.push([], [(id(2), 2)]);
        assert_eq!(
            drain(&mut q),
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
            drain(&mut q),
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
            drain(&mut q),
            vec![
                ArenaOp::Insert(id(1), 1),
                ArenaOp::Insert(id(2), 2),
                ArenaOp::Insert(id(3), 3),
                ArenaOp::Remove(id(9)),
            ]
        );
    }

    #[test]
    fn clear_leaves_nothing_and_later_pushes_behave_normally() {
        let mut q = Uploads::new();
        q.push([id(9)], [(id(1), 1)]);
        q.clear();
        assert_eq!(q.len(), 0);
        assert_eq!(drain(&mut q), vec![], "a step survived the clear");
        q.push([], [(id(1), 2)]);
        q.push([id(2)], []);
        assert_eq!(
            drain(&mut q),
            vec![ArenaOp::Insert(id(1), 2), ArenaOp::Remove(id(2))]
        );
    }

    #[test]
    fn a_requeued_step_goes_to_the_back_unless_something_newer_replaced_it() {
        let mut q = Uploads::new();
        q.push([], [(id(1), 1), (id(2), 2), (id(3), 3)]);
        let first = q.pop().unwrap();
        q.requeue(first);
        let second = q.pop().unwrap();
        assert_eq!(second, ArenaOp::Insert(id(2), 2));
        // A newer step for node 2 supersedes the one being requeued.
        q.push([id(2)], []);
        q.requeue(second);
        assert_eq!(
            drain(&mut q),
            vec![
                ArenaOp::Insert(id(3), 3),
                ArenaOp::Insert(id(1), 1),
                ArenaOp::Remove(id(2)),
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
}
