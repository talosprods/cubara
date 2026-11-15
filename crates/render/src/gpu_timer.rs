//! Pure bookkeeping for a ring of GPU timestamp-query readback slots.
//!
//! A GPU timestamp write plus its async `map_async` readback cannot be read
//! back the same frame it was written -- the callback fires whenever the
//! driver gets around to it, some frames later. A ring of slots lets frame
//! N+1 write into a fresh slot while frame N's slot is still being mapped, so
//! a caller never blocks on the GPU to get a reading.
//!
//! This module is only the state machine that decides which slot is safe to
//! write and which is safe to read -- no wgpu calls -- so it is unit-tested
//! without a device. The GPU-owning half (the actual `QuerySet` and readback
//! buffers) is [`GpuTimer`], below it: one implementation for `--bench` and
//! for the game window's F3 reading (issue #425, Rule 5).

use std::sync::{Arc, Mutex};

use crate::scene::GpuTimestamps;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Slot {
    /// No outstanding GPU work; safe to write new queries into.
    Idle,
    /// Queries written and submitted, `map_async` started. Not safe to read
    /// -- `get_mapped_range` on a buffer whose map hasn't completed panics.
    Mapping,
    /// `map_async`'s callback has fired; safe to read once, then the slot
    /// goes back to `Idle`.
    Ready,
}

/// Round-robins `depth` slots of a timestamp query/readback buffer.
pub struct TimestampRing {
    slots: Vec<Slot>,
}

impl TimestampRing {
    pub fn new(depth: usize) -> Self {
        assert!(depth > 0, "a ring needs at least one slot");
        Self {
            slots: vec![Slot::Idle; depth],
        }
    }

    pub fn depth(&self) -> usize {
        self.slots.len()
    }

    /// Whether `slot` has no outstanding GPU work and can be written this frame.
    pub fn can_write(&self, slot: usize) -> bool {
        self.slots[slot] == Slot::Idle
    }

    /// Call right after submitting `slot`'s query writes and starting its
    /// `map_async`. Panics if `slot` already has outstanding work -- that
    /// would mean a second `map_async` on a buffer that is already mapped.
    pub fn begin_mapping(&mut self, slot: usize) {
        assert!(
            self.can_write(slot),
            "slot {slot} already has outstanding GPU work"
        );
        self.slots[slot] = Slot::Mapping;
    }

    /// Call from `map_async`'s callback once it fires successfully.
    pub fn mark_ready(&mut self, slot: usize) {
        self.slots[slot] = Slot::Ready;
    }

    /// If `slot` is ready, mark it `Idle` again (so it can be reused) and
    /// return `true` -- the caller may now call `get_mapped_range` on it.
    /// Returns `false` (without changing state) if the slot is `Idle`
    /// (nothing was ever written) or `Mapping` (the callback has not fired
    /// yet) -- reading in either case would be wrong: `Idle` has no data,
    /// `Mapping` is not actually mapped yet.
    pub fn take_ready(&mut self, slot: usize) -> bool {
        if self.slots[slot] == Slot::Ready {
            self.slots[slot] = Slot::Idle;
            true
        } else {
            false
        }
    }

    /// Whether any slot still has a `map_async` outstanding. A caller that
    /// wants every already-submitted slot's result before moving on (rather
    /// than sampling whatever happens to be ready right now) polls the
    /// device in a loop until this is `false` -- found necessary in
    /// practice: a single `Maintain::Wait` does not reliably fire every
    /// pending `map_async` callback on every backend (observed on Metal),
    /// so a caller that scans once immediately after `Wait` can see zero
    /// classified slots even though the GPU work is long done.
    pub fn any_mapping(&self) -> bool {
        self.slots.contains(&Slot::Mapping)
    }
}

/// The two query-set indices `slot` writes its begin/end timestamps to.
/// Pulled out of [`GpuTimer`] as plain arithmetic (no `&self`, no wgpu) so it
/// is unit-tested without a device: an off-by-one here would have a slot's
/// `end` alias the next slot's `begin`, silently mixing two frames' readings.
pub fn timestamp_indices(slot: usize) -> (u32, u32) {
    let begin = (slot * 2) as u32;
    (begin, begin + 1)
}

/// What [`GpuTimer::take_ms`] found once a slot's map completed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuSample {
    /// A plausible pass duration.
    Ms(f64),
    /// The two timestamps came back non-increasing -- not a fast pass, a bad
    /// reading. Counted and reported separately, never averaged in as 0ms.
    Invalid,
}

/// Classify a raw `(begin, end)` timestamp pair. `end <= begin` is not a
/// fast pass, it is a bad reading (a non-monotonic pair off a backend that
/// isn't giving real per-pass timing -- observed on Metal, where `end` came
/// back exactly `0` every time) -- reported as [`GpuSample::Invalid`] rather
/// than folded into the average as 0ms via `saturating_sub`, which would
/// hide it. Pulled out of `GpuTimer::take_ms` as plain arithmetic (no
/// `&self`, no wgpu) so the classification itself is unit-tested with fixed
/// inputs, rather than only reachable by getting a real GPU to produce a
/// bad reading on purpose.
pub fn classify_sample(begin: u64, end: u64, period_ns: f64) -> GpuSample {
    if end <= begin {
        GpuSample::Invalid
    } else {
        GpuSample::Ms((end - begin) as f64 * period_ns / 1_000_000.0)
    }
}

/// Whether a backend's pass-scoped timing has been shown to work: at least
/// one valid sample and no invalid one. Not "at least one valid": on Metal a
/// broken backend still produces an occasional pair that classifies as valid
/// (#267), and a working backend produces zero invalid samples.
pub fn timing_proven(valid: u64, invalid: u64) -> bool {
    valid > 0 && invalid == 0
}

/// What the game window does with its GPU timing, given the samples it has
/// classified so far.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Probation {
    /// Too few samples to say: keep timing, show nothing yet.
    Undecided,
    /// The first [`PROBATION_SAMPLES`] were all valid: show the reading.
    Proven,
    /// They were not ([`timing_proven`]): stop paying for it (#267 measured
    /// the cost on Metal at about a fifth of the frame rate).
    Dropped,
}

/// How many classified samples the window judges its GPU timing on. Chosen,
/// not derived: about a second at 60 Hz.
pub const PROBATION_SAMPLES: u64 = 64;

/// The window's verdict after `valid` and `invalid` samples. An invalid one
/// ends the probation at once; nothing is proven before
/// [`PROBATION_SAMPLES`] valid ones.
pub fn probation(valid: u64, invalid: u64) -> Probation {
    if invalid > 0 {
        Probation::Dropped
    } else if valid >= PROBATION_SAMPLES && timing_proven(valid, invalid) {
        Probation::Proven
    } else {
        Probation::Undecided
    }
}

/// Owns a timestamp query set and its readback buffers around the main scene
/// pass, and turns completed reads into milliseconds. The bookkeeping for
/// which slot is safe to write or read is [`TimestampRing`]
/// (above, unit-tested with no GPU involved); this is the
/// thin GPU-owning wrapper around it.
pub struct GpuTimer {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    read_buffers: Vec<wgpu::Buffer>,
    ring: Arc<Mutex<TimestampRing>>,
    period_ns: f64,
}

impl GpuTimer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, depth: usize) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gpu-timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: (depth * 2) as u32,
        });
        // Each slot's region in the resolve buffer must start at a multiple
        // of `wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT` (256 bytes) -- the two
        // 8-byte timestamps it actually holds don't need that much room, but
        // `resolve_query_set` validates the destination offset regardless.
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-timestamps-resolve"),
            size: (depth as u64) * wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read_buffers = (0..depth)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("gpu-timestamps-read"),
                    size: 16,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        Self {
            query_set,
            resolve_buffer,
            read_buffers,
            ring: Arc::new(Mutex::new(TimestampRing::new(depth))),
            period_ns: queue.get_timestamp_period() as f64,
        }
    }

    /// This frame's slot to write into, or `None` if every slot still has
    /// outstanding GPU work -- skip GPU timing this frame rather than stall
    /// waiting for one to free up or overwrite one still in flight.
    pub fn write_slot(&self, frame: u64) -> Option<usize> {
        let slot = (frame as usize) % self.read_buffers.len();
        self.ring.lock().unwrap().can_write(slot).then_some(slot)
    }

    pub fn timestamps(&self, slot: usize) -> GpuTimestamps<'_> {
        let (begin, end) = timestamp_indices(slot);
        GpuTimestamps {
            query_set: &self.query_set,
            begin,
            end,
        }
    }

    /// Resolve `slot`'s two timestamps into the readback buffer -- called
    /// within the same encoder that wrote them, before submit.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder, slot: usize) {
        let src_offset = (slot as u64) * wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT;
        let (begin, end) = timestamp_indices(slot);
        encoder.resolve_query_set(
            &self.query_set,
            begin..end + 1,
            &self.resolve_buffer,
            src_offset,
        );
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            src_offset,
            &self.read_buffers[slot],
            0,
            16,
        );
    }

    /// Call after submit: marks `slot` in-flight and starts its async map.
    pub fn begin_read(&self, slot: usize) {
        self.ring.lock().unwrap().begin_mapping(slot);
        let ring = Arc::clone(&self.ring);
        self.read_buffers[slot]
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                // An error leaves the slot `Mapping` forever -- it just never
                // contributes another GPU sample, which beats panicking a
                // whole benchmark run over one dropped reading.
                if result.is_ok() {
                    ring.lock().unwrap().mark_ready(slot);
                }
            });
    }

    /// If `slot`'s async map has completed, read its two timestamps, unmap,
    /// and classify them ([`classify_sample`]).
    pub fn take_ms(&self, slot: usize) -> Option<GpuSample> {
        if !self.ring.lock().unwrap().take_ready(slot) {
            return None;
        }
        let buf = &self.read_buffers[slot];
        let sample = {
            let data = buf
                .slice(..)
                .get_mapped_range()
                .expect("read back mapped range");
            let raw: &[u64] = bytemuck::cast_slice(&data);
            let (begin, end) = (raw[0], raw[1]);
            if matches!(
                classify_sample(begin, end, self.period_ns),
                GpuSample::Invalid
            ) {
                // Unconditional, not just the first few: `log::debug!` costs
                // nothing unless enabled, and this is exactly the raw data
                // needed to tell "counter never resolved" (0/0) apart from
                // "resolved backwards" from a diagnosis a plain "invalid"
                // count can't -- found necessary on Metal, where every
                // sample came back this way (see the linked issue).
                log::debug!(
                    "GPU timestamp invalid: slot {slot} begin={begin} end={end} \
                     period_ns={period}",
                    period = self.period_ns
                );
            }
            classify_sample(begin, end, self.period_ns)
        };
        buf.unmap();
        Some(sample)
    }

    /// Call once, right after a `device.poll(Maintain::Wait)` -- polls the
    /// device further, up to a 2-second deadline, until every outstanding
    /// slot's `map_async` callback has actually fired
    /// ([`TimestampRing::any_mapping`]), then reads back every slot that has
    /// ever been written. Returns `(valid, invalid)` counts.
    ///
    /// A single `Wait` is not enough on every backend: observed on Metal,
    /// where a scan taken immediately after `Wait` can see zero classified
    /// slots even though the underlying GPU work is long finished --
    /// `Wait` guarantees the *device* has caught up, not that every
    /// `map_async` callback already ran. Warmup's short window (a few
    /// hundred frames) makes this matter in practice in a way the much
    /// longer measured run's own post-loop `Wait` mostly doesn't.
    pub fn drain_after_wait(&self, device: &wgpu::Device) -> (u64, u64) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while self.ring.lock().unwrap().any_mapping() && std::time::Instant::now() < deadline {
            let _ = device.poll(wgpu::PollType::Poll);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let mut valid = 0u64;
        let mut invalid = 0u64;
        for slot in 0..self.read_buffers.len() {
            match self.take_ms(slot) {
                Some(GpuSample::Ms(_)) => valid += 1,
                Some(GpuSample::Invalid) => invalid += 1,
                None => {}
            }
        }
        (valid, invalid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_ring_has_nothing_ready_to_read() {
        let mut ring = TimestampRing::new(3);
        for slot in 0..3 {
            assert!(!ring.take_ready(slot));
        }
    }

    #[test]
    fn a_slot_is_readable_only_after_its_callback_marks_it_ready() {
        let mut ring = TimestampRing::new(3);
        assert!(ring.can_write(0));
        ring.begin_mapping(0);
        assert!(!ring.can_write(0));
        // The callback hasn't fired yet: reading now would be reading an
        // unmapped buffer.
        assert!(!ring.take_ready(0));
        ring.mark_ready(0);
        assert!(ring.take_ready(0));
        // Taken once; the slot goes back to idle rather than being readable twice.
        assert!(!ring.take_ready(0));
        assert!(ring.can_write(0));
    }

    #[test]
    fn any_mapping_is_true_only_while_a_callback_is_outstanding() {
        let mut ring = TimestampRing::new(3);
        assert!(!ring.any_mapping(), "a fresh ring has nothing outstanding");
        ring.begin_mapping(0);
        assert!(ring.any_mapping());
        ring.mark_ready(0);
        // The callback fired; nothing is outstanding any more, even though
        // slot 0 hasn't been taken yet.
        assert!(!ring.any_mapping());
        ring.begin_mapping(1);
        ring.begin_mapping(2);
        ring.mark_ready(1);
        // Slot 2's callback still hasn't fired.
        assert!(ring.any_mapping());
        ring.mark_ready(2);
        assert!(!ring.any_mapping());
    }

    #[test]
    fn slots_cycle_independently() {
        let mut ring = TimestampRing::new(3);
        ring.begin_mapping(0);
        ring.begin_mapping(1);
        // Slot 2 was never written, so it is still writable and not readable.
        assert!(ring.can_write(2));
        assert!(!ring.take_ready(2));
        ring.mark_ready(1);
        // Only slot 1 is ready -- slot 0's callback has not fired.
        assert!(!ring.take_ready(0));
        assert!(ring.take_ready(1));
    }

    #[test]
    #[should_panic]
    fn writing_a_slot_with_outstanding_work_is_a_bug() {
        let mut ring = TimestampRing::new(2);
        ring.begin_mapping(0);
        // Still `Mapping` -- calling `begin_mapping` again without a
        // `take_ready` in between would start a second `map_async` on a
        // buffer that's already mapped, which is the bug this type exists to
        // make impossible to write by accident.
        ring.begin_mapping(0);
    }

    /// The M3's case (#267): one invalid sample, and the window stops timing.
    #[test]
    fn the_window_keeps_gpu_timing_only_after_a_clean_probation() {
        assert_eq!(probation(0, 0), Probation::Undecided);
        assert_eq!(probation(PROBATION_SAMPLES - 1, 0), Probation::Undecided);
        assert_eq!(probation(PROBATION_SAMPLES, 0), Probation::Proven);
        assert_eq!(probation(0, 1), Probation::Dropped);
        assert_eq!(probation(PROBATION_SAMPLES, 1), Probation::Dropped);
        assert!(timing_proven(1, 0));
        assert!(!timing_proven(0, 0) && !timing_proven(5, 1));
    }
}
