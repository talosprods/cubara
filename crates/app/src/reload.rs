//! `--bench reload`: in what order the picture comes back after a jump.
//!
//! The owner, 2026-11-22, after flying far away in Spectator and pressing F4
//! to return to his body: "een verdere laag van het terrein eerder ingeladen
//! was dan een laag die dichterbij was, waardoor je de lagen los van elkaar
//! zag inladen". Leaving Spectator re-centres streaming on the body, which is
//! a jump like `/tp`: everything around the body was unloaded while the
//! camera was away, and has to come back.
//!
//! `--bench flight` cannot see this: it flies a line, and its "loading" share
//! is one number for the whole screen. This measures the order. It loads an
//! eye completely and keeps that frame, jumps the streaming eye
//! [`Reload::away`] blocks off and lets that settle (what the body's
//! surroundings look like to the streamers after a long Spectator flight),
//! jumps back, and then reads every frame until nothing is left to load.
//!
//! Each pixel of the finished frame belongs to one **band**: the voxel level
//! (0 to 3) whose ring draws the ground it shows once loaded, or the far
//! terrain ([`bands`]). Per frame and band, a pixel is a **hole** (nothing
//! drawn where the finished frame shows land), a **stand-in** (something
//! drawn that is not the finished picture: a coarser level, the far terrain
//! over voxel ground) or finished ([`band_shares`]).
//!
//! The owner's symptom is then a frame in which a nearer band still has
//! holes while a farther band is mostly drawn ([`outside_in`]): the horizon
//! standing there with nothing between it and the camera. Loading order
//! done right never shows one -- whatever is drawn far away, something is
//! drawn in front of it.
//!
//! The game's own streamers and renderer ([`crate::flight::Streamed`]), paced
//! like a display, in real time: the mesh workers get exactly the time a
//! player's would.

use std::collections::HashSet;
use std::path::Path;
use std::time::{Duration, Instant};

use cubara_render::Frame;
use cubara_voxel::ChunkCoord;
use cubara_world::node::{self, NodeKey};

use crate::flight::{pixel_ray, save, shows_nothing, surface_hit, Flight, Streamed, TOLERANCE};
use crate::streaming::VERTICAL_LOD_SQUASH;

/// The bands a finished frame's land is split into: voxel levels 0 to 3,
/// nearest first, then the far terrain.
pub const BANDS: usize = 5;

/// The far terrain's band: everything outside the voxel rings.
pub const FAR: usize = 4;

/// What the bands are called in a report.
pub const BAND_NAMES: [&str; BANDS] = ["L0", "L1", "L2", "L3", "far"];

/// A pixel of the finished frame that shows no land: sky. In no band.
pub const SKY: u8 = u8::MAX;

/// A band with fewer pixels than this is not judged: a sliver of a ring
/// behind a hill says nothing about the order the rings loaded in.
pub const MIN_BAND_PIXELS: u32 = 200;

/// The share of a nearer band's pixels showing nothing from which it counts
/// as having a hole: more than a mesh seam, less than one missing node at
/// the ring's near edge.
pub const HOLE_SHARE: f64 = 0.02;

/// The share of a farther band's pixels showing something from which it
/// counts as drawn.
pub const DRAWN_SHARE: f64 = 0.5;

/// The share of a band's pixels showing the finished picture from which the
/// band counts as finished.
pub const DONE_SHARE: f64 = 0.98;

/// How fine the ray march that places a pixel in the world steps, in blocks.
const BAND_RAY_STEP: f32 = 2.0;

/// How long the jump back may take to settle before the measurement gives up.
const RELOAD_TIMEOUT: Duration = Duration::from_secs(120);

/// The band the ground at `point` is drawn by once loaded: the level of the
/// node of `desired` (the ring schedule's nodes around the eye) holding it,
/// or [`FAR`] when none does.
pub fn band_of(point: [f32; 3], desired: &HashSet<NodeKey>) -> usize {
    let chunk = ChunkCoord::from_world_pos(point);
    (0..FAR as u32)
        .find(|&level| desired.contains(&NodeKey::containing(chunk, level)))
        .map_or(FAR, |level| level as usize)
}

/// The band of every pixel of `finished`, the frame an eye at `eye` looking
/// along `look` shows once loaded: [`SKY`] where it shows nothing, else the
/// band of the ground the pixel's ray first meets ([`surface_hit`] on the
/// generator's `height` field, out to `reach` blocks), else -- land in the
/// frame, none within `reach` -- [`FAR`].
///
/// The height field has no trees and no caves, so a pixel showing a crown in
/// front of a ring edge is placed with the ground behind it. Those are a few
/// pixels at an edge, not a band.
pub fn bands(
    finished: &Frame,
    eye: [f32; 3],
    look: [f32; 3],
    reach: f32,
    desired: &HashSet<NodeKey>,
    height: impl Fn(i32, i32) -> i32 + Sync,
) -> Vec<u8> {
    let (width, rows) = (finished.width, finished.height);
    let mut out = vec![SKY; (width * rows) as usize];
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per = (rows as usize).div_ceil(threads).max(1) * width as usize;
    let height = &height;
    std::thread::scope(|scope| {
        for (part, chunk) in out.chunks_mut(per.max(1)).enumerate() {
            scope.spawn(move || {
                for (i, band) in chunk.iter_mut().enumerate() {
                    let index = part * per + i;
                    if shows_nothing(&finished.pixels[index * 4..index * 4 + 4]) {
                        continue;
                    }
                    let (x, y) = (index as u32 % width, index as u32 / width);
                    let dir = pixel_ray(look, width, rows, x, y);
                    *band = match surface_hit(height, eye, dir, reach, BAND_RAY_STEP) {
                        Some(d) => {
                            let p = [0, 1, 2].map(|a| eye[a] + dir[a] * d);
                            band_of(p, desired) as u8
                        }
                        None => FAR as u8,
                    };
                }
            });
        }
    });
    out
}

/// One band of one frame, in pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BandShare {
    /// The band's pixels in the finished frame.
    pub pixels: u32,
    /// Of those, the ones showing nothing.
    pub hole: u32,
    /// Of those, the ones showing something more than [`TOLERANCE`] off the
    /// finished frame: a coarser level, or the far terrain, in its place.
    pub stand_in: u32,
}

impl BandShare {
    fn share(self, n: u32) -> f64 {
        n as f64 / self.pixels.max(1) as f64
    }

    pub fn hole_share(self) -> f64 {
        self.share(self.hole)
    }

    pub fn stand_in_share(self) -> f64 {
        self.share(self.stand_in)
    }

    /// The share showing anything at all.
    pub fn drawn_share(self) -> f64 {
        1.0 - self.hole_share()
    }

    /// The share showing the finished picture.
    pub fn done_share(self) -> f64 {
        1.0 - self.hole_share() - self.stand_in_share()
    }

    /// Enough pixels to say anything about ([`MIN_BAND_PIXELS`]).
    pub fn judged(self) -> bool {
        self.pixels >= MIN_BAND_PIXELS
    }
}

/// What each band of `live` shows against `finished`, the same eye loaded
/// completely; `bands` is [`bands`] of `finished`.
pub fn band_shares(live: &Frame, finished: &Frame, bands: &[u8]) -> [BandShare; BANDS] {
    let mut out = [BandShare::default(); BANDS];
    let pixels = live
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .zip(finished.pixels.as_chunks::<4>().0);
    for ((now, then), &band) in pixels.zip(bands) {
        let Some(share) = out.get_mut(band as usize) else {
            continue;
        };
        share.pixels += 1;
        if shows_nothing(now) {
            share.hole += 1;
        } else if (0..3).any(|c| now[c].abs_diff(then[c]) > TOLERANCE) {
            share.stand_in += 1;
        }
    }
    out
}

/// The owner's symptom in one frame: the nearest band that still has a hole
/// (more than [`HOLE_SHARE`] of it shows nothing) while a band farther away
/// is drawn (at least [`DRAWN_SHARE`] of it shows something), with that
/// farther band. `None` when no frame of land stands behind a hole.
pub fn outside_in(shares: &[BandShare; BANDS]) -> Option<(usize, usize)> {
    (0..BANDS).find_map(|near| {
        let n = shares[near];
        if !n.judged() || n.hole_share() <= HOLE_SHARE {
            return None;
        }
        (near + 1..BANDS)
            .find(|&far| shares[far].judged() && shares[far].drawn_share() >= DRAWN_SHARE)
            .map(|far| (near, far))
    })
}

/// A frame in which a farther band is finished ([`DONE_SHARE`]) while a
/// nearer one is not: detail arriving outside-in. Not a defect by itself --
/// a coarse stand-in near the camera under a finished horizon is what
/// progressive refinement looks like -- so it is reported, not judged.
pub fn refined_outside_in(shares: &[BandShare; BANDS]) -> bool {
    (0..BANDS).any(|near| {
        shares[near].judged()
            && shares[near].done_share() < DONE_SHARE
            && (near + 1..BANDS)
                .any(|far| shares[far].judged() && shares[far].done_share() >= DONE_SHARE)
    })
}

/// One frame after the jump back.
#[derive(Clone, Copy, Debug)]
pub struct Row {
    /// Seconds since the jump back.
    pub t: f32,
    pub shares: [BandShare; BANDS],
    /// Nodes drawn, per voxel level.
    pub drawn: [usize; 4],
    /// Nodes being meshed.
    pub meshing: usize,
    /// Nodes the last search saw.
    pub visible: usize,
    /// Steps waiting in the renderer's upload queue.
    pub uploads: usize,
}

/// What to measure: [`Flight`]'s eye, look, size, pace and quality, and how
/// far away the streaming eye goes before it jumps back.
#[derive(Clone, Debug)]
pub struct Reload {
    pub flight: Flight,
    /// How far off the eye jumps, in blocks, level along the look.
    pub away: f32,
}

/// What a jump back showed.
#[derive(Clone, Debug)]
pub struct ReloadReport {
    pub away: f32,
    /// Every frame from the jump back until nothing was left to load.
    pub rows: Vec<Row>,
    /// Whether it settled inside [`RELOAD_TIMEOUT`].
    pub settled: bool,
    /// The share of the last frame's land pixels still off the frame the
    /// same eye showed before it left: what the bands were measured against
    /// is what came back.
    pub end_differs: f64,
}

impl ReloadReport {
    /// The frames showing the owner's symptom ([`outside_in`]).
    pub fn outside_in_frames(&self) -> usize {
        self.rows
            .iter()
            .filter(|r| outside_in(&r.shares).is_some())
            .count()
    }

    /// When the last such frame was drawn, in seconds after the jump back:
    /// how long the layers were seen loading apart.
    pub fn outside_in_until(&self) -> Option<f32> {
        self.rows
            .iter()
            .rfind(|r| outside_in(&r.shares).is_some())
            .map(|r| r.t)
    }

    /// The frames in which detail was finished outside-in
    /// ([`refined_outside_in`]).
    pub fn refined_outside_in_frames(&self) -> usize {
        self.rows
            .iter()
            .filter(|r| refined_outside_in(&r.shares))
            .count()
    }

    /// The first moment from which `judge` holds for `band` in every later
    /// frame; `None` when the band is not judged or never got there.
    fn since(&self, band: usize, judge: impl Fn(BandShare) -> bool) -> Option<f32> {
        if !self.rows.last()?.shares[band].judged() {
            return None;
        }
        let bad = self.rows.iter().rposition(|r| !judge(r.shares[band]));
        match bad {
            None => self.rows.first().map(|r| r.t),
            Some(i) => self.rows.get(i + 1).map(|r| r.t),
        }
    }

    /// When `band` stopped showing holes ([`HOLE_SHARE`]) for good.
    pub fn covered_at(&self, band: usize) -> Option<f32> {
        self.since(band, |s| s.hole_share() <= HOLE_SHARE)
    }

    /// When `band` showed the finished picture ([`DONE_SHARE`]) for good.
    pub fn done_at(&self, band: usize) -> Option<f32> {
        self.since(band, |s| s.done_share() >= DONE_SHARE)
    }

    /// How many times the visible set changed: the search answers that
    /// moved the frontier.
    pub fn frontier_moves(&self) -> usize {
        self.rows
            .windows(2)
            .filter(|w| w[0].visible != w[1].visible)
            .count()
    }

    /// The frames, of those before the last voxel band was finished, in
    /// which no node was being meshed and none waited to upload: the mesh
    /// workers idle with the picture unfinished, waiting for a search to
    /// say what is seen next. Of how many.
    pub fn idle_frames(&self) -> (usize, usize) {
        let unfinished = |r: &&Row| {
            r.shares[..FAR]
                .iter()
                .any(|s| s.judged() && s.done_share() < DONE_SHARE)
        };
        let loading = self
            .rows
            .iter()
            .rposition(|r| unfinished(&r))
            .map_or(0, |i| i + 1);
        let idle = self.rows[..loading]
            .iter()
            .filter(|r| r.meshing == 0 && r.uploads == 0)
            .count();
        (idle, loading)
    }

    /// The result on one line, `RELOAD:` first, for a script to read.
    pub fn line(&self) -> String {
        let at = |t: Option<f32>| t.map_or("    -".to_string(), |t| format!("{t:5.2}"));
        let per_band = |f: &dyn Fn(usize) -> Option<f32>| {
            (0..BANDS)
                .map(|b| format!("{} {}", BAND_NAMES[b], at(f(b))))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let last = self.rows.last().map_or(0.0, |r| r.t);
        format!(
            "RELOAD: away {:.0} blocks: {} of {} frames outside-in (a nearer band over {:.0}% \
             hole behind a farther band over {:.0}% drawn), the last at {} s; {} frames with \
             detail finished outside-in; no holes from [s]: {}; finished from [s]: {}; {} in \
             {:.2} s ({} frontier moves, the mesh workers idle in {} of the {} frames before \
             the voxels were finished); {:.2}% of the last frame off the frame before the jump",
            self.away,
            self.outside_in_frames(),
            self.rows.len(),
            HOLE_SHARE * 100.0,
            DRAWN_SHARE * 100.0,
            at(self.outside_in_until()),
            self.refined_outside_in_frames(),
            per_band(&|b| self.covered_at(b)),
            per_band(&|b| self.done_at(b)),
            if self.settled {
                "settled"
            } else {
                "NOT settled"
            },
            last,
            self.frontier_moves(),
            self.idle_frames().0,
            self.idle_frames().1,
            self.end_differs * 100.0,
        )
    }

    /// One line of the timeline: the row nearest after `t`.
    fn timeline_row(row: &Row) -> String {
        let bands = (0..BANDS)
            .map(|b| {
                let s = row.shares[b];
                if s.judged() {
                    format!(
                        "{} {:3.0}/{:3.0}",
                        BAND_NAMES[b],
                        s.hole_share() * 100.0,
                        s.stand_in_share() * 100.0
                    )
                } else {
                    format!("{}   -/  -", BAND_NAMES[b])
                }
            })
            .collect::<Vec<_>>()
            .join(" | ");
        format!(
            "reload: t {:5.2} s  hole%/stand-in%: {bands}  drawn {:?}, {} meshing, {} seen, \
             {} uploads{}",
            row.t,
            row.drawn,
            row.meshing,
            row.visible,
            row.uploads,
            match outside_in(&row.shares) {
                Some((near, far)) => format!(
                    "  OUTSIDE-IN ({} behind {})",
                    BAND_NAMES[far], BAND_NAMES[near]
                ),
                None => String::new(),
            }
        )
    }

    /// The timeline, a row every `every` seconds.
    pub fn timeline(&self, every: f32) -> Vec<String> {
        let mut out = Vec::new();
        let mut next = 0.0;
        for row in &self.rows {
            if row.t >= next {
                out.push(Self::timeline_row(row));
                next = row.t + every;
            }
        }
        if let Some(last) = self.rows.last() {
            out.push(Self::timeline_row(last));
        }
        out
    }
}

/// The moments after the jump back a frame is written at with `--frames`.
const DUMP_AT: [f32; 8] = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0];

/// Load `reload.flight.start` completely, jump [`Reload::away`] blocks off
/// and let that settle, jump back, and measure every frame until nothing is
/// left to load. With `frames`, the finished frame and the frames at
/// [`DUMP_AT`] seconds are written there. `None` without a GPU.
pub fn reload(reload: &Reload, frames: Option<&Path>) -> Option<ReloadReport> {
    let flight = &reload.flight;
    let home = flight.start;
    let mut s = Streamed::new(flight, false)?;
    s.settle(flight, home);
    let finished = s.renderer.read_frame()?;

    let gen = cubara_world::WorldGen::new(s.world.seed());
    let center = ChunkCoord::from_world_pos(home);
    let desired: HashSet<NodeKey> =
        node::desired_nodes_3d(center, VERTICAL_LOD_SQUASH, node::DEFAULT_RING_SCHEDULE)
            .into_iter()
            .collect();
    // Far enough to leave the rings' box by any corner.
    let (min, max) = node::covered_box_3d(center, node::DEFAULT_RING_SCHEDULE);
    let reach = (0..3)
        .map(|a| ((max[a] - min[a]) * 16) as f32)
        .fold(0.0f32, |sum, side| sum + side * side)
        .sqrt();
    let bands = bands(&finished, home, flight.look, reach, &desired, |x, z| {
        gen.surface_height(x, z)
    });
    if let Some(dir) = frames {
        if let Err(e) = std::fs::create_dir_all(dir) {
            log::error!("reload: could not create {}: {e}", dir.display());
        }
        save(&dir.join("reload-finished.png"), &finished);
    }

    // Away: level along the look, and above the ground there.
    let along = glam::vec3(flight.look[0], 0.0, flight.look[2]).normalize_or(glam::Vec3::X);
    let off = glam::Vec3::from(home) + along * reload.away;
    let ground = gen.surface_height(off.x.floor() as i32, off.z.floor() as i32) as f32;
    let off = [off.x, off.y.max(ground + 10.0), off.z];
    s.far.reselect();
    s.settle(flight, off);
    let drawn_away = s.nodes.drawn_per_level();
    log::info!(
        "reload: away at {off:.0?}, settled with {drawn_away:?} nodes drawn per level; jumping \
         back to {home:.0?}"
    );

    let budget = Duration::from_secs_f64(1.0 / flight.fps.max(1) as f64);
    let start = Instant::now();
    let mut next_frame = start;
    let mut rows = Vec::new();
    let mut dumps = DUMP_AT.iter().copied().peekable();
    let mut settled = false;
    loop {
        let t = start.elapsed().as_secs_f32();
        s.frame(flight, home);
        let shown = s.renderer.read_frame()?;
        let (_, _, meshing, visible) = s.nodes.counts();
        rows.push(Row {
            t,
            shares: band_shares(&shown, &finished, &bands),
            drawn: s.nodes.drawn_per_level(),
            meshing,
            visible,
            uploads: s.renderer.uploads_pending(),
        });
        if let (Some(dir), Some(&at)) = (frames, dumps.peek()) {
            if t >= at {
                dumps.next();
                save(&dir.join(format!("reload-t{at:05.2}.png")), &shown);
            }
        }
        // The frame just drawn was drawn settled: it is the last one.
        if settled {
            break;
        }
        settled = s.settled();
        if start.elapsed() > RELOAD_TIMEOUT {
            log::warn!("reload: nothing settled in {RELOAD_TIMEOUT:?}");
            break;
        }
        next_frame += budget;
        let now = Instant::now();
        match next_frame.checked_duration_since(now) {
            Some(rest) => std::thread::sleep(rest),
            None => next_frame = now,
        }
    }
    let end = rows.last()?.shares;
    let land: u32 = end.iter().map(|s| s.pixels).sum();
    let off_end: u32 = end.iter().map(|s| s.hole + s.stand_in).sum();
    Some(ReloadReport {
        away: reload.away,
        rows,
        settled,
        end_differs: off_end as f64 / land.max(1) as f64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::far_streaming::FarQuality;

    const LAND: [u8; 4] = [60, 140, 60, 255];
    const OTHER: [u8; 4] = [160, 140, 60, 255];
    const NOTHING: [u8; 4] = [255, 0, 255, 255];

    fn frame(pixels: &[[u8; 4]]) -> Frame {
        Frame {
            width: pixels.len() as u32,
            height: 1,
            pixels: pixels.concat(),
        }
    }

    fn share(pixels: u32, hole: u32, stand_in: u32) -> BandShare {
        BandShare {
            pixels,
            hole,
            stand_in,
        }
    }

    #[test]
    fn a_point_is_in_the_band_of_the_ring_node_holding_it() {
        let center = ChunkCoord::new(0, 0, 0);
        let desired: HashSet<NodeKey> =
            node::desired_nodes_3d(center, VERTICAL_LOD_SQUASH, node::DEFAULT_RING_SCHEDULE)
                .into_iter()
                .collect();
        // The schedule: level 0 to 10 chunks, 1 to 18, 2 to 32, 3 to 64.
        assert_eq!(band_of([8.0, 8.0, 8.0], &desired), 0);
        assert_eq!(band_of([14.5 * 16.0, 8.0, 8.0], &desired), 1);
        assert_eq!(band_of([26.0 * 16.0, 8.0, 8.0], &desired), 2);
        assert_eq!(band_of([50.0 * 16.0, 8.0, 8.0], &desired), 3);
        assert_eq!(band_of([200.0 * 16.0, 8.0, 8.0], &desired), FAR);
    }

    #[test]
    fn a_pixel_is_a_hole_a_stand_in_or_finished_in_its_own_band() {
        // Bands: 0, 0, 1, 1, far, sky.
        let bands = [0, 0, 1, 1, FAR as u8, SKY];
        let finished = frame(&[LAND, LAND, LAND, LAND, LAND, NOTHING]);
        let live = frame(&[NOTHING, LAND, OTHER, LAND, LAND, NOTHING]);
        let shares = band_shares(&live, &finished, &bands);
        assert_eq!(shares[0], share(2, 1, 0), "one hole in band 0");
        assert_eq!(shares[1], share(2, 0, 1), "one stand-in in band 1");
        assert_eq!(shares[FAR], share(1, 0, 0), "the far band is finished");
        assert_eq!(shares[2], share(0, 0, 0));
        assert_eq!(shares[3], share(0, 0, 0), "sky is in no band");
        assert!((shares[0].hole_share() - 0.5).abs() < 1e-9);
        assert!((shares[1].done_share() - 0.5).abs() < 1e-9);
        assert!((shares[0].drawn_share() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn a_colour_within_the_tolerance_is_finished() {
        let near = [LAND[0] + TOLERANCE, LAND[1], LAND[2], 255];
        let off = [LAND[0] + TOLERANCE + 1, LAND[1], LAND[2], 255];
        let shares = band_shares(&frame(&[near, off]), &frame(&[LAND, LAND]), &[0, 0]);
        assert_eq!(shares[0], share(2, 0, 1));
    }

    #[test]
    fn a_hole_in_front_of_drawn_land_is_outside_in() {
        let mut shares = [share(1000, 0, 0); BANDS];
        assert_eq!(outside_in(&shares), None, "everything drawn");
        // Level 1 half missing, the far terrain standing behind it.
        shares[1] = share(1000, 500, 0);
        shares[2] = share(1000, 1000, 0);
        shares[3] = share(1000, 1000, 0);
        assert_eq!(outside_in(&shares), Some((1, FAR)));
        // Nothing drawn behind the hole: loading inside-out, not a symptom.
        shares[FAR] = share(1000, 1000, 0);
        assert_eq!(outside_in(&shares), None);
        // A stand-in is not a hole.
        shares[1] = share(1000, 0, 1000);
        shares[FAR] = share(1000, 0, 0);
        shares[2] = share(1000, 0, 1000);
        shares[3] = share(1000, 0, 1000);
        assert_eq!(outside_in(&shares), None);
    }

    #[test]
    fn the_thresholds_are_where_they_say() {
        let mut shares = [share(1000, 0, 0); BANDS];
        // Exactly the hole share is not yet a hole; one pixel more is.
        shares[0] = share(1000, 20, 0);
        assert_eq!(outside_in(&shares), None);
        shares[0] = share(1000, 21, 0);
        assert_eq!(outside_in(&shares), Some((0, 1)));
        // The band behind counts as drawn from half of it, and is passed
        // over below that for the next one out.
        shares[1] = share(1000, 501, 0);
        assert_eq!(outside_in(&shares), Some((0, 2)));
        shares[1] = share(1000, 500, 0);
        assert_eq!(outside_in(&shares), Some((0, 1)));
        // A sliver of a band is not judged, in front or behind.
        shares[0] = share(MIN_BAND_PIXELS - 1, MIN_BAND_PIXELS - 1, 0);
        shares[1] = share(1000, 0, 0);
        assert_eq!(outside_in(&shares), None);
        shares[0] = share(MIN_BAND_PIXELS, MIN_BAND_PIXELS, 0);
        assert_eq!(outside_in(&shares), Some((0, 1)));
        for band in shares.iter_mut().skip(1) {
            *band = share(MIN_BAND_PIXELS - 1, 0, 0);
        }
        assert_eq!(outside_in(&shares), None);
    }

    #[test]
    fn detail_finished_behind_a_stand_in_is_refined_outside_in() {
        let mut shares = [share(1000, 0, 0); BANDS];
        assert!(!refined_outside_in(&shares));
        shares[0] = share(1000, 0, 21);
        assert!(refined_outside_in(&shares));
        // Not when everything behind it is unfinished too.
        for band in shares.iter_mut().skip(1) {
            *band = share(1000, 0, 21);
        }
        assert!(!refined_outside_in(&shares));
    }

    fn row(t: f32, shares: [BandShare; BANDS], visible: usize) -> Row {
        Row {
            t,
            shares,
            drawn: [0; 4],
            meshing: 0,
            visible,
            uploads: 0,
        }
    }

    #[test]
    fn the_report_counts_the_frames_and_says_when_each_band_was_there() {
        let done = [share(1000, 0, 0); BANDS];
        let mut holed = done;
        holed[0] = share(1000, 900, 0);
        let mut coarse = done;
        coarse[0] = share(1000, 0, 900);
        let report = ReloadReport {
            away: 5000.0,
            rows: vec![
                row(0.0, holed, 1),
                row(0.1, holed, 5),
                row(0.2, coarse, 5),
                row(0.3, done, 9),
            ],
            settled: true,
            end_differs: 0.0,
        };
        assert_eq!(report.outside_in_frames(), 2);
        assert_eq!(report.outside_in_until(), Some(0.1));
        assert_eq!(report.refined_outside_in_frames(), 3);
        assert_eq!(report.covered_at(0), Some(0.2));
        assert_eq!(report.done_at(0), Some(0.3));
        assert_eq!(report.covered_at(FAR), Some(0.0));
        assert_eq!(report.frontier_moves(), 2);
        assert_eq!(report.idle_frames(), (3, 3), "the last frame is finished");
        let line = report.line();
        assert!(
            line.starts_with("RELOAD: away 5000 blocks: 2 of 4 frames outside-in"),
            "{line}"
        );
        assert!(line.contains("the last at  0.10 s"), "{line}");
        assert_eq!(report.timeline(0.15).len(), 3, "t 0, t 0.2 and the last");
    }

    #[test]
    fn a_band_that_never_got_there_has_no_time() {
        let mut holed = [share(1000, 0, 0); BANDS];
        holed[2] = share(1000, 900, 0);
        holed[3] = share(10, 10, 0);
        let report = ReloadReport {
            away: 1.0,
            rows: vec![row(0.0, holed, 0), row(0.1, holed, 0)],
            settled: false,
            end_differs: 0.0,
        };
        assert_eq!(report.covered_at(2), None, "still holed in the last frame");
        assert_eq!(report.covered_at(3), None, "a sliver is not judged");
        assert!(report.line().contains("NOT settled"));
    }

    /// The jump back the owner made: 5 km away and back, the game's own eye
    /// and look, at a small frame so it takes seconds.
    fn jump_back(away: f32) -> Reload {
        Reload {
            flight: Flight {
                start: [8.0, 120.0, 8.0],
                look: [1.0, -0.15, 0.3],
                speed: crate::flight::FLY_SPEED,
                seconds: 0.0,
                fps: 60,
                every: 0.25,
                size: (480, 270),
                quality: FarQuality::High,
                far_quad_scale: crate::far_streaming::FAR_QUAD_SCALE,
                morph: true,
            },
            away,
        }
    }

    /// The owner's observation of 2026-11-22, as a number: after a jump back
    /// from far away no frame may show a farther band drawn behind a hole in
    /// a nearer one. On main at `9ec668a` this fails: the far terrain is
    /// drawn from the first frame, around a hole the size of the voxel
    /// rings, which the voxels then fill from the camera outward over
    /// seconds. Needs a GPU; skipped loudly without one.
    #[test]
    #[ignore = "fails until the reload-order fix (issue TBD)"]
    fn a_jump_back_never_shows_a_farther_band_behind_a_hole() {
        let Some(report) = reload(&jump_back(5000.0), None) else {
            cubara_render::headless::skip_or_require_gpu(
                "a_jump_back_never_shows_a_farther_band_behind_a_hole",
            );
            return;
        };
        println!("{}", report.line());
        for line in report.timeline(0.25) {
            println!("{line}");
        }
        assert!(report.settled, "the jump back never settled");
        assert!(
            report.end_differs < 0.01,
            "what came back is not what was there: {:.2}% of the land differs",
            report.end_differs * 100.0
        );
        assert_eq!(
            report.outside_in_frames(),
            0,
            "{} frames showed a farther band drawn behind a hole in a nearer one, the last \
             {:.2} s after the jump back",
            report.outside_in_frames(),
            report.outside_in_until().unwrap_or(0.0)
        );
    }
}
