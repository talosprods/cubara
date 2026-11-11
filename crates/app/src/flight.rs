//! `--bench flight`: how much of what a player sees while flying is still
//! loading.
//!
//! The owner, 2026-09-27, after flying around: "veel inladen ... zou fijn zijn
//! als dat consistenter goed is". A frame rate says nothing about that, and
//! neither does a golden image of a world loaded up front. So this flies the
//! game's own streaming ([`NodeStreaming`], [`FarStreaming`]) and the game's
//! own renderer, pointed at a texture instead of a window
//! ([`Renderer::offscreen`]), along a straight line at flying speed in real
//! time -- and then flies the same line again, stopping at every sampled
//! point until nothing is left to load. The fraction of pixels that differ
//! between the two at the same eye is what was still loading.
//!
//! Two passes rather than one: settling a second world at each point while
//! the first flies would hand the first's mesh workers free time, and it
//! would look better than a player's.

use std::collections::{BTreeMap, VecDeque};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cubara_render::{CameraPose, Frame, Renderer};
use cubara_world::World;

use crate::far_streaming::{FarQuality, FarStreaming, FAR_VIEW_RADIUS};
use crate::streaming::{BlockBox, NodeStreaming};

/// Blocks per second: `cubara_sim::player`'s free-fly speed.
pub const FLY_SPEED: f32 = 24.0;

/// What frames clear to while measuring: magenta, a colour the land never
/// takes (the fog fades it into the sky's own colour), so a pixel showing it
/// is a pixel nothing was drawn on.
const NOTHING: wgpu::Color = wgpu::Color {
    r: 1.0,
    g: 0.0,
    b: 1.0,
    a: 1.0,
};

/// Whether an RGBA pixel shows [`NOTHING`].
fn shows_nothing(p: &[u8]) -> bool {
    p[0] >= 250 && p[1] <= 5 && p[2] >= 250
}

/// The longest run of frames, at the flight's 60 Hz, that a pixel may show
/// nothing between two frames of land and still count as a hole rather than
/// sky it really looks at: half a second. Longer gaps are caught by the kept
/// frames' comparison with the finished frame.
const HOLE_MAX_FRAMES: u32 = 30;

/// Finds hole **candidates** in a sequence of frames: pixels **inside** land
/// -- land on all four sides -- that show nothing for at most
/// [`HOLE_MAX_FRAMES`], then land again. The owner's "laag detail niveau, dan
/// is het weg, heel kort daarna laadt het op een hoger niveau".
///
/// Inside, because a silhouette is not a hole: a stepped ridge against the sky
/// slides across pixel boundaries as the eye moves, and a pixel on its edge
/// goes land, sky, land within a few frames without anything missing (seen in
/// the first measurement: every such pixel was on a skyline). Sky that is
/// really there is not counted either -- it does not go back to land within
/// half a second on a straight flight -- nor is a run still open when the
/// flight ends. Pure, so the rule is tested without a GPU.
///
/// A candidate, not a hole (issue #384 decision 1): "inside" is judged on the
/// *previous* frame, so a ridge whose silhouette drops two pixels at a detail
/// swap flips a row of pixels whose upper neighbour was land a frame ago --
/// a pop, with nothing missing. A candidate counts as a hole only when the
/// finished frame at that frame's eye shows land at the pixel
/// ([`confirm_holes`]).
pub struct HoleTracker {
    width: usize,
    /// Per pixel: whether it showed land in the last frame.
    land: Vec<bool>,
    /// Per pixel: the frame its current run of nothing began after land, or
    /// `u32::MAX`.
    run_start: Vec<u32>,
    frame: u32,
    /// Per frame: the pixels (indices into the frame) that were candidates
    /// in it.
    pub candidates: Vec<Vec<u32>>,
}

impl HoleTracker {
    /// For frames `width` pixels wide and `pixels` in all.
    pub fn new(width: usize, pixels: usize) -> Self {
        Self {
            width: width.max(1),
            land: vec![false; pixels],
            run_start: vec![u32::MAX; pixels],
            frame: 0,
            candidates: Vec::new(),
        }
    }

    /// Take the next frame: per pixel, whether it shows nothing. Returns the
    /// frames that got their first candidate from this one -- a run closes
    /// when land returns, so they are earlier frames, at most
    /// [`HOLE_MAX_FRAMES`] back -- in order, so the caller can keep those
    /// frames for the finished-frame check.
    pub fn push(&mut self, nothing: impl IntoIterator<Item = bool>) -> Vec<u32> {
        self.candidates.push(Vec::new());
        let before = self.land.clone();
        let (w, n) = (self.width, before.len());
        // Land last frame on all four sides: inside land, not on its edge.
        // The screen's border has a side off screen, so it never counts.
        let inside = |i: usize| {
            let x = i % w;
            x > 0
                && x + 1 < w
                && i >= w
                && i + w < n
                && before[i - 1]
                && before[i + 1]
                && before[i - w]
                && before[i + w]
        };
        let mut newly = Vec::new();
        for (i, empty) in nothing.into_iter().enumerate() {
            if empty {
                if before[i] && inside(i) {
                    self.run_start[i] = self.frame;
                }
                self.land[i] = false;
            } else {
                let start = self.run_start[i];
                if start != u32::MAX && self.frame - start <= HOLE_MAX_FRAMES {
                    for f in start..self.frame {
                        let in_frame = &mut self.candidates[f as usize];
                        if in_frame.is_empty() {
                            newly.push(f);
                        }
                        in_frame.push(i as u32);
                    }
                }
                self.run_start[i] = u32::MAX;
                self.land[i] = true;
            }
        }
        self.frame += 1;
        newly.sort_unstable();
        newly.dedup();
        newly
    }

    /// Per frame, how many candidates it had.
    pub fn counts(&self) -> Vec<u32> {
        self.candidates.iter().map(|c| c.len() as u32).collect()
    }
}

/// Of `candidates` (pixel indices of one frame, [`HoleTracker`]), the ones
/// that are holes: the finished frame at that frame's eye shows land there
/// (issue #384 decision 1). A candidate the finished frame shows sky at was
/// a silhouette moving -- a pop -- not geometry missing.
pub fn confirm_holes(candidates: &[u32], finished: &Frame) -> Vec<u32> {
    let px = finished.pixels.as_chunks::<4>().0;
    candidates
        .iter()
        .copied()
        .filter(|&i| px.get(i as usize).is_some_and(|p| !shows_nothing(&p[..])))
        .collect()
}

/// A box of pixels on the screen, `x1`/`y1` exclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScreenBox {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

impl ScreenBox {
    pub fn contains(&self, x: u32, y: u32) -> bool {
        (self.x0..self.x1).contains(&x) && (self.y0..self.y1).contains(&y)
    }
}

/// How far in front of the eye, in clip-space `w` (blocks along the look),
/// a point must be to project: the near plane [`screen_box`] cuts a box's
/// edges at.
const SCREEN_BOX_NEAR: f32 = 1e-3;

/// Where the world-space box `min..max` lands on a `width` x `height` screen
/// seen through `view_proj`: the screen box around its projected corners,
/// cut to the screen. The part behind the eye projects to nowhere sensible,
/// so the box's edges are first cut at the near plane
/// ([`SCREEN_BOX_NEAR`]) and the cut points projected too: a box around the
/// eye takes the whole screen, one beside the eye only the part in front.
/// `None` when nothing of it is on screen.
pub fn screen_box(
    min: [f32; 3],
    max: [f32; 3],
    view_proj: glam::Mat4,
    width: u32,
    height: u32,
) -> Option<ScreenBox> {
    let (w, h) = (width as f32, height as f32);
    let corners: Vec<glam::Vec4> = (0..8)
        .map(|corner: usize| {
            let pick = |axis: usize| {
                if corner & (1 << axis) == 0 {
                    min[axis]
                } else {
                    max[axis]
                }
            };
            view_proj * glam::vec4(pick(0), pick(1), pick(2), 1.0)
        })
        .collect();
    let in_front = |p: &glam::Vec4| p.w > SCREEN_BOX_NEAR;
    let mut points: Vec<glam::Vec4> = corners.iter().copied().filter(in_front).collect();
    // Each edge joins two corners that differ in one axis' bit; where an
    // edge crosses the near plane, the crossing (linear in clip space)
    // stands in for the corner behind.
    for a in 0..8 {
        for axis in 0..3 {
            let b = a | (1 << axis);
            if b == a || in_front(&corners[a]) == in_front(&corners[b]) {
                continue;
            }
            let (pa, pb) = (corners[a], corners[b]);
            let s = (pa.w - SCREEN_BOX_NEAR) / (pa.w - pb.w);
            points.push(pa + (pb - pa) * s);
        }
    }
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for p in &points {
        let sx = (p.x / p.w + 1.0) * 0.5 * w;
        let sy = (1.0 - p.y / p.w) * 0.5 * h;
        x0 = x0.min(sx);
        y0 = y0.min(sy);
        x1 = x1.max(sx);
        y1 = y1.max(sy);
    }
    let b = ScreenBox {
        x0: x0.floor().clamp(0.0, w) as u32,
        y0: y0.floor().clamp(0.0, h) as u32,
        x1: x1.ceil().clamp(0.0, w) as u32,
        y1: y1.ceil().clamp(0.0, h) as u32,
    };
    (!points.is_empty() && b.x0 < b.x1 && b.y0 < b.y1).then_some(b)
}

/// Per pixel of a `width` x `height` frame, whether it lies inside any of
/// `boxes` -- so a pixel under two boxes is visited once.
fn coverage(boxes: &[ScreenBox], width: u32, height: u32) -> Vec<bool> {
    let mut mask = vec![false; (width * height) as usize];
    for b in boxes {
        for y in b.y0.min(height)..b.y1.min(height) {
            let row = (y * width) as usize;
            mask[row + b.x0.min(width) as usize..row + b.x1.min(width) as usize].fill(true);
        }
    }
    mask
}

/// An RGBA pixel's luminance (Rec. 601), 0..=255.
fn luminance(p: &[u8]) -> u8 {
    ((299 * p[0] as u32 + 587 * p[1] as u32 + 114 * p[2] as u32) / 1000) as u8
}

/// The goldens' per-pixel tolerance (`crates/render/tests/golden.rs`): a
/// luminance change up to this between two frames is noise, not a pop.
pub const POP_TOLERANCE: u8 = 12;

/// Pixels inside any of `boxes` whose luminance changes by more than
/// `tolerance` from `before` to `after`, or that go between land and
/// nothing -- the first half of a pop (issue #384 decision 2). Pure: two
/// frames and the boxes.
///
/// The land/nothing flip is counted on its own because the flight clears to
/// magenta ([`NOTHING`]), whose luminance (105) is that of mid-green grass:
/// a ridge giving way to the clear colour would otherwise be invisible to a
/// luminance test, and that is the pop the owner sees.
pub fn changed_pixels(before: &Frame, after: &Frame, boxes: &[ScreenBox], tolerance: u8) -> u32 {
    let (b, a) = (
        before.pixels.as_chunks::<4>().0,
        after.pixels.as_chunks::<4>().0,
    );
    let inside = coverage(boxes, after.width, after.height);
    a.iter()
        .zip(b)
        .zip(inside)
        .filter(|((pa, pb), inside)| {
            *inside
                && (shows_nothing(&pa[..]) != shows_nothing(&pb[..])
                    || luminance(&pa[..]).abs_diff(luminance(&pb[..])) > tolerance)
        })
        .count() as u32
}

/// The largest vertical shift, in pixels, of the land/sky boundary -- the
/// topmost row showing land in a column -- inside any of `boxes`, from
/// `before` to `after`: the second half of a pop (issue #384 decision 2), a
/// ridge's silhouette dropping or rising at a detail swap. A column with no
/// land in one of the two frames inside the box has no boundary there and
/// is left to [`changed_pixels`].
pub fn silhouette_shift(before: &Frame, after: &Frame, boxes: &[ScreenBox]) -> u32 {
    let w = after.width as usize;
    let skyline = |f: &Frame, x: u32, bx: &ScreenBox| {
        let px = f.pixels.as_chunks::<4>().0;
        (bx.y0..bx.y1).find(|&y| !shows_nothing(&px[y as usize * w + x as usize][..]))
    };
    boxes
        .iter()
        .flat_map(|bx| {
            (bx.x0..bx.x1)
                .filter_map(move |x| Some(skyline(before, x, bx)?.abs_diff(skyline(after, x, bx)?)))
        })
        .max()
        .unwrap_or(0)
}

/// One swap's pop: what changed on screen inside the swapped nodes' boxes
/// between the frame before and the frame of the swap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pop {
    /// The frame of the real-time pass the swap was applied in.
    pub frame: u32,
    /// Nodes that left in it.
    pub nodes: usize,
    /// [`changed_pixels`].
    pub pixels: u32,
    /// [`silhouette_shift`].
    pub silhouette: u32,
}

/// The pops of one flight, summarised.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PopStats {
    pub count: usize,
    pub pixels_p50: u32,
    pub pixels_p99: u32,
    pub pixels_max: u32,
    pub silhouette_p50: u32,
    pub silhouette_p99: u32,
    pub silhouette_max: u32,
}

impl PopStats {
    pub fn of(pops: &[Pop]) -> Self {
        if pops.is_empty() {
            return Self::default();
        }
        let dist = |pick: fn(&Pop) -> u32| {
            let mut sorted: Vec<u32> = pops.iter().map(pick).collect();
            sorted.sort_unstable();
            let at = |per_mille: usize| sorted[(sorted.len() - 1) * per_mille / 1000];
            (at(500), at(990), *sorted.last().unwrap())
        };
        let (pixels_p50, pixels_p99, pixels_max) = dist(|p| p.pixels);
        let (silhouette_p50, silhouette_p99, silhouette_max) = dist(|p| p.silhouette);
        Self {
            count: pops.len(),
            pixels_p50,
            pixels_p99,
            pixels_max,
            silhouette_p50,
            silhouette_p99,
            silhouette_max,
        }
    }
}

/// A pixel counts as still loading when a channel is off by more than this
/// from the finished frame. Far above what a far patch one level coarser or
/// a node one level off moves a pixel by on open ground, far below ground
/// against sky.
const TOLERANCE: u8 = 40;

/// What to fly.
#[derive(Clone, Debug)]
pub struct Flight {
    pub start: [f32; 3],
    /// The direction flown, which is also the one looked in.
    pub look: [f32; 3],
    pub speed: f32,
    pub seconds: f32,
    /// Frames per second the flight is paced at: a monitor's refresh rate,
    /// since the renderer uploads a fixed number of meshes per frame.
    pub fps: u32,
    /// How often a frame is kept to compare, in seconds of flight.
    pub every: f32,
    pub size: (u32, u32),
    pub quality: FarQuality,
}

/// Where the eye is `t` seconds into `flight`.
pub fn eye_at(flight: &Flight, t: f32) -> [f32; 3] {
    let dir = glam::Vec3::from(flight.look).normalize_or_zero();
    (glam::Vec3::from(flight.start) + dir * flight.speed * t).to_array()
}

/// How far from spawn [`toward_peak`] looks for a mountain, in blocks.
const PEAK_SEARCH: i32 = 4096;

/// How far before the peak the flight toward it starts, in blocks: inside the
/// far terrain, so the flight crosses the voxel rings' edges on the way in.
const PEAK_APPROACH: f32 = 900.0;

/// The owner's flight toward a mountain: the tallest surface within
/// [`PEAK_SEARCH`] of spawn, approached level at its summit's height from
/// [`PEAK_APPROACH`] away on the side facing spawn. Returns the start and the
/// direction. Deterministic for a seed: a 32-block grid, ties to the first.
pub fn toward_peak(gen: &cubara_world::WorldGen) -> ([f32; 3], [f32; 3]) {
    let mut best = (i32::MIN, 0, 0);
    for x in (-PEAK_SEARCH..=PEAK_SEARCH).step_by(32) {
        for z in (-PEAK_SEARCH..=PEAK_SEARCH).step_by(32) {
            let h = gen.surface_height(x, z);
            if h > best.0 {
                best = (h, x, z);
            }
        }
    }
    let (h, px, pz) = best;
    let peak = glam::vec2(px as f32, pz as f32);
    let away = (-peak).normalize_or(glam::Vec2::X);
    let start = peak + away * PEAK_APPROACH;
    let y = h as f32 + 3.0;
    let look = glam::vec3(peak.x - start.x, 0.0, peak.y - start.y).normalize();
    log::info!("flight: toward the peak at ({px}, {h}, {pz}), from {start:.0}");
    ([start.x, y, start.y], look.to_array())
}

/// The fraction of `live`'s pixels more than [`TOLERANCE`] off `settled`'s.
pub fn loading_fraction(live: &Frame, settled: &Frame) -> f64 {
    cubara_render::headless::compare(&live.pixels, &settled.pixels, TOLERANCE).differing_fraction
}

/// One kept frame: when, where, and how much of it was still loading.
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub t: f32,
    pub eye: [f32; 3],
    pub loading: f64,
    /// The share of its pixels that showed nothing where the finished frame
    /// shows land: stale terrain ([`stale_pixels`]).
    pub stale: f64,
    /// The share of its pixels that are stale *and* lie inside the screen box
    /// of a node whose swap lag was open on that frame ([`attribute_stale`],
    /// share (a)).
    pub stale_late: f64,
    /// The share of its pixels that are stale, under no open swap, and inside
    /// the screen box of a node the search could see that was not resident
    /// yet ([`attribute_stale`], share (b)): land still loading, with the far
    /// terrain or a coarser node standing in.
    pub stale_loading: f64,
}

/// Where a kept frame's stale pixels fall (issue #398 decision 1): under a
/// node whose swap lag is open (`late`, share (a)); else under a node the
/// search could see that is not resident yet (`loading`, share (b)); else
/// nowhere this measurement can name (`other`, share (c)) -- the far
/// terrain's own error at the join, for one. Counts of pixels; the three sum
/// to the stale pixels given.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StaleShares {
    pub late: usize,
    pub loading: usize,
    pub other: usize,
}

/// Attribute `pixels` (indices into a frame `width` wide, [`stale_pixels`])
/// to the three shares of [`StaleShares`], when `late` are the screen boxes
/// of the nodes whose swap lag was open and `loading` those of the visible
/// nodes not yet resident. A pixel under both counts as late: a held node
/// and the children replacing it cover the same ground, so every late pixel
/// is under a loading box too, and the swap is the lever that moves it.
///
/// A pixel under neither box is asked of `in_strip`: whether what it looks
/// at lies where the voxel nodes should draw but do not yet
/// ([`in_voxel_strip`], the far terrain standing in) -- loading too, by a
/// node no list names yet because the search for the eye's chunk has not
/// answered. Only those pixels are asked, so the caller can afford a ray
/// march per answer.
pub fn attribute_stale(
    pixels: &[u32],
    width: u32,
    late: &[ScreenBox],
    loading: &[ScreenBox],
    mut in_strip: impl FnMut(u32) -> bool,
) -> StaleShares {
    let mut shares = StaleShares::default();
    for &i in pixels {
        let (x, y) = (i % width, i / width);
        let under = |boxes: &[ScreenBox]| boxes.iter().any(|b| b.contains(x, y));
        if under(late) {
            shares.late += 1;
        } else if under(loading) || in_strip(i) {
            shares.loading += 1;
        } else {
            shares.other += 1;
        }
    }
    shares
}

/// Whether the point `p` (blocks) lies inside the box `want` the voxel nodes
/// should draw and outside the box `hole` they draw now
/// (`NodeStreaming::far_strip`): the strip the far terrain stands in for
/// until the nodes there are drawn. Boxes are `max` exclusive.
pub fn in_voxel_strip(p: [f32; 3], hole: BlockBox, want: BlockBox) -> bool {
    in_box(p, want) && !in_box(p, hole)
}

/// Whether `p` lies in the box, `max` exclusive.
fn in_box(p: [f32; 3], (min, max): BlockBox) -> bool {
    (0..3).all(|a| min[a] <= p[a] && p[a] < max[a])
}

/// Where a stale pixel's ray met the ground: how far along it, and the
/// point, in blocks.
pub type Hit = (f32, [f32; 3]);

/// Where share (c)'s pixels look, from their rays' hits ([`surface_hit`]):
/// how many meet ground beyond `reach`, the box the voxel nodes cover (the
/// far terrain's), how many meet nothing (sky), and the nearest and farthest
/// hit. What says whether (c) is the far terrain's (issue #398 decision 3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FarLook {
    pub far: usize,
    pub sky: usize,
    pub nearest: f32,
    pub farthest: f32,
}

impl FarLook {
    pub fn of(hits: impl IntoIterator<Item = Option<Hit>>, reach: BlockBox) -> Self {
        let mut look = Self {
            far: 0,
            sky: 0,
            nearest: f32::MAX,
            farthest: 0.0,
        };
        for hit in hits {
            match hit {
                Some((d, p)) => {
                    look.nearest = look.nearest.min(d);
                    look.farthest = look.farthest.max(d);
                    if !in_box(p, reach) {
                        look.far += 1;
                    }
                }
                None => look.sky += 1,
            }
        }
        look
    }
}

/// What was still loading when a kept frame was drawn (issue #398).
#[derive(Clone, Debug, Default)]
struct Loading {
    /// `NodeStreaming::loading_nodes`: boxes in blocks.
    nodes: Vec<BlockBox>,
    /// `NodeStreaming::far_strip`: the hole the far terrain leaves and the
    /// box the voxels want, while they differ.
    strip: Option<(BlockBox, BlockBox)>,
    /// `NodeStreaming::search_behind`.
    search_behind: bool,
}

/// The pixels (indices) where `live` shows nothing and `settled` shows land
/// on the pixel and on all four sides of it: **stale terrain** (issue #384
/// decision 3) -- a coarse node still drawn where the finished frame has the
/// finer one's land, its lower silhouette showing sky. Inside land, so a
/// skyline drawn a pixel differently (the live pass's far terrain was
/// selected a little behind the eye) is not counted. See [`HoleTracker`].
pub fn stale_pixels(live: &Frame, settled: &Frame) -> Vec<u32> {
    let w = settled.width as usize;
    let land: Vec<bool> = settled
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| !shows_nothing(&p[..]))
        .collect();
    let n = land.len();
    let inside = |i: usize| {
        !i.is_multiple_of(w)
            && i % w + 1 < w
            && i >= w
            && i + w < n
            && land[i]
            && land[i - 1]
            && land[i + 1]
            && land[i - w]
            && land[i + w]
    };
    live.pixels
        .as_chunks::<4>()
        .0
        .iter()
        .enumerate()
        .filter(|(i, l)| shows_nothing(&l[..]) && inside(*i))
        .map(|(i, _)| i as u32)
        .collect()
}

/// How many of `pixels` (indices into a frame `width` wide) lie inside any
/// of `boxes`: the stale pixels under a late swap (issue #384 decision 3),
/// when `boxes` are the screen boxes of the nodes whose swap lag was open.
pub fn under_boxes(pixels: &[u32], width: u32, boxes: &[ScreenBox]) -> usize {
    pixels
        .iter()
        .filter(|&&i| {
            let (x, y) = (i % width, i / width);
            boxes.iter().any(|b| b.contains(x, y))
        })
        .count()
}

/// The direction through the centre of pixel `(x, y)` of a `width` x
/// `height` screen seen from a camera looking along `look` -- the renderer's
/// own projection: a 60-degree vertical field of view, `Y` up. Normalised.
pub fn pixel_ray(look: [f32; 3], width: u32, height: u32, x: u32, y: u32) -> [f32; 3] {
    let forward = glam::Vec3::from(look).normalize_or(glam::Vec3::X);
    let right = forward.cross(glam::Vec3::Y).normalize_or(glam::Vec3::Z);
    let up = right.cross(forward);
    let aspect = width as f32 / height.max(1) as f32;
    let t = 30f32.to_radians().tan();
    let ndc_x = (x as f32 + 0.5) / width.max(1) as f32 * 2.0 - 1.0;
    let ndc_y = 1.0 - (y as f32 + 0.5) / height.max(1) as f32 * 2.0;
    (forward + right * (ndc_x * t * aspect) + up * (ndc_y * t))
        .normalize()
        .to_array()
}

/// How far along `dir` from `eye` the terrain surface is first met, marching
/// `step` blocks at a time out to `max`: the first sample at or below the
/// surface (`height(x, z)` is the surface block's top, so a sample under
/// `height + 1` is in the ground). `None` when the ray leaves `max` without
/// touching ground -- sky. For placing a stale pixel in the world (issue
/// #398): the generator's height field, so no GPU and no streaming state.
pub fn surface_hit(
    height: impl Fn(i32, i32) -> i32,
    eye: [f32; 3],
    dir: [f32; 3],
    max: f32,
    step: f32,
) -> Option<f32> {
    let (eye, dir) = (glam::Vec3::from(eye), glam::Vec3::from(dir));
    let step = step.max(1e-3);
    let mut d = 0.0;
    while d <= max {
        let p = eye + dir * d;
        let h = height(p.x.floor() as i32, p.z.floor() as i32);
        if p.y < h as f32 + 1.0 {
            return Some(d);
        }
        d += step;
    }
    None
}

/// Draw a red ring, two pixels thick, around the pixels at `indices` (into
/// a frame `frame.width` wide) -- 4 to 5 pixels outside their bounding box
/// -- so a frame dump shows where the counted pixels are without covering
/// them, or each other.
pub fn mark_pixels(frame: &mut Frame, indices: &[u32]) {
    let (w, h) = (frame.width as i64, frame.height as i64);
    let Some(&first) = indices.first() else {
        return;
    };
    let at = |i: u32| (i as i64 % w, i as i64 / w);
    let (mut x0, mut y0) = at(first);
    let (mut x1, mut y1) = (x0, y0);
    for &i in indices {
        let (x, y) = at(i);
        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
    }
    for py in y0 - 5..=y1 + 5 {
        for px in x0 - 5..=x1 + 5 {
            let outside = (x0 - px).max(px - x1).max(y0 - py).max(py - y1);
            if outside < 4 || !(0..w).contains(&px) || !(0..h).contains(&py) {
                continue;
            }
            let i = ((py * w + px) * 4) as usize;
            frame.pixels[i..i + 4].copy_from_slice(&[255, 0, 0, 255]);
        }
    }
}

/// The swap-lag distribution for one flight (issue #378): how long a node
/// that should have been swapped for a different level of detail kept being
/// drawn. `NodeStreaming` only counts frames of its own `update` calls (rule
/// 2: no clock inside it); this is where those become milliseconds, with
/// the flight's own pacing, and blocks travelled, with its speed -- the two
/// units decision 3 asks the report to carry.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SwapLagStats {
    pub count: usize,
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
    pub p50_blocks: f64,
    pub p90_blocks: f64,
    pub p99_blocks: f64,
    pub max_blocks: f64,
}

impl SwapLagStats {
    /// `lags` is [`crate::streaming::NodeStreaming::swap_lags`]'s frame
    /// counts; `frames_per_second` and `speed` (blocks/s) convert a frame
    /// count to time and distance the way the flight itself paced and moved.
    /// The rate is the one the flight *achieved*, frames drawn over seconds
    /// flown -- not the one it asked for: a frame that runs late is not made
    /// up (`fly`'s pacing), so under load a frame is longer than 1/`fps` and
    /// the nominal rate would understate the lag.
    pub fn of(lags: &[u64], frames_per_second: f64, speed: f32) -> Self {
        if lags.is_empty() {
            return Self::default();
        }
        let mut sorted = lags.to_vec();
        sorted.sort_unstable();
        let at = |per_mille: usize| sorted[(sorted.len() - 1) * per_mille / 1000] as f64;
        let rate = frames_per_second.max(1e-3);
        let ms_per_frame = 1000.0 / rate;
        let blocks_per_frame = speed as f64 / rate;
        let (p50, p90, p99, max) = (at(500), at(900), at(990), *sorted.last().unwrap() as f64);
        Self {
            count: sorted.len(),
            p50_ms: p50 * ms_per_frame,
            p90_ms: p90 * ms_per_frame,
            p99_ms: p99 * ms_per_frame,
            max_ms: max * ms_per_frame,
            p50_blocks: p50 * blocks_per_frame,
            p90_blocks: p90 * blocks_per_frame,
            p99_blocks: p99 * blocks_per_frame,
            max_blocks: max * blocks_per_frame,
        }
    }
}

/// What a flight found.
#[derive(Debug)]
pub struct FlightReport {
    pub samples: Vec<Sample>,
    /// The part of each frame of the real-time pass the game spent streaming
    /// and drawing, in seconds. Not the whole frame: the rest is this
    /// measurement's sleep standing in for a display, and a sleep wakes when
    /// it likes -- about 4 ms late at p99 on an idle M3.
    pub work_times: Vec<f32>,
    /// Per frame of the real-time pass, the pixels that were holes: candidates
    /// ([`HoleTracker`]) the finished frame confirmed ([`confirm_holes`]).
    pub holes: Vec<u32>,
    /// Per frame, how many of [`holes`](Self::holes) lay inside the screen
    /// box of a node whose swap lag was open on that frame -- the same
    /// attribution as the stale term's, so a "hole" that is a late swap's
    /// silhouette can be told from one that is not.
    pub holes_late: Vec<u32>,
    /// Per frame of the real-time pass, the candidates before confirmation.
    pub candidates: Vec<u32>,
    /// Every how-manieth candidate frame was checked against the finished
    /// frame (issue #384 decision 1): 1 is all of them.
    pub checked_every: usize,
    /// Pixels in a frame.
    pub pixels: usize,
    /// Every swap of the real-time pass, with what it changed on screen.
    pub pops: Vec<Pop>,
    /// How long a node kept being drawn after it should have swapped for a
    /// different level of detail (issue #378) -- the live pass only; the
    /// settled pass jumps around and is not what a player experiences.
    pub swap_lags: SwapLagStats,
    /// How often the world was copied while this ran, and what one copy
    /// weighs (issue #387).
    pub world_clones: WorldClones,
}

/// `World` clones over a run: the count, what one clone copies, and the run's
/// length -- the `clones/s` and `bytes/s` of issue #387, which asked for them
/// to be measured before the snapshot was redesigned. A clone is paid by an
/// `Arc::make_mut` that finds its `Arc` shared, so the count says how often
/// the tick met a mesh job's held snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WorldClones {
    pub clones: u64,
    pub bytes_each: usize,
    pub seconds: f32,
}

impl WorldClones {
    pub fn clones_per_second(&self) -> f64 {
        self.clones as f64 / f64::from(self.seconds.max(f32::EPSILON))
    }

    pub fn bytes_per_second(&self) -> f64 {
        self.clones_per_second() * self.bytes_each as f64
    }

    /// The `WORLD CLONES` term of a bench line.
    pub fn term(&self) -> String {
        format!(
            "WORLD CLONES: {} in {:.1} s = {:.1} clones/s, {:.0} bytes/s ({} bytes each)",
            self.clones,
            self.seconds,
            self.clones_per_second(),
            self.bytes_per_second(),
            self.bytes_each,
        )
    }
}

impl FlightReport {
    /// The share of kept frames with more than `fraction` of the screen still
    /// loading.
    pub fn frames_over(&self, fraction: f64) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let over = self.samples.iter().filter(|s| s.loading > fraction).count();
        over as f64 / self.samples.len() as f64
    }

    /// The mean fraction of the screen still loading.
    pub fn mean(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().map(|s| s.loading).sum::<f64>() / self.samples.len() as f64
    }

    /// The worst kept frame.
    pub fn worst(&self) -> Option<Sample> {
        self.samples
            .iter()
            .copied()
            .max_by(|a, b| a.loading.total_cmp(&b.loading))
    }

    /// The summary line: the frame terms, then the three terms gate item 0
    /// reads (issue #384) -- `HOLES:` (geometry missing, confirmed by the
    /// finished frame), `POPS:` (what a detail swap changed on screen) and
    /// `STALE:` (a kept frame showing nothing where the finished frame has
    /// land, and its three shares, issue #398: under a swap that was running
    /// late, under a node still loading, and under neither) -- and the swap
    /// lag. `scripts/check-phase-gate.sh` parses it; change a
    /// term's wording and the script's pattern together.
    pub fn line(&self) -> String {
        let worst = self.worst().unwrap_or(Sample {
            t: 0.0,
            eye: [0.0; 3],
            loading: 0.0,
            stale: 0.0,
            stale_late: 0.0,
            stale_loading: 0.0,
        });
        let tail = WorkTail::of(&self.work_times);
        let (any, worst_px) = self.hole_frames();
        let candidate_frames = self.candidates.iter().filter(|&&n| n > 0).count();
        let checked = candidate_frames.div_ceil(self.checked_every.max(1));
        let sampled = if self.checked_every > 1 {
            format!(", sampled 1 in {}", self.checked_every)
        } else {
            String::new()
        };
        let stale_frames = self.samples.iter().filter(|s| s.stale > 0.0).count();
        let stale_worst = self.samples.iter().map(|s| s.stale).fold(0.0, f64::max);
        let p = PopStats::of(&self.pops);
        let s = &self.swap_lags;
        format!(
            "FLIGHT: loading mean {:.2}% worst {:.2}% (at {:.2} s, {:.0?}), \
             frames over 1%: {:.0}%, work p99 {:.1} ms p99.9 {:.1} ms max {:.1} ms, \
             over 16.7 ms {} over 33.3 ms {} ({} kept frames) | \
             HOLES: {any} of {} frames show a hole, worst {:.3}%, of which under a \
             late swap {:.0}% ({candidate_frames} candidate frames, {checked} checked \
             against the finished frame{sampled}) | \
             POPS: {} swaps, pixels p50/p99/max {}/{}/{}, \
             silhouette px p50/p99/max {}/{}/{} | \
             STALE: {stale_frames} kept frames with nothing where the finished frame \
             has land, worst {:.3}%, of which under a late swap {:.0}%, under a node \
             still loading {:.0}%, neither {:.0}% | \
             SWAP LAG: n {}, p50/p90/p99/max {:.0}/{:.0}/{:.0}/{:.0} ms \
             ({:.2}/{:.2}/{:.2}/{:.2} blocks) | {}",
            self.mean() * 100.0,
            worst.loading * 100.0,
            worst.t,
            worst.eye,
            self.frames_over(0.01) * 100.0,
            tail.p99 * 1000.0,
            tail.p999 * 1000.0,
            tail.max * 1000.0,
            tail.over_one,
            tail.over_two,
            self.samples.len(),
            self.holes.len(),
            worst_px as f64 * 100.0 / self.pixels.max(1) as f64,
            self.holes_under_late_swap() * 100.0,
            p.count,
            p.pixels_p50,
            p.pixels_p99,
            p.pixels_max,
            p.silhouette_p50,
            p.silhouette_p99,
            p.silhouette_max,
            stale_worst * 100.0,
            self.stale_under_late_swap() * 100.0,
            self.stale_under_loading() * 100.0,
            self.stale_under_neither() * 100.0,
            s.count,
            s.p50_ms,
            s.p90_ms,
            s.p99_ms,
            s.max_ms,
            s.p50_blocks,
            s.p90_blocks,
            s.p99_blocks,
            s.max_blocks,
            self.world_clones.term(),
        )
    }

    /// Frames of the real-time pass with any confirmed hole pixel, and the
    /// most hole pixels in one frame.
    pub fn hole_frames(&self) -> (usize, u32) {
        (
            self.holes.iter().filter(|&&n| n > 0).count(),
            self.holes.iter().copied().max().unwrap_or(0),
        )
    }

    /// Of all the confirmed hole pixels, the share that lay inside the screen
    /// box of a node whose swap lag was open on that frame. Zero when there
    /// were none.
    pub fn holes_under_late_swap(&self) -> f64 {
        let holes: u64 = self.holes.iter().map(|&n| n as u64).sum();
        if holes == 0 {
            return 0.0;
        }
        self.holes_late.iter().map(|&n| n as u64).sum::<u64>() as f64 / holes as f64
    }

    /// Of all the kept frames' stale pixels, the share that lay inside the
    /// screen box of a node whose swap lag was open on that frame: the part
    /// of the stale terrain that is swap lag showing (issue #384 decision 3).
    /// Zero when nothing was stale.
    pub fn stale_under_late_swap(&self) -> f64 {
        self.stale_share(|s| s.stale_late)
    }

    /// Of all the kept frames' stale pixels, the share under no open swap
    /// but inside the screen box of a visible node not yet resident (issue
    /// #398 decision 1, share (b)): terrain still loading. Zero when nothing
    /// was stale.
    pub fn stale_under_loading(&self) -> f64 {
        self.stale_share(|s| s.stale_loading)
    }

    /// Of all the kept frames' stale pixels, the share under neither (share
    /// (c)): what no open swap and no loading node accounts for.
    pub fn stale_under_neither(&self) -> f64 {
        self.stale_share(|s| s.stale - s.stale_late - s.stale_loading)
    }

    /// `part`'s share of the stale pixels, summed over the kept frames.
    fn stale_share(&self, part: impl Fn(&Sample) -> f64) -> f64 {
        let stale: f64 = self.samples.iter().map(|s| s.stale).sum();
        if stale <= 0.0 {
            return 0.0;
        }
        self.samples.iter().map(part).sum::<f64>() / stale
    }
}

/// The slow end of the work per frame, in seconds -- what a player feels as a
/// stutter (block 3.0a; the exit gate's item 0 asks for p99.9 under a 60 Hz
/// interval and no frame over two).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WorkTail {
    pub p99: f32,
    pub p999: f32,
    pub max: f32,
    /// Frames whose work alone took longer than one 60 Hz interval, and two.
    pub over_one: usize,
    pub over_two: usize,
}

/// One frame at 60 Hz, in seconds.
const INTERVAL_60HZ: f32 = 1.0 / 60.0;

impl WorkTail {
    pub fn of(times: &[f32]) -> Self {
        let mut sorted = times.to_vec();
        sorted.sort_by(f32::total_cmp);
        let at = |per_mille: usize| {
            sorted
                .get(sorted.len().saturating_sub(1) * per_mille / 1000)
                .copied()
                .unwrap_or(0.0)
        };
        Self {
            p99: at(990),
            p999: at(999),
            max: sorted.last().copied().unwrap_or(0.0),
            over_one: times.iter().filter(|&&t| t > INTERVAL_60HZ).count(),
            over_two: times.iter().filter(|&&t| t > 2.0 * INTERVAL_60HZ).count(),
        }
    }
}

/// What `--bench walk` found: every frame's whole time, start to GPU done.
pub struct WalkReport {
    pub frame_times: Vec<f32>,
    pub ticks: u64,
    pub distance: f32,
    /// The game's own loop ticks the world and meshes it at once, so this is
    /// where the tick meets a held mesh snapshot (issue #387).
    pub world_clones: WorldClones,
}

impl WalkReport {
    pub fn line(&self) -> String {
        let t = WorkTail::of(&self.frame_times);
        format!(
            "WALK: {} frames, {} ticks, walked {:.0} blocks | frame p99 {:.1} ms p99.9 {:.1} ms \
             max {:.1} ms, over 16.7 ms {} over 33.3 ms {} | {}",
            self.frame_times.len(),
            self.ticks,
            self.distance,
            t.p99 * 1000.0,
            t.p999 * 1000.0,
            t.max * 1000.0,
            t.over_one,
            t.over_two,
            self.world_clones.term(),
        )
    }
}

/// The keys `--bench walk` presses, for whichever loop runs it: W held, Space
/// once a second -- or, with `fly`, F4 on the first frame and then W.
pub struct WalkScript {
    fly: bool,
    phase: WalkPhase,
    next_jump: f32,
    jumping: bool,
}

#[derive(Clone, Copy)]
enum WalkPhase {
    Start,
    ReleaseF4,
    Moving,
}

impl WalkScript {
    pub fn new(fly: bool) -> Self {
        Self {
            fly,
            phase: WalkPhase::Start,
            next_jump: 1.0,
            jumping: false,
        }
    }

    /// Before a frame's `Game::advance`; `t` is seconds since the script began.
    pub fn before(&mut self, game: &mut crate::game::Game, t: f32) {
        use winit::keyboard::KeyCode;
        match self.phase {
            WalkPhase::Start if self.fly => {
                game.key_input(KeyCode::F4, true);
                self.phase = WalkPhase::ReleaseF4;
                return;
            }
            WalkPhase::Start => {
                game.key_input(KeyCode::KeyW, true);
                self.phase = WalkPhase::Moving;
            }
            WalkPhase::ReleaseF4 | WalkPhase::Moving => {}
        }
        if matches!(self.phase, WalkPhase::Moving) && !self.fly && t >= self.next_jump {
            game.key_input(KeyCode::Space, true);
            self.next_jump += 1.0;
            self.jumping = true;
        }
    }

    /// After that frame.
    pub fn after(&mut self, game: &mut crate::game::Game) {
        use winit::keyboard::KeyCode;
        if matches!(self.phase, WalkPhase::ReleaseF4) {
            game.key_input(KeyCode::F4, false);
            game.key_input(KeyCode::KeyW, true);
            self.phase = WalkPhase::Moving;
        }
        if self.jumping {
            game.key_input(KeyCode::Space, false);
            self.jumping = false;
        }
    }
}

/// `--bench walk`: the game's own loop, offscreen -- [`Game::advance`] (the
/// simulation and the server, in process), the voxel and far streaming, the
/// renderer -- while a scripted player walks forward for `seconds`, jumping
/// every second to take steps. Each frame is timed from its start until the
/// GPU has finished it, paced at `fps`: the whole frame the exit gate's item 0
/// asks about (block 3.0a), where `--bench flight` times only streaming and
/// drawing. Starts once the spawn has loaded. `None` without a GPU.
///
/// With `fly`, F4 first: the same loop with the player spectating, flying
/// forward through everything at free-fly speed -- the streaming under load,
/// with the simulation still ticking.
pub fn walk(seconds: f32, size: (u32, u32), fps: u32, fly: bool) -> Option<WalkReport> {
    use crate::game::{load_item_registry, load_recipe_book, Game};
    let mut game = Game::new();
    let (mut renderer, assets) = Renderer::offscreen(size.0, size.1, game.camera_pose())?;
    let layers = assets.layers;
    let registry = Arc::new(assets.registry);
    let items = load_item_registry();
    let recipes = load_recipe_book(&items);
    game.set_assets(Arc::clone(&registry), items, recipes);
    let mut nodes = NodeStreaming::new(
        registry,
        &crate::game::load_structure_registry(),
        &crate::game::load_ore_registry(),
        move |name: &str| layers.layer_of(name),
    );
    let mut far = FarStreaming::new(game.world().seed(), FarQuality::High);
    let budget = Duration::from_secs_f64(1.0 / fps.max(1) as f64);
    let mut frame = |game: &mut Game, dt: f32, renderer: &mut Renderer| {
        for cc in game.advance(dt) {
            nodes.invalidate(game.world(), cc);
        }
        let camera = game.camera_pose();
        let eye = camera.eye.as_vec3().to_array();
        nodes.update(renderer, game.world(), eye);
        let hole = nodes.far_hole(renderer, eye);
        far.update(renderer, eye, hole);
        renderer.render(
            camera,
            game.selected_block(),
            game.cracking(),
            &game.other_players(),
            cubara_render::Hud {
                hotbar: None,
                panel: None,
                health: Some(game.health_view()),
                crosshair: true,
                menu: None,
            },
            FAR_VIEW_RADIUS as f32,
        );
        let _ = renderer.device().poll(wgpu::PollType::wait_indefinitely());
        nodes.settled() && far.settled() && renderer.uploads_pending() == 0
    };
    // Load the spawn first, standing still: what the walk measures is
    // walking through a loaded world, not loading one.
    let waiting = Instant::now();
    while !frame(&mut game, budget.as_secs_f32(), &mut renderer) {
        if waiting.elapsed() > Duration::from_secs(120) {
            log::warn!("walk: the spawn did not settle in two minutes");
            break;
        }
    }
    let start_pos = game.camera_pose().eye;
    let ticks_before = game.sim_tick();
    let clones_before = game.world().clone_generation();
    let mut script = WalkScript::new(fly);
    script.before(&mut game, 0.0); // F4, or W
    if fly {
        frame(&mut game, budget.as_secs_f32(), &mut renderer);
        script.after(&mut game); // F4 up, W down
    }
    let mut frame_times = Vec::new();
    let start = Instant::now();
    let mut last = start;
    let mut next_frame = start;
    while start.elapsed().as_secs_f32() < seconds {
        let t = start.elapsed().as_secs_f32();
        script.before(&mut game, t);
        let began = Instant::now();
        let dt = (began - last).as_secs_f32();
        last = began;
        frame(&mut game, dt, &mut renderer);
        frame_times.push(began.elapsed().as_secs_f32());
        script.after(&mut game);
        next_frame += budget;
        let now = Instant::now();
        match next_frame.checked_duration_since(now) {
            Some(rest) => std::thread::sleep(rest),
            None => next_frame = now,
        }
    }
    Some(WalkReport {
        frame_times,
        ticks: game.sim_tick() - ticks_before,
        distance: (game.camera_pose().eye - start_pos).length() as f32,
        world_clones: WorldClones {
            clones: game.world().clone_generation() - clones_before,
            bytes_each: game.world().clone_bytes(),
            seconds: start.elapsed().as_secs_f32(),
        },
    })
}

/// Everything a flight streams: the world, the game's two streamers, and the
/// game's renderer.
struct Streamed {
    world: Arc<World>,
    nodes: NodeStreaming,
    far: FarStreaming,
    renderer: Renderer,
}

impl Streamed {
    fn new(flight: &Flight) -> Option<Self> {
        let camera = pose(flight, flight.start);
        let (mut renderer, assets) = Renderer::offscreen(flight.size.0, flight.size.1, camera)?;
        renderer.set_clear_color(NOTHING);
        let layers = assets.layers;
        let registry = Arc::new(assets.registry);
        let nodes = NodeStreaming::new(
            registry,
            &crate::game::load_structure_registry(),
            &crate::game::load_ore_registry(),
            move |name: &str| layers.layer_of(name),
        );
        let world = Arc::new(World::new());
        let far = FarStreaming::new(world.seed(), flight.quality);
        Some(Self {
            world,
            nodes,
            far,
            renderer,
        })
    }

    /// Stream for `eye` and draw one frame, the way the game's frame does.
    fn frame(&mut self, flight: &Flight, eye: [f32; 3]) {
        self.nodes.update(&mut self.renderer, &self.world, eye);
        let hole = self.nodes.far_hole(&self.renderer, eye);
        self.far.update(&mut self.renderer, eye, hole);
        self.renderer.render(
            pose(flight, eye),
            None,
            None,
            &[],
            cubara_render::Hud {
                hotbar: None,
                panel: None,
                health: None,
                crosshair: false,
                menu: None,
            },
            FAR_VIEW_RADIUS as f32,
        );
    }

    /// Stream and draw at `eye` until nothing is left to load, then once more
    /// so what arrived last is drawn.
    fn settle(&mut self, flight: &Flight, eye: [f32; 3]) {
        let waiting = Instant::now();
        loop {
            self.frame(flight, eye);
            if self.settled() {
                break;
            }
            if waiting.elapsed() > Duration::from_secs(60) {
                log::warn!("flight: nothing settled at {eye:?} in a minute");
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        self.frame(flight, eye);
    }

    fn settled(&self) -> bool {
        self.nodes.settled() && self.far.settled() && self.renderer.uploads_pending() == 0
    }
}

fn pose(flight: &Flight, eye: [f32; 3]) -> CameraPose {
    CameraPose {
        eye: glam::Vec3::from(eye).as_dvec3(),
        look_dir: glam::Vec3::from(flight.look).normalize_or_zero(),
    }
}

/// Every how-manieth candidate frame the finished frame is rendered for
/// (issue #384 decision 1): 1 is every one. Measured on the Linux machine
/// (GTX 1060 Max-Q, 1280x720): the low flight, with about 500 candidate
/// frames of its 1,200, took 46 s before this check and under 2x that with
/// every candidate frame checked, so nothing is sampled.
const FINISHED_CHECK_EVERY: usize = 1;

/// Counted holes written as frame dumps with `--frames` (issue #384
/// decision 1), so the class of what is counted can be read, not guessed.
const HOLE_DUMPS: usize = 20;

/// Candidates the finished frame rejected, written as frame dumps too -- the
/// class of what is *not* counted (a pop, usually) read the same way.
const REJECTED_DUMPS: usize = 10;

/// Kept frames with stale pixels written as frame dumps, the pixels ringed
/// (issue #398): what the three shares attribute, read rather than guessed.
const STALE_DUMPS: usize = 10;

/// How far a stale pixel's ray is marched for ground ([`surface_hit`]), in
/// blocks, and how far each step is. Eight times the voxel nodes' reach: far
/// enough to tell "beyond the voxels" from "sky" for any ridge big enough to
/// be a stale pixel, cheap enough (2,000 height lookups a pixel) to run on
/// every stale pixel of a kept frame in the settled pass.
const STALE_RAY_MAX: f32 = 8192.0;
const STALE_RAY_STEP: f32 = 4.0;

/// Keeps the live frames a dump may need -- every candidate frame and the
/// frame before it -- as PNG bytes, encoded on its own thread so the
/// real-time pass pays a clone and a channel send per frame rather than an
/// encode. Raw, 500 candidate frames at 1280x720 are 1.8 GB; encoded, the
/// flat colours of a flight frame come to tens of kilobytes each. Only
/// built when `--frames` is given.
struct FrameKeeper {
    send: std::sync::mpsc::Sender<(u32, bool, Frame)>,
    done: std::thread::JoinHandle<BTreeMap<(u32, bool), Vec<u8>>>,
}

impl FrameKeeper {
    fn new() -> Self {
        let (send, recv) = std::sync::mpsc::channel::<(u32, bool, Frame)>();
        let done = std::thread::spawn(move || {
            let mut kept = BTreeMap::new();
            for (frame, before, f) in recv {
                let mut png = std::io::Cursor::new(Vec::new());
                let ok = image::write_buffer_with_format(
                    &mut png,
                    &f.pixels,
                    f.width,
                    f.height,
                    image::ExtendedColorType::Rgba8,
                    image::ImageFormat::Png,
                );
                if ok.is_ok() {
                    kept.insert((frame, before), png.into_inner());
                }
            }
            kept
        });
        Self { send, done }
    }

    /// Keep `frame`'s live frame `f`, or with `before` the frame before it.
    fn keep(&self, frame: u32, before: bool, f: &Frame) {
        let _ = self.send.send((frame, before, f.clone()));
    }

    /// Everything kept, decoded on demand.
    fn finish(self) -> KeptFrames {
        drop(self.send);
        KeptFrames(self.done.join().unwrap_or_default())
    }
}

struct KeptFrames(BTreeMap<(u32, bool), Vec<u8>>);

impl KeptFrames {
    fn get(&self, frame: u32, before: bool) -> Option<Frame> {
        let png = self.0.get(&(frame, before))?;
        let img = image::load_from_memory_with_format(png, image::ImageFormat::Png)
            .ok()?
            .into_rgba8();
        Some(Frame {
            width: img.width(),
            height: img.height(),
            pixels: img.into_raw(),
        })
    }
}

/// Fly `flight` and compare what was on screen with what should have been.
/// With `frames`, the worst kept frame and its finished version are written
/// there as `worst-live.png` and `worst-settled.png`, every kept frame over
/// 1% loading as `tNN.NN-live.png`/`-settled.png`, the first [`HOLE_DUMPS`]
/// counted holes as `hole-fNNNN-live.png`/`-finished.png` with the pixels
/// ringed in red ([`mark_pixels`]), the first [`REJECTED_DUMPS`] rejected
/// candidate frames as `pop-fNNNN-...`, and the first [`STALE_DUMPS`] kept
/// frames with stale pixels as `stale-tNN.NN-live.png`/`-settled.png`, the
/// pixels ringed. `None` without a GPU.
pub fn fly(flight: &Flight, frames: Option<&Path>) -> Option<FlightReport> {
    // The real-time pass: what a player sees -- once the start is loaded, as
    // it is for a player who has been standing there. Loading a world from
    // nothing is a different question, with its own answer.
    let mut live = Streamed::new(flight)?;
    live.settle(flight, flight.start);
    // The flight ticks no simulation: nothing here calls `Arc::make_mut` on
    // the world, so this can only count a clone something else makes.
    let clones_before = live.world.clone_generation();
    let budget = Duration::from_secs_f64(1.0 / flight.fps.max(1) as f64);
    let (width, height) = flight.size;
    let aspect = width as f32 / height.max(1) as f32;
    // Per frame of the real-time pass: when and from where.
    let mut eyes: Vec<(f32, [f32; 3])> = Vec::new();
    // Per frame: the boxes, in blocks, of the nodes whose swap lag was open
    // when it was drawn -- what a stale pixel, or a hole, is laid under.
    let mut open_swaps: Vec<Vec<BlockBox>> = Vec::new();
    // The kept frames, by frame.
    let mut kept: BTreeMap<u32, Frame> = BTreeMap::new();
    // Per kept frame: what was still loading when it was drawn -- what a
    // stale pixel under no open swap is laid under (issue #398 decision 1,
    // share (b)).
    let mut kept_loading: BTreeMap<u32, Loading> = BTreeMap::new();
    // The last `HOLE_MAX_FRAMES + 1` frames, oldest first: a candidate run
    // closes at most that far back, and a pop compares with the frame before.
    let mut recent: VecDeque<Frame> = VecDeque::new();
    // Candidate frames' live frames and the frame before each, for the dumps.
    let keeper = frames.map(|_| FrameKeeper::new());
    let mut work_times = Vec::new();
    let mut holes: Option<HoleTracker> = None;
    let mut pops = Vec::new();
    let start = Instant::now();
    let mut next_frame = start;
    let mut next_keep = 0.0;
    loop {
        let t = start.elapsed().as_secs_f32();
        if t > flight.seconds {
            break;
        }
        let eye = eye_at(flight, t);
        let working = Instant::now();
        live.frame(flight, eye);
        work_times.push(working.elapsed().as_secs_f32());
        // Every frame, for the holes: one lasts a few frames, and the kept
        // ones are a quarter of a second apart. Read after the work is timed,
        // so the readback is not counted as the game's.
        let shown = live.renderer.read_frame()?;
        let frame = eyes.len() as u32;
        eyes.push((t, eye));
        open_swaps.push(live.nodes.open_swaps());
        // A swap this frame applied (nodes left the arena): its pop is what
        // changed inside the leavers' screen boxes since the frame before.
        let swapped = live.renderer.swapped_out();
        if let (false, Some(before)) = (swapped.is_empty(), recent.back()) {
            // The leavers' boxes are relative to the renderer's anchor
            // (issue #379), so they project through the same matrix it drew
            // with.
            let vp = pose(flight, eye).view_proj(aspect, live.renderer.anchor());
            let boxes: Vec<ScreenBox> = swapped
                .iter()
                .filter_map(|b| screen_box(b.min.to_array(), b.max.to_array(), vp, width, height))
                .collect();
            pops.push(Pop {
                frame,
                nodes: swapped.len(),
                pixels: changed_pixels(before, &shown, &boxes, POP_TOLERANCE),
                silhouette: silhouette_shift(before, &shown, &boxes),
            });
        }
        let tracker = holes
            .get_or_insert_with(|| HoleTracker::new(shown.width as usize, shown.pixels.len() / 4));
        let newly = tracker.push(
            shown
                .pixels
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| shows_nothing(p)),
        );
        recent.push_back(shown);
        if recent.len() > HOLE_MAX_FRAMES as usize + 1 {
            recent.pop_front();
        }
        // The frames that just became candidate frames are still in
        // `recent`; keep them, and the frame before each, for their dumps.
        if let Some(keeper) = &keeper {
            for f in newly {
                let at = recent.len() - 1 - (frame - f) as usize;
                keeper.keep(f, false, &recent[at]);
                if at > 0 {
                    keeper.keep(f, true, &recent[at - 1]);
                }
            }
        }
        if t >= next_keep {
            let (resident, held, meshing, visible) = live.nodes.counts();
            log::debug!(
                "flight: t {t:5.2} s: {resident} resident, {held} held, {meshing} meshing, \
                 {visible} visible, {} uploads queued",
                live.renderer.uploads_pending()
            );
            let shown = recent.back()?.clone();
            kept.insert(frame, shown);
            kept_loading.insert(
                frame,
                Loading {
                    nodes: live.nodes.loading_nodes(),
                    strip: live.nodes.far_strip(),
                    search_behind: live.nodes.search_behind(),
                },
            );
            next_keep += flight.every;
        }
        // Paced like a display showing `fps` frames a second: a frame that
        // ran late starts the next at once rather than catching up.
        next_frame += budget;
        let now = Instant::now();
        match next_frame.checked_duration_since(now) {
            Some(rest) => std::thread::sleep(rest),
            None => next_frame = now,
        }
    }
    drop(recent);
    let kept_frames = keeper.map(FrameKeeper::finish);
    // The live pass only (#378): the settled pass below jumps the eye around
    // to reconstruct each kept frame, which is not a swap a player ever
    // waits on.
    let achieved = work_times.len() as f64 / flight.seconds.max(1e-3) as f64;
    let swap_lags = SwapLagStats::of(live.nodes.swap_lags(), achieved, flight.speed);
    let world_clones = WorldClones {
        clones: live.world.clone_generation() - clones_before,
        bytes_each: live.world.clone_bytes(),
        seconds: start.elapsed().as_secs_f32(),
    };
    drop(live);
    let (candidates, counts) = holes.map_or((Vec::new(), Vec::new()), |h| {
        let counts = h.counts();
        (h.candidates, counts)
    });

    // The settled pass: the same eyes, each loaded completely -- every kept
    // frame's, and every candidate frame's, so a candidate is confirmed a
    // hole only where the finished frame shows land (decision 1). In flight
    // order, so the settled world has little to load from one eye to the
    // next.
    let mut settled = Streamed::new(flight)?;
    let mut samples = Vec::with_capacity(kept.len());
    let mut worst: Option<(f64, (Frame, Frame))> = None;
    let mut confirmed = vec![0u32; eyes.len()];
    let mut confirmed_late = vec![0u32; eyes.len()];
    let mut candidate_frames = 0usize;
    let (mut hole_dumps, mut rejected_dumps, mut stale_dumps) = (0usize, 0usize, 0usize);
    let gen = cubara_world::WorldGen::new(settled.world.seed());
    for (frame, &(t, eye)) in eyes.iter().enumerate() {
        let frame = frame as u32;
        let kept_here = kept.remove(&frame);
        let cands = &candidates[frame as usize];
        let check = !cands.is_empty() && {
            candidate_frames += 1;
            (candidate_frames - 1).is_multiple_of(FINISHED_CHECK_EVERY)
        };
        if kept_here.is_none() && !check {
            continue;
        }
        settled.far.reselect();
        settled.settle(flight, eye);
        let reference = settled.renderer.read_frame()?;
        // `open_swaps` holds world-space boxes (the streaming side has no
        // render anchor), so they project through the matrix against the
        // world origin; the flights run near it.
        let vp = pose(flight, eye).view_proj(aspect, cubara_render::anchor::ORIGIN);
        let open = &open_swaps[frame as usize];
        let late_boxes: Vec<ScreenBox> = open
            .iter()
            .filter_map(|&(min, max)| screen_box(min, max, vp, width, height))
            .collect();
        if check {
            let found = confirm_holes(cands, &reference);
            confirmed[frame as usize] = found.len() as u32;
            confirmed_late[frame as usize] = under_boxes(&found, width, &late_boxes) as u32;
            let at = |i: u32| (i % width, i / width);
            log::info!(
                "flight: frame {frame} (t {t:5.2} s at {eye:.0?}): {} candidates, the first at \
                 {:?}; {} holes{}, {} under one of {} late swaps",
                cands.len(),
                at(cands[0]),
                found.len(),
                found
                    .first()
                    .map_or(String::new(), |&i| format!(" (the first at {:?})", at(i))),
                confirmed_late[frame as usize],
                open.len(),
            );
            let dump = if found.is_empty() {
                (rejected_dumps < REJECTED_DUMPS).then(|| {
                    rejected_dumps += 1;
                    ("pop", cands)
                })
            } else {
                (hole_dumps < HOLE_DUMPS).then(|| {
                    hole_dumps += found.len();
                    ("hole", &found)
                })
            };
            if let (Some(dir), Some(kept), Some((kind, marked))) = (frames, &kept_frames, dump) {
                let name = |what: &str| dir.join(format!("{kind}-f{frame:04}-{what}.png"));
                let mut finished = reference.clone();
                mark_pixels(&mut finished, marked);
                save(&name("finished"), &finished);
                for (what, before) in [("live", false), ("before", true)] {
                    if let Some(mut f) = kept.get(frame, before) {
                        mark_pixels(&mut f, marked);
                        save(&name(what), &f);
                    }
                }
            }
        }
        if let Some(shown) = kept_here {
            let loading = loading_fraction(&shown, &reference);
            let stale = stale_pixels(&shown, &reference);
            let still = kept_loading.remove(&frame).unwrap_or_default();
            let loading_boxes: Vec<ScreenBox> = still
                .nodes
                .iter()
                .filter_map(|&(min, max)| screen_box(min, max, vp, width, height))
                .collect();
            // A pixel under neither box: where its ray meets the ground,
            // against the generator's surface (no GPU, no streaming state),
            // and whether that is in the strip the voxels have yet to draw.
            // The hits are kept for the log below, which places share (c)
            // in the world: inside the voxel nodes' reach, beyond it in the
            // far terrain, or at nothing (issue #398 decision 3's question).
            let mut hits: Vec<(Option<Hit>, bool)> = Vec::new();
            let shares = attribute_stale(&stale, width, &late_boxes, &loading_boxes, |i| {
                let dir = pixel_ray(flight.look, width, height, i % width, i / width);
                let hit = surface_hit(
                    |x, z| gen.surface_height(x, z),
                    eye,
                    dir,
                    STALE_RAY_MAX,
                    STALE_RAY_STEP,
                )
                .map(|d| {
                    let p = glam::Vec3::from(eye) + glam::Vec3::from(dir) * d;
                    (d, p.to_array())
                });
                let in_strip = match (hit, still.strip) {
                    (Some((_, p)), Some((hole, want))) => in_voxel_strip(p, hole, want),
                    _ => false,
                };
                hits.push((hit, in_strip));
                in_strip
            });
            let n = shown.pixels.len() as f64 / 4.0;
            log::info!(
                "flight: t {t:5.2} s at {eye:.0?}: {:.2}% loading, {} stale px ({:.3}%): \
                 {} under one of {} late swaps, {} under one of {} loading nodes, {} under \
                 neither{}",
                loading * 100.0,
                stale.len(),
                stale.len() as f64 * 100.0 / n,
                shares.late,
                open.len(),
                shares.loading,
                still.nodes.len(),
                shares.other,
                if still.search_behind {
                    " (the search for this chunk was still running)"
                } else {
                    ""
                },
            );
            if let Some(dir) = frames.filter(|_| loading > 0.01) {
                for (kind, f) in [("live", &shown), ("settled", &reference)] {
                    save(&dir.join(format!("t{:05.2}-{kind}.png", t)), f);
                }
            }
            if shares.other > 0 {
                // Where share (c) looks: the voxel nodes reach the edge of
                // the box they cover around the eye; ground met beyond it is
                // the far terrain's.
                let reach = {
                    let b = cubara_world::node::covered_box_3d(
                        cubara_voxel::ChunkCoord::from_world_pos(eye),
                        cubara_world::node::DEFAULT_RING_SCHEDULE,
                    );
                    (b.0.map(|c| (c * 16) as f32), b.1.map(|c| (c * 16) as f32))
                };
                let look = FarLook::of(
                    hits.iter()
                        .filter(|(_, in_strip)| !in_strip)
                        .map(|(hit, _)| *hit),
                    reach,
                );
                log::info!(
                    "flight: t {t:5.2} s: of the {} stale px under neither, {} look at ground \
                     beyond the voxel nodes' reach (far terrain), {} at nothing within \
                     {STALE_RAY_MAX} blocks; hits {:.0}..{:.0} blocks away",
                    shares.other,
                    look.far,
                    look.sky,
                    look.nearest,
                    look.farthest,
                );
            }
            if let Some(dir) = frames.filter(|_| !stale.is_empty() && stale_dumps < STALE_DUMPS) {
                stale_dumps += 1;
                for (kind, f) in [("live", &shown), ("settled", &reference)] {
                    let mut marked = f.clone();
                    mark_pixels(&mut marked, &stale);
                    save(&dir.join(format!("stale-t{:05.2}-{kind}.png", t)), &marked);
                }
            }
            samples.push(Sample {
                t,
                eye,
                loading,
                stale: stale.len() as f64 / n,
                stale_late: shares.late as f64 / n,
                stale_loading: shares.loading as f64 / n,
            });
            keep_worst(&mut worst, loading, (shown, reference));
        }
    }
    // The measurement's own zero. The first kept frame is at the start, which
    // the live pass loaded completely before it set off, so it must match its
    // finished frame exactly; a difference is the two passes disagreeing about
    // something other than loading -- the camera, the far terrain's selection,
    // the lighting -- and every number here would carry it. (Checked by
    // flying at speed 0: every kept frame 0.00%.)
    if let Some(first) = samples.first().filter(|s| s.loading > 0.0) {
        log::warn!(
            "flight: the start, loaded completely, differs from itself by {:.2}% -- \
             every number below includes that",
            first.loading * 100.0
        );
    }
    if let (Some(dir), Some((_, (live, settled)))) = (frames, worst) {
        for (name, frame) in [("worst-live.png", live), ("worst-settled.png", settled)] {
            save(&dir.join(name), &frame);
        }
    }
    let pixels = (flight.size.0 * flight.size.1) as usize;
    Some(FlightReport {
        samples,
        work_times,
        holes: confirmed,
        holes_late: confirmed_late,
        candidates: counts,
        checked_every: FINISHED_CHECK_EVERY,
        pixels,
        pops,
        swap_lags,
        world_clones,
    })
}

/// Keep `candidate` as the worst so far if it is: more of it was loading.
fn keep_worst<T>(worst: &mut Option<(f64, T)>, loading: f64, candidate: T) {
    if worst.as_ref().is_none_or(|(l, _)| loading > *l) {
        *worst = Some((loading, candidate));
    }
}

fn save(path: &Path, frame: &Frame) {
    if let Err(e) = image::save_buffer(
        path,
        &frame.pixels,
        frame.width,
        frame.height,
        image::ExtendedColorType::Rgba8,
    ) {
        log::error!("flight: could not write {}: {e}", path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flight() -> Flight {
        Flight {
            start: [0.0, 100.0, 0.0],
            look: [3.0, 0.0, 4.0],
            speed: 10.0,
            seconds: 5.0,
            fps: 60,
            every: 0.5,
            size: (64, 32),
            quality: FarQuality::High,
        }
    }

    #[test]
    fn the_eye_moves_along_the_look_at_the_speed() {
        let f = flight();
        assert_eq!(eye_at(&f, 0.0), [0.0, 100.0, 0.0]);
        let e = eye_at(&f, 2.0);
        assert!(
            (e[0] - 12.0).abs() < 1e-4 && (e[2] - 16.0).abs() < 1e-4,
            "{e:?}"
        );
        assert_eq!(e[1], 100.0);
    }

    fn frame(pixels: Vec<u8>) -> Frame {
        Frame {
            width: (pixels.len() / 4) as u32,
            height: 1,
            pixels,
        }
    }

    #[test]
    fn a_pixel_counts_as_loading_only_when_far_off() {
        let settled = frame(vec![100, 100, 100, 255, 100, 100, 100, 255]);
        let near = frame(vec![120, 100, 100, 255, 100, 100, 100, 255]);
        let hole = frame(vec![200, 100, 100, 255, 100, 100, 100, 255]);
        assert_eq!(loading_fraction(&near, &settled), 0.0);
        assert_eq!(loading_fraction(&hole, &settled), 0.5);
    }

    #[test]
    fn the_report_counts_the_frames_over_a_share() {
        let sample = |loading| Sample {
            t: 0.0,
            eye: [0.0; 3],
            loading,
            stale: 0.0,
            stale_late: 0.0,
            stale_loading: 0.0,
        };
        let r = FlightReport {
            samples: vec![sample(0.0), sample(0.02), sample(0.005), sample(0.1)],
            work_times: vec![0.002; 10],
            holes: Vec::new(),
            holes_late: Vec::new(),
            candidates: Vec::new(),
            checked_every: 1,
            pixels: 4,
            pops: Vec::new(),
            swap_lags: SwapLagStats::default(),
            world_clones: WorldClones::default(),
        };
        assert_eq!(r.frames_over(0.01), 0.5);
        assert!((r.mean() - 0.03125).abs() < 1e-12);
        assert_eq!(r.worst().unwrap().loading, 0.1);
    }

    /// Land all round, nothing for a few frames, land again: a hole, counted
    /// in every frame it showed. A pixel on a skyline, one on the screen's
    /// border, and a gap longer than half a second are not.
    #[test]
    fn a_hole_is_land_all_round_then_briefly_nothing_then_land() {
        // 5 x 3 pixels. Index 6 is inside land: a 3-frame hole (frames 1..=3).
        // Index 7 sits under 2, which is sky throughout: a skyline pixel doing
        // the same is not a hole. Index 3, on the top border, neither.
        // Index 8 shows nothing for 32 frames: too long to be a hole.
        let (n, l) = (true, false);
        let base = |f: usize| {
            let mut px = [l; 15];
            px[2] = n;
            if (1..=3).contains(&f) {
                px[6] = n;
                px[7] = n;
                px[3] = n;
            }
            if (1..=32).contains(&f) {
                px[8] = n;
            }
            px
        };
        let frames: Vec<[bool; 15]> = (0..=33).map(base).collect();
        let mut t = HoleTracker::new(5, 15);
        let mut newly = Vec::new();
        for f in &frames {
            newly.push(t.push(f.iter().copied()));
        }
        let mut want = vec![0u32; frames.len()];
        for w in &mut want[1..=3] {
            *w = 1;
        }
        assert_eq!(t.counts(), want);
        assert_eq!(t.candidates[2], vec![6], "the candidate is pixel 6");
        // The run closed on frame 4, which is when frames 1..=3 became
        // candidate frames -- all three at once, in order, no other frame.
        let mut want_newly = vec![Vec::new(); frames.len()];
        want_newly[4] = vec![1, 2, 3];
        assert_eq!(newly, want_newly);
    }

    /// A stale pixel is nothing in the live frame where the finished frame has
    /// land on the pixel and on all four sides; a skyline pixel drawn
    /// differently is not one, and neither is sky in both.
    #[test]
    fn a_stale_pixel_is_nothing_inside_the_finished_frames_land() {
        let (nothing, land) = ([255u8, 0, 255, 255], [90u8, 120, 60, 255]);
        let image = |px: [[u8; 4]; 9]| Frame {
            width: 3,
            height: 3,
            pixels: px.concat(),
        };
        let done = image([land; 9]);
        let mut live = [land; 9];
        live[4] = nothing;
        assert_eq!(stale_pixels(&image(live), &done), vec![4]);
        // The same pixel, but the finished frame has sky above it.
        let mut skyline = [land; 9];
        skyline[1] = nothing;
        assert_eq!(
            stale_pixels(&image(live), &image(skyline)),
            Vec::<u32>::new()
        );
        // Sky in both.
        assert!(stale_pixels(&image([nothing; 9]), &image([nothing; 9])).is_empty());
    }

    const NOTHING_PX: [u8; 4] = [255, 0, 255, 255];
    const LAND_PX: [u8; 4] = [90, 120, 60, 255];

    /// A `width` x `height` frame from a map of rows: `#` land, `.` nothing.
    fn picture(rows: &[&str]) -> Frame {
        let width = rows[0].len() as u32;
        let pixels = rows
            .iter()
            .flat_map(|r| r.bytes())
            .flat_map(|c| if c == b'#' { LAND_PX } else { NOTHING_PX })
            .collect();
        Frame {
            width,
            height: rows.len() as u32,
            pixels,
        }
    }

    /// Feed `frames` to a tracker, return it.
    fn track(frames: &[Frame]) -> HoleTracker {
        let first = &frames[0];
        let mut t = HoleTracker::new(first.width as usize, first.pixels.len() / 4);
        for f in frames {
            t.push(f.pixels.as_chunks::<4>().0.iter().map(|p| shows_nothing(p)));
        }
        t
    }

    /// Issue #384 done-when (a): a ridge against the sky drops two pixels at
    /// frame 1 and stays there. The row that was its top had land above it
    /// in frame 0 (the row above it was also ridge), so the tracker's
    /// previous-frame rule makes those pixels candidates when the ridge rises
    /// again at frame 3. The finished frame at frame 1's eye shows sky there:
    /// the silhouette moved, nothing is missing. Today's tracker counted
    /// this as a hole; confirmed against the finished frame it is 0.
    #[test]
    fn a_ridge_dropping_two_pixels_is_not_a_hole_when_the_finished_frame_shows_sky() {
        let tall = picture(&[".....", ".###.", ".###.", "#####", "#####"]);
        let dropped = picture(&[".....", ".....", ".....", "#####", "#####"]);
        let t = track(&[tall.clone(), dropped.clone(), dropped.clone(), tall]);
        let candidates = &t.candidates[1];
        assert_eq!(
            candidates,
            &vec![12],
            "the tracker's candidate rule itself changed: pixel (2,2) had land on \
             all four sides in frame 0 and went to nothing in frame 1"
        );
        // The finished frame at frame 1's eye is the dropped ridge: sky at
        // the candidate. Not a hole.
        assert_eq!(confirm_holes(candidates, &dropped), Vec::<u32>::new());
        assert_eq!(t.counts()[1], 1, "a candidate, for the record");
    }

    /// Issue #384 done-when (b): a node inside land missing for one frame,
    /// which the finished frame covers with land: one hole, by the tracker's
    /// candidate rule and by the finished frame alike.
    #[test]
    fn a_missing_node_inside_land_the_finished_frame_covers_is_one_hole() {
        let full = picture(&["#####", "#####", "#####", "#####", "#####"]);
        let missing = picture(&["#####", "#####", "##.##", "#####", "#####"]);
        let t = track(&[full.clone(), missing, full.clone()]);
        assert_eq!(t.candidates[1], vec![12]);
        assert_eq!(confirm_holes(&t.candidates[1], &full), vec![12]);
        assert_eq!(t.counts(), vec![0, 1, 0]);
    }

    /// Issue #384 done-when (c): two frames differing by a 3-pixel silhouette
    /// shift inside the box report `silhouette max = 3`, and 0 for a box that
    /// does not cover the shift. The pixel term sees the three rows that
    /// changed inside the box and nothing outside it.
    #[test]
    fn a_three_pixel_silhouette_shift_inside_the_box_is_a_three_pixel_pop() {
        let before = picture(&["........", "........", "........", "....####", "########"]);
        let after = picture(&["....####", "....####", "....####", "....####", "########"]);
        let over_the_ridge = ScreenBox {
            x0: 4,
            y0: 0,
            x1: 8,
            y1: 5,
        };
        let beside_it = ScreenBox {
            x0: 0,
            y0: 0,
            x1: 4,
            y1: 5,
        };
        assert_eq!(silhouette_shift(&before, &after, &[over_the_ridge]), 3);
        assert_eq!(silhouette_shift(&before, &after, &[beside_it]), 0);
        assert_eq!(
            changed_pixels(&before, &after, &[over_the_ridge], POP_TOLERANCE),
            12
        );
        assert_eq!(
            changed_pixels(&before, &after, &[beside_it], POP_TOLERANCE),
            0
        );
        // A luminance change within the tolerance is not a pop.
        let mut shaded = before.clone();
        shaded.pixels[(4 * 8 + 4) * 4] += POP_TOLERANCE;
        assert_eq!(
            changed_pixels(&before, &shaded, &[over_the_ridge], POP_TOLERANCE),
            0
        );
        shaded.pixels[(4 * 8 + 4) * 4] = 255;
        assert_eq!(
            changed_pixels(&before, &shaded, &[over_the_ridge], POP_TOLERANCE),
            1
        );
        // Luminance weights the channels as the eye does (Rec. 601: red
        // 0.299, green 0.587, blue 0.114). A blue change of 80 is 9 of
        // luminance, under the tolerance; the same change in green is 47,
        // a pop; and a red change of 30 is 9 where green's would be 18.
        let change = |channel: usize, by: u8| {
            let mut f = before.clone();
            f.pixels[(4 * 8 + 4) * 4 + channel] += by;
            changed_pixels(&before, &f, &[over_the_ridge], POP_TOLERANCE)
        };
        assert_eq!(change(2, 80), 0, "blue weighs little");
        assert_eq!(change(1, 80), 1, "green weighs much");
        assert_eq!(change(0, 30), 0, "red weighs less than green");
    }

    /// Issue #384 done-when (d): a stale pixel lies under an open swap's box
    /// and is attributed; one under a swap that already closed (its box is
    /// not among the open ones) is not.
    #[test]
    fn a_stale_pixel_is_attributed_to_an_open_swap_lag_not_a_closed_one() {
        let finished = picture(&["#####", "#####", "#####", "#####"]);
        let live = picture(&["#####", "#.###", "###.#", "#####"]);
        let stale = stale_pixels(&live, &finished);
        assert_eq!(stale, vec![6, 13]);
        let open = ScreenBox {
            x0: 0,
            y0: 0,
            x1: 2,
            y1: 2,
        };
        let closed = ScreenBox {
            x0: 2,
            y0: 2,
            x1: 5,
            y1: 4,
        };
        assert_eq!(
            under_boxes(&stale, 5, &[open]),
            1,
            "pixel (1,1) is under the open swap"
        );
        assert_eq!(under_boxes(&stale, 5, &[]), 0, "no swap is open");
        assert_eq!(
            under_boxes(&stale, 5, &[open, closed]),
            2,
            "were the closed one still open, (3,2) would count too"
        );
    }

    /// A box in front of the eye lands where the camera puts it: a 2-block
    /// cube 10 blocks ahead, centred, covers a box around the screen's
    /// centre; off to the side it is `None`; straddling the eye it is the
    /// whole screen.
    #[test]
    fn a_world_box_projects_to_the_screen_box_around_its_corners() {
        let vp = CameraPose {
            eye: glam::DVec3::ZERO,
            look_dir: glam::Vec3::NEG_Z,
        }
        .view_proj(16.0 / 9.0, cubara_render::anchor::ORIGIN);
        let centred = screen_box([-1.0, -1.0, -11.0], [1.0, 1.0, -9.0], vp, 1600, 900).unwrap();
        assert!(centred.x0 < 800 && 800 < centred.x1, "{centred:?}");
        assert!(centred.y0 < 450 && 450 < centred.y1, "{centred:?}");
        assert!(centred.x1 - centred.x0 < 400, "{centred:?}");
        let higher = screen_box([-1.0, 2.0, -11.0], [1.0, 4.0, -9.0], vp, 1600, 900).unwrap();
        assert!(
            higher.y1 <= centred.y0,
            "higher in the world is higher on screen"
        );
        assert_eq!(
            screen_box([50.0, 0.0, -11.0], [52.0, 2.0, -9.0], vp, 1600, 900),
            None
        );
        assert_eq!(
            screen_box([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], vp, 1600, 900),
            Some(ScreenBox {
                x0: 0,
                y0: 0,
                x1: 1600,
                y1: 900
            }),
            "a box around the eye is the whole screen"
        );
        assert_eq!(
            screen_box([-1.0, -1.0, 9.0], [1.0, 1.0, 11.0], vp, 1600, 900),
            None,
            "a box wholly behind the eye is not on screen"
        );
        // A box reaching from behind the eye to ahead of it, off to the
        // right and outside the 60-degree view: its part in front is still
        // off screen, so it is `None` rather than the whole screen.
        assert_eq!(
            screen_box([5.0, -1.0, -1.0], [7.0, 1.0, 1.0], vp, 1600, 900),
            None
        );
        // The same reach past the eye but centred: the part in front fills
        // the screen's width, since the near cut is right at the eye.
        let straddling = screen_box([-1.0, -1.0, -3.0], [1.0, 1.0, 1.0], vp, 1600, 900).unwrap();
        assert_eq!((straddling.x0, straddling.x1), (0, 1600), "{straddling:?}");
    }

    /// Two boxes over the same pixels count a changed pixel once.
    #[test]
    fn overlapping_boxes_do_not_double_count_a_changed_pixel() {
        let before = picture(&["....", "####"]);
        let after = picture(&["..##", "####"]);
        let b = ScreenBox {
            x0: 0,
            y0: 0,
            x1: 4,
            y1: 2,
        };
        assert_eq!(changed_pixels(&before, &after, &[b, b], POP_TOLERANCE), 2);
    }

    #[test]
    fn pop_stats_are_the_distribution_of_each_term() {
        let pop = |pixels, silhouette| Pop {
            frame: 0,
            nodes: 1,
            pixels,
            silhouette,
        };
        let s = PopStats::of(&[pop(10, 0), pop(20, 2), pop(30, 1), pop(400, 5)]);
        assert_eq!(s.count, 4);
        assert_eq!((s.pixels_p50, s.pixels_p99, s.pixels_max), (20, 30, 400));
        assert_eq!(
            (s.silhouette_p50, s.silhouette_p99, s.silhouette_max),
            (1, 2, 5)
        );
        assert_eq!(PopStats::of(&[]), PopStats::default());
    }

    /// Issue #398 decision 1: every stale pixel falls under exactly one of
    /// the three shares -- late swap first, then a loading node, then
    /// neither -- and a pixel under both a late swap and a loading node is
    /// the late swap's (a held node and its replacements share a box).
    #[test]
    fn a_stale_pixel_is_a_late_swap_else_a_loading_node_else_neither() {
        let finished = picture(&["######", "######", "######", "######"]);
        let live = picture(&["######", "#.#.##", "###..#", "######"]);
        let stale = stale_pixels(&live, &finished);
        assert_eq!(stale, vec![7, 9, 15, 16]);
        let late = ScreenBox {
            x0: 0,
            y0: 0,
            x1: 2,
            y1: 2,
        };
        // Covers the late box's pixel too, and (3,1) and (3,2); not (4,2).
        let loading = ScreenBox {
            x0: 0,
            y0: 0,
            x1: 4,
            y1: 3,
        };
        let never = |_: u32| false;
        assert_eq!(
            attribute_stale(&stale, 6, &[late], &[loading], never),
            StaleShares {
                late: 1,
                loading: 2,
                other: 1,
            }
        );
        assert_eq!(
            attribute_stale(&stale, 6, &[], &[loading], never),
            StaleShares {
                late: 0,
                loading: 3,
                other: 1,
            },
            "with no swap open, the shared pixel is the loading node's"
        );
        assert_eq!(
            attribute_stale(&stale, 6, &[], &[], never),
            StaleShares {
                late: 0,
                loading: 0,
                other: 4,
            },
            "nothing to lay them under: all three shares' remainder"
        );
        assert_eq!(
            attribute_stale(&[], 6, &[late], &[loading], never),
            StaleShares::default()
        );
        // The strip is asked only about pixels under neither box, and a
        // pixel it claims is loading: (4,2) is the one under neither.
        let mut asked = Vec::new();
        let shares = attribute_stale(&stale, 6, &[late], &[loading], |i| {
            asked.push(i);
            i == 16
        });
        assert_eq!(asked, vec![16], "only the pixel under neither box");
        assert_eq!(
            shares,
            StaleShares {
                late: 1,
                loading: 3,
                other: 0,
            }
        );
    }

    /// Share (c) placed in the world: a hit beyond the voxel nodes' reach is
    /// the far terrain's, one inside it is not, a miss is sky, and the
    /// nearest and farthest hits are the nearest and farthest.
    #[test]
    fn where_the_unattributed_pixels_look_is_told_by_their_hits() {
        let reach = ([-1024.0, -1024.0, -1024.0], [1024.0, 1024.0, 1024.0]);
        let look = FarLook::of(
            [
                Some((1800.0, [1700.0, 200.0, 500.0])),
                Some((530.0, [500.0, 100.0, 150.0])),
                None,
                Some((1500.0, [1023.0, 100.0, 1100.0])),
            ],
            reach,
        );
        assert_eq!(
            look,
            FarLook {
                far: 2,
                sky: 1,
                nearest: 530.0,
                farthest: 1800.0,
            }
        );
        assert_eq!(FarLook::of([], reach).far, 0);
    }

    /// The strip the far terrain stands in for is what the voxels want and
    /// do not draw yet: a point in the new part of the box is in it, one in
    /// the part already drawn is not, and neither is one outside both.
    #[test]
    fn the_voxel_strip_is_the_wanted_box_less_the_drawn_one() {
        let hole = ([0.0, 0.0, 0.0], [128.0, 128.0, 128.0]);
        let want = ([64.0, 0.0, 0.0], [192.0, 128.0, 128.0]);
        assert!(
            in_voxel_strip([150.0, 10.0, 10.0], hole, want),
            "new ground"
        );
        assert!(
            !in_voxel_strip([100.0, 10.0, 10.0], hole, want),
            "drawn already"
        );
        assert!(
            !in_voxel_strip([10.0, 10.0, 10.0], hole, want),
            "no longer wanted"
        );
        assert!(
            !in_voxel_strip([300.0, 10.0, 10.0], hole, want),
            "far terrain"
        );
        // `max` is exclusive, `min` inclusive: the hole's far face is the
        // strip's first block.
        assert!(in_voxel_strip([128.0, 0.0, 0.0], hole, want));
        assert!(!in_voxel_strip([192.0, 0.0, 0.0], hole, want));
    }

    /// The report's three shares are shares of the stale pixels over all
    /// kept frames, they sum to one whenever anything was stale, and are all
    /// zero when nothing was.
    #[test]
    fn the_three_stale_shares_sum_to_one_over_the_kept_frames() {
        let sample = |stale, stale_late, stale_loading| Sample {
            t: 0.0,
            eye: [0.0; 3],
            loading: 0.0,
            stale,
            stale_late,
            stale_loading,
        };
        let report = |samples| FlightReport {
            samples,
            work_times: Vec::new(),
            holes: Vec::new(),
            holes_late: Vec::new(),
            candidates: Vec::new(),
            checked_every: 1,
            pixels: 1000,
            pops: Vec::new(),
            swap_lags: SwapLagStats::default(),
            world_clones: WorldClones::default(),
        };
        // 10 stale pixels per mille in all: 2 late, 5 loading, 3 neither.
        let r = report(vec![sample(0.004, 0.002, 0.002), sample(0.006, 0.0, 0.003)]);
        assert!((r.stale_under_late_swap() - 0.2).abs() < 1e-9);
        assert!((r.stale_under_loading() - 0.5).abs() < 1e-9);
        assert!((r.stale_under_neither() - 0.3).abs() < 1e-9);
        let clean = report(vec![sample(0.0, 0.0, 0.0)]);
        assert_eq!(clean.stale_under_late_swap(), 0.0);
        assert_eq!(clean.stale_under_loading(), 0.0);
        assert_eq!(clean.stale_under_neither(), 0.0);
    }

    /// A pixel's ray follows the renderer's projection: the screen's centre
    /// looks along `look`, the top centre 30 degrees above it (half the
    /// vertical field of view), and the right-hand side to the right of a
    /// camera looking along -Z. Where a pixel's ray goes agrees with where
    /// `screen_box` puts a point: the two are inverses.
    #[test]
    fn a_pixel_ray_inverts_the_projection() {
        let look = [0.0, 0.0, -1.0];
        let (w, h) = (1600, 900);
        let centre = glam::Vec3::from(pixel_ray(look, w, h, 800, 450));
        assert!(centre.abs_diff_eq(glam::Vec3::NEG_Z, 2e-3), "{centre}");
        let top = glam::Vec3::from(pixel_ray(look, w, h, 800, 0));
        let elevation = top.y.atan2(-top.z).to_degrees();
        assert!((elevation - 30.0).abs() < 0.1, "{elevation} degrees");
        let right = glam::Vec3::from(pixel_ray(look, w, h, 1599, 450));
        assert!(right.x > 0.5, "the right of the screen is +X here: {right}");
        // The inverse of `screen_box`: a point 20 blocks along the ray of
        // pixel (1200, 300) projects to a box containing that pixel.
        let ray = glam::Vec3::from(pixel_ray(look, w, h, 1200, 300)) * 20.0;
        let vp = CameraPose {
            eye: glam::DVec3::ZERO,
            look_dir: glam::Vec3::NEG_Z,
        }
        .view_proj(w as f32 / h as f32, cubara_render::anchor::ORIGIN);
        let b = screen_box((ray - 0.01).to_array(), (ray + 0.01).to_array(), vp, w, h).unwrap();
        assert!(b.contains(1200, 300), "{b:?}");
    }

    /// Marching for ground: a ray at 45 degrees down from 10 blocks above a
    /// flat surface meets it about 10 * sqrt 2 blocks out, within a step; a
    /// level ray never does; and the surface block's own height counts as
    /// ground, not the air above it.
    #[test]
    fn a_ray_meets_the_surface_where_it_drops_below_it() {
        let flat = |_: i32, _: i32| 0;
        let eye = [0.0, 11.0, 0.0];
        let down = [
            std::f32::consts::FRAC_1_SQRT_2,
            -std::f32::consts::FRAC_1_SQRT_2,
            0.0,
        ];
        let d = surface_hit(flat, eye, down, 100.0, 0.5).unwrap();
        assert!((d - 10.0 * std::f32::consts::SQRT_2).abs() <= 0.5, "{d}");
        assert_eq!(surface_hit(flat, eye, [1.0, 0.0, 0.0], 100.0, 0.5), None);
        // A wall 20 blocks tall 5 blocks ahead stops a level ray at 5.
        let wall = |x: i32, _: i32| if x >= 5 { 20 } else { 0 };
        assert_eq!(
            surface_hit(wall, eye, [1.0, 0.0, 0.0], 100.0, 1.0),
            Some(5.0)
        );
        // Beyond `max` is sky, even where ground would be.
        assert_eq!(surface_hit(wall, eye, [1.0, 0.0, 0.0], 4.0, 1.0), None);
    }

    /// The summary line carries the three terms gate item 0 parses
    /// (`scripts/check-phase-gate.sh`), in the words its patterns look for,
    /// and the stale shares are shares of stale pixels, not of the screen.
    #[test]
    fn the_line_prints_holes_pops_and_stale_as_three_terms() {
        let sample = |stale, stale_late, stale_loading| Sample {
            t: 0.0,
            eye: [0.0; 3],
            loading: 0.0,
            stale,
            stale_late,
            stale_loading,
        };
        let r = FlightReport {
            samples: vec![
                sample(0.0, 0.0, 0.0),
                sample(0.004, 0.003, 0.0),
                sample(0.002, 0.0, 0.0015),
            ],
            work_times: vec![0.002; 10],
            holes: vec![0, 4, 0, 0],
            holes_late: vec![0, 1, 0, 0],
            candidates: vec![0, 5, 3, 0],
            checked_every: 1,
            pixels: 1000,
            pops: vec![Pop {
                frame: 1,
                nodes: 2,
                pixels: 7,
                silhouette: 3,
            }],
            swap_lags: SwapLagStats::default(),
            world_clones: WorldClones {
                clones: 30,
                bytes_each: 140,
                seconds: 10.0,
            },
        };
        let line = r.line();
        assert!(
            line.contains("HOLES: 1 of 4 frames show a hole, worst 0.400%, of which under a late swap 25% (2 candidate frames, 2 checked against the finished frame)"),
            "{line}"
        );
        assert!(
            line.contains(
                "POPS: 1 swaps, pixels p50/p99/max 7/7/7, silhouette px p50/p99/max 3/3/3"
            ),
            "{line}"
        );
        assert!(
            line.contains("STALE: 2 kept frames with nothing where the finished frame has land, worst 0.400%, of which under a late swap 50%, under a node still loading 25%, neither 25%"),
            "{line}"
        );
        assert!(!line.contains("sampled"));
        assert!(
            line.contains(
                "WORLD CLONES: 30 in 10.0 s = 3.0 clones/s, 420 bytes/s (140 bytes each)"
            ),
            "{line}"
        );
        let sampled = FlightReport {
            checked_every: 3,
            ..r
        };
        assert!(
            sampled.line().contains(
                "(2 candidate frames, 1 checked against the finished frame, sampled 1 in 3)"
            ),
            "{}",
            sampled.line()
        );
    }

    /// The ring goes around the marked pixels' bounding box, not each pixel:
    /// two neighbours share one ring, so neither covers the other.
    #[test]
    fn marking_pixels_rings_their_box_in_red_without_covering_them() {
        let mut f = picture(&["#################"; 15]);
        // (6,5) and (8,7): box (6..=8, 5..=7); ring at 4..=5 outside it.
        mark_pixels(&mut f, &[5 * 17 + 6, 7 * 17 + 8]);
        let red = [255u8, 0, 0, 255];
        let px = |x: usize, y: usize| &f.pixels[(y * 17 + x) * 4..(y * 17 + x) * 4 + 4];
        assert_eq!(px(6, 5), &LAND_PX, "a marked pixel is covered");
        assert_eq!(px(8, 7), &LAND_PX, "a marked pixel is covered");
        assert_eq!(px(7, 6), &LAND_PX, "between them is covered");
        assert_eq!(
            px(7, 9),
            &LAND_PX,
            "2 outside the box: too close to be ring"
        );
        assert_eq!(px(7, 11), &red, "4 outside the box: ring");
        assert_eq!(px(7, 12), &red, "5 outside: ring");
        assert_eq!(px(7, 13), &LAND_PX, "6 outside: beyond the ring");
        assert_eq!(px(1, 0), &red, "the corner of the ring");
        assert_eq!(px(0, 0), &LAND_PX, "beyond the corner");
        // Near the border, the ring is clipped rather than wrapped.
        let mut g = picture(&["#################"; 15]);
        mark_pixels(&mut g, &[0]);
        let gx = |x: usize, y: usize| &g.pixels[(y * 17 + x) * 4..(y * 17 + x) * 4 + 4];
        assert_eq!(gx(16, 4), &LAND_PX, "wrapped round");
        assert_eq!(gx(4, 0), &red);
        mark_pixels(&mut g, &[]);
    }

    /// The tail reads the slow end: p99.9 and the worst, and counts frames
    /// over one and two 60 Hz intervals -- not frames at exactly one.
    #[test]
    fn the_work_tail_is_the_slow_end() {
        let mut times = vec![0.002f32; 1000];
        times[10] = 0.020; // over one interval
        times[20] = 0.040; // over two
        times[30] = 1.0 / 60.0; // exactly one: not over
        let t = WorkTail::of(&times);
        assert_eq!((t.over_one, t.over_two), (2, 1));
        assert_eq!(t.max, 0.040);
        assert_eq!(t.p999, 0.020, "p99.9 of 1,000 frames is the second worst");
        assert_eq!(t.p99, 0.002);
    }

    #[test]
    fn the_frames_kept_are_the_worst_ones() {
        let mut worst = None;
        for (loading, name) in [(0.02, "a"), (0.3, "b"), (0.1, "c"), (0.3, "d")] {
            keep_worst(&mut worst, loading, name);
        }
        assert_eq!(worst, Some((0.3, "b")), "kept a later or a lesser frame");
    }

    /// Issue #378: frames become milliseconds at the flight's own pace, and
    /// blocks at its speed -- at 60 FPS and 24 blocks/s (`FLY_SPEED`), one
    /// frame is 1000/60 ms and 24/60 of a block.
    #[test]
    fn swap_lag_stats_convert_frames_with_the_flights_own_pace_and_speed() {
        let lags = [1u64, 2, 10, 100];
        let s = SwapLagStats::of(&lags, 60.0, 24.0);
        assert_eq!(s.count, 4);
        assert!((s.max_ms - 100.0 * 1000.0 / 60.0).abs() < 1e-9);
        assert!((s.max_blocks - 100.0 * 24.0 / 60.0).abs() < 1e-9);
        // p99 of these 4 sorted values is index (4-1)*990/1000 = 2: `10`.
        assert!((s.p99_ms - 10.0 * 1000.0 / 60.0).abs() < 1e-9);
    }

    /// A flight that only managed 30 frames a second under load: the same
    /// 30-frame lag is a whole second and 24 blocks, not the half second the
    /// 60 FPS it asked for would claim.
    #[test]
    fn a_slow_flight_converts_at_the_rate_it_achieved() {
        let s = SwapLagStats::of(&[30], 30.0, 24.0);
        assert!((s.max_ms - 1000.0).abs() < 1e-9);
        assert!((s.max_blocks - 24.0).abs() < 1e-9);
    }

    #[test]
    fn swap_lag_stats_of_nothing_is_the_default() {
        assert_eq!(SwapLagStats::of(&[], 60.0, 24.0), SwapLagStats::default());
    }
}
