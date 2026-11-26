//! The far terrain's GPU-free half (`docs/PROPOSAL_FAR_VIEW.md` §3.2).
//!
//! Beyond the voxel rings, the land is drawn as a height field. Which squares
//! of it -- *patches* -- are drawn around an eye, and what heights they hold,
//! are pure functions of the eye and the seed, and live here with their
//! tests. The renderer only ever sees the heights.
//!
//! A patch is a quadtree node: [`PATCH_QUADS`] quads of `2^level` blocks on a
//! side. Near the eye patches are small and their quads fine; far away they
//! are large and coarse, so that a quad stays about the same size *on screen*
//! wherever it is drawn -- the property the voxel rings have too, and the one
//! [`FarView::split_ratio`] sets.
//!
//! The heights are the terrain's own: [`WorldGen::surface_height`], averaged
//! over the ground each vertex stands for. The terrain is a height field with
//! caves beneath it (`PHASE1_ARCHITECTURE.md` §8.1), so from kilometres away
//! this is the whole of what can be seen -- and unlike a cube-shaped LOD node,
//! a height here is a number, not a cell boundary, so it does not flatten with
//! distance (`PROPOSAL_FAR_VIEW.md` §2A, table B).
//!
//! **What is drawn is blocks** (issue #422). The height field stays the data;
//! [`mesh`] turns it into what a voxel mesher would make of the same ground:
//! every quad a flat top at a whole number of blocks ([`quad_top`], the one
//! place that height is decided), a vertical wall wherever two neighbouring
//! tops differ, exactly as tall as the step, and equal neighbours merged into
//! one face. A quad of level `L` stands on the lattice of `2^L` blocks -- the
//! lattice the voxel rings' coarse cells stand on -- so where the two meet
//! their steps fall on the same lines.

use crate::WorldGen;

/// Quads along one side of a patch.
pub const PATCH_QUADS: usize = 32;

/// Heights along one side of a patch: one more than the quads.
pub const PATCH_VERTS: usize = PATCH_QUADS + 1;

/// One square of the far terrain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PatchKey {
    /// A quad is `2^level` blocks wide.
    pub level: u32,
    /// Position on this level's grid of patches.
    pub x: i32,
    pub z: i32,
}

impl PatchKey {
    pub const fn new(level: u32, x: i32, z: i32) -> Self {
        Self { level, x, z }
    }

    /// The width of one quad, in blocks.
    pub fn quad(self) -> i64 {
        1i64 << self.level
    }

    /// The width of the whole patch, in blocks.
    pub fn size(self) -> i64 {
        PATCH_QUADS as i64 * self.quad()
    }

    /// The patch's lowest `(x, z)` corner, in blocks.
    pub fn origin(self) -> [i64; 2] {
        [self.x as i64 * self.size(), self.z as i64 * self.size()]
    }

    /// The patch at `level` that holds the block column `(x, z)`.
    pub fn containing(level: u32, x: i64, z: i64) -> Self {
        let size = PATCH_QUADS as i64 * (1i64 << level);
        Self::new(level, x.div_euclid(size) as i32, z.div_euclid(size) as i32)
    }

    /// The four patches one level finer that tile this one.
    pub fn children(self) -> [PatchKey; 4] {
        let (level, x, z) = (self.level - 1, self.x * 2, self.z * 2);
        [
            Self::new(level, x, z),
            Self::new(level, x + 1, z),
            Self::new(level, x, z + 1),
            Self::new(level, x + 1, z + 1),
        ]
    }
}

/// **The finest level a far patch is ever drawn at**: the level of the voxel
/// rings' coarsest cells, whose ground the far terrain meets at the join. A
/// far quad finer than that cell would put a fine band between two coarse
/// ones (level 3's cells of 8 blocks, the far terrain's quads of 4 or 2, then
/// 16 and wider), which no screen size or quality asks for: the finest quad
/// is the coarsest cell, at any screen height and any [`FarView`]. The same
/// number for both sides of the join is what lets
/// [`quad_top`] equal the voxel cell's top there ([`EXACT_TOP_QUAD`]).
pub const FINEST_LEVEL: u32 = {
    let schedule = crate::node::DEFAULT_RING_SCHEDULE;
    schedule[schedule.len() - 1].0
};

/// A box the far terrain leaves to something else: the voxel rings, which
/// draw everything inside it themselves. Axis-aligned, around the eye's chunk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hole {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl Hole {
    fn contains_box(&self, min: [f64; 3], max: [f64; 3]) -> bool {
        (0..3).all(|a| self.min[a] <= min[a] && max[a] <= self.max[a])
    }

    /// The same box with its ground -- `x` and `z`, not `y` -- grown by
    /// `by` blocks on every side (shrunk, for a negative `by`; `None` when
    /// that leaves nothing).
    pub fn grown_across(self, by: f64) -> Option<Hole> {
        let mut out = self;
        for a in [0, 2] {
            out.min[a] -= by;
            out.max[a] += by;
        }
        [0, 2]
            .iter()
            .all(|&a| out.min[a] < out.max[a])
            .then_some(out)
    }
}

/// Everything [`select`] needs: where the eye is, how finely to draw, how far,
/// and what not to draw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FarView {
    /// The camera, in blocks.
    pub eye: [f64; 3],
    /// A patch is split into four while the eye is nearer to it than
    /// `split_ratio` times its width. A quad then never appears wider than
    /// `1 / (split_ratio * PATCH_QUADS)` radians -- see [`split_ratio_for`].
    pub split_ratio: f64,
    /// ...and while its own [`PatchHeights::error`] would appear taller than
    /// this, in radians: the *height* bound, which puts fine patches where
    /// the land is rough (mountains) rather than everywhere at that distance.
    pub max_error: f64,
    /// ...but never into quads narrower than this, in radians. Without a
    /// floor the height bound chases the terrain's one-block steps, which a
    /// smooth surface cannot follow to within a block, down to single columns
    /// wherever a pixel is about a block wide.
    pub min_quad: f64,
    /// How far the far terrain reaches, in blocks, measured across the ground.
    pub radius: f64,
    /// Left to the voxel rings.
    pub hole: Option<Hole>,
}

/// The `split_ratio` that keeps a quad at most `pixels` wide on a screen
/// `screen_height` pixels tall with a vertical field of view of `fov_y`
/// radians.
pub fn split_ratio_for(pixels: f64, screen_height: f64, fov_y: f64) -> f64 {
    let pixel = fov_y / screen_height;
    1.0 / (pixels * pixel * PATCH_QUADS as f64)
}

/// The lowest and highest top any level draws, in blocks: one above the
/// lowest and the highest ground a column can have
/// ([`WorldGen::column_top_range`], pit floors included). Every far face's
/// top, and a wall's foot, stands within it: a top is the mean of such
/// columns, or a summit among them, plus one.
pub fn ground_range() -> (f64, f64) {
    let (lo, hi) = WorldGen::column_top_range();
    (lo as f64 + 1.0, hi as f64 + 1.0)
}

/// The lowest and highest surface the generator can produce -- the vertical
/// extent of a patch whose heights are not known yet.
pub fn surface_bounds() -> (f64, f64) {
    let (lo, hi) = WorldGen::surface_range();
    // `+ 1`: a surface block's top face is one above the block itself.
    (lo as f64, hi as f64 + 1.0)
}

/// The patches drawn around `view.eye`, each covering ground no other one
/// does.
///
/// A quadtree walk from roots big enough to reach `view.radius`. A patch is
/// dropped when it lies wholly beyond `radius` or wholly inside the hole. It
/// is split when the eye is within `split_ratio` of its width (measured to its
/// box, with the terrain's full height range as its vertical extent), or when
/// `error_of` knows its error and that error would appear taller than
/// `max_error`. Otherwise it is kept.
///
/// `error_of` answers only for patches already generated, so a caller streams
/// toward the answer: select, generate what is new, select again. A patch
/// whose error is not known yet is judged by distance alone.
pub fn select(view: &FarView, error_of: impl Fn(PatchKey) -> Option<f32>) -> Vec<PatchKey> {
    select_walk(view, error_of).patches
}

/// What one walk of [`select`] found: the patches, the ones split on the
/// way to them, and how far the eye may move before any of it can change.
#[derive(Clone, Debug, PartialEq)]
pub struct Selection {
    /// What [`select`] returns.
    pub patches: Vec<PatchKey>,
    /// Every patch the walk split: the ancestors of `patches`, up to the
    /// roots, that were not dropped. The patches a nearer or farther eye
    /// could keep whole (`FarPlan`'s prefetch in the app).
    pub split: Vec<PatchKey>,
    /// **How far the eye may move, in blocks, before the walk could decide
    /// anything differently** -- with the same errors known and the same
    /// hole. Every decision compares a distance from the eye to a box
    /// with a fixed threshold, and such a distance changes by at most what
    /// the eye moves; this is the smallest gap between a distance and its
    /// threshold the walk met (#398's plan A). An eye nearer than this to
    /// `view.eye` selects the same patches.
    pub slack: f64,
}

/// [`select`], with what else the walk found ([`Selection`]). One walk: the
/// patches it returns are `select`'s.
pub fn select_walk(view: &FarView, error_of: impl Fn(PatchKey) -> Option<f32>) -> Selection {
    let root_level = root_level_for(view.radius);
    let root_size = ((PATCH_QUADS as i64) << root_level) as f64;
    let [ex, _, ez] = view.eye;
    // Every root that can reach within the radius, and one more on every
    // side, which the radius test drops: so which roots are walked never
    // decides anything, and a root coming within reach as the eye moves is
    // one the walk already measured -- its gap to the radius is in the
    // slack. (The range was worked out from the eye truncated to whole
    // blocks, and a root within the radius by less than the eye's fraction
    // of a block was not walked; at the edge of the far terrain, out of
    // sight, but a decision the slack did not know about.)
    let lo = |c: f64| ((c - view.radius) / root_size).floor() as i64 - 1;
    let hi = |c: f64| ((c + view.radius) / root_size).floor() as i64 + 1;
    let mut out = Selection {
        patches: Vec::new(),
        split: Vec::new(),
        slack: f64::INFINITY,
    };
    for x in lo(ex)..=hi(ex) {
        for z in lo(ez)..=hi(ez) {
            walk(
                view,
                &error_of,
                PatchKey::new(root_level, x as i32, z as i32),
                &mut out,
            );
        }
    }
    out
}

/// The coarsest level whose patches are at least `radius` wide, so a 3 x 3
/// block of them always covers the view: the level [`select`] starts its
/// walk from, and so the coarsest patch whose error can decide a split.
pub fn root_level_for(radius: f64) -> u32 {
    let mut level = 0;
    while ((PATCH_QUADS as i64) << level) < radius.ceil() as i64 {
        level += 1;
    }
    level
}

fn walk(
    view: &FarView,
    error_of: &impl Fn(PatchKey) -> Option<f32>,
    key: PatchKey,
    out: &mut Selection,
) {
    let (min, max) = patch_box(key);
    // Each threshold below is met by a distance from the eye, so the gap
    // between the two is how far the eye can move before that decision
    // flips: the walk's slack is the smallest of them. A decision the eye
    // cannot move (the hole) has none.
    let ground = ground_distance(view.eye, min, max);
    out.slack = out.slack.min((ground - view.radius).abs());
    if ground > view.radius {
        return;
    }
    if view.hole.is_some_and(|h| h.contains_box(min, max)) {
        return;
    }
    let splittable = key.level > FINEST_LEVEL;
    let d = distance(view.eye, min, max);
    let near_within = view.split_ratio * key.size() as f64;
    let near = d < near_within;
    let children_wide_enough = (key.quad() as f64 / 2.0) >= view.min_quad * d;
    let error = error_of(key);
    let rough = children_wide_enough && error.is_some_and(|e| e as f64 > view.max_error * d);
    if splittable {
        out.slack = out.slack.min((d - near_within).abs());
        if view.min_quad > 0.0 {
            let wide_within = key.quad() as f64 / 2.0 / view.min_quad;
            out.slack = out.slack.min((d - wide_within).abs());
        }
        // An error of 0 is never "rough" at any distance (`e > max_error *
        // d` with `d >= 0`), so it has no threshold to meet.
        if let Some(e) = error.filter(|&e| e > 0.0 && view.max_error > 0.0) {
            out.slack = out.slack.min((d - e as f64 / view.max_error).abs());
        }
    }
    if splittable && (near || rough) {
        out.split.push(key);
        for child in key.children() {
            walk(view, error_of, child, out);
        }
    } else {
        out.patches.push(key);
    }
}

/// A patch's box: its square of ground, and every height the terrain can
/// reach.
fn patch_box(key: PatchKey) -> ([f64; 3], [f64; 3]) {
    let [ox, oz] = key.origin();
    let size = key.size() as f64;
    let (lo, hi) = surface_bounds();
    (
        [ox as f64, lo, oz as f64],
        [ox as f64 + size, hi, oz as f64 + size],
    )
}

/// From `p` to the nearest point of the box.
fn distance(p: [f64; 3], min: [f64; 3], max: [f64; 3]) -> f64 {
    (0..3)
        .map(|a| (min[a] - p[a]).max(p[a] - max[a]).max(0.0).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// The same, across the ground only.
fn ground_distance(p: [f64; 3], min: [f64; 3], max: [f64; 3]) -> f64 {
    [0, 2]
        .iter()
        .map(|&a| (min[a] - p[a]).max(p[a] - max[a]).max(0.0).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// A patch's heights, row by row (`z` outer, `x` inner), [`PATCH_VERTS`] on a
/// side: the top of the ground at each vertex, in blocks.
#[derive(Clone, Debug, PartialEq)]
pub struct PatchHeights {
    pub key: PatchKey,
    pub heights: Vec<f32>,
    pub min: f32,
    pub max: f32,
    /// How far, in blocks, this patch's surface is from the next finer one's,
    /// measured at the middle of every quad and quad edge -- where the finer
    /// patch has a vertex of its own. This is the 99th percentile of those
    /// gaps; [`error_max`](Self::error_max) is the worst. What [`select`]
    /// weighs against `max_error`. Zero at `level` 0.
    ///
    /// The 99th, not the 90th: a mountain is a few percent of a patch, and a
    /// 90th percentile let its worst columns reach 3.9 px while the patch
    /// looked fine (review of #284). Not the maximum either -- that splits
    /// to the floor almost everywhere (`PROPOSAL_FAR_VIEW.md` §3.2).
    pub error: f32,
    /// The worst of the same gaps.
    pub error_max: f32,
    /// What every quad of the patch shows besides its height
    /// ([`QuadGround::of`]), row by row, [`PATCH_QUADS`] on a side.
    pub ground: Vec<QuadGround>,
    /// The top of every quad, in whole blocks ([`quad_top`]), row by row,
    /// [`TOPS_SIDE`] on a side: the patch's own quads and one ring of its
    /// neighbours' around them (the *apron*), so a step on the patch's edge
    /// is as exact as one inside it.
    pub tops: Vec<i32>,
    /// The patch as blocks: what [`mesh`] makes of `tops`.
    pub faces: Vec<Face>,
}

/// Quads along one side of [`PatchHeights::tops`]: the patch's own and one
/// more on either side.
pub const TOPS_SIDE: usize = PATCH_QUADS + 2;

/// Up to this width, in blocks, a quad's top is worked out from every one of
/// its columns ([`WorldGen::cell_top`]); wider, from [`TOP_SAMPLES`] squared
/// sample columns. Eight: the width of a cell of the voxel rings' coarsest
/// level, so that where the far terrain meets them a quad and a cell over
/// the same columns have the same top, exactly. **Chosen, not derived**
/// beyond that: an exact top costs a surface height per column, 64 a quad
/// here and four times that for every level above.
pub const EXACT_TOP_QUAD: i64 = 8;

/// Sample columns along a side of a quad wider than [`EXACT_TOP_QUAD`].
/// **Chosen, not derived**: what the height field's own vertices are sampled
/// with ([`FOOTPRINT_SAMPLES`]). A sampled top can miss a peak narrower than
/// the spacing, and a quad's top can stand above its parent's
/// (`measure::a_child_above_its_parent`).
pub const TOP_SAMPLES: i32 = 4;

/// Sample columns along a side of the quads of `key` for their top, and for
/// what the top wears: every column up to [`EXACT_TOP_QUAD`],
/// [`TOP_SAMPLES`] beyond.
fn top_samples(key: PatchKey) -> i32 {
    if key.quad() <= EXACT_TOP_QUAD {
        key.quad() as i32
    } else {
        TOP_SAMPLES
    }
}

/// Sample columns along a side of a quad for what its faces show
/// ([`WorldGen::ground_faces`]): the wall mix and the ambient occlusion.
/// Four up to twice [`EXACT_TOP_QUAD`] (16 blocks, where the far terrain
/// takes over from the voxels' coarsest cells), where a difference in colour
/// is seen side by side; two beyond. **Chosen, not derived**: each sample
/// column costs nine surface heights. Measured by `far_rows_report` in
/// `crates/render/tests/lod_consistency`: at quads of 16 two samples left
/// the tops' colour 3.8/255 off level 0's and the stone share of the walls
/// 2.5 points off on this seed's footprints (the voxel levels hold 3.0 and
/// 1.25); four leave 2.6 and 1.1.
fn face_samples(key: PatchKey) -> i32 {
    if key.quad() <= 2 * EXACT_TOP_QUAD {
        4
    } else {
        2
    }
}

/// **The one place a quad's top is decided** (the seam issue #422 asks for):
/// the height, in whole blocks, of the flat top the quad `(i, j)` of the
/// patch `key` is drawn with -- `-1` and [`PATCH_QUADS`] being its
/// neighbours' nearest quads.
///
/// It is the rule that one ground serves every level (issue #423,
/// [`WorldGen::cell_top_sampled`]): the mean of the level-0 surfaces of the
/// quad's columns, rounded to the block -- and, where the quad holds a
/// summit, the highest mean of its four half-width cells (#437) -- plus one
/// for the top face. So at level 0 a quad's top is its column's, and a
/// level-3 quad's is the level-3 voxel cell's over the same 64 columns, to
/// the block. Nothing here is the far terrain's own: a change to the
/// voxels' rule is a change to this.
pub fn quad_top(gen: &WorldGen, key: PatchKey, i: i32, j: i32) -> i32 {
    let q = key.quad();
    let [ox, oz] = key.origin();
    let (x0, z0) = (ox + i as i64 * q, oz + j as i64 * q);
    gen.cell_top_sampled(x0 as i32, z0 as i32, q as i32, top_samples(key)) + 1
}

/// What a quad's faces show besides their height, from what level 0 shows
/// over the same ground ([`WorldGen::ground_faces`]), in 255ths.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QuadGround {
    /// What the quad's top wears: the share of it that is the surface
    /// block's top, and that is soil; the rest is stone. All of it the
    /// surface block, but for a quad that shows the floor of a pit
    /// ([`WorldGen::top_layer`]): all soil, or all stone.
    pub top: [u8; 2],
    /// How open level 0's top faces are to the sky.
    pub top_open: u8,
    /// Per direction ([`crate::worldgen::GROUND_DIRECTIONS`]): the share of
    /// level 0's risers that is the surface block's side, and that is soil.
    /// The rest is stone.
    pub sides: [[u8; 2]; 4],
    /// Per direction, how open those risers are.
    pub side_open: [u8; 4],
}

impl QuadGround {
    /// Ground with nothing known about it: open, and the surface block's
    /// side all the way down.
    pub const PLAIN: QuadGround = QuadGround {
        top: [255, 0],
        top_open: 255,
        sides: [[255, 0]; 4],
        side_open: [255; 4],
    };

    /// **The one place a face's material and occlusion are decided**, on
    /// this side of the renderer: level 0's own faces over the quad's
    /// columns, by [`WorldGen::ground_faces`] -- so a wall is the mix of
    /// surface, soil and stone that level 0's risers facing its way are, by
    /// depth below each column's *own* surface (issue #423 decision 2), and
    /// not by how tall the wall is.
    ///
    /// Where level 0 has no riser facing a direction among the sampled
    /// columns, a wall facing it takes the mix of all four directions
    /// together, and the surface block's side if there is no riser at all.
    ///
    /// **The top wears what a voxel cell's top wears** over the same
    /// columns ([`WorldGen::top_layer`], the voxels' `top_block`): what
    /// level 0 shows from above one of them, picked by the cell's hash --
    /// the surface block, or a pit's floor in the share the columns are pit.
    /// `top` is the quad's own ([`quad_top`]): the hash is the cell's, and a
    /// cell is its place and its top.
    pub fn of(gen: &WorldGen, key: PatchKey, i: i32, j: i32, top: i32) -> Self {
        use crate::worldgen::GroundLayer;
        let q = key.quad();
        let [ox, oz] = key.origin();
        let (x0, z0) = (ox + i as i64 * q, oz + j as i64 * q);
        // `top - 1`: the top block, which the top face is one above.
        let wears = match gen.top_layer(x0 as i32, z0 as i32, q as i32, top_samples(key), top - 1) {
            GroundLayer::Surface => [255, 0],
            GroundLayer::Soil => [0, 255],
            GroundLayer::Stone => [0, 0],
        };
        let faces = gen.ground_faces(x0 as i32, z0 as i32, q as i32, face_samples(key));
        let unit = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        let all: [f32; 3] = [0, 1, 2].map(|l| faces.sides.iter().map(|s| s[l]).sum());
        let total: f32 = all.iter().sum();
        let fallback = if total > 0.0 {
            [unit(all[0] / total), unit(all[1] / total)]
        } else {
            [255, 0]
        };
        let sides = [0, 1, 2, 3].map(|d| {
            match (
                faces.share(d, GroundLayer::Surface),
                faces.share(d, GroundLayer::Soil),
            ) {
                (Some(surface), Some(soil)) => [unit(surface), unit(soil)],
                _ => fallback,
            }
        });
        QuadGround {
            top: wears,
            top_open: unit(faces.top_open),
            sides,
            side_open: faces.side_open.map(unit),
        }
    }
}
/// Samples per axis averaged into one vertex's height, at most.
const FOOTPRINT_SAMPLES: i64 = 4;

/// The heights of `key`, from the generator.
///
/// Each vertex stands for a quad's width of ground around it, and its height
/// is the mean top of that ground, sampled on a grid of at most
/// [`FOOTPRINT_SAMPLES`] a side. A single column would alias: at a 256-block
/// quad, one sample decides whether a vertex is on the peak or beside it, and
/// the answer changes as the patch moves. At `level` 0 a vertex is one
/// column, taken exactly.
pub fn generate(gen: &WorldGen, key: PatchKey) -> PatchHeights {
    let q = key.quad();
    let [ox, oz] = key.origin();
    let mut heights = Vec::with_capacity(PATCH_VERTS * PATCH_VERTS);
    for j in 0..PATCH_VERTS as i64 {
        for i in 0..PATCH_VERTS as i64 {
            heights.push(footprint_mean(gen, ox + i * q, oz + j * q, q));
        }
    }
    let mut tops = Vec::with_capacity(TOPS_SIDE * TOPS_SIDE);
    for j in -1..=PATCH_QUADS as i32 {
        for i in -1..=PATCH_QUADS as i32 {
            tops.push(quad_top(gen, key, i, j));
        }
    }
    let mut ground = Vec::with_capacity(PATCH_QUADS * PATCH_QUADS);
    for j in 0..PATCH_QUADS as i32 {
        for i in 0..PATCH_QUADS as i32 {
            let top = tops[(j + 1) as usize * TOPS_SIDE + (i + 1) as usize];
            ground.push(QuadGround::of(gen, key, i, j, top));
        }
    }
    let min = heights.iter().copied().fold(f32::INFINITY, f32::min);
    let max = heights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let (error_max, error) = if q < 2 {
        (0.0, 0.0)
    } else {
        finer_gap(gen, &heights, ox, oz, q)
    };
    let mut patch = PatchHeights {
        key,
        heights,
        min,
        max,
        error,
        error_max,
        ground,
        tops,
        faces: Vec::new(),
    };
    patch.faces = mesh(key, &patch.tops, &patch.ground);
    patch
}

/// The mean top of the ground in a `q`-block square centred on `(x, z)`,
/// sampled on a grid of at most [`FOOTPRINT_SAMPLES`] a side.
fn footprint_mean(gen: &WorldGen, x: i64, z: i64, q: i64) -> f32 {
    let samples = q.clamp(1, FOOTPRINT_SAMPLES);
    let step = q.max(1) / samples;
    let offset = |s: i64| s * step + step / 2 - q / 2;
    let mut sum = 0i64;
    for a in 0..samples {
        for b in 0..samples {
            sum += gen.surface_height((x + offset(a)) as i32, (z + offset(b)) as i32) as i64;
        }
    }
    // `+ 1`: the top face of the surface block.
    sum as f32 / (samples * samples) as f32 + 1.0
}

/// The gaps between what this patch draws and the vertex the next finer
/// patch has there: at every vertex (where the finer one averages half the
/// ground), at the middle of every quad edge, and at the middle of every quad.
fn finer_gap(gen: &WorldGen, heights: &[f32], ox: i64, oz: i64, q: i64) -> (f32, f32) {
    let h = |i: usize, j: usize| heights[j * PATCH_VERTS + i];
    let half = q / 2;
    let mut gaps = Vec::with_capacity(3 * PATCH_VERTS * PATCH_VERTS);
    let mut check = |x: i64, z: i64, blend: f32| {
        gaps.push((footprint_mean(gen, x, z, half) - blend).abs());
    };
    for j in 0..PATCH_VERTS {
        for i in 0..PATCH_VERTS {
            let (x, z) = (ox + i as i64 * q, oz + j as i64 * q);
            // The finer patch has a vertex here too, averaged over half the
            // ground: a peak narrower than a quad shows up here, not between
            // vertices (found in review of #284 -- 3.9 times the p90 gap).
            check(x, z, h(i, j));
            if i < PATCH_QUADS {
                check(x + half, z, (h(i, j) + h(i + 1, j)) / 2.0);
            }
            if j < PATCH_QUADS {
                check(x, z + half, (h(i, j) + h(i, j + 1)) / 2.0);
            }
            if i < PATCH_QUADS && j < PATCH_QUADS {
                let centre = (h(i, j) + h(i + 1, j) + h(i, j + 1) + h(i + 1, j + 1)) / 4.0;
                check(x + half, z + half, centre);
            }
        }
    }
    gaps.sort_by(f32::total_cmp);
    (gaps[gaps.len() - 1], gaps[gaps.len() * 99 / 100])
}

/// Everything a one-shot caller needs -- a screenshot, the bench, a golden
/// test: select, generate what is new, select again, until nothing new is
/// asked for. The patches drawn, in [`select`]'s order.
///
/// On one thread. A live game streams instead, spreading the same loop over
/// frames and workers; a one-shot caller has one frame to be right in.
pub fn build(gen: &WorldGen, view: &FarView) -> Vec<PatchHeights> {
    let mut have: std::collections::HashMap<PatchKey, PatchHeights> =
        std::collections::HashMap::new();
    loop {
        let keys = select(view, |k| have.get(&k).map(|p| p.error));
        let new: Vec<PatchKey> = keys
            .iter()
            .copied()
            .filter(|k| !have.contains_key(k))
            .collect();
        if new.is_empty() {
            return keys.iter().filter_map(|k| have.remove(k)).collect();
        }
        for k in new {
            have.insert(k, generate(gen, k));
        }
    }
}

/// Which way a [`Face`] looks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FaceDir {
    /// A quad's flat top.
    Top,
    /// A wall, by the direction it faces: the side of the higher of two
    /// neighbouring tops, looking out over the lower.
    PosX,
    NegX,
    PosZ,
    NegZ,
}

/// One rectangle of a patch drawn as blocks: a flat top, or a wall.
///
/// In quads from the patch's low corner, so every corner is on the lattice
/// of `2^level` blocks. A top spans `i0..i1` by `j0..j1` at height `top`. A
/// wall stands on one quad line -- `i0 == i1` for one facing along x,
/// `j0 == j1` for one facing along z -- runs along the other axis, and hangs
/// from `top` down to [`bottom`](Self::bottom).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Face {
    pub dir: FaceDir,
    pub i0: u8,
    pub j0: u8,
    pub i1: u8,
    pub j1: u8,
    /// The height of a top, or of a wall's upper edge, in blocks.
    pub top: i32,
    /// A wall: how many blocks of step it shows -- the difference between
    /// the two tops it stands between, exactly. Zero for a top, and for a
    /// wall on the patch's edge whose neighbour is no lower.
    pub rise: u32,
    /// A wall on the patch's own edge: the step down to the neighbouring
    /// patch's quad as this level sees it (the apron), and the patch's skirt
    /// below that, because the neighbour drawn may be of another level and
    /// its top there a little different ([`PatchHeights::skirt_blocks`]).
    pub border: bool,
    /// The share of the face, in 255ths, that is the surface block (its side
    /// on a wall, its top on a top) and that is soil; the rest is stone. A
    /// wall's is [`QuadGround::sides`] for the quad it belongs to and the
    /// way it faces; a top's is [`QuadGround::top`].
    pub mix: [u8; 2],
    /// How open the face is to the sky, in 255ths ([`QuadGround`]).
    pub open: u8,
}

impl Face {
    /// A wall's lower edge, in blocks, for a patch whose skirt is `skirt`
    /// deep. A top's own height.
    pub fn bottom(&self, skirt: i32) -> i32 {
        self.top - self.rise as i32 - if self.border { skirt } else { 0 }
    }
}

/// No face crosses a line of this many blocks, on either ground axis, and a
/// top that touches one is a single quad deep: a chunk.
///
/// The voxel rings' box -- the hole the far terrain leaves -- has its sides
/// on chunk edges (`node::covered_box_3d`: whole nodes of the schedule's
/// coarsest level, 128 blocks in the game and as little as one chunk for a
/// small bench or test scene), and the join there
/// (`cubara_render::far::join`) works on the first quad inside the box. A
/// face merged across the side, or a wide one against it, would be half in
/// the hole and half out. So the mesher keeps to the lines everywhere, since
/// any of them can be the box's side next.
///
/// A quad a chunk wide or wider (level 4 and up) is a cell of its own: it
/// merges with nothing. That costs little, measured: on the generated
/// terrain more than half of all quad edges carry a step at every level, and
/// two tops merge only if they are as open to the sky as each other, so
/// there is next to nothing to merge (`measure::blocks_against_smooth`).
pub const JOIN_GRID: i64 = 16;

/// What may merge with what: two quads are in the same group when one face
/// may cover both. Within a [`JOIN_GRID`] cell that is its inner quads
/// together, each of its four edge rows on its own, and each corner quad
/// alone.
fn merge_groups(key: PatchKey) -> Vec<u32> {
    // Quads a cell; a quad wider than a cell is one itself.
    let n = (JOIN_GRID / key.quad()).max(1);
    // Where a quad is in its cell along one axis: the cell, and whether it
    // is the cell's first (1) or last (2) quad, or neither (0).
    let along = |patch: i32, k: usize| {
        let g = patch as i64 * PATCH_QUADS as i64 + k as i64;
        let (cell, at) = (g.div_euclid(n), g.rem_euclid(n));
        let cell = (cell - (patch as i64 * PATCH_QUADS as i64).div_euclid(n)) as u32;
        let edge = if at == 0 {
            1
        } else if at == n - 1 {
            2
        } else {
            0
        };
        (cell, edge)
    };
    let mut groups = Vec::with_capacity(PATCH_QUADS * PATCH_QUADS);
    for j in 0..PATCH_QUADS {
        for i in 0..PATCH_QUADS {
            let ((ci, ei), (cj, ej)) = (along(key.x, i), along(key.z, j));
            groups.push(if ei != 0 && ej != 0 {
                // A corner quad: alone.
                0x8000_0000 | (j * PATCH_QUADS + i) as u32
            } else {
                ci | cj << 6 | ei << 12 | ej << 14
            });
        }
    }
    groups
}

/// Whether quad lines `k` and `k + 1` along one axis of a patch at `patch`
/// on its level's grid are in the same [`JOIN_GRID`] cell -- whether a wall
/// may run across the quad line between them.
fn same_cell(key: PatchKey, patch: i32, k: usize) -> bool {
    let n = (JOIN_GRID / key.quad()).max(1);
    let g = patch as i64 * PATCH_QUADS as i64 + k as i64;
    g.div_euclid(n) == (g + 1).div_euclid(n)
}

/// **A patch as blocks** (issue #422): from the tops of its quads and of the
/// ring of neighbours around them (`tops`, [`TOPS_SIDE`] a side, as
/// [`PatchHeights::tops`]), the faces a voxel mesher would make of the same
/// columns.
///
/// - **Tops.** Every quad is flat at its top. Neighbours of the same height
///   that wear the same and are as open to the sky merge into one rectangle,
///   greedily, row by row -- within the limits of [`JOIN_GRID`].
/// - **Walls.** Wherever two neighbouring tops differ by `k` blocks there is
///   a wall exactly `k` blocks tall, standing on the line between them,
///   belonging to the higher and facing the lower. Walls in a row with the
///   same top and the same height merge.
/// - **The edge.** Along the patch's own edge every quad has a wall facing
///   out, whether or not the neighbour is lower: what the skirt was. It
///   shows the step down to the neighbour's quad as this level knows it,
///   and hangs a skirt's depth below that ([`Face::border`]). Both patches
///   on an edge draw one, back to back; whichever is higher covers the step.
///
/// Every coordinate is a whole number of quads and blocks, so the faces of
/// one patch meet exactly, and so do those of two patches of one level.
/// Deterministic: the faces come out tops first, then the walls facing along
/// x line by line, then those along z.
pub fn mesh(key: PatchKey, tops: &[i32], ground: &[QuadGround]) -> Vec<Face> {
    assert_eq!(tops.len(), TOPS_SIDE * TOPS_SIDE, "a patch's tops");
    assert_eq!(ground.len(), PATCH_QUADS * PATCH_QUADS, "a patch's ground");
    const N: usize = PATCH_QUADS;
    // By quad, -1 and N being the apron.
    let top = |i: i32, j: i32| tops[(j + 1) as usize * TOPS_SIDE + (i + 1) as usize];
    let groups = merge_groups(key);
    let mut faces = Vec::new();

    // Tops: the widest run of equal quads along x, then as many equal rows
    // of it along z as there are.
    let mut done = [false; N * N];
    let same = |a: usize, b: usize, done: &[bool]| {
        !done[b]
            && groups[a] == groups[b]
            && ground[a].top == ground[b].top
            && ground[a].top_open == ground[b].top_open
            && top((a % N) as i32, (a / N) as i32) == top((b % N) as i32, (b / N) as i32)
    };
    for j in 0..N {
        for i in 0..N {
            let first = j * N + i;
            if done[first] {
                continue;
            }
            let mut w = 1;
            while i + w < N && same(first, first + w, &done) {
                w += 1;
            }
            let mut d = 1;
            while j + d < N && (0..w).all(|k| same(first, (j + d) * N + i + k, &done)) {
                d += 1;
            }
            for dj in 0..d {
                for di in 0..w {
                    done[(j + dj) * N + i + di] = true;
                }
            }
            faces.push(Face {
                dir: FaceDir::Top,
                i0: i as u8,
                j0: j as u8,
                i1: (i + w) as u8,
                j1: (j + d) as u8,
                top: top(i as i32, j as i32),
                rise: 0,
                border: false,
                mix: ground[first].top,
                open: ground[first].top_open,
            });
        }
    }

    // Walls. `along_x`: the walls facing along x, standing on the lines of
    // constant x and running along z; otherwise the other way round.
    for along_x in [true, false] {
        let (patch_run, pos, neg) = if along_x {
            (key.z, FaceDir::PosX, FaceDir::NegX)
        } else {
            (key.x, FaceDir::PosZ, FaceDir::NegZ)
        };
        // What the quad `at` along the run, on the low (`false`) or high
        // side of `line`, shows on a wall facing `dir`: the wall is that
        // quad's.
        let shown = |line: i32, k: usize, high: bool, dir: FaceDir| {
            let across = if high { line } else { line - 1 } as usize;
            let g = if along_x {
                ground[k * N + across]
            } else {
                ground[across * N + k]
            };
            let d = dir as usize - 1;
            (g.sides[d], g.side_open[d])
        };
        for line in 0..=N as i32 {
            // The wall on this line beside quad `k` of the run, if any:
            // (direction, top, rise, border, what it shows).
            let wall = |k: usize| {
                let (below, above) = if along_x {
                    (top(line - 1, k as i32), top(line, k as i32))
                } else {
                    (top(k as i32, line - 1), top(k as i32, line))
                };
                if line == 0 {
                    // This patch's quad is the one above the line.
                    let rise = (above - below).max(0) as u32;
                    Some((neg, above, rise, true, shown(line, k, true, neg)))
                } else if line == N as i32 {
                    let rise = (below - above).max(0) as u32;
                    Some((pos, below, rise, true, shown(line, k, false, pos)))
                } else if below > above {
                    let rise = (below - above) as u32;
                    Some((pos, below, rise, false, shown(line, k, false, pos)))
                } else if above > below {
                    let rise = (above - below) as u32;
                    Some((neg, above, rise, false, shown(line, k, true, neg)))
                } else {
                    None
                }
            };
            let mut k = 0;
            while k < N {
                let Some(this) = wall(k) else {
                    k += 1;
                    continue;
                };
                let mut end = k + 1;
                while end < N && same_cell(key, patch_run, end - 1) && wall(end) == Some(this) {
                    end += 1;
                }
                let (dir, top, rise, border, (mix, open)) = this;
                let (line, k0, k1) = (line as u8, k as u8, end as u8);
                let (i0, j0, i1, j1) = if along_x {
                    (line, k0, line, k1)
                } else {
                    (k0, line, k1, line)
                };
                faces.push(Face {
                    dir,
                    i0,
                    j0,
                    i1,
                    j1,
                    top,
                    rise,
                    border,
                    mix,
                    open,
                });
                k = end;
            }
        }
    }
    faces
}

impl PatchHeights {
    /// The top of the quad `(i, j)`, in whole blocks; `-1` and
    /// [`PATCH_QUADS`] are the neighbouring patches' nearest quads.
    pub fn top(&self, i: i32, j: i32) -> i32 {
        self.tops[(j + 1) as usize * TOPS_SIDE + (i + 1) as usize]
    }

    /// [`skirt`](Self::skirt) in whole blocks, rounded up: how far a wall on
    /// the patch's edge hangs below the step it shows.
    pub fn skirt_blocks(&self) -> i32 {
        self.skirt().ceil() as i32
    }

    /// How far the patch's skirt drops below its edge, in blocks: enough to
    /// cover the gap to a neighbour two levels coarser (a split for roughness
    /// can put one there), whose edge is a straight line across four of this
    /// patch's quads -- a slope of one block per block opens at most that --
    /// plus this patch's own worst gap to its finer self.
    pub fn skirt(&self) -> f32 {
        4.0 * self.key.quad() as f32 + self.error_max
    }

    /// The drawn surface at block column `(x, z)`: the bilinear blend of the
    /// four vertices around it, which is what the rasterizer draws between
    /// them (to within the diagonal each quad is split along).
    pub fn height_at(&self, x: f64, z: f64) -> f64 {
        let [ox, oz] = self.key.origin();
        let q = self.key.quad() as f64;
        let fx = ((x - ox as f64) / q).clamp(0.0, PATCH_QUADS as f64);
        let fz = ((z - oz as f64) / q).clamp(0.0, PATCH_QUADS as f64);
        let (i, j) = (
            (fx.floor() as usize).min(PATCH_QUADS - 1),
            (fz.floor() as usize).min(PATCH_QUADS - 1),
        );
        let (tx, tz) = (fx - i as f64, fz - j as f64);
        let h = |i: usize, j: usize| self.heights[j * PATCH_VERTS + i] as f64;
        let top = h(i, j) * (1.0 - tx) + h(i + 1, j) * tx;
        let bottom = h(i, j + 1) * (1.0 - tx) + h(i + 1, j + 1) * tx;
        top * (1.0 - tz) + bottom * tz
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::HashMap;

    const SEED: u64 = 0x005E_ED00_00C0_FFEE;

    /// 1080p, the 60° field of view the camera uses, quads of 8 px.
    fn ratio() -> f64 {
        split_ratio_for(8.0, 1080.0, 60f64.to_radians())
    }

    fn view(eye: [f64; 3], radius: f64) -> FarView {
        FarView {
            eye,
            split_ratio: ratio(),
            max_error: PIXEL,
            min_quad: 2.0 * PIXEL,
            radius,
            hole: None,
        }
    }

    /// One pixel at 1080p with a 60° field of view, in radians.
    const PIXEL: f64 = std::f64::consts::FRAC_PI_3 / 1080.0;

    /// What a streaming caller does: select, generate what is new, select
    /// again, until nothing new is asked for.
    pub(super) fn converge(
        gen: &WorldGen,
        view: &FarView,
    ) -> (Vec<PatchKey>, HashMap<PatchKey, PatchHeights>) {
        let mut have: HashMap<PatchKey, PatchHeights> = HashMap::new();
        loop {
            let keys = select(view, |k| have.get(&k).map(|p| p.error));
            let new: Vec<PatchKey> = keys
                .iter()
                .copied()
                .filter(|k| !have.contains_key(k))
                .collect();
            if new.is_empty() {
                return (keys, have);
            }
            // A selection that never settles fails here rather than running
            // until somebody notices (a broken distance splits to level 0).
            assert!(
                have.len() < 20_000,
                "{} patches and still asking",
                have.len()
            );
            for k in new {
                have.insert(k, generate(gen, k));
            }
        }
    }

    /// Which selected patch covers each ground cell of `cell` blocks.
    fn coverage(keys: &[PatchKey], cell: i64, lo: i64, hi: i64) -> HashMap<(i64, i64), usize> {
        let mut map: HashMap<(i64, i64), usize> = HashMap::new();
        for k in keys {
            let [ox, oz] = k.origin();
            let s = k.size();
            let mut x = (ox.max(lo)).div_euclid(cell) * cell;
            while x < (ox + s).min(hi) {
                let mut z = (oz.max(lo)).div_euclid(cell) * cell;
                while z < (oz + s).min(hi) {
                    if x >= ox && z >= oz {
                        *map.entry((x, z)).or_default() += 1;
                    }
                    z += cell;
                }
                x += cell;
            }
        }
        map
    }

    #[test]
    fn a_ratio_gives_quads_of_the_asked_size() {
        // 8 px at 1080p / 60 degrees: a pixel is 0.97 mrad, a quad 7.8 mrad.
        let r = ratio();
        let quad_angle = 1.0 / (r * PATCH_QUADS as f64);
        let pixel = 60f64.to_radians() / 1080.0;
        assert!(
            (quad_angle / pixel - 8.0).abs() < 1e-9,
            "{}",
            quad_angle / pixel
        );
    }

    /// Every patch drawn, other than the finest, is at least `split_ratio`
    /// widths away -- so none of its quads looks wider than asked. This is
    /// the pixel bound the proposal's criterion rests on (§3.1).
    #[test]
    fn no_quad_is_drawn_wider_than_asked() {
        for eye in [[8.0, 40.0, 8.0], [8.0, 300.0, 8.0], [8.0, 3000.0, 8.0]] {
            let v = view(eye, 262_144.0);
            for key in select(&v, |_| None) {
                // The finest patches are the join's: never finer than the
                // voxel rings' coarsest cell, so as wide as they are.
                if key.level <= FINEST_LEVEL {
                    continue;
                }
                let (min, max) = patch_box(key);
                let d = distance(eye, min, max);
                let angle = key.quad() as f64 / d;
                let limit = 1.0 / (v.split_ratio * PATCH_QUADS as f64);
                assert!(
                    angle <= limit * (1.0 + 1e-9),
                    "{key:?} from {eye:?}: quad {angle} rad, limit {limit}"
                );
            }
        }
    }

    /// The patches tile the ground: every cell within the radius belongs to
    /// exactly one of them.
    #[test]
    fn the_patches_tile_the_ground_exactly_once() {
        let eye = [8.0, 300.0, 8.0];
        let radius = 20_000.0;
        let keys = select(&view(eye, radius), |_| None);
        // Cells of 512 blocks, out to where the view certainly reaches. A
        // multiple of the cell, so the cells checked are the cells counted.
        let reach = 12_288;
        let map = coverage(&keys, 512, -reach, reach);
        for x in (-reach..reach).step_by(512) {
            for z in (-reach..reach).step_by(512) {
                let n = map.get(&(x, z)).copied().unwrap_or(0);
                assert_eq!(n, 1, "cell ({x}, {z}) covered {n} times");
            }
        }
    }

    /// Neighbouring patches differ by at most one level, which is what lets
    /// a vertex on a shared edge blend toward its coarser neighbour without a
    /// crack (the geomorphing in `PROPOSAL_FAR_VIEW.md` §3.2).
    #[test]
    fn neighbours_differ_by_at_most_one_level() {
        // By distance alone. A split for roughness can put a fine patch next
        // to a coarse one two levels up; the skirts cover that edge, and the
        // error bound keeps what it hides under a pixel.
        let keys = select(&view([8.0, 40.0, 8.0], 60_000.0), |_| None);
        // Level of each 32-block cell along both axes through the eye, and a
        // diagonal line, sampled finely enough to see every boundary.
        let level_at = |x: i64, z: i64| {
            keys.iter()
                .find(|k| {
                    let [ox, oz] = k.origin();
                    (ox..ox + k.size()).contains(&x) && (oz..oz + k.size()).contains(&z)
                })
                .map(|k| i64::from(k.level))
        };
        for line in [(1, 0), (0, 1), (1, 1)] {
            let mut prev: Option<i64> = None;
            for t in (-50_000i64..50_000).step_by(32) {
                let here = level_at(t * line.0, t * line.1);
                if let (Some(a), Some(b)) = (prev, here) {
                    assert!(
                        (a - b).abs() <= 1,
                        "levels {a} and {b} meet at {t} on {line:?}"
                    );
                }
                prev = here;
            }
        }
    }

    #[test]
    fn nothing_is_selected_inside_the_hole() {
        let hole = Hole {
            min: [-1024.0, -512.0, -1024.0],
            max: [1024.0, 512.0, 1024.0],
        };
        let keys = select(
            &FarView {
                hole: Some(hole),
                ..view([8.0, 40.0, 8.0], 20_000.0)
            },
            |_| None,
        );
        assert!(!keys.is_empty());
        for k in &keys {
            let (min, max) = patch_box(*k);
            assert!(
                !hole.contains_box(min, max),
                "{k:?} lies wholly in the hole"
            );
        }
        // And something still covers the hole's edge, where the rings end.
        let map = coverage(&keys, 64, 960, 1088);
        assert!(map.contains_key(&(1024, 1024)), "a gap where the rings end");
    }

    /// **The far terrain's finest quad is the voxel rings' coarsest cell**,
    /// at any screen height (#422, the review of #423): on a tall display the
    /// pixel floor lets quads of 4 reach out to 1,589 blocks, finer than the
    /// level-3 cells at the join, which made a fine band between two coarse
    /// ones. Rough ground everywhere (`error_of` huge) is the worst case.
    #[test]
    fn the_far_terrain_is_never_finer_than_the_join() {
        let hole = Hole {
            min: [-1024.0, -1024.0, -1024.0],
            max: [1024.0, 1024.0, 1024.0],
        };
        // The coarsest voxel cell is 8 blocks: the level the far terrain stops at.
        assert_eq!(1i64 << FINEST_LEVEL, 8);
        for rows in [1080.0, 1664.0, 2880.0, 4320.0] {
            let pixel = std::f64::consts::FRAC_PI_3 / rows;
            let keys = select(
                &FarView {
                    eye: [8.0, 40.0, 8.0],
                    split_ratio: split_ratio_for(16.0, rows, std::f64::consts::FRAC_PI_3),
                    max_error: 0.0,
                    min_quad: 4.0 * pixel,
                    radius: 20_000.0,
                    hole: Some(hole),
                },
                |_| Some(1e9),
            );
            assert!(!keys.is_empty());
            let finest = keys.iter().map(|k| k.level).min().expect("patches");
            assert_eq!(finest, FINEST_LEVEL, "{rows} rows");
        }
    }

    /// From 3 km up the voxel rings reach nothing, and the far terrain is all
    /// there is below: it must reach right under the eye.
    #[test]
    fn from_high_up_the_ground_below_is_covered() {
        let keys = select(&view([8.0, 3000.0, 8.0], 262_144.0), |_| None);
        let under = keys.iter().any(|k| {
            let [ox, oz] = k.origin();
            (ox..ox + k.size()).contains(&8) && (oz..oz + k.size()).contains(&8)
        });
        assert!(under, "nothing covers the ground under the eye");
    }

    /// At level 0 a vertex is one column, exactly: the top face of its surface
    /// block. Every coarser level averages this, so an offset here is an
    /// offset everywhere -- one block, which the pixel tests below would call
    /// close enough at a kilometre.
    #[test]
    fn a_level_0_vertex_is_the_top_of_its_column() {
        let gen = WorldGen::new(SEED);
        for key in [PatchKey::new(0, 0, 0), PatchKey::new(0, -3, 7)] {
            let patch = generate(&gen, key);
            let [ox, oz] = key.origin();
            for (i, j) in [(0, 0), (5, 17), (PATCH_QUADS, PATCH_QUADS)] {
                let top = gen.surface_height((ox + i as i64) as i32, (oz + j as i64) as i32) + 1;
                assert_eq!(
                    patch.heights[j * PATCH_VERTS + i],
                    top as f32,
                    "{key:?} ({i}, {j})"
                );
            }
            assert_eq!(patch.heights.len(), (PATCH_QUADS + 1) * (PATCH_QUADS + 1));
        }
    }

    /// `error_max` is the worst gap, found independently: at every vertex,
    /// quad centre and edge midpoint, the finer footprint's mean against this
    /// patch's surface there. The skirt is sized from it, so a smaller number here is
    /// a seam that opens on the roughest edge.
    #[test]
    fn error_max_is_the_worst_gap_there_is() {
        let gen = WorldGen::new(SEED);
        // A patch in the mountains, where the gaps are uneven.
        let key = PatchKey::containing(5, -848, 4832);
        let p = generate(&gen, key);
        let [ox, oz] = key.origin();
        let q = key.quad();
        let h = |i: usize, j: usize| p.heights[j * PATCH_VERTS + i];
        let mut worst = 0f32;
        for j in 0..PATCH_VERTS {
            for i in 0..PATCH_VERTS {
                let (x, z) = (ox + i as i64 * q, oz + j as i64 * q);
                worst = worst.max((footprint_mean(&gen, x, z, q / 2) - h(i, j)).abs());
            }
        }
        for j in 0..PATCH_QUADS {
            for i in 0..PATCH_QUADS {
                let (x, z) = (ox + i as i64 * q, oz + j as i64 * q);
                let centre = (h(i, j) + h(i + 1, j) + h(i, j + 1) + h(i + 1, j + 1)) / 4.0;
                for (fx, fz, blend) in [
                    (x + q / 2, z, (h(i, j) + h(i + 1, j)) / 2.0),
                    (x, z + q / 2, (h(i, j) + h(i, j + 1)) / 2.0),
                    (x + q / 2, z + q / 2, centre),
                ] {
                    worst = worst.max((footprint_mean(&gen, fx, fz, q / 2) - blend).abs());
                }
            }
        }
        assert!(
            worst > 1.0,
            "a mountain patch with no gap to speak of: {worst}"
        );
        assert_eq!(p.error_max, worst);
        assert!(p.error <= p.error_max);
    }

    /// Two patches side by side share an edge, and must agree on it height
    /// for height, or the seam between them opens. At every level.
    #[test]
    fn neighbouring_patches_agree_along_their_shared_edge() {
        let gen = WorldGen::new(SEED);
        for level in [0, 3, 7] {
            let a = generate(&gen, PatchKey::new(level, 2, 5));
            let east = generate(&gen, PatchKey::new(level, 3, 5));
            let south = generate(&gen, PatchKey::new(level, 2, 6));
            for k in 0..PATCH_VERTS {
                let at = |p: &PatchHeights, i: usize, j: usize| p.heights[j * PATCH_VERTS + i];
                assert_eq!(
                    at(&a, PATCH_QUADS, k),
                    at(&east, 0, k),
                    "level {level}, east edge {k}"
                );
                assert_eq!(
                    at(&a, k, PATCH_QUADS),
                    at(&south, k, 0),
                    "level {level}, south edge {k}"
                );
            }
        }
    }

    /// Where the far terrain starts: the voxel rings draw everything nearer
    /// (`PROPOSAL_FAR_VIEW.md` §3.2), so no patch is ever seen from closer.
    const FAR_TERRAIN_STARTS: f64 = 1024.0;

    /// A patch whose error would show is split -- and stops splitting once
    /// its quads would get narrower than `min_quad`, however rough the land.
    #[test]
    fn rough_patches_are_split_down_to_the_floor_and_no_further() {
        let v = FarView {
            radius: 40_000.0,
            ..view([8.0, 300.0, 8.0], 40_000.0)
        };
        let flat = select(&v, |_| Some(0.0));
        let rough = select(&v, |_| Some(1.0e9));
        // Measured 5,300 against 1,690: rough land gets about three times the
        // patches, and the floor is what stops it at three rather than more.
        assert!(
            rough.len() > flat.len() * 2,
            "{} vs {}",
            rough.len(),
            flat.len()
        );
        for key in &rough {
            let (min, max) = patch_box(*key);
            let d = distance(v.eye, min, max);
            // Its children would have been narrower than the floor.
            assert!(
                key.level <= FINEST_LEVEL || (key.quad() as f64 / 2.0) < v.min_quad * d,
                "{key:?} could have been split and was not"
            );
            // And the floor is honoured by the patch itself.
            assert!(
                key.quad() as f64 >= v.min_quad * d / 2.0 || key.level <= FINEST_LEVEL,
                "{key:?} split below the floor"
            );
        }
    }

    /// Per level, the far terrain's height against the ground a pixel covers,
    /// in pixels, at the nearest distance that level is drawn from: sampled
    /// over a grid of columns across flat land and mountains alike.
    fn height_error_px(gen: &WorldGen, view: &FarView) -> Vec<f64> {
        let (keys, have) = converge(gen, view);
        let mut errors = Vec::new();
        for sx in 0..24i64 {
            for sz in 0..24i64 {
                let (x, z) = (sx * 1579 - 18_000, sz * 1433 - 17_000);
                let Some(key) = keys.iter().find(|k| {
                    let [ox, oz] = k.origin();
                    (ox..ox + k.size()).contains(&x) && (oz..oz + k.size()).contains(&z)
                }) else {
                    continue;
                };
                let drawn = have[key].height_at(x as f64, z as f64);
                let d = {
                    let dx = x as f64 - view.eye[0];
                    let dz = z as f64 - view.eye[2];
                    (dx * dx + dz * dz).sqrt().max(FAR_TERRAIN_STARTS)
                };
                // The ground one pixel covers there, averaged.
                let foot = (d * PIXEL).max(1.0) as i64;
                let n = foot.min(8);
                let step = (foot / n).max(1);
                let mut sum = 0.0;
                for a in 0..n {
                    for b in 0..n {
                        sum += gen.surface_height((x + a * step) as i32, (z + b * step) as i32)
                            as f64
                            + 1.0;
                    }
                }
                errors.push((drawn - sum / (n * n) as f64).abs() / (d * PIXEL));
            }
        }
        errors.sort_by(f64::total_cmp);
        errors
    }

    #[test]
    #[ignore]
    fn print_error_per_config() {
        let gen = WorldGen::new(SEED);
        for (px, err) in [
            (32.0, f64::INFINITY),
            (32.0, 2.0),
            (16.0, f64::INFINITY),
            (16.0, 2.0),
            (16.0, 1.0),
            (8.0, f64::INFINITY),
            (8.0, 1.0),
            (4.0, f64::INFINITY),
        ] {
            let v = FarView {
                max_error: err * PIXEL,
                split_ratio: split_ratio_for(px, 1080.0, std::f64::consts::FRAC_PI_3),
                ..view([8.0, 300.0, 8.0], 25_000.0)
            };
            let e = height_error_px(&gen, &v);
            let mean = e.iter().sum::<f64>() / e.len() as f64;
            println!(
                "{px} px / {err} px: mean {mean:.2}, p90 {:.2}, p99 {:.2}, max {:.2}",
                e[e.len() * 9 / 10],
                e[e.len() * 99 / 100],
                e[e.len() - 1]
            );
        }
    }

    /// **The point of this module** (`PROPOSAL_FAR_VIEW.md` §2A, table B): a
    /// cube-shaped LOD node drew plain ground at y = 28 as 48-64 at 1-8 km --
    /// 20-36 blocks, up to 25 px -- and at y = 0 from 8 km. Here, with 16 px
    /// quads and a 1 px height bound, the drawn height is under a pixel from
    /// the ground it stands for in nine columns out of ten, out to 25 km in
    /// every direction, mountains included. With the shipped settings -- 2 px
    /// on each patch's 99th-percentile gap, quads no narrower than 4 px --
    /// measured: mean 0.39 px, p90 1.05, p99 2.72, worst 3.60.
    #[test]
    fn the_drawn_height_is_within_a_pixel_of_the_ground() {
        let gen = WorldGen::new(SEED);
        let v = FarView {
            max_error: 2.0 * PIXEL,
            min_quad: 4.0 * PIXEL,
            split_ratio: split_ratio_for(16.0, 1080.0, std::f64::consts::FRAC_PI_3),
            ..view([8.0, 300.0, 8.0], 25_000.0)
        };
        let e = height_error_px(&gen, &v);
        let mean = e.iter().sum::<f64>() / e.len() as f64;
        let p90 = e[e.len() * 9 / 10];
        assert!(e.len() > 400, "only {} columns sampled", e.len());
        let p99 = e[e.len() * 99 / 100];
        assert!(mean < 0.45, "mean {mean:.2} px");
        assert!(p90 < 1.2, "p90 {p90:.2} px");
        assert!(p99 < 3.0, "p99 {p99:.2} px");
    }

    /// And the height bound is what keeps it there: the same view with
    /// patches chosen by distance alone is measurably worse. Without this,
    /// the test above could pass with the error never consulted.
    #[test]
    fn the_height_bound_is_what_keeps_the_error_down() {
        let gen = WorldGen::new(SEED);
        let base = FarView {
            split_ratio: split_ratio_for(16.0, 1080.0, std::f64::consts::FRAC_PI_3),
            ..view([8.0, 300.0, 8.0], 25_000.0)
        };
        let bounded = height_error_px(
            &gen,
            &FarView {
                max_error: 2.0 * PIXEL,
                min_quad: 4.0 * PIXEL,
                ..base
            },
        );
        let unbounded = height_error_px(
            &gen,
            &FarView {
                max_error: f64::INFINITY,
                ..base
            },
        );
        let p90 = |e: &[f64]| e[e.len() * 9 / 10];
        // Measured 1.05 px against 1.58.
        assert!(
            p90(&unbounded) > p90(&bounded) * 1.4,
            "bounded p90 {:.2} px, unbounded {:.2} px",
            p90(&bounded),
            p90(&unbounded)
        );
    }

    /// A fixed, cheap stand-in for a patch's error: 0 to 8 blocks from a
    /// small integer hash of its key. Known for every patch, so the pin does
    /// not need generated heights.
    fn pinned_error(key: PatchKey) -> Option<f32> {
        let h = (key.level as u64)
            .wrapping_mul(0x9E37_79B9)
            .wrapping_add((key.x as i64 as u64).wrapping_mul(0x85EB_CA6B))
            .wrapping_add((key.z as i64 as u64).wrapping_mul(0xC2B2_AE35));
        Some(((h >> 7) % 9) as f32)
    }

    /// FNV-1a over each key's level, x and z as little-endian bytes, in order.
    fn fnv1a(keys: &[PatchKey]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for k in keys {
            let bytes = k
                .level
                .to_le_bytes()
                .into_iter()
                .chain(k.x.to_le_bytes())
                .chain(k.z.to_le_bytes());
            for b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    }

    /// Rule 6: `select` is shared by the game, `far::build`, the screenshot,
    /// the census and the bench, and the streaming work of #398 touches how it
    /// is called. Its output, for eyes the game uses and a few it does not
    /// (the origin, a half-block off it, far from it), is pinned as a count
    /// and a hash of the keys in order. The constants were recorded by
    /// running this on `main`'s `select`, before any change -- and re-recorded
    /// deliberately twice since. For #422: no patch is finer than the voxel
    /// rings' coarsest cell ([`FINEST_LEVEL`]), which removes every level 0
    /// to 2 patch (the counts fell from 3,000 and more to 1,300 and 2,100).
    /// For #398: the walk starts from every root within reach of the eye
    /// itself, not of the eye cut to whole blocks, and one more on every
    /// side (the slack needs it, `following_the_slack_keeps_...`). Two pins
    /// of twelve moved, both at the game's radius and the eye at the
    /// origin, by four patches each: roots whose ground starts exactly at
    /// the radius (the eye at 0) or half a block inside it (the eye at
    /// -0.5), which the old range left out -- 262 km off, past where the
    /// fog ends (`fog_finishes_inside_the_far_terrain`), so nothing on
    /// screen changes.
    #[test]
    fn select_output_is_pinned() {
        let eyes = [
            [8.0, 120.0, 8.0],
            [8.0, 400.0, 8.0],
            [8.0, 450.0, 8.0],
            [0.0, 120.0, 0.0],
            [-0.5, 120.0, -0.5],
            [100000.3, 300.0, -70000.6],
        ];
        let mut got = Vec::new();
        for radius in [20_000.0, 262_144.0] {
            for eye in eyes {
                let keys = select(&view(eye, radius), pinned_error);
                got.push((keys.len(), fnv1a(&keys)));
            }
        }
        println!("{got:#?}");
        assert_eq!(got, PINNED.to_vec());
    }

    /// The sizes the game draws at (High, quads of 16 pixels, an error
    /// bound of 2 and a floor of 4, three times as wide: the app's
    /// `FAR_QUAD_SCALE`), which this crate cannot read (Rule 3). Coarser
    /// than [`view`]'s: fewer patches, and thresholds further apart.
    fn game_view(eye: [f64; 3], radius: f64) -> FarView {
        FarView {
            split_ratio: split_ratio_for(48.0, 1080.0, std::f64::consts::FRAC_PI_3),
            max_error: 6.0 * PIXEL,
            min_quad: 12.0 * PIXEL,
            hole: Some(voxel_box()),
            ..view(eye, radius)
        }
    }

    /// The voxels' box around the low flight's start, in blocks.
    fn voxel_box() -> Hole {
        Hole {
            min: [-1024.0, -1024.0, -1024.0],
            max: [1152.0, 1024.0, 1152.0],
        }
    }

    /// **Holding a selection until the eye has moved its slack keeps it
    /// exact at every step** (#398's plan A): along lines of flight at the
    /// flight's 0.4 block a frame, a selection is worked out again only
    /// once the eye has moved as far as the last one's slack, and at every
    /// step in between the held one is what the eye there selects. Each of
    /// the walk's thresholds gets a flight that crosses it: three of the
    /// gate's lines with the errors pinned (the split distance and the
    /// error's), the low flight over ground rough everywhere (the width
    /// floor decides every split beyond the split distance), and a radius
    /// of 1,500 blocks the flight moves across. Not vacuous: the selection
    /// changes on the way, and fewer selections are made than steps.
    #[test]
    fn following_the_slack_keeps_the_selection_exact_at_every_step() {
        fn rough_everywhere(_: PatchKey) -> Option<f32> {
            Some(1e9)
        }
        type ErrorOf = fn(PatchKey) -> Option<f32>;
        type ViewAt<'a> = &'a dyn Fn([f64; 3]) -> FarView;
        type Flight<'a> = ([f64; 3], [f64; 3], ViewAt<'a>, ErrorOf);
        let gate = |eye: [f64; 3]| game_view(eye, 20_000.0);
        let small = |eye: [f64; 3]| FarView {
            hole: None,
            ..game_view(eye, 1_500.0)
        };
        let flights: [Flight; 5] = [
            ([8.0, 120.0, 8.0], [1.0, -0.15, 0.3], &gate, pinned_error),
            (
                [300.5, 90.0, -41.25],
                [-0.7, 0.2, 0.68],
                &gate,
                pinned_error,
            ),
            ([8.0, 450.0, 8.0], [0.4, -0.8, 0.2], &gate, pinned_error),
            (
                [8.0, 120.0, 8.0],
                [1.0, -0.15, 0.3],
                &gate,
                rough_everywhere,
            ),
            ([8.0, 120.0, 8.0], [1.0, 0.0, 0.0], &small, pinned_error),
        ];
        for (start, dir, at_eye, error_of) in flights {
            let n = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
            let eye_at = |step: usize| [0, 1, 2].map(|a| start[a] + dir[a] / n * 0.4 * step as f64);
            let mut held = select_walk(&at_eye(start), error_of);
            let mut held_at = start;
            let (mut selects, mut changes) = (1, 0);
            let steps = 1500;
            for step in 1..=steps {
                let eye = eye_at(step);
                let moved = (0..3)
                    .map(|a| (eye[a] - held_at[a]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if moved >= held.slack {
                    let new = select_walk(&at_eye(eye), error_of);
                    changes += usize::from(new.patches != held.patches);
                    selects += 1;
                    held = new;
                    held_at = eye;
                }
                assert_eq!(
                    held.patches,
                    select(&at_eye(eye), error_of),
                    "{start:?} along {dir:?}, step {step}: the held selection is not the \
                     eye's own ({moved} blocks from where it was made, slack {})",
                    held.slack
                );
            }
            println!(
                "{start:?} along {dir:?}: {selects} selections in {steps} steps, {changes} \
                 changes"
            );
            assert!(
                selects < steps / 2,
                "{start:?}: {selects} selections in {steps} steps -- a slack this small \
                 holds nothing, and the test tests nothing"
            );
            assert!(
                changes >= 1,
                "{start:?} along {dir:?}: the selection never changed"
            );
        }
    }

    /// **The slack is the smallest gap of a decision that can change, and
    /// no other**: with a radius of 200 blocks every root is a level-3
    /// patch, which no distance splits, so the walk's only decisions are
    /// which roots the radius drops, and its slack is the nearest root's
    /// gap to the radius -- worked out here by hand. The eye stands 0.1
    /// block from the split distance of the root beside it: a slack that
    /// counted a level-3 patch's split thresholds would be 0.1 (found by
    /// `check-tests-can-fail.sh`: nothing saw `if splittable` go).
    #[test]
    fn the_slack_counts_only_decisions_that_can_change() {
        let eye = [84.0, 100.0, 128.0];
        let radius = 200.0;
        let v = FarView {
            hole: None,
            ..game_view(eye, radius)
        };
        assert_eq!(root_level_for(radius), FINEST_LEVEL, "roots of level 3");
        let near_within = v.split_ratio * PatchKey::new(FINEST_LEVEL, 1, 0).size() as f64;
        assert!(
            (256.0 - eye[0] - near_within).abs() < 0.2,
            "the root at +x is not at its split distance ({near_within})"
        );
        let (lo, hi) = surface_bounds();
        assert!(
            (lo..hi).contains(&eye[1]),
            "the eye is outside the roots' height"
        );
        let mut gap = f64::INFINITY;
        for x in -2i64..=2 {
            for z in -2i64..=2 {
                let (x0, z0) = ((x * 256) as f64, (z * 256) as f64);
                let dx = (x0 - eye[0]).max(eye[0] - (x0 + 256.0)).max(0.0);
                let dz = (z0 - eye[2]).max(eye[2] - (z0 + 256.0)).max(0.0);
                gap = gap.min(((dx * dx + dz * dz).sqrt() - radius).abs());
            }
        }
        let s = select_walk(&v, pinned_error);
        assert!(
            (s.slack - gap).abs() < 1e-9,
            "slack {} where the nearest root is {gap} from the radius",
            s.slack
        );
        assert!(
            gap > 10.0,
            "the radius's gap {gap} hides the split distance's"
        );
    }

    /// The patches a walk split are exactly the ancestors of its patches
    /// that it did not drop: every patch's parent is among them (but for the
    /// roots), each of them has its four children among the patches, the
    /// split or the dropped, and no patch is both.
    #[test]
    fn the_split_patches_are_the_ancestors_the_walk_kept() {
        let v = FarView {
            hole: Some(voxel_box()),
            ..view([8.0, 120.0, 8.0], 20_000.0)
        };
        let s = select_walk(&v, pinned_error);
        let root = root_level_for(v.radius);
        let split: std::collections::HashSet<PatchKey> = s.split.iter().copied().collect();
        let patches: std::collections::HashSet<PatchKey> = s.patches.iter().copied().collect();
        assert_eq!(split.len(), s.split.len(), "a patch split twice");
        assert!(split.is_disjoint(&patches), "a patch both kept and split");
        let parent = |k: PatchKey| PatchKey::new(k.level + 1, k.x >> 1, k.z >> 1);
        for k in s.patches.iter().chain(&s.split) {
            if k.level < root {
                assert!(split.contains(&parent(*k)), "{k:?}'s parent was not split");
            }
        }
        let dropped = |k: PatchKey| {
            let (min, max) = patch_box(k);
            ground_distance(v.eye, min, max) > v.radius
                || v.hole.is_some_and(|h| h.contains_box(min, max))
        };
        for k in &s.split {
            for c in k.children() {
                assert!(
                    patches.contains(&c) || split.contains(&c) || dropped(c),
                    "{c:?}, a child of the split {k:?}, is nowhere"
                );
            }
        }
        assert!(s.split.len() > 20, "{} split", s.split.len());
    }

    /// `Hole::grown_across` moves `x` and `z` and never `y`, and a box shrunk
    /// to nothing is none.
    #[test]
    fn a_box_grows_across_the_ground_only() {
        let b = Hole {
            min: [0.0, 5.0, 10.0],
            max: [100.0, 50.0, 30.0],
        };
        let g = b.grown_across(8.0).unwrap();
        assert_eq!((g.min, g.max), ([-8.0, 5.0, 2.0], [108.0, 50.0, 38.0]));
        let s = b.grown_across(-5.0).unwrap();
        assert_eq!((s.min, s.max), ([5.0, 5.0, 15.0], [95.0, 50.0, 25.0]));
        assert_eq!(b.grown_across(-10.0), None, "z shrinks to nothing");
    }

    const PINNED: [(usize, u64); 12] = [
        (1354, 0x978e180d13ca8d0c),
        (1351, 0x5eb0a082c904d27f),
        (1342, 0xa5355f221dec017a),
        (1342, 0x51d201868e34f8b4),
        (1342, 0x51d201868e34f8b4),
        (1355, 0x515ee101814d02d6),
        (2106, 0xa6596bd608972c34),
        (2103, 0x35464759fed1d207),
        (2094, 0x1cc170268a38e04a),
        (2098, 0x8e80c3edccb13d74),
        (2094, 0x0edd665fec8c136c),
        (2097, 0x18b30fa785776798),
    ];
}

#[cfg(test)]
mod measure {
    use super::*;

    /// Measurement, not a check: how often a quad's top stands above its
    /// parent quad's, and by how much, per level of the child. Up to
    /// [`EXACT_TOP_QUAD`] a top is exact and a parent is the highest of its
    /// children; beyond it both are sampled, and a child looks at columns
    /// its parent skips. Where it happens, a far patch swapping for its
    /// coarser parent lowers that ground by this much.
    #[test]
    #[ignore]
    fn a_child_above_its_parent() {
        let gen = WorldGen::new(0x005E_ED00_00C0_FFEE);
        println!("| child level | quad | parent quads | a child above | mean excess | worst |");
        println!("|--:|--:|--:|--:|--:|--:|");
        for level in 2..11u32 {
            let (mut parents, mut above, mut sum, mut worst) = (0u32, 0u32, 0i64, 0i32);
            // Plain and mountains alike: patches from the origin to the range.
            for (x, z) in [(8, 8), (-848, 4832), (5000, -3000), (-20_000, 9000)] {
                let parent = PatchKey::containing(level + 1, x, z);
                let child = parent.children()[0];
                for j in 0..PATCH_QUADS as i32 / 2 {
                    for i in 0..PATCH_QUADS as i32 / 2 {
                        let up = quad_top(&gen, parent, i, j);
                        let down = [(0, 0), (1, 0), (0, 1), (1, 1)]
                            .map(|(a, b)| quad_top(&gen, child, 2 * i + a, 2 * j + b))
                            .into_iter()
                            .max()
                            .unwrap();
                        parents += 1;
                        if down > up {
                            above += 1;
                            sum += (down - up) as i64;
                            worst = worst.max(down - up);
                        }
                    }
                }
            }
            println!(
                "| {level} | {} | {parents} | {:.1}% | {:.1} | {worst} |",
                1 << level,
                100.0 * above as f64 / parents as f64,
                if above > 0 {
                    sum as f64 / above as f64
                } else {
                    0.0
                },
            );
        }
    }

    /// Measurement, not a check: triangles per level, the smooth surface
    /// (2,312 a patch, skirt included) against the blocks, over everything
    /// the three eyes select at High.
    #[test]
    #[ignore]
    fn blocks_against_smooth() {
        let gen = WorldGen::new(0x005E_ED00_00C0_FFEE);
        let pixel = std::f64::consts::FRAC_PI_3 / 1080.0;
        for eye in [[8.0, 40.0, 8.0], [8.0, 300.0, 8.0], [8.0, 3000.0, 8.0]] {
            let view = FarView {
                eye,
                split_ratio: split_ratio_for(16.0, 1080.0, std::f64::consts::FRAC_PI_3),
                max_error: 2.0 * pixel,
                min_quad: 4.0 * pixel,
                radius: 262_144.0,
                hole: Some(Hole {
                    min: [-1024.0, eye[1] - 1024.0, -1024.0],
                    max: [1152.0, eye[1] + 1024.0, 1152.0],
                }),
            };
            let patches = build(&gen, &view);
            let mut by_level = std::collections::BTreeMap::<u32, [usize; 4]>::new();
            for p in &patches {
                let e = by_level.entry(p.key.level).or_default();
                let count = |f: &dyn Fn(&Face) -> bool| p.faces.iter().filter(|x| f(x)).count();
                e[0] += 1;
                e[1] += count(&|f| f.dir == FaceDir::Top);
                e[2] += count(&|f| f.dir != FaceDir::Top && !f.border);
                e[3] += count(&|f| f.border);
            }
            let smooth = patches.len() * 2312;
            let blocks: usize = patches.iter().map(|p| p.faces.len() * 2).sum();
            println!(
                "eye {eye:?}: {} patches, smooth {smooth}, blocks {blocks} ({:+.0}%)",
                patches.len(),
                100.0 * (blocks as f64 / smooth as f64 - 1.0)
            );
            for (l, [n, t, w, b]) in by_level {
                println!(
                    "  level {l:>2}: {n:>4} patches; a patch: {:>5} tops {:>5} walls {:>4} edge \
                     = {:>5} triangles",
                    t / n,
                    w / n,
                    b / n,
                    2 * (t + w + b) / n
                );
            }
        }
    }

    /// Measurement, not a check: what the three gate eyes select at 262 km,
    /// with the voxel rings' cube as the hole. How much of it is in view at
    /// once is the bench's to measure, with the real frustum (Rule 1 keeps
    /// trigonometry out of this crate).
    #[test]
    #[ignore]
    fn what_the_gate_eyes_select() {
        let gen = WorldGen::new(0x005E_ED00_00C0_FFEE);
        let pixel = std::f64::consts::FRAC_PI_3 / 1080.0;
        // (quad px by distance, height error px -- 0 for none)
        for (px, err) in [
            (8.0, 0.0),
            (16.0, 0.0),
            (16.0, 1.0),
            (16.0, 2.0),
            (8.0, 1.0),
        ] {
            for eye in [[8.0, 40.0, 8.0], [8.0, 300.0, 8.0], [8.0, 3000.0, 8.0]] {
                let view = FarView {
                    eye,
                    split_ratio: split_ratio_for(px, 1080.0, std::f64::consts::FRAC_PI_3),
                    max_error: if err > 0.0 {
                        err * pixel
                    } else {
                        f64::INFINITY
                    },
                    min_quad: 2.0 * pixel,
                    radius: 262_144.0,
                    hole: Some(Hole {
                        min: [-1024.0, eye[1] - 512.0, -1024.0],
                        max: [1040.0, eye[1] + 528.0, 1040.0],
                    }),
                };
                let (keys, have) = super::tests::converge(&gen, &view);
                let tris = |n: usize| (n * PATCH_QUADS * PATCH_QUADS * 2) as f64 / 1e6;
                println!(
                    "{px:>4} px, error {err} px, eye y {:>5}: {:>5} patches ({:.1}M tris); {} generated",
                    eye[1],
                    keys.len(),
                    tris(keys.len()),
                    have.len(),
                );
            }
        }
    }
}

/// The far terrain as blocks (issue #422): [`quad_top`], [`mesh`], and what
/// holds two patches together.
#[cfg(test)]
mod block_tests {
    use super::*;

    const SEED: u64 = 0x005E_ED00_00C0_FFEE;

    /// [`mesh`] of ground about which nothing but its tops is known.
    fn mesh_plain(key: PatchKey, tops: &[i32]) -> Vec<Face> {
        mesh(key, tops, &[QuadGround::PLAIN; PATCH_QUADS * PATCH_QUADS])
    }

    /// A patch's tops, apron and all, from a function of the quad.
    fn tops_from(f: impl Fn(i32, i32) -> i32) -> Vec<i32> {
        let mut tops = Vec::with_capacity(TOPS_SIDE * TOPS_SIDE);
        for j in -1..=PATCH_QUADS as i32 {
            for i in -1..=PATCH_QUADS as i32 {
                tops.push(f(i, j));
            }
        }
        tops
    }

    /// A cheap, fixed hash of a quad, for tops with steps of every size.
    fn rough(seed: u64, span: i32) -> impl Fn(i32, i32) -> i32 {
        move |i, j| {
            let h = (i as i64 as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add((j as i64 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
                .wrapping_add(seed)
                .wrapping_mul(0x2545_F491_4F6C_DD1D);
            20 + ((h >> 40) % span as u64) as i32
        }
    }

    /// The keys the properties are tried on: levels with whole cells of
    /// [`JOIN_GRID`] inside a patch, the level where every quad is a cell,
    /// and two above the grid altogether.
    const KEYS: [PatchKey; 5] = [
        PatchKey::new(0, 1, 2),
        PatchKey::new(2, -3, 7),
        PatchKey::new(4, 0, 0),
        PatchKey::new(7, 2, -1),
        PatchKey::new(9, 1, 1),
    ];

    fn walls(faces: &[Face]) -> impl Iterator<Item = &Face> {
        faces.iter().filter(|f| f.dir != FaceDir::Top)
    }

    /// **One ground for every level** (issue #423): a quad's top is the top
    /// face of the voxel cell over the same columns (`WorldGen::cell_top`,
    /// whatever that rule is: the mean, and a summit at its highest half's
    /// mean since #437), never above the highest of them and the column's
    /// own top at level 0 (what the sky sees of it, `column_top`: the floor
    /// of a pit under one a cave has opened, §9.9) -- in the mountains,
    /// where the three differ.
    #[test]
    fn a_quads_top_is_the_voxel_cells_top() {
        let gen = WorldGen::new(SEED);
        let highest = |x0: i64, z0: i64, q: i64| {
            let mut top = i32::MIN;
            for z in z0..z0 + q {
                for x in x0..x0 + q {
                    top = top.max(gen.column_top(x as i32, z as i32));
                }
            }
            top
        };
        // Levels 0 to 2: whole blocks, exactly: a column's own top at
        // level 0, and the voxel cell's `cell_top` (never above the highest
        // column) at levels 1 and 2.
        for level in 0..3 {
            let key = PatchKey::containing(level, -848, 4832);
            let (q, [ox, oz]) = (key.quad(), key.origin());
            for (i, j) in [(-1, 3), (0, 0), (7, 19), (31, 31), (32, 8)] {
                let (x0, z0) = (ox + i as i64 * q, oz + j as i64 * q);
                let top = quad_top(&gen, key, i, j);
                assert_eq!(top, gen.cell_top(x0 as i32, z0 as i32, q as i32) + 1);
                assert!(top <= highest(x0, z0, q) + 1);
                if level == 0 {
                    assert_eq!(top, highest(x0, z0, q) + 1);
                }
            }
        }
        // Level 3: eight columns a side, every one looked at -- the level-3
        // voxel cell over the same columns, `cell_top(.., 8)`. Quads that a
        // pit sinks are among them (§9.9): under the same rule over the
        // surfaces, which is what a quad's top was until then, they stand
        // higher.
        let key = PatchKey::containing(3, -848, 4832);
        let [ox, oz] = key.origin();
        let mut sunk = 0;
        for j in -1..=PATCH_QUADS as i32 {
            for i in -1..=PATCH_QUADS as i32 {
                let (x0, z0) = (ox + i as i64 * 8, oz + j as i64 * 8);
                let top = quad_top(&gen, key, i, j);
                assert_eq!(top, gen.cell_top(x0 as i32, z0 as i32, 8) + 1);
                assert!(top <= highest(x0, z0, 8) + 1, "({i}, {j}): {top}");
                let whole = WorldGen::top_over(
                    &|x, z| gen.surface_height(x, z),
                    x0 as i32,
                    z0 as i32,
                    8,
                    8,
                );
                sunk += (top < whole + 1) as u32;
            }
        }
        assert!(sunk >= 20, "only {sunk} quads of 8 stand lower for a pit");
        // Wider: the sampled form of the same rule.
        let key = PatchKey::containing(6, -848, 4832);
        let [ox, oz] = key.origin();
        for (i, j) in [(0, 0), (9, 30), (31, 2)] {
            let (x0, z0) = (ox + i as i64 * 64, oz + j as i64 * 64);
            let top = quad_top(&gen, key, i, j);
            assert_eq!(
                top,
                gen.cell_top_sampled(x0 as i32, z0 as i32, 64, TOP_SAMPLES) + 1
            );
        }
    }

    /// **A far crest grows few towers** (#422 on #437's summit rule). The
    /// zoom pairs of 2026-11-21 saw "thin vertical spires on peak tops and
    /// ridges" in the far terrain at 1,200 blocks: quads that held a summit
    /// stood at their highest column while their neighbours stood at their
    /// means. A quad's top is the voxel cell's ([`quad_top`]), so the far
    /// terrain follows the voxels' rule to the highest half's mean with
    /// nothing changed here -- and this holds it there, at the far terrain's
    /// own widths, the sampled ones included: over 1,792 blocks square
    /// around the default seed's highest peak, the quads that stand at
    /// least their own width above all four quads beside them.
    ///
    /// Measured: 12, 8 and 5 at quads of 8, 16 and 32. Under the highest
    /// column (#432's rule, which this is red on) they were 53, 24 and 9.
    ///
    /// A quad is held against what the four beside it draw of the ground
    /// with no cave in it (the rule over the surfaces): a quad beside one a
    /// pit has sunk stands over the pit at level 0 too, and that is the
    /// pit's wall, not a tower (§9.9; the voxels'
    /// `a_coarse_crest_grows_few_towers` says the same).
    #[test]
    fn a_far_crest_grows_few_towers() {
        /// Per far level, the most towers allowed. **Chosen, not derived**:
        /// the measured counts with a little room.
        const MOST: [(u32, usize); 3] = [(3, 14), (4, 10), (5, 6)];
        const SPAN: i32 = 1792;
        let gen = WorldGen::new(SEED);
        let (x_lo, z_lo) = (-3072 - 1024, -3232 - 1120);
        for (level, most) in MOST {
            let key = PatchKey::new(level, 0, 0);
            let q = key.quad() as i32;
            let n = SPAN / q;
            // Any quad of the level's grid, counted from the patch at the
            // origin: `quad_top` takes them all.
            let tops: Vec<i32> = (0..n * n)
                .map(|k| quad_top(&gen, key, x_lo / q + k % n, z_lo / q + k / n))
                .collect();
            let at = |i: i32, j: i32| tops[(j * n + i) as usize];
            // The top face the quad `(i, j)` would have with no cave in the
            // ground: the same rule, on the same samples, over the surfaces.
            let [ox, oz] = key.origin();
            let ground = |i: i32, j: i32| {
                WorldGen::top_over(
                    &|x, z| gen.surface_height(x, z),
                    ox as i32 + (x_lo / q + i) * q,
                    oz as i32 + (z_lo / q + j) * q,
                    q,
                    top_samples(key),
                ) + 1
            };
            let towers = (1..n - 1)
                .flat_map(|j| (1..n - 1).map(move |i| (i, j)))
                .filter(|&(i, j)| {
                    let beside = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)];
                    beside.iter().all(|&(a, b)| at(i, j) - at(a, b) >= q)
                        && beside.iter().all(|&(a, b)| at(i, j) - ground(a, b) >= q)
                })
                .count();
            println!("quads of {q}: {towers} towers");
            assert!(
                towers <= most,
                "quads of {q}: {towers} quads stand alone at least {q} blocks above all four beside them, allowed {most}"
            );
        }
    }

    /// Where the default seed's mountains are riddled with pits: the ground
    /// the voxels' own pit test measures (`worldgen`'s
    /// `a_coarse_top_shows_pit_floors_in_the_share_level_0_does`).
    const PITS: (i32, i32) = (-848, 4832);

    /// **A far quad of 8 wears what the level-3 cell over the same columns
    /// wears** (#422 on the colour morph's `top_block`): the same pick of
    /// the same column, so where the far terrain hands the ground to level 3
    /// no top changes colour -- a pit's floor stays where it was. Held
    /// against the node the game generates, cell by cell, over four level-3
    /// nodes of the pitted mountains; both kinds of top must turn up.
    #[test]
    fn a_far_top_wears_what_the_voxel_cell_wears() {
        use crate::worldgen::{OreSet, TerrainBlocks};
        use cubara_voxel::{BlockId, Chunk};
        let gen = WorldGen::new(SEED);
        let blocks = TerrainBlocks {
            oak: None,
            ores: OreSet::EMPTY,
            grass: BlockId(1),
            soil: BlockId(2),
            stone: BlockId(3),
        };
        let (step, v) = (8, 4);
        assert_eq!(1i64 << FINEST_LEVEL, step as i64, "level 3's cell");
        let size = Chunk::SIZE as i32 * step;
        let (mut surface, mut floors) = (0, 0);
        let mut nodes = std::collections::BTreeMap::new();
        for (nx, nz) in [(-1, -1), (0, -1), (-1, 0), (0, 0)] {
            let (ox, oz) = (
                PITS.0.div_euclid(size) * size + nx * size,
                PITS.1.div_euclid(size) * size + nz * size,
            );
            for lz in 0..Chunk::SIZE as i32 {
                for lx in 0..Chunk::SIZE as i32 {
                    let (x0, z0) = (ox + lx * step, oz + lz * step);
                    let top = gen.cell_top(x0, z0, step);
                    let oy = top.div_euclid(size) * size;
                    let node = nodes
                        .entry((ox, oy, oz))
                        .or_insert_with(|| gen.generate_aniso([ox, oy, oz], step, v, blocks));
                    let block = node.get(lx as usize, ((top - oy) / v) as usize, lz as usize);
                    if block == BlockId::AIR {
                        continue; // every column a pit: level 3 draws the hole
                    }
                    let key = PatchKey::containing(FINEST_LEVEL, x0 as i64, z0 as i64);
                    let [px, pz] = key.origin();
                    let (i, j) = (
                        ((x0 as i64 - px) / step as i64) as i32,
                        ((z0 as i64 - pz) / step as i64) as i32,
                    );
                    let far = QuadGround::of(&gen, key, i, j, quad_top(&gen, key, i, j)).top;
                    let voxel = match block {
                        b if b == blocks.grass => [255, 0],
                        b if b == blocks.soil => [0, 255],
                        _ => [0, 0],
                    };
                    assert_eq!(
                        far, voxel,
                        "the cell at ({x0}, {z0}): level 3 wears {block:?} on top"
                    );
                    if far == [255, 0] {
                        surface += 1;
                    } else {
                        floors += 1;
                    }
                }
            }
        }
        assert!(
            surface > 500 && floors > 30,
            "{surface} tops wear the surface block, {floors} a pit's floor"
        );
    }

    /// **A far top shows a pit's floor in the share level 0's columns are
    /// pits**, at every quad width (#422 on `WorldGen::top_block`'s rule):
    /// over 1,024 blocks square of the pitted mountains, the tops that wear
    /// something other than the surface block number what the pit share of
    /// the ground says -- every column of it counted, so a sampled quad is
    /// held to the ground and not to its own samples. Before the rule the
    /// far terrain showed none.
    #[test]
    fn a_far_top_shows_pit_floors_in_the_share_level_0_does() {
        /// Per far level, how far the count may be from the expectation, as
        /// a share of it. **Chosen, not derived**: the pick is a hash per
        /// quad, so the count is a sample -- 1,010.5, 252.6 and 63.2 quads
        /// expected here at 8, 16 and 32 blocks (6.17% of the columns are
        /// pit), 1,010, 261 and 66 measured; the bounds are about three
        /// standard deviations of a count that size.
        const WITHIN: [(u32, f64); 3] = [(3, 0.1), (4, 0.2), (5, 0.35)];
        const SPAN: i32 = 1024;
        let gen = WorldGen::new(SEED);
        let (x_lo, z_lo) = (
            PITS.0.div_euclid(SPAN) * SPAN - SPAN / 2,
            PITS.1.div_euclid(SPAN) * SPAN - SPAN / 2,
        );
        let pit: Vec<bool> = (0..SPAN * SPAN)
            .map(|k| {
                let (x, z) = (x_lo + k % SPAN, z_lo + k / SPAN);
                gen.pit_floor(x, z, gen.surface_height(x, z)).is_some()
            })
            .collect();
        let expected_share = pit.iter().filter(|p| **p).count() as f64 / pit.len() as f64;
        for (level, within) in WITHIN {
            let q = 1i32 << level;
            let quads = (SPAN / q) * (SPAN / q);
            let expected = expected_share * quads as f64;
            let mut shown = 0;
            for k in 0..quads {
                let (x0, z0) = (x_lo + k % (SPAN / q) * q, z_lo + k / (SPAN / q) * q);
                let key = PatchKey::containing(level, x0 as i64, z0 as i64);
                let [px, pz] = key.origin();
                let (i, j) = (
                    ((x0 as i64 - px) / q as i64) as i32,
                    ((z0 as i64 - pz) / q as i64) as i32,
                );
                let wears = QuadGround::of(&gen, key, i, j, quad_top(&gen, key, i, j)).top;
                shown += (wears != [255, 0]) as u32;
            }
            println!(
                "quads of {q}: {shown} of {quads} tops show a pit's floor, {expected:.1} expected ({:.2}% of the columns are pit)",
                100.0 * expected_share
            );
            assert!(expected > 8.0, "quads of {q}: too few pits to measure");
            assert!(
                (shown as f64 - expected).abs() <= within * expected,
                "quads of {q}: {shown} tops show a pit's floor, {expected:.1} expected"
            );
        }
    }

    /// A quad over ground with no riser at all (a column no neighbour is
    /// lower than) shows the surface block's side on a wall facing any way:
    /// `QuadGround::of`'s last fallback, the one a division by no risers at
    /// all would turn into nothing.
    #[test]
    fn a_wall_over_ground_with_no_risers_is_the_surface_blocks_side() {
        let gen = WorldGen::new(SEED);
        let key = PatchKey::new(0, 0, 0);
        let mut found = 0;
        for j in 1..31 {
            for i in 1..31 {
                let [ox, oz] = key.origin();
                let (x, z) = (ox as i32 + i, oz as i32 + j);
                let h = gen.column_top(x, z);
                let level = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    .iter()
                    .all(|&(dx, dz)| gen.column_top(x + dx, z + dz) >= h);
                if level {
                    found += 1;
                    assert_eq!(
                        QuadGround::of(&gen, key, i, j, quad_top(&gen, key, i, j)).sides,
                        [[255, 0]; 4],
                        "the column at ({x}, {z})"
                    );
                }
            }
        }
        assert!(found > 20, "only {found} columns without a riser");
    }

    /// **A wall is what level 0's risers facing its way are made of**
    /// (issue #423 decision 2), not what its own height would suggest: its
    /// mix and openness are its quad's [`QuadGround`] for its direction, and
    /// a quad's ground is `WorldGen::ground_faces` of its columns.
    #[test]
    fn a_wall_shows_what_level_0_shows_that_way() {
        use crate::worldgen::GroundLayer;
        let gen = WorldGen::new(SEED);
        let key = PatchKey::containing(3, -848, 4832);
        let patch = generate(&gen, key);
        let [ox, oz] = key.origin();
        let unit = |v: f32| (v * 255.0).round() as u8;
        let (mut below, mut differing) = (0, 0);
        for f in &patch.faces {
            // The quad the face belongs to.
            let (i, j) = match f.dir {
                FaceDir::Top | FaceDir::NegX | FaceDir::NegZ => (f.i0, f.j0),
                FaceDir::PosX => (f.i0 - 1, f.j0),
                FaceDir::PosZ => (f.i0, f.j0 - 1),
            };
            let g = patch.ground[j as usize * PATCH_QUADS + i as usize];
            let level0 =
                gen.ground_faces((ox + i as i64 * 8) as i32, (oz + j as i64 * 8) as i32, 8, 4);
            if f.dir == FaceDir::Top {
                assert_eq!((f.mix, f.open), (g.top, unit(level0.top_open)), "{f:?}");
                continue;
            }
            let d = f.dir as usize - 1;
            assert_eq!((f.mix, f.open), (g.sides[d], g.side_open[d]), "{f:?}");
            assert_eq!(f.open, unit(level0.side_open[d]), "{f:?}");
            if let Some(surface) = level0.share(d, GroundLayer::Surface) {
                let soil = level0.share(d, GroundLayer::Soil).unwrap();
                assert_eq!(f.mix, [unit(surface), unit(soil)], "{f:?}");
                if f.mix[1] > 0 {
                    below += 1;
                }
                // Its direction's own mix, not another's.
                if g.sides.iter().any(|s| *s != g.sides[d]) {
                    differing += 1;
                }
            }
        }
        assert!(
            below > 200,
            "only {below} walls show anything under the surface block"
        );
        assert!(
            differing > 200,
            "only {differing} walls tell one direction from another"
        );
    }

    /// **The far terrain's blocks are on the world's block grid**, and on
    /// the grid of the voxel rings' coarse cells where the two meet: a quad
    /// of level `L` has its corners on multiples of `2^L` blocks, and at
    /// level 0 its top is the top face of the column it is.
    #[test]
    fn the_steps_are_on_the_block_grid() {
        let gen = WorldGen::new(SEED);
        for key in [
            PatchKey::new(0, 3, -2),
            PatchKey::new(3, -5, 1),
            PatchKey::new(6, 0, 0),
        ] {
            let [ox, oz] = key.origin();
            assert_eq!(ox.rem_euclid(key.quad()), 0);
            assert_eq!(oz.rem_euclid(key.quad()), 0);
        }
        // A level-3 quad is one 8-block cell of the coarsest voxel ring;
        // coarser quads are whole numbers of them.
        for level in 3..9 {
            assert_eq!(PatchKey::new(level, 1, 1).quad() % 8, 0);
        }
        let key = PatchKey::new(0, 3, -2);
        let patch = generate(&gen, key);
        let [ox, oz] = key.origin();
        for j in -1..=PATCH_QUADS as i32 {
            for i in -1..=PATCH_QUADS as i32 {
                let column = gen.column_top((ox + i as i64) as i32, (oz + j as i64) as i32);
                assert_eq!(patch.top(i, j), column + 1, "quad ({i}, {j})");
            }
        }
    }

    /// **A step of `k` blocks is a wall of exactly `k` blocks**: over tops
    /// with steps of every size, each line between two quads of a patch has
    /// one wall where they differ -- on the line, as tall as the difference,
    /// hanging from the higher, facing the lower -- and none where they are
    /// level.
    #[test]
    fn a_step_of_k_blocks_is_a_wall_of_exactly_k_blocks() {
        const N: i32 = PATCH_QUADS as i32;
        let mut steps = 0;
        for (n, key) in KEYS.into_iter().enumerate() {
            for span in [2, 3, 40] {
                let top = rough(n as u64 * 31 + span as u64, span);
                let faces = mesh_plain(key, &tops_from(&top));
                // Every wall, by the unit of quad line it stands on:
                // (faces along x?, line, quad along it) -> the wall.
                let mut on = std::collections::HashMap::new();
                for f in walls(&faces) {
                    let along_x = matches!(f.dir, FaceDir::PosX | FaceDir::NegX);
                    assert_eq!(f.i0 == f.i1, along_x, "{f:?}");
                    assert_eq!(f.j0 == f.j1, !along_x, "{f:?}");
                    let (line, run) = if along_x {
                        (f.i0, f.j0..f.j1)
                    } else {
                        (f.j0, f.i0..f.i1)
                    };
                    assert!(!run.is_empty(), "{f:?} has no length");
                    for k in run {
                        let before = on.insert((along_x, line as i32, k as i32), *f);
                        assert!(before.is_none(), "two walls on one line: {f:?} {before:?}");
                    }
                }
                for along_x in [true, false] {
                    for line in 1..N {
                        for k in 0..N {
                            let (below, above) = if along_x {
                                (top(line - 1, k), top(line, k))
                            } else {
                                (top(k, line - 1), top(k, line))
                            };
                            let wall = on.get(&(along_x, line, k));
                            if below == above {
                                assert!(wall.is_none(), "a wall between level quads: {wall:?}");
                                continue;
                            }
                            steps += 1;
                            let wall = wall.unwrap_or_else(|| {
                                panic!("{key:?}: no wall on line {line} at {k}, {below} to {above}")
                            });
                            let want = match (along_x, below > above) {
                                (true, true) => FaceDir::PosX,
                                (true, false) => FaceDir::NegX,
                                (false, true) => FaceDir::PosZ,
                                (false, false) => FaceDir::NegZ,
                            };
                            assert_eq!(wall.dir, want, "{wall:?}");
                            assert_eq!(wall.top, below.max(above), "{wall:?}");
                            assert_eq!(wall.rise, below.abs_diff(above), "{wall:?}");
                            assert!(!wall.border, "{wall:?}");
                            // Exactly the step: its foot is the lower top.
                            assert_eq!(wall.bottom(99), below.min(above), "{wall:?}");
                        }
                    }
                }
            }
        }
        assert!(steps > 20_000, "only {steps} steps tried");
    }

    /// The tops are the quads: each quad is under exactly one top, at its
    /// own height.
    #[test]
    fn every_quad_is_under_one_top_at_its_height() {
        for (n, key) in KEYS.into_iter().enumerate() {
            for span in [1, 2, 40] {
                let top = rough(n as u64 * 17 + span as u64, span);
                let faces = mesh_plain(key, &tops_from(&top));
                let mut under = [0u8; PATCH_QUADS * PATCH_QUADS];
                for f in faces.iter().filter(|f| f.dir == FaceDir::Top) {
                    assert!(f.i0 < f.i1 && f.j0 < f.j1, "{f:?}");
                    assert_eq!((f.rise, f.border), (0, false), "{f:?}");
                    for j in f.j0..f.j1 {
                        for i in f.i0..f.i1 {
                            under[j as usize * PATCH_QUADS + i as usize] += 1;
                            assert_eq!(f.top, top(i as i32, j as i32), "{f:?} over ({i}, {j})");
                        }
                    }
                }
                assert!(
                    under.iter().all(|&n| n == 1),
                    "{key:?}: a quad under {under:?}"
                );
            }
        }
    }

    /// **Equal neighbours merge**, within a cell of [`JOIN_GRID`]. A level-0
    /// patch is two cells by two: level ground is nine tops a cell -- its
    /// 14 x 14 inside as one, its four edge rows, its four corners -- where
    /// the unmerged cell has 256, and one wall along each cell's stretch of
    /// the patch's edge. Two plateaus are that again on either side of one
    /// wall a cell long. Ground with no two neighbours alike merges nothing.
    #[test]
    fn equal_neighbours_merge() {
        let key = PatchKey::new(0, 0, 0);
        let count = |faces: &[Face], dir: FaceDir| faces.iter().filter(|f| f.dir == dir).count();
        let flat = mesh_plain(key, &tops_from(|_, _| 30));
        assert_eq!(count(&flat, FaceDir::Top), 4 * 9);
        // The inside of the first cell, in one piece.
        assert!(
            flat.contains(&Face {
                dir: FaceDir::Top,
                i0: 1,
                j0: 1,
                i1: 15,
                j1: 15,
                top: 30,
                rise: 0,
                border: false,
                mix: [255, 0],
                open: 255,
            }),
            "{flat:?}"
        );
        assert_eq!(walls(&flat).count(), 8);
        for wall in walls(&flat) {
            assert!(wall.border && wall.rise == 0, "{wall:?}");
            assert_eq!(
                (wall.i1 - wall.i0) + (wall.j1 - wall.j0),
                16,
                "{wall:?} does not run its cell's length"
            );
        }

        // Two plateaus, the step along x = 5: inside the first cells, so
        // it cuts their insides in two and adds nothing else.
        let two = mesh_plain(key, &tops_from(|i, _| if i < 5 { 30 } else { 37 }));
        let inner: Vec<_> = walls(&two).filter(|f| !f.border).copied().collect();
        let wall = |j0: u8, j1: u8| Face {
            dir: FaceDir::NegX,
            i0: 5,
            j0,
            i1: 5,
            j1,
            top: 37,
            rise: 7,
            border: false,
            mix: [255, 0],
            open: 255,
        };
        assert_eq!(inner, vec![wall(0, 16), wall(16, 32)]);
        // Each of the two cells it crosses: the inside in two, and its two
        // edge rows along x in two.
        assert_eq!(count(&two, FaceDir::Top), 4 * 9 + 2 * 3);

        // Level ground of which one quad is shut in a little more than the
        // rest: that quad is a top of its own, with its own openness.
        let mut ground = [QuadGround::PLAIN; PATCH_QUADS * PATCH_QUADS];
        ground[5 * PATCH_QUADS + 7].top_open = 200;
        let shaded = mesh(key, &tops_from(|_, _| 30), &ground);
        let own: Vec<_> = shaded.iter().filter(|f| f.open == 200).collect();
        assert_eq!(own.len(), 1, "{own:?}");
        assert_eq!((own[0].i0, own[0].j0, own[0].i1, own[0].j1), (7, 5, 8, 6));
        assert!(count(&shaded, FaceDir::Top) > 4 * 9);

        // Level ground of which one quad shows a pit's floor: a top of its
        // own again, wearing the floor, and no other top wears it.
        let mut ground = [QuadGround::PLAIN; PATCH_QUADS * PATCH_QUADS];
        ground[9 * PATCH_QUADS + 3].top = [0, 0];
        let pitted = mesh(key, &tops_from(|_, _| 30), &ground);
        let own: Vec<_> = pitted
            .iter()
            .filter(|f| f.dir == FaceDir::Top && f.mix != [255, 0])
            .collect();
        assert_eq!(own.len(), 1, "{own:?}");
        assert_eq!(
            (own[0].i0, own[0].j0, own[0].i1, own[0].j1, own[0].mix),
            (3, 9, 4, 10, [0, 0])
        );
        assert!(count(&pitted, FaceDir::Top) > 4 * 9);

        // Above the grid a quad is a cell: level ground is a top a quad.
        let above = mesh_plain(PatchKey::new(4, 0, 0), &tops_from(|_, _| 30));
        assert_eq!(count(&above, FaceDir::Top), PATCH_QUADS * PATCH_QUADS);
        assert!(walls(&above).all(|f| f.border));

        // Nothing alike: a top a quad.
        let none = mesh_plain(key, &tops_from(|i, j| i * 40 + j));
        assert_eq!(count(&none, FaceDir::Top), PATCH_QUADS * PATCH_QUADS);
        assert_eq!(walls(&none).filter(|f| !f.border).count(), 2 * 31 * 32);
    }

    /// **No face crosses a line of [`JOIN_GRID`], and a top that touches one
    /// is a single quad deep there** -- what the join to the voxels rests on
    /// (`cubara_render::far::join` moves the corners a quad inside the box's
    /// side, and a wall is in the box or out of it as a whole). On level
    /// ground, where everything that may merge does.
    #[test]
    fn no_face_crosses_the_join_grid() {
        for key in [
            PatchKey::new(0, 3, 2),
            PatchKey::new(1, -1, 0),
            PatchKey::new(2, 5, -4),
            PatchKey::new(3, 0, 0),
            PatchKey::new(4, -2, 1),
            PatchKey::new(6, 0, 3),
        ] {
            let q = key.quad();
            let [ox, oz] = key.origin();
            let faces = mesh_plain(key, &tops_from(|_, _| 30));
            let mut merged = false;
            for f in &faces {
                // In blocks.
                let (x0, x1) = (ox + f.i0 as i64 * q, ox + f.i1 as i64 * q);
                let (z0, z1) = (oz + f.j0 as i64 * q, oz + f.j1 as i64 * q);
                // A quad wider than the grid crosses its lines as it is; a
                // face must not be more than one of those.
                let cell = JOIN_GRID.max(q);
                for (lo, hi) in [(x0, x1), (z0, z1)] {
                    assert!(
                        hi == lo || lo.div_euclid(cell) == (hi - 1).div_euclid(cell),
                        "{key:?}: {f:?} crosses a line, {lo}..{hi}"
                    );
                }
                merged |= (x1 - x0) > q || (z1 - z0) > q;
                if f.dir != FaceDir::Top {
                    continue;
                }
                let on_line = |v: i64| v.rem_euclid(JOIN_GRID) == 0;
                if on_line(x0) || on_line(x1) {
                    assert_eq!(x1 - x0, q, "{key:?}: {f:?} is wide against a line");
                }
                if on_line(z0) || on_line(z1) {
                    assert_eq!(z1 - z0, q, "{key:?}: {f:?} is deep against a line");
                }
            }
            // And the limits are limits, not a ban: below the level where a
            // quad is a cell, something did merge.
            assert_eq!(merged, key.level < 4, "{key:?}");
        }
    }

    /// Two patches of one level side by side agree on the quads along their
    /// shared edge: each one's apron is the other's own row. So the wall on
    /// that edge is exact, as inside a patch.
    #[test]
    fn neighbouring_patches_agree_on_the_quads_along_their_edge() {
        let gen = WorldGen::new(SEED);
        const N: i32 = PATCH_QUADS as i32;
        // (5, -27, 151) is in the mountains.
        for key in [
            PatchKey::new(0, 2, 5),
            PatchKey::new(3, 2, 5),
            PatchKey::new(5, -27, 151),
        ] {
            let a = generate(&gen, key);
            let east = generate(&gen, PatchKey::new(key.level, key.x + 1, key.z));
            let south = generate(&gen, PatchKey::new(key.level, key.x, key.z + 1));
            let mut differ = false;
            for k in 0..N {
                assert_eq!(a.top(N, k), east.top(0, k), "{key:?} east, {k}");
                assert_eq!(a.top(N - 1, k), east.top(-1, k), "{key:?} east, {k}");
                assert_eq!(a.top(k, N), south.top(k, 0), "{key:?} south, {k}");
                assert_eq!(a.top(k, N - 1), south.top(k, -1), "{key:?} south, {k}");
                differ |= a.top(N, k) != a.top(N - 1, k);
            }
            if key.level == 5 {
                assert!(differ, "level ground in the mountains proves nothing");
            }
        }
    }

    /// The patch covering the block column `(x, z)`, if any.
    fn patch_at(patches: &[PatchHeights], x: i64, z: i64) -> Option<&PatchHeights> {
        patches.iter().find(|p| {
            let [ox, oz] = p.key.origin();
            (ox..ox + p.key.size()).contains(&x) && (oz..oz + p.key.size()).contains(&z)
        })
    }

    /// The drawn top of the column `(x, z)` of `p`.
    fn top_at(p: &PatchHeights, x: i64, z: i64) -> i32 {
        let [ox, oz] = p.key.origin();
        let q = p.key.quad();
        p.top(((x - ox) / q) as i32, ((z - oz) / q) as i32)
    }

    /// How far short of closing the worst edge falls, over everything a view
    /// selects: for every stretch of every patch's edge with another patch
    /// beyond it, the wall there must reach from this patch's top down to
    /// the other's, whatever the other's level. Returns `(stretches checked,
    /// of them between levels, the worst shortfall in blocks -- zero or less
    /// when all close)`. `skirt_of` is the skirt a patch's edge walls are
    /// given.
    fn worst_gap(
        patches: &[PatchHeights],
        skirt_of: impl Fn(&PatchHeights) -> i32,
    ) -> (usize, usize, i32) {
        const N: i64 = PATCH_QUADS as i64;
        let (mut checked, mut between, mut worst) = (0, 0, i32::MIN);
        for p in patches {
            let [ox, oz] = p.key.origin();
            let q = p.key.quad();
            // Each edge: the wall's direction, and for quad `k` along it, the
            // column just inside and the step to the column just outside.
            let edges = [
                (FaceDir::NegX, 0u8, true),
                (FaceDir::PosX, N as u8, true),
                (FaceDir::NegZ, 0u8, false),
                (FaceDir::PosZ, N as u8, false),
            ];
            for (dir, line, along_x) in edges {
                let inside = if line == 0 { 0 } else { N - 1 };
                let outside = if line == 0 { -1 } else { N * q };
                for k in 0..N {
                    let wall = p
                        .faces
                        .iter()
                        .find(|f| {
                            f.border && f.dir == dir && {
                                let (at, run) = if along_x {
                                    (f.i0, f.j0..f.j1)
                                } else {
                                    (f.j0, f.i0..f.i1)
                                };
                                at == line && run.contains(&(k as u8))
                            }
                        })
                        .unwrap_or_else(|| panic!("{:?}: no wall on edge {dir:?} at {k}", p.key));
                    let mine = if along_x {
                        p.top(inside as i32, k as i32)
                    } else {
                        p.top(k as i32, inside as i32)
                    };
                    assert_eq!(wall.top, mine, "{:?}: {wall:?}", p.key);
                    let foot = wall.bottom(skirt_of(p));
                    // Along this quad's edge, every neighbour beyond it.
                    let mut s = 0;
                    while s < q {
                        let (x, z) = if along_x {
                            (ox + outside, oz + k * q + s)
                        } else {
                            (ox + k * q + s, oz + outside)
                        };
                        let Some(other) = patch_at(patches, x, z) else {
                            // The edge of the view, or the voxels' hole.
                            s += q;
                            continue;
                        };
                        let theirs = top_at(other, x, z);
                        checked += 1;
                        if other.key.level != p.key.level {
                            between += 1;
                        }
                        if mine > theirs {
                            worst = worst.max(foot - theirs);
                        }
                        s += q.min(other.key.quad());
                    }
                }
            }
        }
        (checked, between, worst)
    }

    /// The views the watertightness property is tried on: three seeds, an
    /// eye on the ground, one on a hill and one in flight, to 40 km, with
    /// the voxels' hole -- so rough land splits some patches two levels
    /// below their neighbours.
    fn watertight_views() -> Vec<(u64, FarView)> {
        let pixel = std::f64::consts::FRAC_PI_3 / 1080.0;
        let mut out = Vec::new();
        for (seed, eye) in [
            (SEED, [8.0, 40.0, 8.0]),
            (0x0BAD_5EED, [-3000.0, 300.0, 5200.0]),
            (1_881_928_537_856_268_101, [640.0, 900.0, -410.0]),
        ] {
            out.push((
                seed,
                FarView {
                    eye,
                    split_ratio: split_ratio_for(16.0, 1080.0, std::f64::consts::FRAC_PI_3),
                    max_error: 2.0 * pixel,
                    min_quad: 4.0 * pixel,
                    radius: 20_000.0,
                    hole: Some(Hole {
                        min: [eye[0] - 1024.0, eye[1] - 1024.0, eye[2] - 1024.0],
                        max: [eye[0] + 1024.0, eye[1] + 1024.0, eye[2] + 1024.0],
                    }),
                },
            ));
        }
        out
    }

    /// **Watertight between patches, whatever their levels** (issue #422
    /// decision 4: the skirts are walls). Over everything three seeded views
    /// select: along every edge shared by two patches, the higher one's wall
    /// reaches down to the lower one's top -- exactly, between two of one
    /// level, and through its skirt between two of different levels. With
    /// no skirt at all the same views leave gaps, so the test can tell.
    #[test]
    fn no_gap_between_patches_of_any_level() {
        let (mut checked, mut between) = (0, 0);
        let mut unskirted = i32::MIN;
        for (seed, view) in watertight_views() {
            let patches = build(&WorldGen::new(seed), &view);
            let (n, m, worst) = worst_gap(&patches, PatchHeights::skirt_blocks);
            assert!(
                worst <= 0,
                "seed {seed:#x}: an edge wall stops {worst} blocks above its neighbour's top"
            );
            checked += n;
            between += m;
            unskirted = unskirted.max(worst_gap(&patches, |_| 0).2);
            // Between two of one level the wall is exact: no skirt needed.
            let same: Vec<PatchHeights> = patches
                .iter()
                .filter(|p| p.key.level == patches[0].key.level)
                .cloned()
                .collect();
            let (_, m, worst) = worst_gap(&same, |_| 0);
            assert_eq!(m, 0);
            assert!(
                worst <= 0,
                "seed {seed:#x}: {worst} between two of one level"
            );
        }
        println!("{checked} stretches, {between} between levels; unskirted worst {unskirted}");
        assert!(between > 2_000, "only {between} stretches between levels");
        assert!(
            unskirted > 0,
            "no edge needed its skirt, so this proves nothing about it"
        );
    }

    /// The same edges have no T-junction left open in the geometry itself:
    /// every wall stands on whole quads of its own patch, and a finer
    /// patch's quads divide its coarser neighbour's exactly, so the corners
    /// of one lie on the lines of the other -- what is left is rounding in
    /// the projection, which `face.wgsl`'s `CRACK_SEAL` covers, as it does
    /// for the voxels' greedy mesh.
    #[test]
    fn a_finer_patchs_quads_divide_its_neighbours() {
        let (seed, view) = watertight_views().remove(1);
        let patches = build(&WorldGen::new(seed), &view);
        let mut levels = std::collections::BTreeSet::new();
        for p in &patches {
            levels.insert(p.key.level);
            let [ox, oz] = p.key.origin();
            let q = p.key.quad();
            // Its edges are on its own grid, and on every finer one.
            for finer in 0..=p.key.level {
                assert_eq!(ox.rem_euclid(1 << finer), 0);
                assert_eq!(oz.rem_euclid(1 << finer), 0);
                assert_eq!(q.rem_euclid(1 << finer), 0);
            }
            for coarser in &patches {
                if coarser.key.level <= p.key.level {
                    continue;
                }
                // A finer patch is inside one quad line pair of a coarser
                // one or outside it, never across.
                let cq = coarser.key.quad();
                let [cx, cz] = coarser.key.origin();
                let size = coarser.key.size();
                let overlaps = ox < cx + size && cx < ox + p.key.size();
                let overlaps_z = oz < cz + size && cz < oz + p.key.size();
                assert!(
                    !(overlaps && overlaps_z),
                    "{:?} over {:?}",
                    p.key,
                    coarser.key
                );
                assert_eq!((ox - cx).rem_euclid(cq.min(p.key.size())), 0);
                assert_eq!((oz - cz).rem_euclid(cq.min(p.key.size())), 0);
            }
        }
        assert!(levels.len() >= 5, "only levels {levels:?}");
    }
}
