//! **A candidate measured and not built: the far terrain hidden behind
//! itself** -- an occlusion cull of the far terrain's tiles
//! (`cubara_render::far::TILE_QUADS`), worked out on the CPU from where the
//! eye is. `far_census_visible` measures what it would leave out, and holds
//! it to a software depth buffer (`docs/PHASE3_ARCHITECTURE.md` §9.8 has the
//! numbers and why it is not in the renderer).
//!
//! Seen from an eye near the ground, most of the far terrain is behind
//! nearer far terrain. This finds tiles that cannot own a pixel --
//! conservatively: a tile is hidden only when every point of it is under
//! the horizon that solid ground nearer the eye draws.
//!
//! **The horizon.** Round the eye, in [`HORIZON_BINS`] directions, the
//! steepest slope `(y - eye.y) / d` (`d` the distance across the ground)
//! under which everything is known to be inside the ground. A tile whose
//! quads all stand at least `floor` high is solid from `floor` down over its
//! whole square, so a line of sight that passes over its square lower than
//! `floor` has gone into the ground -- and into the ground means through a
//! face of it first, facing the eye, which is drawn: the far terrain is
//! watertight, and where a line of sight leaves the voxels' box under the
//! far terrain's skirt on the box's side, it has gone into the voxels'
//! ground (`far_census_visible` draws a stand-in for it; without one, such
//! lines were the only ground this ever hid wrongly). Under the square,
//! along one direction, that is every slope up to `(floor - eye.y) / d1`
//! where `floor` is above the eye and `d1` is where the line enters the
//! square -- up to `(floor - eye.y) / d2`, `d2` where it leaves it, where
//! `floor` is below -- for whatever lies beyond `d2`. Both are bounded here
//! by the square's farthest and nearest corners, the safe way round.
//!
//! **The order.** Tiles are tested nearest first; a tile that is not hidden
//! joins the horizon once the sweep has passed its far corner, so that
//! everything it hides lies wholly behind it. A tile the voxels' box reaches
//! is never hidden and never hides (its corners are the join's to move),
//! and neither is one the eye stands over.
//!
//! **The margins.** A tile is tested with its box grown by a block and by
//! `1e-4` of its distance on every side -- more than the crack seal moves a
//! corner (`face.wgsl`'s `CRACK_SEAL`, `6e-5` of the distance) and than
//! `f32` loses in the shader -- and against one more direction on either
//! side; a floor counts a block lower than it is. A direction is only
//! raised where the square covers all of it.

use cubara_render::Aabb;
use glam::Vec3;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// The directions round the eye the horizon is kept in. A tile of the far
/// terrain is some 0.1 radian wide on screen at Medium; this is 0.003.
/// **Chosen, not derived**: finer costs time per frame for every tile, and
/// a tile only ever raises the directions it covers whole.
pub const HORIZON_BINS: usize = 2048;

/// What the occlusion cull knows of one tile, in the eye's frame of
/// reference (both anchor-relative, or both in world blocks).
#[derive(Clone, Copy, Debug)]
pub struct HorizonTile {
    /// Every point of every face of the tile is inside this box.
    pub bounds: Aabb,
    /// The tile's quads, `[x0, z0, x1, z1]`: the ground is solid under
    /// [`floor`](Self::floor) over all of this.
    pub square: [f32; 4],
    /// The lowest top among the tile's quads; `f32::NEG_INFINITY` when that
    /// is not known (then it hides nothing).
    pub floor: f32,
    /// Neither hidden nor hiding: a tile the voxels' box reaches.
    pub exempt: bool,
}

/// The horizon's buffers, kept from frame to frame so that a frame
/// allocates nothing.
#[derive(Debug, Default)]
pub struct Horizon {
    slope: Vec<f64>,
    order: Vec<(u64, u32)>,
    pending: BinaryHeap<Reverse<(u64, u32)>>,
}

/// A direction across the ground as a number in `[0, 4)` that grows with
/// the angle from `+x` towards `+z` -- the angle without the trigonometry
/// (a quarter turn is one).
pub fn pseudo_angle(dx: f64, dz: f64) -> f64 {
    let s = dx.abs() + dz.abs();
    let (x, z) = (dx / s, dz / s);
    if z >= 0.0 {
        if x > 0.0 {
            z
        } else {
            1.0 - x
        }
    } else if x < 0.0 {
        2.0 - z
    } else {
        3.0 + x
    }
}

/// How a rectangle `[x0, z0, x1, z1]` lies from an eye at `(ex, ez)`: the
/// distance to its nearest point and to its farthest corner, and the
/// directions it covers as `[lo, hi]` in [`pseudo_angle`]s (`hi - lo < 2`,
/// either may be outside `[0, 4)`). `None` when the eye is over it.
pub fn view_of(r: [f64; 4], ex: f64, ez: f64) -> Option<(f64, f64, [f64; 2])> {
    if ex >= r[0] && ex <= r[2] && ez >= r[1] && ez <= r[3] {
        return None;
    }
    let dx = (r[0] - ex).max(ex - r[2]).max(0.0);
    let dz = (r[1] - ez).max(ez - r[3]).max(0.0);
    let near = (dx * dx + dz * dz).sqrt();
    let corners = [(r[0], r[1]), (r[2], r[1]), (r[0], r[3]), (r[2], r[3])];
    let far = corners
        .iter()
        .map(|&(x, z)| ((x - ex).powi(2) + (z - ez).powi(2)).sqrt())
        .fold(0.0, f64::max);
    let centre = pseudo_angle((r[0] + r[2]) / 2.0 - ex, (r[1] + r[3]) / 2.0 - ez);
    let (mut lo, mut hi) = (0.0f64, 0.0f64);
    for (x, z) in corners {
        let off = (pseudo_angle(x - ex, z - ez) - centre + 2.0).rem_euclid(4.0) - 2.0;
        lo = lo.min(off);
        hi = hi.max(off);
    }
    Some((near, far, [centre + lo, centre + hi]))
}

/// The bin of a pseudo-angle, unwrapped (an index past either end of
/// `0..HORIZON_BINS` is the same direction modulo it).
fn bin(a: f64) -> i64 {
    (a * HORIZON_BINS as f64 / 4.0).floor() as i64
}

impl Horizon {
    /// Whether each of `tiles` can show an eye at `eye` anything: `shown[k]`
    /// is `false` only for a tile every point of which is under the
    /// horizon of the ground nearer to the eye (the module's comment).
    pub fn cull(&mut self, eye: Vec3, tiles: &[HorizonTile], shown: &mut Vec<bool>) {
        let (ex, ey, ez) = (eye.x as f64, eye.y as f64, eye.z as f64);
        shown.clear();
        shown.resize(tiles.len(), true);
        self.slope.clear();
        self.slope.resize(HORIZON_BINS, f64::NEG_INFINITY);
        self.pending.clear();
        self.order.clear();
        // The tile as tested: its box, grown by the margin.
        let grown = |t: &HorizonTile| {
            let b = t.bounds;
            let reach = ((b.max.x as f64 - ex).abs().max((b.min.x as f64 - ex).abs()))
                .max((b.max.z as f64 - ez).abs().max((b.min.z as f64 - ez).abs()));
            let m = 1.0 + 1e-4 * reach * std::f64::consts::SQRT_2;
            (
                [
                    b.min.x as f64 - m,
                    b.min.z as f64 - m,
                    b.max.x as f64 + m,
                    b.max.z as f64 + m,
                ],
                b.max.y as f64 + m,
            )
        };
        for (k, t) in tiles.iter().enumerate() {
            if t.exempt {
                continue;
            }
            if let Some((near, _, _)) = view_of(grown(t).0, ex, ez) {
                self.order.push((near.to_bits(), k as u32));
            }
        }
        self.order.sort_unstable();
        for at in 0..self.order.len() {
            let (near_bits, k) = self.order[at];
            let near_t = f64::from_bits(near_bits);
            // Everything whose square ends before this tile begins.
            while let Some(&Reverse((far_bits, o))) = self.pending.peek() {
                if f64::from_bits(far_bits) > near_t {
                    break;
                }
                self.pending.pop();
                self.raise(&tiles[o as usize], ex, ey, ez);
            }
            let tile = &tiles[k as usize];
            let (rect, top) = grown(tile);
            let Some((near, far, [lo, hi])) = view_of(rect, ex, ez) else {
                continue;
            };
            let rise = top - ey;
            let steepest = if rise >= 0.0 { rise / near } else { rise / far };
            let hidden = hi - lo < 2.0
                && (bin(lo) - 1..=bin(hi) + 1).all(|b| {
                    self.slope[b.rem_euclid(HORIZON_BINS as i64) as usize] > steepest
                });
            if hidden {
                shown[k as usize] = false;
            } else if tile.floor.is_finite() {
                let s = tile.square.map(f64::from);
                if let Some((_, far, _)) = view_of(s, ex, ez) {
                    self.pending.push(Reverse((far.to_bits(), k)));
                }
            }
        }
    }

    /// Raise the horizon to what `tile`'s square of solid ground hides, in
    /// every direction the square covers whole.
    fn raise(&mut self, tile: &HorizonTile, ex: f64, ey: f64, ez: f64) {
        let s = tile.square.map(f64::from);
        let Some((near, far, [lo, hi])) = view_of(s, ex, ez) else {
            return;
        };
        if hi - lo >= 2.0 || near <= 0.0 {
            return;
        }
        let rise = tile.floor as f64 - 1.0 - ey;
        let slope = if rise >= 0.0 { rise / far } else { rise / near };
        // The bins wholly inside `[lo, hi]`.
        let first = (lo * HORIZON_BINS as f64 / 4.0).ceil() as i64;
        let last = bin(hi) - 1;
        for b in first..=last {
            let at = &mut self.slope[b.rem_euclid(HORIZON_BINS as i64) as usize];
            *at = at.max(slope);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pseudo_angle_grows_with_the_angle_all_the_way_round() {
        let mut last = -1.0;
        for k in 0..3600 {
            let a = k as f64 / 3600.0 * std::f64::consts::TAU;
            let p = pseudo_angle(a.cos(), a.sin());
            assert!((0.0..4.0).contains(&p), "{a}: {p}");
            assert!(p > last, "{a}: {p} after {last}");
            last = p;
        }
        // A quarter turn is one, and opposite directions are two apart.
        for (dx, dz, p) in [(1.0, 0.0, 0.0), (0.0, 1.0, 1.0), (-1.0, 0.0, 2.0), (0.0, -1.0, 3.0)] {
            assert_eq!(pseudo_angle(dx, dz), p);
        }
    }

    fn tile(square: [f32; 4], floor: f32, top: f32) -> HorizonTile {
        HorizonTile {
            bounds: Aabb {
                min: Vec3::new(square[0], floor - 50.0, square[1]),
                max: Vec3::new(square[2], top, square[3]),
            },
            square,
            floor,
            exempt: false,
        }
    }

    /// A ridge hides the low ground behind it and not the high ground, and
    /// hides nothing that is not behind it.
    #[test]
    fn a_ridge_hides_what_is_lower_behind_it() {
        let eye = Vec3::new(0.0, 100.0, 0.0);
        let tiles = [
            // A ridge 1,000 to 1,100 out along +x, 150 high.
            tile([1000.0, -400.0, 1100.0, 400.0], 150.0, 150.0),
            // Behind it, low: hidden.
            tile([2000.0, -50.0, 2100.0, 50.0], 100.0, 160.0),
            // Behind it, but standing above its line: shown.
            tile([2000.0, 100.0, 2100.0, 200.0], 100.0, 260.0),
            // Low, but off to the side of the ridge: shown.
            tile([2000.0, 1000.0, 2100.0, 1100.0], 0.0, 10.0),
            // Low, in front of the ridge: shown.
            tile([500.0, -50.0, 600.0, 50.0], 0.0, 10.0),
        ];
        let mut shown = Vec::new();
        Horizon::default().cull(eye, &tiles, &mut shown);
        assert_eq!(shown, vec![true, false, true, true, true]);
    }

    /// An exempt tile neither hides nor is hidden, and a tile the eye stands
    /// over is never hidden.
    #[test]
    fn the_voxels_box_and_the_eye_are_left_alone() {
        let eye = Vec3::new(0.0, 100.0, 0.0);
        let mut ridge = tile([1000.0, -400.0, 1100.0, 400.0], 150.0, 150.0);
        let behind = tile([2000.0, -50.0, 2100.0, 50.0], 100.0, 160.0);
        let mut shown = Vec::new();
        ridge.exempt = true;
        Horizon::default().cull(eye, &[ridge, behind], &mut shown);
        assert_eq!(shown, vec![true, true], "an exempt ridge hides nothing");
        let mut low = behind;
        low.exempt = true;
        ridge.exempt = false;
        Horizon::default().cull(eye, &[ridge, low], &mut shown);
        assert_eq!(shown, vec![true, true], "an exempt tile is not hidden");
        let under = tile([-10.0, -10.0, 10.0, 10.0], -500.0, -400.0);
        Horizon::default().cull(eye, &[ridge, under], &mut shown);
        assert_eq!(shown, vec![true, true], "the tile under the eye");
    }
}
