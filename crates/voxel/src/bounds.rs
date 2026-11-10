//! Axis-aligned bounds: a meshed node's, local to the node (issue #379), and
//! the world-space or anchor-relative boxes made from them.
//!
//! Pure geometry — no GPU, no world/streaming knowledge (`ARCHITECTURE.md` Rule
//! 4) — shared between `cubara-world` (which computes a meshed node's bounds,
//! `mesh_with_bounds` below) and `cubara-render` (which culls against them,
//! `crate::culling` there). Living here, next to `Mesh`, is what lets both
//! depend on it without either depending on the other.

use glam::Vec3;

use crate::aniso::NodeChunk;
use crate::mesh::Mesh;
use crate::voxel::{Chunk, MeshContext};

/// An axis-aligned bounding box. Which space it is in is the holder's to
/// say: [`build_mesh_bounded`] gives node-local ones, [`Aabb::translated`]
/// moves one into whatever space its offset is measured from.
#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// This box moved by `by`: the one "local bounds + where the node stands"
    /// step (`ARCHITECTURE.md` Rule 5) that turns a node-local box from
    /// [`build_mesh_bounded`] into a world-space or render-anchor-relative
    /// one. The caller chooses what `by` is measured from.
    pub fn translated(self, by: Vec3) -> Self {
        Self::new(self.min + by, self.max + by)
    }

    /// A node-local box placed at the node's origin `block`, in world space:
    /// [`translated`](Self::translated) by the block as `f32`. Exact below
    /// 2^24 blocks; a renderer with a render anchor subtracts the anchor
    /// from the block first and translates by the difference instead.
    pub fn at_block(self, block: [i32; 3]) -> Self {
        self.translated(Vec3::new(block[0] as f32, block[1] as f32, block[2] as f32))
    }

    /// The corner that extends furthest in the direction of `normal` — the corner
    /// most likely to remain inside the half-space `normal` points into. Used by
    /// `cubara-render`'s frustum test (`Frustum::intersects_aabb`).
    pub fn positive_vertex(&self, normal: Vec3) -> Vec3 {
        Vec3::new(
            if normal.x >= 0.0 {
                self.max.x
            } else {
                self.min.x
            },
            if normal.y >= 0.0 {
                self.max.y
            } else {
                self.min.y
            },
            if normal.z >= 0.0 {
                self.max.z
            } else {
                self.min.z
            },
        )
    }
}

/// Greedy-mesh `chunk` and compute its **node-local** bounds -- relative to
/// the chunk/node's own minimum corner, in world units: `scale` is how big
/// one of its lattice cells is (1.0 for an ordinary chunk; `2^level` for an
/// LOD node, whose `16^3` lattice covers `2^level` chunks per axis at the
/// node's own coarser sample spacing). Returns `None` for a chunk that
/// produces no geometry.
///
/// Local, not world-space, on purpose (issue #379): the node's origin is an
/// integer block the renderer subtracts its render anchor from exactly, and
/// a box that had the origin added in `f32` here would have to have it
/// subtracted in `f32` there -- lossy past 2^24 blocks. Whoever needs the
/// box in world space (`cubara-world`'s `NodeGeometry::world_aabb`) or
/// relative to an anchor (`cubara-render`'s `NodeBox`) adds the offset it
/// wants with [`Aabb::translated`].
///
/// This is the one shared step between two otherwise-separate meshing paths: an
/// explicit `(ChunkCoord, Chunk)` pair with no `World` involved (`cubara-render`'s
/// `render_chunks`/`upload_node`, `scale` always 1.0), and a `World`-driven
/// `NodeKey` (`cubara-world`'s `mesh::mesh_node`, `scale` from the node's own
/// level) — both just need "mesh this chunk, this big," so it lives here
/// rather than being duplicated in both crates.
pub fn build_mesh_bounded(chunk: &Chunk, ctx: &MeshContext, scale: f32) -> Option<(Mesh, Aabb)> {
    build_mesh_bounded_occluded(chunk, ctx, scale, |_, _, _| false, |_, _, _| true)
}

/// [`build_mesh_bounded`] with [`Chunk::build_mesh_with_neighbours`]'s view of
/// the neighbours: covered border faces left out, skirts only where needed.
pub fn build_mesh_bounded_occluded(
    chunk: &Chunk,
    ctx: &MeshContext,
    scale: f32,
    occluded: impl Fn(i32, i32, i32) -> bool,
    may_be_solid: impl Fn(i32, i32, i32) -> bool,
) -> Option<(Mesh, Aabb)> {
    let mesh = chunk.build_mesh_with_neighbours(ctx, occluded, may_be_solid);
    if mesh.indices.is_empty() {
        return None;
    }
    let aabb = mesh_aabb(&mesh, Vec3::splat(scale));
    Some((mesh, aabb))
}

/// [`build_mesh_bounded_occluded`], for a [`NodeChunk`] (R1b, decision 6) --
/// the anisotropic coarse-node path alongside the plain-`Chunk` one above.
///
/// There is no `scale` parameter: a `NodeChunk::Aniso` already bakes its own
/// per-axis scale into every vertex position it emits (decision 3 --
/// positions in blocks, so `node_origins.w` is 1.0 for these), and a
/// `NodeChunk::Uniform` is exactly a level-0 `Chunk`, whose positions are
/// already node-local block coordinates too (its own scale is 1 on every
/// axis). So the node-local bounds here are the mesh's own positions,
/// unscaled -- the same place [`mesh_aabb`]'s `scale` argument would be
/// `[1.0; 3]` if this took one.
pub fn build_mesh_bounded_occluded_node(
    chunk: &NodeChunk,
    ctx: &MeshContext,
    occluded: impl Fn(i32, i32, i32) -> bool,
    may_be_solid: impl Fn(i32, i32, i32) -> bool,
) -> Option<(Mesh, Aabb)> {
    let mesh = chunk.build_mesh_with_neighbours(ctx, occluded, may_be_solid);
    if mesh.indices.is_empty() {
        return None;
    }
    let aabb = mesh_aabb(&mesh, Vec3::ONE);
    Some((mesh, aabb))
}

/// The node-local bounds of `mesh`'s vertex positions, scaled per axis by
/// `scale` -- the one "mesh + scale -> AABB" computation
/// [`build_mesh_bounded_occluded`] and [`build_mesh_bounded_occluded_node`]
/// both reduce to (`ARCHITECTURE.md` Rule 5), so a future third caller has
/// one place to share rather than a second copy of the loop. Exact: lattice
/// positions are small integers and `scale` is a power of two.
fn mesh_aabb(mesh: &Mesh, scale: Vec3) -> Aabb {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for v in &mesh.vertices {
        let p = Vec3::new(v.x() as f32, v.y() as f32, v.z() as f32);
        min = min.min(p);
        max = max.max(p);
    }
    Aabb::new(min * scale, max * scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translated_moves_both_corners_by_the_same_offset() {
        let b = Aabb::new(Vec3::new(0.0, 1.0, 2.0), Vec3::new(16.0, 9.0, 18.0));
        let t = b.translated(Vec3::new(1_048_576.0, -32.0, 64.0));
        assert_eq!(t.min, Vec3::new(1_048_576.0, -31.0, 66.0));
        assert_eq!(t.max, Vec3::new(1_048_592.0, -23.0, 82.0));
    }

    #[test]
    fn positive_vertex_picks_the_far_corner_per_axis_sign() {
        let b = Aabb::new(Vec3::new(-1.0, -2.0, -3.0), Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(
            b.positive_vertex(Vec3::new(1.0, 1.0, 1.0)),
            Vec3::new(1.0, 2.0, 3.0)
        );
        assert_eq!(
            b.positive_vertex(Vec3::new(-1.0, -1.0, -1.0)),
            Vec3::new(-1.0, -2.0, -3.0)
        );
    }
}
