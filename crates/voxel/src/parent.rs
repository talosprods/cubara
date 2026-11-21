//! What the next coarser level draws, for the colour morph (issue #380's
//! follow-up, `docs/PHASE3_ARCHITECTURE.md` §9.6).
//!
//! The distance morph moves a node's vertices to where its parent level
//! draws them, so that when the ring edge swaps the two nothing moves. The
//! *shape* then agrees; the colour did not -- a riser wore its own material
//! and its own baked shade, the parent's top its own shade band -- and that
//! was most of what a swap still changed on screen. So each vertex also
//! carries what the parent draws at its fully morphed position
//! ([`crate::Vertex::with_parent`]), and `mesh.wgsl` blends toward it by the
//! same factor that moves it.
//!
//! **What the parent draws is read off the parent's own mesh**, made by the
//! same mesher (`ARCHITECTURE.md` Rule 5): which quad covers a point, its
//! texture layer, and its openness there interpolated over the triangle that
//! holds the point, as the rasterizer interpolates it. Reading it off the
//! mesh rather than working it out again from the parent's cells is what
//! makes it exact where the parent's mesher merges rows into one ramp of
//! shade (a coarse wall's band, `wall_ao`): that ramp is a property of the
//! merged quad, not of any one cell.

use std::collections::HashMap;

use crate::mesh::{Face, Mesh};

/// One quad of the parent's mesh, in the child node's frame: its texture
/// layer and its two triangles, each corner as `(a, b, openness)` with `a`,
/// `b` the plane's two in-plane axes (`(d + 1) % 3` and `(d + 2) % 3` for a
/// face along `d`, the mesher's sweep axes).
#[derive(Clone, Debug, PartialEq)]
struct ParentQuad {
    layer: u32,
    tris: [[[f32; 3]; 3]; 2],
    /// Its openness where every corner has the same; `None` where it varies.
    uniform: Option<f32>,
}

/// The parent level's faces over a child node ([`ParentFaces::from_mesh`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParentFaces {
    quads: Vec<ParentQuad>,
    /// `(face, plane, a, b)` -> quad, one entry per block of each quad's
    /// area: the quad that covers the block `[a, a + 1) x [b, b + 1)` of the
    /// plane `plane` (in blocks along the face's axis).
    cells: HashMap<(u8, i32, i32, i32), u32>,
}

/// What a child face shows of its parent ([`ParentFaces::look_up`]), the
/// part of a mesher cell's identity that two cells must share to merge: the
/// parent's layer, and its openness where that is the same all over the
/// quad -- so that cells of two parent quads that look the same still merge
/// -- or else the quad itself, whose openness the merged child quad reads at
/// its own corners ([`ParentFaces::openness_at`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParentLook {
    /// Every corner of the parent quad is equally open: `(layer, openness
    /// in 1/255ths)`.
    Uniform(u32, u8),
    /// The parent quad's openness varies: `(layer, quad)`.
    Quad(u32, u32),
}

impl ParentLook {
    /// The parent's texture layer.
    pub fn layer(self) -> u32 {
        match self {
            ParentLook::Uniform(layer, _) | ParentLook::Quad(layer, _) => layer,
        }
    }
}

/// The in-plane axes of a face along `d`.
fn plane_axes(d: usize) -> (usize, usize) {
    ((d + 1) % 3, (d + 2) % 3)
}

impl ParentFaces {
    /// The faces of `parent`, a mesh made by the same mesher, whose node
    /// origin lies `offset` blocks from the child's (parent origin minus
    /// child origin); only the blocks of the planes whose in-plane extent
    /// overlaps `[lo, hi)` (child frame, in blocks, all three axes) are
    /// indexed -- the child's own box, with whatever margin its faces may
    /// reach at full morph.
    ///
    /// Quads with no area (a latent riser, which only the parent's own
    /// morph opens) are left out: at rest they show nothing.
    pub fn from_mesh(parent: &Mesh, offset: [i32; 3], lo: [i32; 3], hi: [i32; 3]) -> Self {
        let mut out = ParentFaces::default();
        for quad in parent.indices.chunks(6) {
            let verts: Vec<_> = quad.iter().map(|&i| parent.vertices[i as usize]).collect();
            let face = verts[0].face();
            let d = face as usize / 2;
            let (ua, va) = plane_axes(d);
            let pos = |k: usize| {
                let v = verts[k];
                let p = [v.x() as i32, v.y() as i32, v.z() as i32];
                [p[0] + offset[0], p[1] + offset[1], p[2] + offset[2]]
            };
            let plane = pos(0)[d];
            let corner = |k: usize| -> [f32; 3] {
                let p = pos(k);
                [p[ua] as f32, p[va] as f32, verts[k].openness()]
            };
            let tris = [
                [corner(0), corner(1), corner(2)],
                [corner(3), corner(4), corner(5)],
            ];
            let (mut a0, mut a1, mut b0, mut b1) = (i32::MAX, i32::MIN, i32::MAX, i32::MIN);
            for k in 0..6 {
                let p = pos(k);
                a0 = a0.min(p[ua]);
                a1 = a1.max(p[ua]);
                b0 = b0.min(p[va]);
                b1 = b1.max(p[va]);
            }
            if a0 == a1 || b0 == b1 {
                continue;
            }
            if plane < lo[d] || plane > hi[d] {
                continue;
            }
            let opens: Vec<f32> = (0..6).map(|k| verts[k].openness()).collect();
            let uniform = opens
                .iter()
                .all(|&o| o == opens[0])
                .then_some(opens[0]);
            let id = out.quads.len() as u32;
            out.quads.push(ParentQuad {
                layer: verts[0].tex_layer(),
                tris,
                uniform,
            });
            for a in a0.max(lo[ua])..a1.min(hi[ua]) {
                for b in b0.max(lo[va])..b1.min(hi[va]) {
                    out.cells.insert((face as u8, plane, a, b), id);
                }
            }
        }
        out
    }

    /// The parent quad facing `face` that covers `inside` (child frame, in
    /// blocks): a point in the interior of a child face at its fully morphed
    /// place, on the face's plane. `None` where the parent draws no such
    /// face there.
    pub fn look_up(&self, face: Face, inside: [f32; 3]) -> Option<ParentLook> {
        let d = face as usize / 2;
        let (ua, va) = plane_axes(d);
        let key = (
            face as u8,
            inside[d].round() as i32,
            inside[ua].floor() as i32,
            inside[va].floor() as i32,
        );
        let &id = self.cells.get(&key)?;
        let q = &self.quads[id as usize];
        Some(match q.uniform {
            Some(open) => {
                ParentLook::Uniform(q.layer, (open * 255.0).round().clamp(0.0, 255.0) as u8)
            }
            None => ParentLook::Quad(q.layer, id),
        })
    }

    /// How open `look`'s quad is at `p` (child frame, in blocks, on its
    /// plane), interpolated over whichever of its two triangles holds
    /// `toward` -- a point a little inside the child face whose corner `p`
    /// is, which decides at the quad's diagonal and its edges.
    pub fn openness_at(&self, look: ParentLook, face: Face, p: [f32; 3], toward: [f32; 3]) -> f32 {
        let id = match look {
            ParentLook::Uniform(_, open) => return open as f32 / 255.0,
            ParentLook::Quad(_, id) => id,
        };
        let d = face as usize / 2;
        let (ua, va) = plane_axes(d);
        let q = &self.quads[id as usize];
        let at = |pt: [f32; 3]| [pt[ua], pt[va]];
        let weights = |tri: &[[f32; 3]; 3], pt: [f32; 2]| -> [f32; 3] {
            let [x0, y0, _] = tri[0];
            let [x1, y1, _] = tri[1];
            let [x2, y2, _] = tri[2];
            let det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
            let w0 = ((y1 - y2) * (pt[0] - x2) + (x2 - x1) * (pt[1] - y2)) / det;
            let w1 = ((y2 - y0) * (pt[0] - x2) + (x0 - x2) * (pt[1] - y2)) / det;
            [w0, w1, 1.0 - w0 - w1]
        };
        // The triangle that holds `toward`: the one whose smallest weight
        // there is the larger (both hold it on the diagonal itself, where
        // they agree).
        let toward = at(toward);
        let inside = |tri: &[[f32; 3]; 3]| {
            let w = weights(tri, toward);
            w[0].min(w[1]).min(w[2])
        };
        let tri = if inside(&q.tris[0]) >= inside(&q.tris[1]) {
            &q.tris[0]
        } else {
            &q.tris[1]
        };
        let w = weights(tri, at(p));
        (w[0] * tri[0][2] + w[1] * tri[1][2] + w[2] * tri[2][2]).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Vertex;

    /// A mesh of one quad facing `face`, corners in the mesher's order (two
    /// triangles 0-1-2 and 0-2-3), each `(position, ao)`, in `layer`.
    fn one_quad(face: Face, corners: [([u32; 3], u32); 4], layer: u32) -> Mesh {
        let vertices = corners
            .iter()
            .map(|&([x, y, z], ao)| Vertex::new(x, y, z, ao, face, layer, 0, 0))
            .collect();
        Mesh {
            vertices,
            indices: vec![0, 1, 2, 0, 2, 3],
            face_indices: [0; 6],
        }
    }

    const ALL: ([i32; 3], [i32; 3]) = ([-100, -100, -100], [100, 100, 100]);

    /// A top four blocks square at y = 8, dark at its corner (0, 0) and open
    /// at the others: found under any point of it, and read as the
    /// rasterizer reads it -- the corners' own openness, and on the
    /// diagonal from (0, 0) to (4, 4) the mean of its two ends.
    #[test]
    fn a_parent_quad_is_found_under_its_points_and_read_as_drawn() {
        // Top faces sweep (u, v) = (z, x).
        let mesh = one_quad(
            Face::PosY,
            [([0, 8, 0], 0), ([0, 8, 4], 3), ([4, 8, 4], 3), ([4, 8, 0], 3)],
            7,
        );
        let faces = ParentFaces::from_mesh(&mesh, [0, 0, 0], ALL.0, ALL.1);
        let look = faces.look_up(Face::PosY, [1.5, 8.0, 2.5]).expect("under the quad");
        assert_eq!(look, ParentLook::Quad(7, 0));
        assert_eq!(look.layer(), 7);
        let at = |p: [f32; 3], toward: [f32; 3]| faces.openness_at(look, Face::PosY, p, toward);
        let inside = [2.0, 8.0, 1.0];
        assert_eq!(at([0.0, 8.0, 0.0], inside), 0.0);
        assert_eq!(at([4.0, 8.0, 4.0], inside), 1.0);
        assert!((at([2.0, 8.0, 2.0], inside) - 0.5).abs() < 1e-6);
        // Off the quad, the other way up, or another plane: nothing.
        assert_eq!(faces.look_up(Face::PosY, [4.5, 8.0, 1.0]), None);
        assert_eq!(faces.look_up(Face::NegY, [1.5, 8.0, 2.5]), None);
        assert_eq!(faces.look_up(Face::PosY, [1.5, 9.0, 2.5]), None);
    }

    /// Which triangle holds a corner is decided by the point `toward`: at
    /// the corner (4, 0), shared by both triangles' edges but a vertex of
    /// only one, the openness is read off the one `toward` lies in.
    #[test]
    fn the_triangle_that_holds_the_child_face_is_the_one_read() {
        let mesh = one_quad(
            Face::PosY,
            [([0, 8, 0], 3), ([0, 8, 4], 0), ([4, 8, 4], 3), ([4, 8, 0], 0)],
            1,
        );
        let faces = ParentFaces::from_mesh(&mesh, [0, 0, 0], ALL.0, ALL.1);
        let look = faces.look_up(Face::PosY, [1.0, 8.0, 1.0]).unwrap();
        // (x, z) = (2, 2) is on the diagonal: both triangles say 1.
        // (x, z) = (1, 3) lies in triangle 0-1-2, (3, 1) in 0-2-3; the point
        // (2, 1) read toward each is that triangle's plane extended.
        let p = [2.0, 8.0, 1.0];
        let a = faces.openness_at(look, Face::PosY, p, [1.0, 8.0, 3.0]);
        let b = faces.openness_at(look, Face::PosY, p, [3.0, 8.0, 1.0]);
        assert!((a - b).abs() > 0.2, "both read one triangle: {a} {b}");
    }

    /// A quad equally open all over is read as its openness, so cells of
    /// two such quads can merge; the offset moves the parent into the
    /// child's frame; and only the window is indexed.
    #[test]
    fn a_uniform_quad_reads_as_its_openness_and_the_offset_and_window_hold() {
        let mesh = one_quad(
            Face::PosX,
            [([16, 0, 0], 3), ([16, 2, 0], 3), ([16, 2, 2], 3), ([16, 0, 2], 3)],
            3,
        );
        let faces = ParentFaces::from_mesh(&mesh, [-16, 4, 0], ALL.0, ALL.1);
        assert_eq!(
            faces.look_up(Face::PosX, [0.0, 5.5, 1.5]),
            Some(ParentLook::Uniform(3, 255))
        );
        assert_eq!(faces.look_up(Face::PosX, [16.0, 1.5, 1.5]), None);
        let window = ParentFaces::from_mesh(&mesh, [-16, 4, 0], [0, 0, 0], [16, 16, 1]);
        assert!(window.look_up(Face::PosX, [0.0, 5.5, 0.5]).is_some());
        assert_eq!(window.look_up(Face::PosX, [0.0, 5.5, 1.5]), None, "z 1 is out");
    }
}
