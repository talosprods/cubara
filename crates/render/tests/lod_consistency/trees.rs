//! **A tree looks the same at every distance** (issue #433, items 2 to 4):
//! the definition, as a measurement. No GPU.
//!
//! Level 0 draws a tree as blocks in its chunk's mesh. Levels 1 and up draw
//! it as an instance of a static model, standing on the top of the cell
//! that draws the ground under it. Over a seeded sample of the trees the
//! default world grows, each tree is drawn as every representation draws
//! it:
//!
//! - **level 0**: the tree's quads in the meshes the game makes of the
//!   level-0 chunks around it (`cubara_world::mesh::mesh_node`), each its
//!   texture's mean colour, lit by [`face_light`] and shaded by its own
//!   baked AO through [`ao_fraction`] -- a quad measured exactly as the
//!   ground harness measures one (`harness.rs`);
//! - **an instance at level `L`**: the model the game draws there, where
//!   the game places it, lit as `trees.wgsl` lights it;
//! - **an instance at the end of its band**: the same, where the distance
//!   morph has taken it when the ring edge swaps it for the next level.
//!
//! Each is then compared with another drawing of the same tree, tree by
//! tree:
//!
//! - **height**: the tree's top (its highest face) and its foot (its
//!   lowest), in blocks;
//! - **outline**: the silhouette from five views (three level, two from
//!   above), each rasterised at an eighth of a block, as the area one
//!   drawing covers and the other does not, in square blocks;
//! - **brightness**: the mean colour of what each view sees of the tree
//!   (the nearest face at each sample), and the area-weighted lit colour
//!   per face direction, both /255 in linear light (`colour_delta`).
//!
//! Every view and every colour is of the tree alone: the ground it stands
//! on is the ground harness's (`main.rs`), and is not drawn here.

use cubara_render::aggregate::{ao_fraction, face_light};
use cubara_voxel::Face;
use cubara_world::node::NodeKey;
use cubara_world::{TreeSite, WorldSnapshot};
use glam::Vec3;

use super::harness::{colour_delta, Rng, Setup};

/// The six directions a face of a tree can point, in the order every array
/// here uses.
pub const TREE_DIRECTIONS: [Face; 6] = [
    Face::PosY,
    Face::NegY,
    Face::PosX,
    Face::NegX,
    Face::PosZ,
    Face::NegZ,
];

pub fn tree_direction_name(d: usize) -> &'static str {
    ["+Y", "-Y", "+X", "-X", "+Z", "-Z"][d]
}

fn direction_of(n: Vec3) -> usize {
    TREE_DIRECTIONS
        .iter()
        .position(|f| Vec3::from(f.normal()).dot(n) > 0.99)
        .unwrap_or_else(|| panic!("a tree face whose normal {n:?} is not on an axis"))
}

/// One face of a drawn tree, in world blocks.
#[derive(Clone, Copy, Debug)]
pub struct TreeQuad {
    pub corners: [Vec3; 4],
    /// Index into [`TREE_DIRECTIONS`].
    pub dir: usize,
    /// The face's lit colour, linear: its texture's mean (or the model's
    /// flat colour), its light, and the mean of its shade over the two
    /// triangles it is drawn as.
    pub lit: Vec3,
}

impl TreeQuad {
    pub fn normal(&self) -> Vec3 {
        Vec3::from(TREE_DIRECTIONS[self.dir].normal())
    }

    pub fn area(&self) -> f64 {
        let [a, b, _, d] = self.corners;
        ((b - a).cross(d - a)).length() as f64
    }
}

/// A tree as one representation draws it.
#[derive(Clone, Debug, Default)]
pub struct DrawnTree {
    pub quads: Vec<TreeQuad>,
}

impl DrawnTree {
    /// The highest face, in blocks.
    pub fn top(&self) -> f32 {
        self.quads
            .iter()
            .flat_map(|q| q.corners.iter().map(|c| c.y))
            .fold(f32::MIN, f32::max)
    }

    /// The lowest face (the foot of the trunk), in blocks.
    pub fn foot(&self) -> f32 {
        self.quads
            .iter()
            .flat_map(|q| q.corners.iter().map(|c| c.y))
            .fold(f32::MAX, f32::min)
    }

    /// Per direction, the face area and its lit colour times area.
    pub fn faces(&self) -> ([f64; 6], [[f64; 3]; 6]) {
        let (mut area, mut lit) = ([0f64; 6], [[0f64; 3]; 6]);
        for q in &self.quads {
            let a = q.area();
            area[q.dir] += a;
            for c in 0..3 {
                lit[q.dir][c] += q.lit[c] as f64 * a;
            }
        }
        (area, lit)
    }

    /// This tree moved by `by` blocks: what a test builds a wrong drawing
    /// from on purpose.
    pub fn moved(&self, by: Vec3) -> DrawnTree {
        DrawnTree {
            quads: self
                .quads
                .iter()
                .map(|q| TreeQuad {
                    corners: q.corners.map(|c| c + by),
                    ..*q
                })
                .collect(),
        }
    }
}

/// The mean over a quad of a value given at its corners, as the rasteriser
/// interpolates it over the two triangles the quad is drawn as: the corners
/// on the shared diagonal count in both. `diagonal_02` says which diagonal
/// that is (the first index of the six is the quad's first corner).
fn quad_mean(v: [f64; 4], diagonal_02: bool) -> f64 {
    if diagonal_02 {
        (2.0 * (v[0] + v[2]) + v[1] + v[3]) / 6.0
    } else {
        (2.0 * (v[1] + v[3]) + v[0] + v[2]) / 6.0
    }
}

/// What level 0 draws of `site`: its quads in the game's own meshes of the
/// level-0 chunks it stands in, those of the oak's textures inside its
/// bounds. Also returns how many oak quads were left out because they
/// reach outside those bounds (two canopies that touch merge their faces).
pub fn level_0_tree(setup: &Setup, snapshot: &WorldSnapshot, site: TreeSite) -> (DrawnTree, usize) {
    let blocks = setup.tree_blocks();
    let tree_layers = ["oak_log_side", "oak_log_top", "oak_leaves"].map(|n| setup.layer_of(n));
    let r = site.canopy_radius;
    let lo = Vec3::new((site.x - r) as f32, site.base as f32, (site.z - r) as f32);
    let hi = Vec3::new(
        (site.x + r + 1) as f32,
        (site.base + site.height + 1) as f32,
        (site.z + r + 1) as f32,
    );
    let inside = |p: Vec3| p.cmpge(lo - 1e-3).all() && p.cmple(hi + 1e-3).all();
    let (cx, cz) = (site.x.div_euclid(16), site.z.div_euclid(16));
    let layer_of = |name: &str| setup.layer_of(name);
    let mut quads = Vec::new();
    let mut stray = 0;
    for cy in (site.base - 1).div_euclid(16)..=(site.base + site.height + 1).div_euclid(16) {
        let node = NodeKey::new(0, [cx, cy, cz]);
        let Some(geometry) =
            cubara_world::mesh::mesh_node(snapshot, &setup.registry, &layer_of, node, blocks)
        else {
            continue;
        };
        let origin = Vec3::from(geometry.origin.map(|c| c as f32));
        let mesh = &geometry.mesh;
        for six in mesh.indices.chunks(6) {
            let base = *six.iter().min().expect("six indices") as usize;
            let corners = &mesh.vertices[base..base + 4];
            let layer = corners[0].tex_layer();
            if !tree_layers.contains(&layer) {
                continue;
            }
            let pos: [Vec3; 4] = std::array::from_fn(|k| {
                let c = corners[k];
                origin + Vec3::new(c.x() as f32, c.y() as f32, c.z() as f32)
            });
            if !pos.iter().all(|&p| inside(p)) {
                stray += 1;
                continue;
            }
            let ao = std::array::from_fn(|k| {
                let c = corners[k];
                ao_fraction(c.ao(), c.ao_width_log2(), setup.lighting.ao_floor) as f64
            });
            let shade = quad_mean(ao, six[0] as usize == base) as f32;
            let normal = Vec3::from(corners[0].face().normal());
            quads.push(TreeQuad {
                corners: pos,
                dir: direction_of(normal),
                lit: setup.colour(layer) * face_light(normal, &setup.lighting) * shade,
            });
        }
    }
    (DrawnTree { quads }, stray)
}

/// A tree model's quads (`cubara_render`'s `TreeModelMesh`) standing with
/// its local origin -- the trunk's base block -- at `at`, lit as
/// `trees.wgsl` lights them.
pub fn model_tree(setup: &Setup, model: &cubara_render::TreeModelMesh, at: Vec3) -> DrawnTree {
    let mut quads = Vec::new();
    for six in model.indices.chunks(6) {
        let base = *six.iter().min().expect("six indices") as usize;
        let vs = &model.vertices[base..base + 4];
        let normal = Vec3::from(vs[0].normal);
        let colour = if vs[0].layer >= 0.0 {
            setup.colour(vs[0].layer as u32)
        } else {
            Vec3::from(vs[0].color)
        };
        let shade = quad_mean(
            std::array::from_fn(|k| model_shade(setup, &vs[k])),
            six[0] as usize == base,
        ) as f32;
        quads.push(TreeQuad {
            corners: std::array::from_fn(|k| Vec3::from(vs[k].position) + at),
            dir: direction_of(normal),
            lit: colour * face_light(normal, &setup.lighting) * shade,
        });
    }
    DrawnTree { quads }
}

/// How much of its light `trees.wgsl` leaves a model vertex:
/// `mix(ao_floor, 1, open)`, `ao_fraction` of a one-block cell written for
/// an openness -- the same line `mesh.wgsl` lights a block by.
fn model_shade(setup: &Setup, v: &cubara_render::TreeVertex) -> f64 {
    let floor = setup.lighting.ao_floor as f64;
    floor + (1.0 - floor) * v.open as f64
}

/// The model the game draws a tree of `site` with at `level`: the oak of
/// its own trunk height at levels 1 and 2, its silhouette from level 3 --
/// built by the game's own builder (`tree_model_meshes`).
pub fn model_for(setup: &Setup, level: u32, site: TreeSite) -> cubara_render::TreeModelMesh {
    let layer_of = |name: &str| setup.layer_of(name);
    let (oaks, silhouettes) =
        cubara_render::tree_model_meshes(&setup.registry, &setup.structures, &layer_of);
    let shapes = cubara_render::TreeShapes::of(&setup.structures);
    let k = shapes.index(site.height);
    if level <= 2 {
        oaks[k].clone()
    } else {
        silhouettes[k].clone()
    }
}

/// The node of `level` that holds the ground under a tree: the one whose
/// rows hold the top block of the cell its trunk stands on.
pub fn node_under(setup: &Setup, level: u32, site: TreeSite) -> NodeKey {
    let size = 16 << level;
    let w = 1 << level;
    let top = setup
        .gen
        .cell_top(site.x.div_euclid(w) * w, site.z.div_euclid(w) * w, w);
    NodeKey::new(
        level,
        [
            site.x.div_euclid(size),
            top.div_euclid(size),
            site.z.div_euclid(size),
        ],
    )
}

/// Where the game stands the instance of `site` at `level`: the trunk's
/// base block, and how far the distance morph lifts it at the end of its
/// band, in blocks -- `cubara_world::mesh::node_trees`, what a built node
/// carries, for the node that holds the ground under it.
pub fn placed(setup: &Setup, snapshot: &WorldSnapshot, level: u32, site: TreeSite) -> ([i32; 3], i32) {
    let node = node_under(setup, level, site);
    let found = cubara_world::mesh::node_trees(snapshot, node, setup.tree_blocks(), None)
        .into_iter()
        .find(|t| (t.site.x, t.site.z) == (site.x, site.z))
        .unwrap_or_else(|| panic!("the node {node:?} does not place the tree at {site:?}"));
    assert_eq!(
        found.site.height, site.height,
        "a node placed the tree at ({}, {}) with another height",
        site.x, site.z
    );
    ([found.site.x, found.site.base, found.site.z], found.morph)
}

/// One way of drawing a tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rep {
    Level0,
    /// The instance at `level`; `morphed` at the end of its band.
    Instance { level: u32, morphed: bool },
}

impl Rep {
    pub fn name(self) -> String {
        match self {
            Rep::Level0 => "L0".into(),
            Rep::Instance {
                level,
                morphed: false,
            } => format!("L{level}"),
            Rep::Instance {
                level,
                morphed: true,
            } => format!("L{level} at full morph"),
        }
    }
}

/// `site` drawn as `rep` draws it.
pub fn draw(setup: &Setup, snapshot: &WorldSnapshot, site: TreeSite, rep: Rep) -> DrawnTree {
    match rep {
        Rep::Level0 => level_0_tree(setup, snapshot, site).0,
        Rep::Instance { level, morphed } => {
            let (at, morph) = placed(setup, snapshot, level, site);
            let y = at[1] + if morphed { morph } else { 0 };
            model_tree(
                setup,
                &model_for(setup, level, site),
                Vec3::new(at[0] as f32, y as f32, at[2] as f32),
            )
        }
    }
}

/// A view of a tree: the direction it is looked at along.
pub const VIEWS: [(&str, [f32; 3]); 5] = [
    ("level, from +x", [-1.0, 0.0, 0.0]),
    ("level, from +z", [0.0, 0.0, -1.0]),
    ("level, diagonal", [-1.0, 0.0, -1.0]),
    ("from above, oblique", [-1.0, -0.8, -0.6]),
    ("from straight above", [0.0, -1.0, 0.0]),
];

/// Samples per block of a view.
const RES: f32 = 8.0;
/// Half the side of a view, in blocks: room for the tallest tree and the
/// steps the ground moves it by.
const HALF: f32 = 12.0;
const SIDE: usize = (2.0 * HALF * RES) as usize;

/// What one view sees of a tree: per sample, the nearest face's depth and
/// lit colour, or nothing.
pub struct Image {
    pub seen: Vec<Option<(f32, Vec3)>>,
}

impl Image {
    pub fn area(&self) -> f64 {
        self.seen.iter().filter(|s| s.is_some()).count() as f64 / (RES * RES) as f64
    }

    /// The mean colour of what is seen.
    pub fn mean(&self) -> Option<Vec3> {
        let (sum, n) = self
            .seen
            .iter()
            .flatten()
            .fold((Vec3::ZERO, 0usize), |(s, n), (_, c)| (s + *c, n + 1));
        (n > 0).then(|| sum / n as f32)
    }
}

fn basis(look: Vec3) -> (Vec3, Vec3) {
    let look = look.normalize();
    let up = if look.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
    let right = look.cross(up).normalize();
    (right, right.cross(look).normalize())
}

/// `tree` seen along `look` by a parallel projection centred on `centre`.
pub fn image(tree: &DrawnTree, centre: Vec3, look: Vec3) -> Image {
    let look = look.normalize();
    let (right, up) = basis(look);
    let mut seen: Vec<Option<(f32, Vec3)>> = vec![None; SIDE * SIDE];
    let cell = |u: f32| ((u + HALF) * RES - 0.5).ceil() as i64;
    let last = |u: f32| ((u + HALF) * RES - 0.5).floor() as i64;
    for q in &tree.quads {
        let n = q.normal();
        let facing = n.dot(look);
        if facing >= -1e-6 {
            continue;
        }
        let p: [(f32, f32); 4] = q
            .corners
            .map(|c| ((c - centre).dot(right), (c - centre).dot(up)));
        let (umin, umax) = p.iter().fold((f32::MAX, f32::MIN), |(a, b), &(u, _)| (a.min(u), b.max(u)));
        let (vmin, vmax) = p.iter().fold((f32::MAX, f32::MIN), |(a, b), &(_, v)| (a.min(v), b.max(v)));
        for j in cell(vmin).max(0)..=last(vmax).min(SIDE as i64 - 1) {
            let v = (j as f32 + 0.5) / RES - HALF;
            for i in cell(umin).max(0)..=last(umax).min(SIDE as i64 - 1) {
                let u = (i as f32 + 0.5) / RES - HALF;
                let side = |k: usize| {
                    let (a, b) = (p[k], p[(k + 1) % 4]);
                    (b.0 - a.0) * (v - a.1) - (b.1 - a.1) * (u - a.0)
                };
                let s = [side(0), side(1), side(2), side(3)];
                if !(s.iter().all(|&x| x >= 0.0) || s.iter().all(|&x| x <= 0.0)) {
                    continue;
                }
                let o = centre + right * u + up * v;
                let t = n.dot(q.corners[0] - o) / facing;
                let slot = &mut seen[j as usize * SIDE + i as usize];
                if slot.is_none_or(|(d, _)| t < d) {
                    *slot = Some((t, q.lit));
                }
            }
        }
    }
    Image { seen }
}

/// One view of a tree, one drawing against another.
#[derive(Clone, Copy, Debug, Default)]
pub struct ViewCompare {
    /// What one covers and the other does not, in square blocks.
    pub xor: f64,
    /// The reference's silhouette, in square blocks.
    pub area: f64,
    /// The mean colours seen, /255 apart.
    pub seen: f64,
}

/// One tree, one drawing against another.
#[derive(Clone, Debug)]
pub struct TreeCompare {
    pub site: TreeSite,
    /// The drawing's top minus the reference's, in blocks.
    pub top: f32,
    /// The drawing's foot minus the reference's, in blocks.
    pub foot: f32,
    pub views: [ViewCompare; VIEWS.len()],
    /// Per face direction, the lit colours /255 apart; `None` where the
    /// reference shows under [`MIN_TREE_AREA`] that way.
    pub t3: [Option<f64>; 6],
}

/// The least area of a direction a tree's colour is taken from, in square
/// blocks. **Chosen, not derived**: the trunk's top face is one block, a
/// colour from one face is one sample.
pub const MIN_TREE_AREA: f64 = 4.0;

impl TreeCompare {
    pub fn worst_view_xor(&self) -> f64 {
        self.views.iter().map(|v| v.xor).fold(0.0, f64::max)
    }

    pub fn worst_view_seen(&self) -> f64 {
        self.views.iter().map(|v| v.seen).fold(0.0, f64::max)
    }

    pub fn worst_t3(&self) -> f64 {
        self.t3.iter().flatten().copied().fold(0.0, f64::max)
    }
}

/// The centre every view of `site` is taken around, whatever draws it: the
/// trunk's column, half way up the tree level 0 draws.
fn centre_of(site: TreeSite) -> Vec3 {
    Vec3::new(
        site.x as f32 + 0.5,
        site.base as f32 + (site.height + 2) as f32 / 2.0,
        site.z as f32 + 0.5,
    )
}

/// `other` against `reference`, both drawings of `site`.
pub fn compare_trees(site: TreeSite, reference: &DrawnTree, other: &DrawnTree) -> TreeCompare {
    let centre = centre_of(site);
    let views = std::array::from_fn(|k| {
        let look = Vec3::from(VIEWS[k].1);
        let (a, b) = (image(reference, centre, look), image(other, centre, look));
        let xor = a
            .seen
            .iter()
            .zip(&b.seen)
            .filter(|(x, y)| x.is_some() != y.is_some())
            .count() as f64
            / (RES * RES) as f64;
        let seen = match (a.mean(), b.mean()) {
            (Some(x), Some(y)) => colour_delta(x.as_dvec3().into(), y.as_dvec3().into()),
            (None, None) => 0.0,
            _ => f64::INFINITY,
        };
        ViewCompare {
            xor,
            area: a.area(),
            seen,
        }
    });
    let (ra, rl) = reference.faces();
    let (oa, ol) = other.faces();
    let t3 = std::array::from_fn(|d| {
        (ra[d] >= MIN_TREE_AREA).then(|| {
            if oa[d] <= 0.0 {
                return f64::INFINITY;
            }
            colour_delta(rl[d].map(|c| c / ra[d]), ol[d].map(|c| c / oa[d]))
        })
    });
    TreeCompare {
        site,
        top: other.top() - reference.top(),
        foot: other.foot() - reference.foot(),
        views,
        t3,
    }
}

/// Whether level 0 meshes `site` in more than one chunk: the rows from the
/// ground under it to its apex cross a chunk's floor. The level-0 mesher
/// bakes a chunk's AO without looking into the next one
/// (`docs/PHASE3_ARCHITECTURE.md` §9.3), so the faces just under such a
/// floor come out more open there than the corner rule says -- than the
/// instance, meshed whole, says.
pub fn crosses_a_chunk_floor(site: TreeSite) -> bool {
    (site.base - 1).div_euclid(16) != (site.base + site.height).div_euclid(16)
}

/// What the ground under a tree is like: how far its 5 x 5 columns' tops
/// spread, in blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TreeGround {
    /// All 25 columns at one height.
    Flat,
    /// One or two blocks between the lowest and the highest.
    Gentle,
    /// Three blocks or more.
    Steep,
}

pub const TREE_GROUNDS: [TreeGround; 3] = [TreeGround::Flat, TreeGround::Gentle, TreeGround::Steep];

pub fn tree_ground(setup: &Setup, site: TreeSite) -> TreeGround {
    let r = site.canopy_radius;
    let hs: Vec<i32> = (-r..=r)
        .flat_map(|dz| (-r..=r).map(move |dx| (dx, dz)))
        .map(|(dx, dz)| setup.gen.surface_height(site.x + dx, site.z + dz))
        .collect();
    let spread = hs.iter().max().unwrap() - hs.iter().min().unwrap();
    match spread {
        0 => TreeGround::Flat,
        1..=2 => TreeGround::Gentle,
        _ => TreeGround::Steep,
    }
}

/// The seed the trees are drawn from. Printed by every failure.
pub const TREE_SEED: u64 = 0x0433_7EE5_C0FF_0001;

/// `per_kind` trees on each kind of ground, drawn from `seed`: random chunk
/// columns within 6,000 blocks of the origin, each kept if it grows a tree
/// and its kind still has room. Not hand-picked -- the seed decides.
pub fn draw_trees(setup: &Setup, seed: u64, per_kind: usize) -> Vec<(TreeGround, TreeSite)> {
    let blocks = setup.tree_blocks();
    let mut rng = Rng(seed);
    let mut out: Vec<(TreeGround, TreeSite)> = Vec::new();
    let range = 6000 / 16;
    for _ in 0..2_000_000 {
        if out.len() == per_kind * TREE_GROUNDS.len() {
            break;
        }
        let (cx, cz) = (rng.within(range), rng.within(range));
        let Some(site) = setup.gen.tree_site(cx, cz, blocks) else {
            continue;
        };
        let kind = tree_ground(setup, site);
        if out.iter().filter(|(k, _)| *k == kind).count() < per_kind
            && !out.iter().any(|(_, s)| *s == site)
        {
            out.push((kind, site));
        }
    }
    out.sort_by_key(|(k, s)| (*k, s.x, s.z));
    out
}

/// What is compared: each instance level against level 0, and each ring
/// edge's two sides.
pub fn comparisons() -> Vec<(String, Rep, Rep)> {
    let inst = |level, morphed| Rep::Instance { level, morphed };
    vec![
        ("L1 vs L0".into(), Rep::Level0, inst(1, false)),
        ("L2 vs L0".into(), Rep::Level0, inst(2, false)),
        ("L3 vs L0".into(), Rep::Level0, inst(3, false)),
        ("edge 160 m: L1 vs L0".into(), Rep::Level0, inst(1, false)),
        ("edge 288 m: L2 vs L1 at full morph".into(), inst(1, true), inst(2, false)),
        ("edge 512 m: L3 vs L2 at full morph".into(), inst(2, true), inst(3, false)),
    ]
}

/// Every comparison of every tree: per comparison, per tree.
pub fn measure_trees(setup: &Setup, trees: &[(TreeGround, TreeSite)]) -> Vec<Vec<(TreeGround, TreeCompare)>> {
    let snapshot = setup.world.snapshot();
    let comparisons = comparisons();
    let mut out: Vec<Vec<(TreeGround, TreeCompare)>> = vec![Vec::new(); comparisons.len()];
    for &(kind, site) in trees {
        let mut drawn: Vec<(Rep, DrawnTree)> = Vec::new();
        let mut get = |rep: Rep| -> DrawnTree {
            if let Some((_, t)) = drawn.iter().find(|(r, _)| *r == rep) {
                return t.clone();
            }
            let t = draw(setup, &snapshot, site, rep);
            drawn.push((rep, t.clone()));
            t
        };
        for (k, (_, a, b)) in comparisons.iter().enumerate() {
            let (ta, tb) = (get(*a), get(*b));
            out[k].push((kind, compare_trees(site, &ta, &tb)));
        }
    }
    out
}

fn stats(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let v: Vec<f64> = values.collect();
    if v.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    (mean, v.iter().copied().fold(0.0, f64::max))
}

/// The table: per comparison, the mean and the worst tree of each number.
pub fn tree_report(measured: &[Vec<(TreeGround, TreeCompare)>]) -> String {
    let mut s = String::new();
    s.push_str(
        "| comparison | ground | trees | top: mean abs / worst, moved | foot: mean abs / worst | \
         outline, worst view: mean / worst (sq blocks) | silhouette (sq blocks) | seen colour, worst \
         view: mean / worst | T3 worst dir: mean / worst |\n|---|---|---|---|---|---|---|---|---|\n",
    );
    for (k, (name, _, _)) in comparisons().iter().enumerate() {
        let rows = &measured[k];
        let mut kinds: Vec<(String, Box<dyn Fn(TreeGround, &TreeCompare) -> bool>)> =
            vec![("all".into(), Box::new(|_, _| true))];
        for g in TREE_GROUNDS {
            kinds.push((format!("{g:?}"), Box::new(move |k, _| k == g)));
        }
        kinds.push((
            "all, in one chunk".into(),
            Box::new(|_, c| !crosses_a_chunk_floor(c.site)),
        ));
        kinds.push((
            "Flat, in one chunk".into(),
            Box::new(|g, c| g == TreeGround::Flat && !crosses_a_chunk_floor(c.site)),
        ));
        for (kind, keep) in kinds {
            let these: Vec<&TreeCompare> = rows
                .iter()
                .filter(|(g, c)| keep(*g, c))
                .map(|(_, c)| c)
                .collect();
            if these.is_empty() {
                continue;
            }
            let (top_mean, top_worst) = stats(these.iter().map(|c| c.top.abs() as f64));
            let moved = these.iter().filter(|c| c.top != 0.0).count();
            let (foot_mean, foot_worst) = stats(these.iter().map(|c| c.foot.abs() as f64));
            let (xor_mean, xor_worst) = stats(these.iter().map(|c| c.worst_view_xor()));
            let (area, _) = stats(these.iter().map(|c| {
                c.views.iter().map(|v| v.area).sum::<f64>() / VIEWS.len() as f64
            }));
            let (seen_mean, seen_worst) = stats(these.iter().map(|c| c.worst_view_seen()));
            let (t3_mean, t3_worst) = stats(these.iter().map(|c| c.worst_t3()));
            s.push_str(&format!(
                "| {name} | {} | {} | {top_mean:.2} / {top_worst:.0}, {moved} | {foot_mean:.2} / {foot_worst:.0} | \
                 {xor_mean:.1} / {xor_worst:.1} | {area:.1} | {seen_mean:.1} / {seen_worst:.1} | \
                 {t3_mean:.1} / {t3_worst:.1} |\n",
                kind,
                these.len(),
            ));
        }
    }
    s.push_str("\nThe worst trees (seen colour or T3 over 2/255, or the outline over 4 square blocks):\n");
    for (k, (name, _, _)) in comparisons().iter().enumerate() {
        for (g, c) in &measured[k] {
            if c.worst_view_seen() > 2.0 || c.worst_t3() > 2.0 || c.worst_view_xor() > 4.0 {
                let worst_dir = (0..6)
                    .filter_map(|d| c.t3[d].map(|v| (d, v)))
                    .fold((0, 0.0), |a, b| if b.1 > a.1 { b } else { a });
                s.push_str(&format!(
                    "- {name}, {g:?} tree{} at ({}, {}, base {}, height {}): top {:+}, foot {:+}, \
                     outline {:.1}, seen {:.1}, T3 {:.1} ({})\n",
                    if crosses_a_chunk_floor(c.site) { " across a chunk floor" } else { "" },
                    c.site.x,
                    c.site.z,
                    c.site.base,
                    c.site.height,
                    c.top,
                    c.foot,
                    c.worst_view_xor(),
                    c.worst_view_seen(),
                    worst_dir.1,
                    tree_direction_name(worst_dir.0),
                ));
            }
        }
    }
    s
}

/// The bounds a tree is held to (#433's done-when). **All chosen, not
/// derived**, each a little over what the build measures (the PR's table)
/// and well under what any of the defects it was written against shows.
pub mod bounds {
    /// A drawing's outline in any view against the other's, where both
    /// stand on the same foot on level ground, in square blocks: sixteen
    /// samples of an eighth of a block, so a sample on an edge that rounds
    /// the other way is not a failure. Measured 0.0. On a slope level 0
    /// leaves out the faces of a tree that touch the ground beside it --
    /// a trunk's foot against the uphill column, leaves inside it -- which
    /// the ground hides on screen anyway: 1 to 11 square blocks there,
    /// reported and not held (the foot and the height are, exactly).
    pub const OUTLINE: f64 = 0.25;
    /// Per face direction, and the mean colour each view sees, /255, on
    /// level ground for a tree level 0 meshes in one chunk (elsewhere the
    /// slope's own blocks and the chunk floor shade level 0's tree, and
    /// the instance cannot know). Measured 0.0.
    pub const FLAT_COLOUR: f64 = 0.5;
    /// The same, over every tree level 0 meshes in one chunk, the mean of
    /// the trees. Measured 0.4.
    pub const MEAN_COLOUR: f64 = 1.0;
    /// The worst single tree level 0 meshes in one chunk, any direction:
    /// the trunk of a tree on a slope stands beside ground that shades it
    /// at level 0. Measured 3.4 (T3) and 4.2 (seen; a tree whose foot also
    /// moves a block with its ground).
    pub const WORST_COLOUR: f64 = 5.0;
    /// At the 512 m edge the silhouette against the oak: per direction it
    /// is the oak's colour by construction (measured 0.0); what a view
    /// sees differs where the oak's darker corners are the ones hidden
    /// (measured 0.9 against the oak, 1.0 against level 0, mean and worst
    /// alike). `SILHOUETTE_SEEN` stands in for `FLAT_COLOUR` and
    /// `MEAN_COLOUR` for the seen colour of level 3.
    pub const SILHOUETTE_T3: f64 = 0.1;
    pub const SILHOUETTE_SEEN: f64 = 1.5;
    /// At the 288 and 512 m edges, where the morph has taken the near
    /// level's tree: the most a tree's foot may still move, in blocks, and
    /// how many of the sample may move at all. A column beside a cave or
    /// pit, or across a node's floor, does not morph (`cubara_world::morph`),
    /// and its tree swaps with it. Measured 1 tree of 48, by 1.
    pub const EDGE_STEP: f32 = 1.0;
    pub const EDGE_MOVED: usize = 2;
}

/// The bound on the colour a view sees of `rep`'s tree: `otherwise`, or the
/// silhouette's own at level 3.
fn seen_bound(rep: Rep, otherwise: f64) -> f64 {
    match rep {
        Rep::Instance { level: 3.., .. } => bounds::SILHOUETTE_SEEN,
        _ => otherwise,
    }
}

/// Everything [`bounds`] holds `measured` (by [`comparisons`]) to, as the
/// list of what fails: empty when the trees are the same at every distance.
/// A function, so that a test can hand it a wrong drawing and see it fail.
pub fn check(setup: &Setup, measured: &[Vec<(TreeGround, TreeCompare)>]) -> Vec<String> {
    let mut failures = Vec::new();
    let names = comparisons();
    for (k, (name, a, b)) in names.iter().enumerate() {
        let rows = &measured[k];
        for (ground, c) in rows {
            let at = format!(
                "{name}: the {ground:?} tree at ({}, {}) (seed {TREE_SEED:#x})",
                c.site.x, c.site.z
            );
            // Its own height, wherever it stands.
            if c.top != c.foot {
                failures.push(format!(
                    "{at}: its top moved {:+} and its foot {:+} -- it is not its own height",
                    c.top, c.foot
                ));
            }
            // Standing where its ground is: the foot moves by what the
            // ground under its trunk moves, and nothing else.
            if *a == Rep::Level0 {
                if let Rep::Instance { level, .. } = b {
                    let w = 1 << level;
                    let s = c.site;
                    let step = setup.gen.cell_top(s.x.div_euclid(w) * w, s.z.div_euclid(w) * w, w)
                        - setup.gen.surface_height(s.x, s.z);
                    if c.foot != step as f32 {
                        failures.push(format!(
                            "{at}: its foot moved {:+}, its ground {step:+}",
                            c.foot
                        ));
                    }
                }
            }
            // On the same foot, the same outline from every side.
            if *ground == TreeGround::Flat && c.foot == 0.0 && c.worst_view_xor() > bounds::OUTLINE
            {
                failures.push(format!(
                    "{at}: on the same foot its outline differs by {:.2} square blocks",
                    c.worst_view_xor()
                ));
            }
            let one_chunk = !crosses_a_chunk_floor(c.site);
            if *a == Rep::Level0 && one_chunk {
                let flat = *ground == TreeGround::Flat && c.foot == 0.0;
                if flat && c.worst_t3() > bounds::FLAT_COLOUR {
                    failures.push(format!(
                        "{at}: its colour per direction is {:.2}/255 off level 0's",
                        c.worst_t3()
                    ));
                }
                if flat && c.worst_view_seen() > seen_bound(*b, bounds::FLAT_COLOUR) {
                    failures.push(format!(
                        "{at}: the colour seen of it is {:.2}/255 off level 0's",
                        c.worst_view_seen()
                    ));
                }
                if c.worst_t3() > bounds::WORST_COLOUR {
                    failures.push(format!(
                        "{at}: its colour per direction is {:.2}/255 off level 0's",
                        c.worst_t3()
                    ));
                }
            }
        }
        if *a == Rep::Level0 {
            let these: Vec<&TreeCompare> = rows
                .iter()
                .map(|(_, c)| c)
                .filter(|c| !crosses_a_chunk_floor(c.site))
                .collect();
            for (what, value, bound) in [
                (
                    "T3",
                    stats(these.iter().map(|c| c.worst_t3())).0,
                    bounds::MEAN_COLOUR,
                ),
                (
                    "seen colour",
                    stats(these.iter().map(|c| c.worst_view_seen())).0,
                    seen_bound(*b, bounds::MEAN_COLOUR),
                ),
            ] {
                if value > bound {
                    failures.push(format!(
                        "{name}: the mean tree's {what} is {value:.2}/255 off level 0's"
                    ));
                }
            }
        }
        if let Rep::Instance { morphed: true, .. } = a {
            let moved: Vec<f32> = rows
                .iter()
                .map(|(_, c)| c.foot.abs())
                .filter(|&m| m > 0.0)
                .collect();
            if moved.len() > bounds::EDGE_MOVED || moved.iter().any(|&m| m > bounds::EDGE_STEP) {
                failures.push(format!(
                    "{name}: {} of {} trees still move at the swap, by {moved:?}",
                    moved.len(),
                    rows.len()
                ));
            }
            if matches!(b, Rep::Instance { level: 3, .. }) {
                for (_, c) in rows {
                    if c.worst_t3() > bounds::SILHOUETTE_T3
                        || c.worst_view_seen() > bounds::SILHOUETTE_SEEN
                    {
                        failures.push(format!(
                            "{name}: the silhouette at ({}, {}) is T3 {:.2}, seen {:.2}/255 off the oak",
                            c.site.x,
                            c.site.z,
                            c.worst_t3(),
                            c.worst_view_seen()
                        ));
                    }
                }
            }
        }
    }
    failures
}

/// The two boxes level 3 drew a tree with before #433 (trunk and canopy,
/// sized to the oak's bounds, in the log's and the leaves' mean colours,
/// unshaded): rebuilt here only to show the check would see them.
pub fn two_boxes(setup: &Setup, site: TreeSite) -> cubara_render::TreeModelMesh {
    let colour = |name| cubara_render::materials::mean_color(name).expect("a texture");
    let (top, r) = (site.height - 1, site.canopy_radius as f32);
    let mut mesh = cubara_render::TreeModelMesh::default();
    let mut push_box = |min: [f32; 3], max: [f32; 3], color: [f32; 3]| {
        for axis in 0..3 {
            for (side, plane) in [(1.0f32, max[axis]), (-1.0, min[axis])] {
                let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
                let corner = |a: f32, b: f32| {
                    let mut p = [0.0f32; 3];
                    p[axis] = plane;
                    p[u] = a;
                    p[v] = b;
                    p
                };
                let mut corners = [
                    corner(min[u], min[v]),
                    corner(max[u], min[v]),
                    corner(max[u], max[v]),
                    corner(min[u], max[v]),
                ];
                if side < 0.0 {
                    corners.reverse();
                }
                let mut normal = [0.0f32; 3];
                normal[axis] = side;
                let base = mesh.vertices.len() as u32;
                for position in corners {
                    mesh.vertices.push(cubara_render::TreeVertex {
                        position,
                        normal,
                        uv: [0.0, 0.0],
                        layer: -1.0,
                        color,
                        open: 1.0,
                    });
                }
                mesh.indices
                    .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            }
        }
    };
    let _ = setup;
    push_box([0.0, 0.0, 0.0], [1.0, site.height as f32, 1.0], colour("oak_log_side"));
    push_box(
        [-r, (top - site.canopy_radius) as f32, -r],
        [r + 1.0, (top + 2) as f32, r + 1.0],
        colour("oak_leaves"),
    );
    mesh
}
