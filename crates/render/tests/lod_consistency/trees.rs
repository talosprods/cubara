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

/// How much of its light `trees.wgsl` leaves a model vertex.
fn model_shade(_setup: &Setup, _v: &cubara_render::TreeVertex) -> f64 {
    // No AO on the instances (#332 decision 4).
    1.0
}

/// The model the game draws a tree of `site` with at `level`.
pub fn model_for(setup: &Setup, level: u32, _site: TreeSite) -> cubara_render::TreeModelMesh {
    let layer_of = |name: &str| setup.layer_of(name);
    if level <= 2 {
        cubara_render::build_oak_mesh(&setup.registry, &setup.structures, &layer_of)
            .expect("oak.ron ships")
    } else {
        let (height, r) = cubara_render::box_shape(&setup.structures);
        let colour = |name| cubara_render::materials::mean_color(name).expect("a texture");
        cubara_render::build_box_mesh(height, r, colour("oak_log_side"), colour("oak_leaves"))
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
/// band, in blocks.
pub fn placed(setup: &Setup, snapshot: &WorldSnapshot, level: u32, site: TreeSite) -> ([i32; 3], i32) {
    let node = node_under(setup, level, site);
    let found = snapshot
        .node_trees(node, setup.tree_blocks())
        .into_iter()
        .find(|t| (t.x, t.z) == (site.x, site.z))
        .unwrap_or_else(|| panic!("the node {node:?} does not place the tree at {site:?}"));
    ([found.x, found.base, found.z], 0)
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
        let mut kinds: Vec<Option<TreeGround>> = vec![None];
        kinds.extend(TREE_GROUNDS.iter().map(|&g| Some(g)));
        for kind in kinds {
            let these: Vec<&TreeCompare> = rows
                .iter()
                .filter(|(g, _)| kind.is_none_or(|k| k == *g))
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
                kind.map_or("all".to_string(), |k| format!("{k:?}")),
                these.len(),
            ));
        }
    }
    s
}
