//! Which nodes the camera could possibly see: everything else need not be
//! meshed, uploaded or drawn.
//!
//! **The rule this must never break: if any straight line from the camera
//! reaches a node through open air, that node is in the set.** It may include
//! nodes nobody can see -- that only costs time -- but leaving out one that can
//! be seen is a hole in the world.
//!
//! # How
//!
//! A search outward from the camera's node, from node to neighbouring node
//! across faces, entering a node through one face and leaving through another
//! only if air inside joins the two ([`FaceLinks`]). What makes it cull anything
//! at all is one more rule: **never step in the direction opposite one already
//! taken.** A path that has gone down may not come back up.
//!
//! Why that loses nothing: a straight line moves in one direction along each
//! axis, so it never steps against a direction it already took, and inside
//! every node it crosses, the air along it joins the face it came in by to the
//! face it leaves by. So each straight line of sight is one of the paths the
//! search follows.
//!
//! When two paths reach a node through the same face, the one that has taken
//! fewer directions forbids less; keeping the *intersection* of what each has
//! taken forbids no more than either, so merging can never block a line of
//! sight either.
//!
//! The camera's own direction of view plays no part, so the set only changes
//! when the camera moves to another node or a node's contents change -- not
//! every frame. The frustum is culled separately, per frame, as before.
//!
//! This is the approach known from other voxel engines as cave culling; the
//! proof above is why it is safe here.

use std::collections::{HashMap, HashSet, VecDeque};

use cubara_voxel::{BlockRegistry, Chunk, ChunkCoord, Face, FaceLinks, NodeChunk};

use crate::node::NodeKey;

/// Sub-blocks per node along each axis. The search runs over these rather
/// than over whole nodes: a far node is 128 blocks wide, and one cave touching
/// two of its faces used to make all of it see-through. Measured on the radius-64
/// scene with caves at every level of detail: 1.74M triangles reachable at
/// grain 1, 1.25M at grain 2, 1.12M at grain 4 -- which also searched nine
/// times longer, for a tenth of the gain.
pub const GRAIN: usize = 2;

/// What joins what inside each of a node's `GRAIN`³ sub-blocks, indexed
/// `(z * GRAIN + y) * GRAIN + x`.
pub type NodeLinks = [FaceLinks; GRAIN * GRAIN * GRAIN];

/// Every sub-block of an empty node is open.
pub const OPEN: NodeLinks = [FaceLinks::ALL; GRAIN * GRAIN * GRAIN];

/// The sub-block links of a generated node; `None` (nothing generated, it is
/// empty) is [`OPEN`].
///
/// Takes [`NodeChunk`] (R1b, issue #327 decision 6), not a plain [`Chunk`]:
/// a coarse node's lattice has its own vertical cell count now, different
/// from its horizontal one, so each of the `GRAIN`³ sub-blocks' own region
/// size is per-axis -- `chunk.mesh_dims()[axis] / GRAIN` -- rather than one
/// shared `Chunk::SIZE / GRAIN`. `GRAIN` divides every axis's cell count
/// evenly for every level/`v` combination R1b produces (checked by
/// `node_links_region_size_divides_every_level_evenly` below), so this never
/// rounds.
///
/// **The links are those of the lattice the node is drawn on**
/// ([`NodeChunk::mesh_dims`], [`NodeChunk::drawn`]), which for a level-3
/// node is not the one it stores: its rows of four are cut off at each
/// column's top (issue #423), and the blocks above a top are air. The rule
/// in this module's header rests on the links describing that air; worked
/// out on the stored rows they called it rock, and the search left out
/// nodes that the drawn lattice keeps (14 distinct level-3 nodes over 48
/// eye positions, review of #432). `the_links_of_a_node_are_those_of_what_it_draws`
/// holds the two together.
pub fn node_links(chunk: Option<&NodeChunk>, registry: &BlockRegistry) -> NodeLinks {
    let Some(chunk) = chunk else {
        return OPEN;
    };
    let dims = chunk.mesh_dims();
    let size = [dims[0] / GRAIN, dims[1] / GRAIN, dims[2] / GRAIN];
    std::array::from_fn(|i| {
        let (x, y, z) = (i % GRAIN, i / GRAIN % GRAIN, i / GRAIN / GRAIN);
        let min = [x * size[0], y * size[1], z * size[2]];
        FaceLinks::of_region_node(chunk, registry, min, size)
    })
}

const FACES: [Face; 6] = [
    Face::PosX,
    Face::NegX,
    Face::PosY,
    Face::NegY,
    Face::PosZ,
    Face::NegZ,
];

fn opposite(face: Face) -> Face {
    match face {
        Face::PosX => Face::NegX,
        Face::NegX => Face::PosX,
        Face::PosY => Face::NegY,
        Face::NegY => Face::PosY,
        Face::PosZ => Face::NegZ,
        Face::NegZ => Face::PosZ,
    }
}

fn step_of(face: Face) -> [i32; 3] {
    let n = face.normal();
    [n[0] as i32, n[1] as i32, n[2] as i32]
}

/// The coarsest level a region's nodes are looked for at.
const MAX_LEVEL: u32 = 8;

/// The nodes being searched, numbered, so per-sub-block state is a flat array.
struct Index<'a> {
    nodes: Vec<NodeKey>,
    of: HashMap<NodeKey, u32>,
    set: &'a HashSet<NodeKey>,
    /// Whether a step may go into finer nodes. From a camera inside the
    /// region it never needs to: distance from the camera only grows along a
    /// line from it. From outside, a line enters through the coarse outer
    /// nodes and travels towards finer ones, so it must.
    finer: bool,
}

impl<'a> Index<'a> {
    fn new(set: &'a HashSet<NodeKey>, finer: bool) -> Self {
        let mut nodes: Vec<NodeKey> = set.iter().copied().collect();
        nodes.sort();
        let of = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (*n, i as u32))
            .collect();
        Self {
            nodes,
            of,
            set,
            finer,
        }
    }

    /// The sub-block containing world block `p`, looked for in a node of
    /// `level` or the one coarser -- the only neighbours a line of sight can
    /// step into.
    fn locate(&self, p: [i32; 3], levels: &[u32]) -> Option<(u32, usize)> {
        let chunk = ChunkCoord::from_block(p[0], p[1], p[2]);
        let node = levels
            .iter()
            .map(|&l| NodeKey::containing(chunk, l))
            .find(|n| self.set.contains(n))?;
        let origin = node.world_origin();
        let size = (Chunk::SIZE / GRAIN) as i32 * node.extent_chunks();
        let i = |k: usize| ((p[k] - origin[k]) / size) as usize;
        Some((self.of[&node], (i(2) * GRAIN + i(1)) * GRAIN + i(0)))
    }

    /// A block just outside the face `d` of sub-block `c` of `node`, `along`
    /// quarters of the way across it.
    fn beyond(&self, node: NodeKey, c: [usize; 3], d: [i32; 3], along: [i32; 2]) -> [i32; 3] {
        let origin = node.world_origin();
        let size = (Chunk::SIZE / GRAIN) as i32 * node.extent_chunks();
        let mut t = 0;
        [0, 1, 2].map(|k| {
            let min = origin[k] + c[k] as i32 * size;
            match d[k] {
                1 => min + size,
                -1 => min - 1,
                _ => {
                    let at = min + size * along[t] / 4;
                    t += 1;
                    at
                }
            }
        })
    }

    /// Whether `face` of sub-block `sub` of node `n` is the outside of the
    /// whole region: no node of any size holds the block beyond it. (That
    /// [`neighbours`](Self::neighbours) finds nothing there does not say so --
    /// from a camera inside the region it never looks into finer nodes.)
    fn leaves_region(&self, n: u32, sub: usize, face: Face) -> bool {
        let node = self.nodes[n as usize];
        let c = [sub % GRAIN, sub / GRAIN % GRAIN, sub / GRAIN / GRAIN];
        let p = self.beyond(node, c, step_of(face), [2, 2]);
        let chunk = ChunkCoord::from_block(p[0], p[1], p[2]);
        !(0..=MAX_LEVEL).any(|l| self.set.contains(&NodeKey::containing(chunk, l)))
    }

    /// The sub-blocks across `face` of sub-block `sub` of node `n`: its
    /// sibling, or the same-level or one-coarser sub-block beyond the face --
    /// or, when [`finer`](Self::finer) steps are allowed and neither exists,
    /// the finer sub-blocks touching it.
    fn neighbours(&self, n: u32, sub: usize, face: Face) -> Vec<(u32, usize)> {
        let node = self.nodes[n as usize];
        let c = [sub % GRAIN, sub / GRAIN % GRAIN, sub / GRAIN / GRAIN];
        let d = step_of(face);
        let inside = (0..3).all(|k| (0..GRAIN as i32).contains(&(c[k] as i32 + d[k])));
        if inside {
            let m = [0, 1, 2].map(|k| (c[k] as i32 + d[k]) as usize);
            return vec![(n, (m[2] * GRAIN + m[1]) * GRAIN + m[0])];
        }
        let beyond = |along: [i32; 2]| self.beyond(node, c, d, along);
        if let Some(found) = self.locate(beyond([2, 2]), &[node.level, node.level + 1]) {
            return vec![found];
        }
        if !self.finer || node.level == 0 {
            return Vec::new();
        }
        let mut found = Vec::with_capacity(4);
        for along in [[1, 1], [3, 1], [1, 3], [3, 3]] {
            if let Some(v) = self.locate(beyond(along), &[node.level - 1]) {
                if !found.contains(&v) {
                    found.push(v);
                }
            }
        }
        found
    }
}

/// What joins what in a whole node, from its sub-blocks: face `a` joins face
/// `b` when air joins a sub-block on side `a` to one on side `b`, through the
/// sub-blocks between.
fn whole(subs: &NodeLinks) -> FaceLinks {
    let at = |i: usize| [i % GRAIN, i / GRAIN % GRAIN, i / GRAIN / GRAIN];
    let axis = |f: Face| (0..3).find(|&k| step_of(f)[k] != 0).expect("an axis");
    let on_side = |i: usize, f: Face| {
        let k = axis(f);
        if step_of(f)[k] > 0 {
            at(i)[k] == GRAIN - 1
        } else {
            at(i)[k] == 0
        }
    };
    FaceLinks::from_fn(|a, b| {
        let mut seen = [[false; 6]; GRAIN * GRAIN * GRAIN];
        let mut stack: Vec<(usize, Face)> = (0..subs.len())
            .filter(|&i| on_side(i, a))
            .map(|i| (i, a))
            .collect();
        while let Some((i, entered)) = stack.pop() {
            if std::mem::replace(&mut seen[i][entered as usize], true) {
                continue;
            }
            for out in FACES {
                if !subs[i].joins(entered, out) {
                    continue;
                }
                if on_side(i, out) {
                    if out == b {
                        return true;
                    }
                    continue;
                }
                let mut c = at(i);
                let k = axis(out);
                c[k] = (c[k] as i32 + step_of(out)[k]) as usize;
                stack.push(((c[2] * GRAIN + c[1]) * GRAIN + c[0], opposite(out)));
            }
        }
        false
    })
}

/// What to search a node through while its own links are not known yet: what
/// the nodes that drew the same space before it said.
///
/// **Why this exists: the holes of 2026-09-27.** An unknown node used to be a
/// wall. Flying forward, every node crossing the edge of a ring splits into
/// eight new ones, so a whole curtain of unknown nodes stood across the view --
/// and everything behind it left the visible set, and was unloaded, until the
/// curtain had been meshed. Hundreds of blocks of ground vanished and came
/// back every few seconds (`crate`'s caller measures it: `--bench flight`).
///
/// - **It split** (its parent, or a coarser ancestor, is known): the part of
///   the ancestor it is, as the ancestor drew it -- one of the ancestor's
///   sub-blocks, for every sub-block of this node.
/// - **It merged** (some of its children are known): each sub-block is one
///   child, as that child drew it whole; a child not known yet is taken as
///   open.
/// - **Neither**: new ground, at the edge of what is streamed or on the first
///   frame. It stays a wall, so what is generated grows outward along what can
///   be seen.
fn stand_in(node: NodeKey, links: &impl Fn(NodeKey) -> Option<NodeLinks>) -> Option<NodeLinks> {
    // A child is exactly one of its parent's sub-blocks only at grain 2.
    const _: () = assert!(GRAIN == 2, "stand_in takes a sub-block to be a child");
    let mut ancestor = node;
    for up in 1..=3 {
        ancestor = ancestor.parent();
        if let Some(known) = links(ancestor) {
            // The ancestor's sub-block holding this node: the bit `up - 1` of
            // its position along each axis.
            let bit = |p: i32| ((p >> (up - 1)) & 1) as usize;
            let [x, y, z] = node.pos.map(bit);
            return Some([known[(z * GRAIN + y) * GRAIN + x]; GRAIN * GRAIN * GRAIN]);
        }
    }
    let children: Vec<Option<NodeLinks>> = node.children().into_iter().map(links).collect();
    if children.iter().any(Option::is_some) {
        let mut merged = OPEN;
        for (i, child) in children.iter().enumerate() {
            if let Some(child) = child {
                merged[i] = whole(child);
            }
        }
        return Some(merged);
    }
    None
}

/// What a search from the camera found: [`search`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sight {
    /// Every node a line of sight from anywhere in the camera's node could
    /// reach.
    pub nodes: HashSet<NodeKey>,
    /// Whether a line of sight could leave the region for the sky: out of
    /// its top or one of its sides, where the far terrain is drawn
    /// (`docs/PROPOSAL_FAR_VIEW.md` §3.4).
    ///
    /// `false` only when the search is sure: every path it followed ended at
    /// rock, inside the region. An eye shut in underground -- the owner's
    /// cave of 2026-11-14 (issue #424). Anything the search cannot tell is
    /// `true`: a path stopping at a node nothing is known about (it may be
    /// air), a camera outside the region, an empty region.
    ///
    /// The floor does not count. Nothing is drawn below it, so a cave that
    /// runs out of the bottom of the region shows no far terrain.
    ///
    /// This is exact in the way the search is: the same paths, so by the
    /// proof in this module's header every straight line to the sky is one of
    /// them. A fixed depth below the surface would be wrong at a cave mouth
    /// and under an overhang.
    pub open_to_sky: bool,
}

/// Every node in `nodes` a line of sight from anywhere in `camera`'s node could
/// reach: [`search`]'s [`Sight::nodes`].
pub fn visible_nodes(
    camera: ChunkCoord,
    nodes: &HashSet<NodeKey>,
    links: impl Fn(NodeKey) -> Option<NodeLinks>,
) -> HashSet<NodeKey> {
    search(camera, nodes, links).nodes
}

/// Every node in `nodes` a line of sight from anywhere in `camera`'s node could
/// reach, and whether one could reach the sky ([`Sight::open_to_sky`]).
///
/// `links(node)` is what joins what inside each of a node's sub-blocks; `None`
/// means not known yet (not generated). An unknown node is included -- it may
/// well be seen -- and searched through as what drew its space before it did
/// ([`stand_in`]); only a node nothing is known about is not searched through,
/// which is what lets a caller generate outward along what can be seen instead
/// of generating everything.
///
/// The search starts from **every** sub-block of the camera's node, not only
/// the one the camera is in, so the result holds for any position in that
/// node and only needs working out again when the camera changes node.
pub fn search(
    camera: ChunkCoord,
    nodes: &HashSet<NodeKey>,
    links: impl Fn(NodeKey) -> Option<NodeLinks>,
) -> Sight {
    let mut visible = HashSet::new();
    let start = (0..=MAX_LEVEL)
        .map(|level| NodeKey::containing(camera, level))
        .find(|n| nodes.contains(n));
    if nodes.is_empty() {
        return Sight {
            nodes: visible,
            open_to_sky: true,
        };
    }
    // From outside the region the camera is already under the sky.
    let mut open_to_sky = start.is_none();
    let index = Index::new(nodes, start.is_none());
    let subs = GRAIN * GRAIN * GRAIN;
    let known: Vec<Option<NodeLinks>> = index
        .nodes
        .iter()
        .map(|&n| links(n).or_else(|| stand_in(n, &links)))
        .collect();
    let mut seen = vec![false; index.nodes.len()];
    // For each (node, sub-block, face entered by): the directions taken to get
    // there, kept as the intersection over every path that has arrived.
    // `NONE_YET` is a value no set of six directions can be.
    const NONE_YET: u8 = 0xFF;
    let mut arrived = vec![NONE_YET; index.nodes.len() * subs * 6];
    let mut queue: VecDeque<(u32, usize, Face, u8)> = VecDeque::new();
    let bit = |f: Face| 1u8 << f as u8;

    match start {
        Some(start) => {
            let start_n = index.of[&start];
            seen[start_n as usize] = true;
            for sub in 0..subs {
                for face in FACES {
                    for (n, s) in index.neighbours(start_n, sub, face) {
                        queue.push_back((n, s, opposite(face), bit(face)));
                    }
                }
            }
        }
        // **A camera outside the region altogether.** A line of sight from
        // there enters through an outer face on the camera's side of it, moving
        // inward; which other ways it is moving is unknown, so only inward is
        // taken. Every such face is a place to start.
        None => {
            for (n, node) in index.nodes.iter().enumerate() {
                let origin = node.chunk_origin();
                let extent = node.extent_chunks();
                let o = [origin.x, origin.y, origin.z];
                let c = [camera.x, camera.y, camera.z];
                for sub in 0..subs {
                    for face in FACES {
                        if !index.neighbours(n as u32, sub, face).is_empty() {
                            continue;
                        }
                        let d = step_of(face);
                        let k = (0..3).find(|&k| d[k] != 0).expect("an axis");
                        let facing_camera = if d[k] > 0 {
                            c[k] >= o[k] + extent
                        } else {
                            c[k] < o[k]
                        };
                        if facing_camera {
                            queue.push_back((n as u32, sub, face, bit(opposite(face))));
                        }
                    }
                }
            }
        }
    }

    while let Some((n, sub, entered_by, taken)) = queue.pop_front() {
        seen[n as usize] = true;
        let slot = &mut arrived[(n as usize * subs + sub) * 6 + entered_by as usize];
        let taken = match *slot {
            // Nothing new: an earlier path already forbids no more than this.
            before if before != NONE_YET && before & !taken == 0 => continue,
            before if before != NONE_YET => before & taken,
            _ => taken,
        };
        *slot = taken;
        let Some(inside) = known[n as usize] else {
            // Nothing is known of it: it may be open air.
            open_to_sky = true;
            continue;
        };
        for out in FACES {
            if taken & bit(opposite(out)) != 0 || !inside[sub].joins(entered_by, out) {
                continue;
            }
            let next = index.neighbours(n, sub, out);
            if next.is_empty() && out != Face::NegY && index.leaves_region(n, sub, out) {
                open_to_sky = true;
            }
            for (m, s) in next {
                queue.push_back((m, s, opposite(out), taken | bit(out)));
            }
        }
    }
    for (i, was) in seen.into_iter().enumerate() {
        if was {
            visible.insert(index.nodes[i]);
        }
    }
    Sight {
        nodes: visible,
        open_to_sky,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::desired_nodes_3d;
    use crate::{TerrainBlocks, World};
    use cubara_voxel::{BlockId, BlockRegistry, Chunk, DropRule, Faces, Interact, Material, Shape};

    /// `node_links` divides each axis's own cell count by `GRAIN` (2); this
    /// must never round, at any level/`v` combination R1b produces, or
    /// `node_links` silently drops cells off the edge of its sub-block
    /// regions. Level 0's `[16, 16, 16]` divides trivially; levels 1-3 are
    /// R1b's real anisotropic shapes.
    #[test]
    fn node_links_region_size_divides_every_level_evenly() {
        for level in 0u32..=3 {
            let dims = NodeKey::new(level, [0, 0, 0]).lattice_dims();
            for (axis, &n) in dims.iter().enumerate() {
                assert_eq!(
                    n % GRAIN as i32,
                    0,
                    "level {level} axis {axis}: {n} cells does not divide evenly by GRAIN={GRAIN}"
                );
            }
        }
    }

    /// **The links of a node are those of what it draws** (review of #432,
    /// B2). A level-3 node stores rows of four blocks and is drawn cut off
    /// at each column's top, so the lattice it stores is not the one the
    /// camera sees. Every generated level-3 node around the default seed's
    /// mountains, three ways:
    ///
    /// - as it is: `node_links` of the node the game makes;
    /// - **as drawn**: a lattice of block rows built from `NodeChunk::drawn`,
    ///   with nothing left to cut;
    /// - as stored: its rows of four, drawn whole.
    ///
    /// The first must be the second, node for node. And the third must
    /// differ from them somewhere, and only ever by calling air rock -- or
    /// this could not tell which lattice `node_links` reads. (It read the
    /// stored one until the review: the search then left out nodes the
    /// drawn lattice keeps.)
    #[test]
    fn the_links_of_a_node_are_those_of_what_it_draws() {
        use cubara_voxel::AnisoChunk;
        let registry = registry();
        let blocks = TerrainBlocks::from_registry(&registry);
        let world = World::new();
        let (mut nodes, mut differ) = (0, 0);
        // The mountain the golden images frame: chunk column (-56, 304).
        for nx in -11..-3 {
            for nz in 34..42 {
                for ny in 0..3 {
                    let node = NodeKey::new(3, [nx, ny, nz]);
                    let Some(chunk) = world.node_at(node, blocks) else {
                        continue;
                    };
                    let NodeChunk::Aniso(lattice) = &chunk else {
                        panic!("a level-3 node is an anisotropic lattice");
                    };
                    assert_eq!(lattice.scale()[1], 4, "level 3 stores rows of four");
                    let as_drawn = NodeChunk::Aniso(AnisoChunk::from_fn(
                        chunk.mesh_dims(),
                        chunk.mesh_scale(),
                        |x, y, z| chunk.drawn(x, y, z),
                    ));
                    let as_stored = NodeChunk::Aniso(AnisoChunk::from_fn(
                        lattice.dims(),
                        lattice.scale(),
                        |x, y, z| lattice.get(x, y, z),
                    ));
                    let links = node_links(Some(&chunk), &registry);
                    let drawn = node_links(Some(&as_drawn), &registry);
                    let stored = node_links(Some(&as_stored), &registry);
                    assert_eq!(links, drawn, "{node:?}: the links are not what it draws");
                    nodes += 1;
                    if stored != drawn {
                        differ += 1;
                        // A stored row is solid wherever a drawn one is: it
                        // can only join less.
                        for (s, d) in stored.iter().zip(&drawn) {
                            for a in FACES {
                                for b in FACES {
                                    assert!(
                                        !s.joins(a, b) || d.joins(a, b),
                                        "{node:?}: the stored rows join {a:?} to {b:?} and the drawn ones do not"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        println!("{nodes} level-3 nodes, {differ} whose stored rows join less than what they draw");
        assert!(
            nodes >= 60 && differ >= 3,
            "{nodes} nodes, {differ} of them told apart: not enough to say which lattice the links read"
        );
    }

    fn registry() -> BlockRegistry {
        let material = |name: &str| {
            (
                std::path::PathBuf::from("test-fixture.ron"),
                Material {
                    name: name.to_string(),
                    solid: true,
                    faces: Faces::All(name.to_string()),
                    shapes: vec![Shape::Full],
                    drops: DropRule::SameName,
                    requires_tier: 0,
                    hardness: Some(1),
                    interact: Interact::None,
                },
            )
        };
        BlockRegistry::from_materials(vec![
            material("cubara:grass"),
            material("cubara:soil"),
            material("cubara:stone"),
        ])
        .expect("fixture registry")
    }

    /// A small world region and what every node in it draws.
    struct Scene {
        nodes: HashSet<NodeKey>,
        chunks: HashMap<NodeKey, Option<NodeChunk>>,
        links: HashMap<NodeKey, NodeLinks>,
    }

    impl Scene {
        fn around(world: &World, camera: ChunkCoord, registry: &BlockRegistry) -> Self {
            Self::of(world, camera, registry, 2, &[(0u32, 3), (1, 6), (2, 12)])
        }

        fn of(
            world: &World,
            camera: ChunkCoord,
            registry: &BlockRegistry,
            squash: i32,
            schedule: &[(u32, i32)],
        ) -> Self {
            let blocks = TerrainBlocks::from_registry(registry);
            let nodes: HashSet<NodeKey> = desired_nodes_3d(camera, squash, schedule)
                .into_iter()
                .collect();
            let mut chunks = HashMap::new();
            let mut links = HashMap::new();
            for &n in &nodes {
                let chunk = world.node_at(n, blocks);
                links.insert(n, node_links(chunk.as_ref(), registry));
                chunks.insert(n, chunk);
            }
            Self {
                nodes,
                chunks,
                links,
            }
        }

        /// The node drawing block `b`, and whether it draws it solid.
        fn at(&self, b: [i32; 3]) -> Option<(NodeKey, bool)> {
            let chunk = ChunkCoord::from_block(b[0], b[1], b[2]);
            let node = (0..=2)
                .map(|l| NodeKey::containing(chunk, l))
                .find(|n| self.nodes.contains(n))?;
            let o = node.world_origin();
            let (sy, sh) = (node.cell_height_blocks(), node.extent_chunks());
            let i = |k: usize, step: i32| ((b[k] - o[k]) / step) as usize;
            let solid = self.chunks[&node]
                .as_ref()
                .is_some_and(|c| c.get(i(0, sh), i(1, sy), i(2, sh)) != BlockId::AIR);
            Some((node, solid))
        }
    }

    /// Walk the blocks a ray passes through (Amanatides & Woo), returning the
    /// first node that draws one of them solid, if the ray hits anything
    /// before leaving the scene.
    fn first_hit(scene: &Scene, eye: [f32; 3], dir: [f32; 3]) -> Option<NodeKey> {
        let mut cell = [
            eye[0].floor() as i32,
            eye[1].floor() as i32,
            eye[2].floor() as i32,
        ];
        let mut step = [0i32; 3];
        let mut t_max = [f32::INFINITY; 3];
        let mut t_delta = [f32::INFINITY; 3];
        for k in 0..3 {
            if dir[k] > 0.0 {
                step[k] = 1;
                t_max[k] = (cell[k] as f32 + 1.0 - eye[k]) / dir[k];
                t_delta[k] = 1.0 / dir[k];
            } else if dir[k] < 0.0 {
                step[k] = -1;
                t_max[k] = (eye[k] - cell[k] as f32) / -dir[k];
                t_delta[k] = 1.0 / -dir[k];
            }
        }
        let mut entered = false;
        for _ in 0..4000 {
            let Some((node, solid)) = scene.at(cell) else {
                // Outside the scene: before entering, keep going; after
                // leaving, the ray is done.
                if entered {
                    return None;
                }
                let k = if t_max[0] < t_max[1] && t_max[0] < t_max[2] {
                    0
                } else if t_max[1] < t_max[2] {
                    1
                } else {
                    2
                };
                cell[k] += step[k];
                t_max[k] += t_delta[k];
                continue;
            };
            entered = true;
            if solid {
                return Some(node);
            }
            let k = if t_max[0] < t_max[1] && t_max[0] < t_max[2] {
                0
            } else if t_max[1] < t_max[2] {
                1
            } else {
                2
            };
            cell[k] += step[k];
            t_max[k] += t_delta[k];
        }
        None
    }

    /// **Nothing a straight line of sight reaches is left out.** About 1,200
    /// rays each from a camera above the ground, one inside a cave and one high in
    /// the air, through a real world with caves and levels of detail.
    /// Links joining exactly `a` and `b`.
    fn pair(a: Face, b: Face) -> FaceLinks {
        FaceLinks::from_fn(|x, y| (x, y) == (a, b) || (x, y) == (b, a))
    }

    #[test]
    fn a_whole_node_joins_what_its_sub_blocks_join_between_them() {
        assert_eq!(whole(&OPEN), FaceLinks::ALL);
        assert_eq!(whole(&[FaceLinks::NONE; 8]), FaceLinks::NONE);
        // In at -x through sub-block 0, on into sub-block 1, up into 3, and
        // out at +y: -x joins +y, and nothing else joins anything.
        let mut subs = [FaceLinks::NONE; 8];
        subs[0] = pair(Face::NegX, Face::PosX);
        subs[1] = pair(Face::NegX, Face::PosY);
        subs[3] = pair(Face::NegY, Face::PosY);
        let w = whole(&subs);
        for a in FACES {
            for b in FACES {
                let want = matches!((a, b), (Face::NegX, Face::PosY) | (Face::PosY, Face::NegX));
                assert_eq!(w.joins(a, b), want, "{a:?} {b:?}");
            }
        }
        // A sub-block that joins nothing breaks the path.
        subs[1] = FaceLinks::NONE;
        assert_eq!(whole(&subs), FaceLinks::NONE);
    }

    #[test]
    fn a_split_node_is_searched_as_its_ancestor_drew_it() {
        let parent = NodeKey::new(2, [3, -1, 0]);
        // Only the sub-block at x 1, y 0, z 1 (index 5) is open.
        let mut drawn = [FaceLinks::NONE; 8];
        drawn[5] = FaceLinks::ALL;
        let links = |n: NodeKey| (n == parent).then_some(drawn);
        let children = parent.children();
        assert_eq!(stand_in(children[5], &links), Some(OPEN));
        assert_eq!(stand_in(children[0], &links), Some([FaceLinks::NONE; 8]));
        // Two levels down, inside the open half and outside it.
        let inside = children[5].children()[2];
        let outside = children[4].children()[1];
        assert_eq!(stand_in(inside, &links), Some(OPEN));
        assert_eq!(stand_in(outside, &links), Some([FaceLinks::NONE; 8]));
    }

    #[test]
    fn a_merged_node_is_searched_as_its_children_drew_it() {
        let node = NodeKey::new(1, [0, 2, -3]);
        let children = node.children();
        let solid = children[3];
        let unknown = children[6];
        let links = |n: NodeKey| {
            if n == solid {
                Some([FaceLinks::NONE; 8])
            } else if n == unknown || !children.contains(&n) {
                None
            } else {
                Some(OPEN)
            }
        };
        let merged = stand_in(node, &links).expect("its children are known");
        assert_eq!(merged[3], FaceLinks::NONE);
        assert_eq!(merged[6], FaceLinks::ALL, "an unknown child is open");
        assert_eq!(merged[0], FaceLinks::ALL);
        assert_eq!(
            stand_in(NodeKey::new(1, [50, 50, 50]), &links),
            None,
            "new ground stands in for nothing"
        );
    }

    /// Whether all of `node` lies below the ground's surface.
    fn underground(world: &World, node: NodeKey) -> bool {
        let [x, y, z] = node.world_origin();
        let size = node.extent_chunks() * Chunk::SIZE as i32;
        (x..x + size)
            .flat_map(|x| (z..z + size).map(move |z| (x, z)))
            .all(|(x, z)| world.surface_height(x, z) >= y + size)
    }

    /// **The holes of 2026-09-27.** The moment the camera crosses into another
    /// chunk, some nodes around it split and some merge, and none of those is
    /// meshed yet. Everything the camera could see before and still can must
    /// stay in the set meanwhile -- dropping it unloads it, and that was
    /// hundreds of blocks of ground vanishing while flying.
    #[test]
    fn a_node_split_or_merged_since_the_last_search_hides_nothing_behind_it() {
        let registry = registry();
        let world = World::new();
        let surface = world.surface_height(8, 8);
        let before = ChunkCoord::from_world_pos([8.3, surface as f32 + 20.7, 8.1]);
        let known = Scene::around(&world, before, &registry);
        for step in [[1, 0, 0], [0, -2, 0], [0, 2, 0], [-2, 0, 1]] {
            let after = ChunkCoord::new(before.x + step[0], before.y + step[1], before.z + step[2]);
            let now = Scene::around(&world, after, &registry);
            assert!(
                now.nodes.iter().any(|n| !known.links.contains_key(n)),
                "{step:?} split or merged nothing, so this proves nothing"
            );
            let full = visible_nodes(after, &now.nodes, |n| now.links.get(&n).copied());
            // What the streamer knows the moment the camera has moved: the
            // links of everything from before, and nothing of what is new.
            let partial = visible_nodes(after, &now.nodes, |n| known.links.get(&n).copied());
            // Lost is allowed in two places only. New ground at the edge of
            // the region, which nothing drew before -- it is generated as the
            // search reaches it. And inside a cave, where a node's parent drew
            // a passage too narrow for its coarser blocks: a glimpse into the
            // dark for the moment the node takes to mesh, never open ground.
            let stand_in_of = |n: NodeKey| stand_in(n, &|k| known.links.get(&k).copied());
            let lost: Vec<NodeKey> = full
                .iter()
                .filter(|n| !partial.contains(n))
                .filter(|&&n| stand_in_of(n).is_some() && !underground(&world, n))
                .copied()
                .collect();
            assert!(
                lost.is_empty(),
                "{step:?}: {} of {} visible nodes dropped until the new ones are meshed: {lost:?}",
                lost.len(),
                full.len()
            );
        }
    }

    #[test]
    fn every_node_a_line_of_sight_hits_is_visible() {
        let registry = registry();
        let blocks = TerrainBlocks::from_registry(&registry);
        let world = World::new();
        let surface = world.surface_height(8, 8);
        // A cave: the first air below the surface near the origin.
        let cave = (0..400)
            .flat_map(|i| {
                let (x, z) = (i % 20, i / 20);
                (surface - 60..surface - 4).map(move |y| [x, y, z])
            })
            .find(|p| !world.is_solid_at(p[0], p[1], p[2], blocks))
            .expect("a cave below the origin");
        // Each camera with the centre of the region it looks at. The last is
        // far outside its region, the way the bench's orbit is.
        let cases = [
            ([8.3, surface as f32 + 6.7, 8.1], None),
            (
                [
                    cave[0] as f32 + 0.37,
                    cave[1] as f32 + 0.41,
                    cave[2] as f32 + 0.53,
                ],
                None,
            ),
            ([5.2, 190.6, -3.3], None),
            ([350.4, 420.3, -280.7], Some(ChunkCoord::new(0, 1, 0))),
        ];

        let mut culled_something = false;
        for (eye, around) in cases {
            let camera = ChunkCoord::from_world_pos(eye);
            let scene = Scene::around(&world, around.unwrap_or(camera), &registry);
            let visible = visible_nodes(camera, &scene.nodes, |n| scene.links.get(&n).copied());
            culled_something |= visible.len() < scene.nodes.len();

            // Every direction on a lattice, nudged off it so no ray runs exactly
            // along an axis or through an edge. The walk does not need them
            // normalised, and this crate keeps floating-point trigonometry out
            // of simulation code (`scripts/check-architecture.sh`, Rule 1).
            let mut dirs = Vec::new();
            for x in -7i32..=7 {
                for y in -7i32..=7 {
                    for z in -7i32..=7 {
                        if x.abs().max(y.abs()).max(z.abs()) == 7 {
                            dirs.push([x as f32 + 0.137, y as f32 + 0.071, z as f32 + 0.229]);
                        }
                    }
                }
            }
            for dir in dirs {
                if let Some(hit) = first_hit(&scene, eye, dir) {
                    assert!(
                        visible.contains(&hit),
                        "eye {eye:?}: a ray along {dir:?} sees {hit:?}, which was culled"
                    );
                }
            }
        }
        assert!(
            culled_something,
            "nothing was culled anywhere, so this proves nothing"
        );
    }

    /// A cube of level-0 nodes around `camera`, each solid rock unless
    /// `links` says otherwise -- and whether a search from `camera` finds a
    /// way out to the sky.
    fn sky_from(camera: ChunkCoord, links: impl Fn(NodeKey) -> Option<NodeLinks>) -> bool {
        let nodes: HashSet<NodeKey> = desired_nodes_3d(camera, 1, &[(0u32, 4)])
            .into_iter()
            .collect();
        assert!(nodes.contains(&NodeKey::containing(camera, 0)));
        search(camera, &nodes, links).open_to_sky
    }

    const ROCK: NodeLinks = [FaceLinks::NONE; GRAIN * GRAIN * GRAIN];

    /// **Issue #424's rule, on worlds small enough to read.** The far terrain
    /// is drawn for an eye that a line of sight could leave the region from,
    /// by its top or a side; it is not for one shut in.
    #[test]
    fn only_a_way_out_of_the_region_is_open_to_the_sky() {
        let camera = ChunkCoord::new(3, -5, 7);
        let column = |n: NodeKey| n.pos[0] == camera.x && n.pos[2] == camera.z;
        let row = |n: NodeKey| n.pos[1] == camera.y && n.pos[2] == camera.z;

        // Rock on every side.
        assert!(!sky_from(camera, |_| Some(ROCK)), "shut in");
        // A shaft straight up to the top of the region.
        let shaft = |n: NodeKey| {
            Some(if column(n) && n.pos[1] > camera.y {
                OPEN
            } else {
                ROCK
            })
        };
        assert!(sky_from(camera, shaft), "a shaft up");
        // The same shaft, blocked one node short of the top.
        let top = (camera.y + 1..)
            .take_while(|&y| shaft_node_exists(camera, y))
            .last()
            .expect("nodes above the camera");
        let blocked = |n: NodeKey| {
            if n.pos[1] == top {
                Some(ROCK)
            } else {
                shaft(n)
            }
        };
        assert!(!sky_from(camera, blocked), "a shaft that stops short");
        // A tunnel out of a side.
        let tunnel = |n: NodeKey| {
            Some(if row(n) && n.pos[0] < camera.x {
                OPEN
            } else {
                ROCK
            })
        };
        assert!(sky_from(camera, tunnel), "a tunnel out of a side");
        // A shaft down and out of the floor: nothing is drawn below it.
        let pit = |n: NodeKey| {
            Some(if column(n) && n.pos[1] < camera.y {
                OPEN
            } else {
                ROCK
            })
        };
        assert!(!sky_from(camera, pit), "a pit out of the floor");
        // A node nothing is known about may be air.
        let unknown = NodeKey::new(0, [camera.x, camera.y + 1, camera.z]);
        assert!(
            sky_from(camera, |n| (n.level == 0 && n != unknown).then_some(ROCK)),
            "an unknown node next to the camera"
        );
        // ...but not one the search never reaches.
        let unreached = NodeKey::new(0, [camera.x, camera.y + 3, camera.z]);
        assert!(
            !sky_from(camera, |n| (n.level == 0 && n != unreached).then_some(ROCK)),
            "an unknown node behind rock"
        );
        // From outside the region, or with no region, nothing is sure.
        let nodes: HashSet<NodeKey> = desired_nodes_3d(camera, 1, &[(0u32, 4)])
            .into_iter()
            .collect();
        let outside = ChunkCoord::new(camera.x + 40, camera.y, camera.z);
        assert!(search(outside, &nodes, |_| Some(ROCK)).open_to_sky);
        assert!(search(camera, &HashSet::new(), |_| Some(ROCK)).open_to_sky);
    }

    /// A cavern spanning two levels of detail, with rock all round it, is
    /// still shut in: the search carries the answer across a change of level.
    ///
    /// (This does not tell `leaves_region` from "no neighbour found". The two
    /// differ only where a path steps from a coarse node towards finer ones,
    /// which a path that never reverses does not do while detail only
    /// coarsens away from the camera. The check stays because paths are
    /// merged by intersecting what they have taken, and nothing proves a
    /// merged path keeps that property.)
    #[test]
    fn a_coarse_node_beside_finer_ones_is_not_the_edge_of_the_region() {
        let camera = ChunkCoord::new(3, -5, 7);
        let nodes: HashSet<NodeKey> = desired_nodes_3d(camera, 1, &[(0u32, 3), (1, 10)])
            .into_iter()
            .collect();
        // Open: every node wholly within 6 chunks of the camera.
        let near = |n: NodeKey| {
            let (o, e) = (n.chunk_origin(), n.extent_chunks());
            let c = [camera.x, camera.y, camera.z];
            [o.x, o.y, o.z]
                .iter()
                .zip(c)
                .all(|(&lo, c)| lo >= c - 6 && lo + e - 1 <= c + 6)
        };
        let open: Vec<NodeKey> = nodes.iter().copied().filter(|&n| near(n)).collect();
        assert!(
            open.iter().any(|n| n.level == 0) && open.iter().any(|n| n.level == 1),
            "the cavern is all one level, so this proves nothing"
        );
        let sight = search(camera, &nodes, |n| Some(if near(n) { OPEN } else { ROCK }));
        assert!(open.iter().all(|n| sight.nodes.contains(n)));
        assert!(!sight.open_to_sky, "a cavern walled in by rock");
    }

    /// Whether the cube around `camera` has a node at height `y` in the
    /// camera's column.
    fn shaft_node_exists(camera: ChunkCoord, y: i32) -> bool {
        desired_nodes_3d(camera, 1, &[(0u32, 4)])
            .contains(&NodeKey::new(0, [camera.x, y, camera.z]))
    }

    /// **The owner's cave, and its mouth** (issue #424), in the real world:
    /// a room dug out of solid rock under the ground sees no sky, the surface
    /// above it does, and so does the same room once a shaft is dug from it
    /// to the surface -- the cave mouth a depth threshold gets wrong.
    #[test]
    fn a_room_underground_is_shut_in_until_a_shaft_opens_it() {
        let registry = registry();
        let mut world = World::new();
        let gen = crate::WorldGen::new(world.seed());
        let size = Chunk::SIZE as i32;
        let solid_chunk = |c: ChunkCoord| {
            let o = [c.x * size, c.y * size, c.z * size];
            (0..size).all(|x| {
                (0..size).all(|y| (0..size).all(|z| gen.is_solid(o[0] + x, o[1] + y, o[2] + z)))
            })
        };
        // The first chunk under the ground near the origin with rock on all
        // six sides: nothing natural can lead out of a room inside it.
        let room = (0..400)
            .map(|i| {
                let (cx, cz) = (i % 20 - 10, i / 20 - 10);
                let lowest = (0..size)
                    .flat_map(|x| (0..size).map(move |z| (x, z)))
                    .map(|(x, z)| world.surface_height(cx * size + x, cz * size + z))
                    .min()
                    .expect("a chunk has columns");
                ChunkCoord::new(cx, lowest.div_euclid(size) - 2, cz)
            })
            .find(|&c| {
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, -1, 0],
                    [0, 0, 1],
                    [0, 0, -1],
                ]
                .into_iter()
                .all(|d| solid_chunk(ChunkCoord::new(c.x + d[0], c.y + d[1], c.z + d[2])))
            })
            .expect("solid rock under the ground somewhere near the origin");
        let o = [room.x * size, room.y * size, room.z * size];
        for x in 4..12 {
            for y in 4..12 {
                for z in 4..12 {
                    world.set_block(o[0] + x, o[1] + y, o[2] + z, BlockId::AIR);
                }
            }
        }
        let sky = |world: &World, camera: ChunkCoord| {
            let scene = Scene::of(world, camera, &registry, 1, &[(0u32, 4)]);
            search(camera, &scene.nodes, |n| scene.links.get(&n).copied()).open_to_sky
        };
        assert!(!sky(&world, room), "a room in solid rock sees no sky");
        // Standing on the ground above it.
        let surface = world.surface_height(o[0] + 8, o[2] + 8);
        let above = ChunkCoord::from_block(o[0] + 8, surface + 3, o[2] + 8);
        assert!(sky(&world, above), "the surface is under the sky");
        // The shaft: the room's own width, up through everything above it.
        for x in 4..12 {
            for z in 4..12 {
                for y in o[1] + 12..=surface + 2 {
                    world.set_block(o[0] + x, y, o[2] + z, BlockId::AIR);
                }
            }
        }
        assert!(
            sky(&world, room),
            "the same room, with a shaft to the surface"
        );
    }
}
