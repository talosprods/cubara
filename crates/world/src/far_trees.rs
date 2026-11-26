//! **Trees on the far terrain** (issue #433 item 1): which trees a far patch
//! draws and where they stand -- the GPU-free half.
//!
//! Beyond the voxel rings the far terrain draws the ground (`crate::far`);
//! the trees on it are the same trees the voxels grow, from the same
//! placement ([`WorldGen::tree_site`], one per chunk column at most): no
//! second generator (`ARCHITECTURE.md` Rule 5). What is the far terrain's
//! own is only where a tree stands: **on the top of the far quad its trunk
//! is in** ([`crate::far::quad_top`]), the ground the far terrain draws
//! there -- so a far trunk neither floats nor sinks, and at quads of 8
//! blocks, where a far quad is a level-3 cell, a far tree stands where level
//! 3 stands it (`WorldGen::node_trees`: the cell's top plus one).

use crate::far::{PatchKey, TOPS_SIDE};
use crate::{TerrainBlocks, WorldGen};

/// One tree the far terrain draws: its trunk's column, the first trunk
/// block's `y` -- the top of the far quad under it -- and its trunk's
/// height, as the world grows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FarTree {
    pub x: i32,
    pub z: i32,
    pub base: i32,
    pub height: i32,
}

/// Every tree whose trunk stands in the patch `key`, standing on the top of
/// its quad: `tops` is the patch's [`crate::far::PatchHeights::tops`]
/// ([`TOPS_SIDE`] a side, its own quads inside a ring of its neighbours').
/// In chunk-column order (`z` outer, `x` inner), so the list is a function
/// of the seed and the key alone.
///
/// A patch is a whole number of chunk columns wide (32 quads of at least one
/// block), and a trunk stands inside its chunk column
/// ([`WorldGen::tree_site`] keeps it a canopy's reach from the column's
/// edge), so every tree of the world is in exactly one patch of a level.
pub fn patch_trees(
    gen: &WorldGen,
    key: PatchKey,
    tops: &[i32],
    blocks: TerrainBlocks,
) -> Vec<FarTree> {
    assert_eq!(tops.len(), TOPS_SIDE * TOPS_SIDE, "a patch's tops");
    if blocks.oak.is_none() {
        return Vec::new();
    }
    let chunk = cubara_voxel::Chunk::SIZE as i64;
    let [ox, oz] = key.origin();
    let (q, chunks) = (key.quad(), key.size() / chunk);
    let (cx0, cz0) = (ox.div_euclid(chunk), oz.div_euclid(chunk));
    let mut out = Vec::new();
    for cz in cz0..cz0 + chunks {
        for cx in cx0..cx0 + chunks {
            let Some(site) = gen.tree_site(cx as i32, cz as i32, blocks) else {
                continue;
            };
            let i = (site.x as i64 - ox).div_euclid(q) as usize;
            let j = (site.z as i64 - oz).div_euclid(q) as usize;
            out.push(FarTree {
                x: site.x,
                z: site.z,
                base: tops[(j + 1) * TOPS_SIDE + i + 1],
                height: site.height,
            });
        }
    }
    out
}

/// **How densely trees stand where every column may grow one**, per trunk
/// height (the oak's, lowest first): trees per square block. The placement's
/// own numbers ([`WorldGen::tree_site`]): one chunk column in `oak.density`
/// carries a tree, its trunk height drawn evenly from the oak's range by the
/// same hash. Times a quad's wooded share ([`crate::far::QuadGround::trees`])
/// it is how densely the quad is wooded -- what the far terrain's tint
/// stands for (issue #433 item 1). Empty for a world without trees.
pub fn density(blocks: TerrainBlocks) -> Vec<f32> {
    let Some(oak) = blocks.oak else {
        return Vec::new();
    };
    let (lo, hi) = oak.height;
    let heights = (hi - lo + 1).max(1);
    let column = (cubara_voxel::Chunk::SIZE * cubara_voxel::Chunk::SIZE) as f32;
    vec![1.0 / (oak.density as f32 * column * heights as f32); heights as usize]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::far::{generate_with, quad_top, PatchKey, WithTrees, PATCH_QUADS};
    use crate::worldgen::{Oak, OreSet};
    use cubara_voxel::BlockId;

    /// The real oak's numbers (`assets/structures/oak.ron`) on a palette of
    /// plain ids: what the world grows, without loading a registry.
    fn oak_blocks() -> TerrainBlocks {
        TerrainBlocks {
            oak: Some(Oak {
                log: BlockId(10),
                leaves: BlockId(11),
                grows_on: BlockId(1),
                height: (4, 6),
                canopy_radius: 2,
                density: 12,
            }),
            ores: OreSet::EMPTY,
            grass: BlockId(1),
            soil: BlockId(2),
            stone: BlockId(3),
        }
    }

    fn with_trees(up_to_level: u32) -> Option<WithTrees> {
        Some(WithTrees {
            blocks: oak_blocks(),
            up_to_level,
        })
    }

    /// Patches of every kind of ground near spawn: flat land, hills, a
    /// mountain's flank (the seed's nearest range is within 600 blocks).
    fn some_keys(level: u32) -> Vec<PatchKey> {
        (-2..2)
            .flat_map(|x| (-2..2).map(move |z| PatchKey::new(level, x, z)))
            .collect()
    }

    /// **At quads of 8 a far tree is level 3's tree**, standing where level
    /// 3 stands it: the same trunk, the same height, the same base -- the
    /// level-3 cell's top plus one (`WorldGen::node_trees`), since a far
    /// quad of 8 blocks is that cell (`far::EXACT_TOP_QUAD`). What makes the
    /// hand-over at the ring edge move no tree.
    #[test]
    fn a_far_tree_at_quads_of_8_stands_where_level_3_stands_it() {
        let gen = WorldGen::new(0x5EED_0000_C0FF_EE);
        let blocks = oak_blocks();
        let mut compared = 0;
        for key in some_keys(3) {
            let patch = generate_with(&gen, key, with_trees(3));
            let mut far = patch.trees.clone();
            // The level-3 nodes over the patch: 128 blocks each, 2 x 2.
            let mut level3: Vec<FarTree> = Vec::new();
            let [ox, oz] = key.origin();
            for nz in 0..2 {
                for nx in 0..2 {
                    let node = crate::node::NodeKey::new(
                        3,
                        [(ox / 128) as i32 + nx, 0, (oz / 128) as i32 + nz],
                    );
                    level3.extend(gen.node_trees(node, blocks).into_iter().map(|s| FarTree {
                        x: s.x,
                        z: s.z,
                        base: s.base,
                        height: s.height,
                    }));
                }
            }
            far.sort();
            level3.sort();
            assert_eq!(far, level3, "the patch {key:?}");
            compared += far.len();
        }
        assert!(compared > 50, "only {compared} trees compared");
    }

    /// **A far tree stands on its quad's top** at every level -- the ground
    /// the far terrain draws under it, [`quad_top`] -- and is the world's
    /// tree there ([`WorldGen::tree_site`]): its own column and height.
    #[test]
    fn a_far_tree_stands_on_its_quads_top() {
        let gen = WorldGen::new(0x5EED_0000_C0FF_EE);
        let blocks = oak_blocks();
        for level in [4, 5, 6] {
            for key in some_keys(level).into_iter().take(4) {
                let patch = generate_with(&gen, key, with_trees(8));
                assert!(!patch.trees.is_empty(), "no trees in {key:?}");
                let [ox, oz] = key.origin();
                for t in &patch.trees {
                    let (i, j) = (
                        (t.x as i64 - ox).div_euclid(key.quad()) as i32,
                        (t.z as i64 - oz).div_euclid(key.quad()) as i32,
                    );
                    assert!((0..PATCH_QUADS as i32).contains(&i), "{t:?} outside {key:?}");
                    assert!((0..PATCH_QUADS as i32).contains(&j), "{t:?} outside {key:?}");
                    assert_eq!(t.base, quad_top(&gen, key, i, j), "{t:?} in {key:?}");
                    let site = gen
                        .tree_site(t.x.div_euclid(16), t.z.div_euclid(16), blocks)
                        .expect("the world grows this tree");
                    assert_eq!((site.x, site.z, site.height), (t.x, t.z, t.height));
                }
            }
        }
    }

    /// **Every tree is in exactly one patch of a level**: a patch's trees
    /// are its four children's, the same trunks and heights -- only the
    /// ground under them is the coarser quad's.
    #[test]
    fn a_patch_holds_the_trees_of_its_children() {
        let gen = WorldGen::new(0x5EED_0000_C0FF_EE);
        for key in some_keys(5).into_iter().take(3) {
            let site = |t: &FarTree| (t.x, t.z, t.height);
            let mut parent: Vec<_> = generate_with(&gen, key, with_trees(8))
                .trees
                .iter()
                .map(site)
                .collect();
            let mut children: Vec<_> = key
                .children()
                .into_iter()
                .flat_map(|c| generate_with(&gen, c, with_trees(8)).trees)
                .map(|t| site(&t))
                .collect();
            parent.sort();
            children.sort();
            assert_eq!(parent, children, "{key:?}");
        }
    }

    /// Only the levels asked for carry trees one by one, and a world
    /// without the oak carries none at any level -- nor any wooded share.
    #[test]
    fn trees_are_carried_where_asked_for() {
        let gen = WorldGen::new(0x5EED_0000_C0FF_EE);
        let key = PatchKey::new(5, 0, 0);
        assert!(!generate_with(&gen, key, with_trees(5)).trees.is_empty());
        let coarse = generate_with(&gen, key, with_trees(4));
        assert!(coarse.trees.is_empty());
        assert!(coarse.ground.iter().any(|g| g.trees > 0), "the share is every level's");
        let bare = generate_with(&gen, key, None);
        assert!(bare.trees.is_empty());
        assert!(bare.ground.iter().all(|g| g.trees == 0));
        let no_oak = TerrainBlocks {
            oak: None,
            ..oak_blocks()
        };
        let none = generate_with(
            &gen,
            key,
            Some(WithTrees {
                blocks: no_oak,
                up_to_level: 8,
            }),
        );
        assert!(none.trees.is_empty());
        assert!(none.ground.iter().all(|g| g.trees == 0));
    }

    /// **A quad's wooded share is the share of its columns a tree may grow
    /// on** -- by the placement's own rule: the surface block is the one the
    /// oak grows on and no cave has carved it away
    /// (`WorldGen::tree_in_column`'s test, asked here through its density).
    /// At quads of 8, every column.
    #[test]
    fn a_quads_share_is_the_columns_a_tree_may_grow_on() {
        let gen = WorldGen::new(0x5EED_0000_C0FF_EE);
        let (mut pits, mut quads) = (0, 0);
        // A mountain's flank, where caves open pits (the seed's nearest
        // range is within 600 blocks of the origin), and flat land.
        for key in [PatchKey::new(3, 1, 1), PatchKey::new(3, -2, 0), PatchKey::new(3, 0, -2)] {
            let patch = generate_with(&gen, key, with_trees(3));
            let [ox, oz] = key.origin();
            for j in 0..PATCH_QUADS {
                for i in 0..PATCH_QUADS {
                    let (x0, z0) = (ox as i32 + 8 * i as i32, oz as i32 + 8 * j as i32);
                    let mut grows = 0;
                    for dz in 0..8 {
                        for dx in 0..8 {
                            let (x, z) = (x0 + dx, z0 + dz);
                            let surface = gen.surface_height(x, z);
                            if gen.solid_on(x, surface, z, surface) {
                                grows += 1;
                            }
                        }
                    }
                    let want = (grows as f32 / 64.0 * 255.0).round() as u8;
                    assert_eq!(patch.ground[j * PATCH_QUADS + i].trees, want, "{key:?} ({i}, {j})");
                    pits += usize::from(grows < 64);
                    quads += 1;
                }
            }
        }
        assert!(pits > 0, "no quad of the {quads} has a pit: this pins little");
    }

    /// **The density is what the world grows**: over the trees of a wide
    /// stretch of land (patches of quads of 64, 16,384 chunk columns each),
    /// the trees counted per trunk height against the density times the
    /// ground a tree may grow on -- each quad's wooded share times its area.
    /// Within 3% per height (**chosen, not derived**: the count's own
    /// spread is about 1.5% at a few thousand trees a height).
    #[test]
    fn the_density_is_what_the_world_grows() {
        let gen = WorldGen::new(0x5EED_0000_C0FF_EE);
        let blocks = oak_blocks();
        let per_height = density(blocks);
        assert_eq!(per_height.len(), 3);
        let mut counted = [0f64; 3];
        let mut expected = [0f64; 3];
        for key in some_keys(6) {
            let patch = generate_with(&gen, key, with_trees(6));
            for t in &patch.trees {
                counted[(t.height - 4) as usize] += 1.0;
            }
            let area = (key.quad() * key.quad()) as f64;
            let grows: f64 = patch.ground.iter().map(|g| g.trees as f64 / 255.0 * area).sum();
            for (e, d) in expected.iter_mut().zip(&per_height) {
                *e += grows * *d as f64;
            }
        }
        for h in 0..3 {
            let off = (counted[h] - expected[h]).abs() / expected[h];
            assert!(
                off < 0.03,
                "trunk height {}: {} trees where the density says {:.0}",
                h + 4,
                counted[h],
                expected[h]
            );
        }
        assert!(density(TerrainBlocks { oak: None, ..blocks }).is_empty());
    }
}
