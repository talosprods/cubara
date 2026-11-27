//! **A coarse level never opens a cave through ground that level 0 draws**
//! (`docs/PHASE3_ARCHITECTURE.md` §9.12).
//!
//! The owner's floating block (the player-view tour, 2026-11-27): standing
//! at (5376, 3408) and looking north, a stone-topped block floated over a
//! gap of sky 346 blocks away, and went as the player came within 288. It
//! was a level-2 cell at (5460, 3068) whose rows 135 and 136 were cave --
//! every block of its ground in them is cave at level 0, so the cave rule of
//! §9.4 carved them -- seen past the cell in front of it, (5460, 3072),
//! standing at the mean of its columns, 134, where level 0's columns stand
//! at 136 to 138 and close the cave. The cave was right and the top was
//! right; together they opened what level 0 keeps shut.
//!
//! **The class, as these tests measure it.** For every block row a coarse
//! level draws as cave -- air at or under its cell's top -- and for every
//! cell that may be drawn beside it or over it and leaves that row open to
//! the sky -- a cell beside it whose top is under the row, of the cell's own
//! level or, at its node's edge, of the level finer or coarser (rings are
//! one level apart); the cell itself over its own top; at a node's top, the
//! cell over it as the next node's level draws it -- the sky sees that row
//! across that cell at level 0 too: none of its columns stands at or over
//! the row. Where one does, level 0 has ground there -- the block itself, or
//! a roof over a cave -- and a level that leaves it open shows the cave
//! through it: the sky, where the cave runs through. Level 0 never breaks
//! it: it draws its own blocks.
//!
//! What a level draws is the game's own: the generated nodes in the first
//! test, `WorldGen::cell_draws_solid` in the sweep (held to the generated
//! nodes block by block by `certainly_solid_is_what_a_coarse_node_draws`).
//! A cell's top is `WorldGen::cell_top`, level 0's ground
//! `WorldGen::is_solid`; neither is the rule under test.

use std::collections::HashMap;

use cubara_voxel::{BlockId, ChunkCoord};
use cubara_world::node::NodeKey;
use cubara_world::{CellGround, OreSet, TerrainBlocks, World, WorldGen};

/// The coarsest voxel level. Past it is the far terrain, whose quads at the
/// join are its cells (§9.3): no coarser level is drawn beside it.
const COARSEST: u32 = 3;

/// A node is this many cells wide, deep, and cells' widths tall.
const NODE_CELLS: i32 = 16;

/// The ground the checks ask about, each cell's top worked out once.
struct Ground {
    gen: WorldGen,
    tops: HashMap<(i32, i32, i32), i32>,
    peaks: HashMap<(i32, i32, i32), i32>,
}

impl Ground {
    fn new(gen: WorldGen) -> Self {
        Self {
            gen,
            tops: HashMap::new(),
            peaks: HashMap::new(),
        }
    }

    /// The top of the cell `w` blocks wide that holds the column `(x, z)`.
    fn top(&mut self, x: i32, z: i32, w: i32) -> i32 {
        let (x0, z0) = (x.div_euclid(w) * w, z.div_euclid(w) * w);
        let gen = self.gen;
        *self
            .tops
            .entry((x0, z0, w))
            .or_insert_with(|| gen.cell_top(x0, z0, w))
    }

    /// The highest column top of the cell `w` blocks wide that holds the
    /// column `(x, z)`: at level 0 the sky sees every row over it there,
    /// and no row at or under it.
    fn peak(&mut self, x: i32, z: i32, w: i32) -> i32 {
        let (x0, z0) = (x.div_euclid(w) * w, z.div_euclid(w) * w);
        let gen = self.gen;
        *self.peaks.entry((x0, z0, w)).or_insert_with(|| {
            (0..w * w)
                .map(|k| gen.column_top(x0 + k % w, z0 + k / w))
                .max()
                .expect("a cell covers a column")
        })
    }

    /// The cell `w` wide holding `(x, z)` leaves row `r` open to the sky
    /// where level 0 has ground at or over it.
    fn unroofs(&mut self, x: i32, z: i32, w: i32, r: i32) -> bool {
        self.top(x, z, w) < r && self.peak(x, z, w) >= r
    }
}

/// A cave row a level draws open through ground that level 0 draws there.
struct Opening {
    level: u32,
    cell: [i32; 2],
    row: i32,
    /// `"+x"`, `"-x"`, `"+z"`, `"-z"` or `"above"`.
    toward: &'static str,
    /// The level whose cell leaves it open.
    by: u32,
}

impl std::fmt::Debug for Opening {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            level,
            cell: [x, z],
            row,
            toward,
            by,
        } = self;
        write!(
            f,
            "level {level}, cell ({x}, {z}), row {row}, toward {toward}, left open by level {by}"
        )
    }
}

/// Every opening through level 0's ground of the cave rows `caves` of the
/// cell of `level` at `(x0, z0)`.
fn openings(g: &mut Ground, level: u32, [x0, z0]: [i32; 2], caves: &[i32]) -> Vec<Opening> {
    let w = 1i32 << level;
    let (i, j) = (
        x0.div_euclid(w).rem_euclid(NODE_CELLS),
        z0.div_euclid(w).rem_euclid(NODE_CELLS),
    );
    let mut found = Vec::new();
    for &r in caves {
        for (toward, dx, dz, edge) in [
            ("+x", 1, 0, i == NODE_CELLS - 1),
            ("-x", -1, 0, i == 0),
            ("+z", 0, 1, j == NODE_CELLS - 1),
            ("-z", 0, -1, j == 0),
        ] {
            // The levels that may be drawn against this side.
            let mut levels = vec![level];
            if edge && level > 1 {
                levels.push(level - 1);
            }
            if edge && level < COARSEST {
                levels.push(level + 1);
            }
            // The columns just across the side.
            let across = |t: i32| match (dx, dz) {
                (1, _) => (x0 + w, z0 + t),
                (-1, _) => (x0 - 1, z0 + t),
                (_, 1) => (x0 + t, z0 + w),
                _ => (x0 + t, z0 - 1),
            };
            let through = levels.into_iter().find(|&k| {
                (0..w).any(|t| {
                    let (x, z) = across(t);
                    g.unroofs(x, z, 1 << k, r)
                })
            });
            if let Some(by) = through {
                found.push(Opening {
                    level,
                    cell: [x0, z0],
                    row: r,
                    toward,
                    by,
                });
            }
        }
        // Over it: the cell itself inside its node; at the node's top, the
        // next node's level, which may be this one, finer or coarser.
        let over = r + 1;
        let levels = if over.rem_euclid(NODE_CELLS * w) != 0 {
            vec![level]
        } else {
            let mut l = vec![level];
            if level > 1 {
                l.push(level - 1);
            }
            if level < COARSEST {
                l.push(level + 1);
            }
            l
        };
        let through = levels
            .into_iter()
            .find(|&k| (0..w * w).any(|t| g.unroofs(x0 + t % w, z0 + t / w, 1 << k, over)));
        if let Some(by) = through {
            found.push(Opening {
                level,
                cell: [x0, z0],
                row: r,
                toward: "above",
                by,
            });
        }
    }
    found
}

fn test_blocks() -> TerrainBlocks {
    TerrainBlocks {
        oak: None,
        ores: OreSet::EMPTY,
        grass: BlockId(1),
        soil: BlockId(2),
        stone: BlockId(3),
    }
}

/// **The owner's floating block is rock where level 0's ground hides it.**
/// The nodes the game generates at levels 1, 2 and 3 around block (5461,
/// 138, 3072) -- the one the tour found floating at level 2 -- and the
/// nodes beside them: no cave row in them opens through level 0's ground,
/// and the level-2 cell at (5460, 3068) draws its rows 135 and 136, which
/// were the gap, as ground.
///
/// Those two rows are cave in every block of their ground at level 0 --
/// asserted here, so that what keeps them closed is seen to be the cell in
/// front of them and nothing else: it stands at 134, under its own columns,
/// which stand over both rows at level 0.
#[test]
fn the_owners_floating_block_is_rock_where_level_0_hides_it() {
    let world = World::new();
    let gen = WorldGen::new(world.seed());
    let blocks = test_blocks();
    let mut g = Ground::new(gen);
    let block = [5461.0, 138.0, 3072.0];
    let mut caves_seen = 0;
    let mut failures = Vec::new();
    for level in 1..=COARSEST {
        let w = 1i32 << level;
        let centre = NodeKey::containing(ChunkCoord::from_world_pos(block), level);
        for dz in -1..=1 {
            for dx in -1..=1 {
                for dy in -1..=1 {
                    let [x, y, z] = centre.pos;
                    let node = NodeKey::new(level, [x + dx, y + dy, z + dz]);
                    let Some(chunk) = world.node_at(node, blocks) else {
                        continue;
                    };
                    let origin = node.world_origin();
                    let dims = chunk.mesh_dims();
                    for lz in 0..dims[2] {
                        for lx in 0..dims[0] {
                            let (x0, z0) = (origin[0] + lx as i32 * w, origin[2] + lz as i32 * w);
                            let top = g.top(x0, z0, w);
                            let caves: Vec<i32> = (0..dims[1])
                                .map(|ly| (ly, origin[1] + ly as i32))
                                .filter(|&(ly, y)| {
                                    y <= top && chunk.drawn(lx, ly, lz) == BlockId::AIR
                                })
                                .map(|(_, y)| y)
                                .collect();
                            caves_seen += caves.len();
                            failures.extend(openings(&mut g, level, [x0, z0], &caves));
                        }
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} cave rows open through level 0's ground, the first: {:?}",
        failures.len(),
        &failures[..failures.len().min(8)]
    );
    assert!(caves_seen > 0, "no cave rows in these nodes to check");

    // The gap itself: level 2's cell at (5460, 3068), rows 135 and 136 (in
    // the node beside the block's, which starts at z = 3072).
    let node = NodeKey::containing(ChunkCoord::from_world_pos([5460.0, 135.0, 3068.0]), 2);
    let chunk = world.node_at(node, blocks).expect("the hill top's node");
    let origin = node.world_origin();
    let (lx, lz) = (
        ((5460 - origin[0]) / 4) as usize,
        ((3068 - origin[2]) / 4) as usize,
    );
    assert_eq!(
        g.top(5460, 3068, 4),
        140,
        "the cell holds the hill's summit"
    );
    assert_eq!(
        g.top(5460, 3072, 4),
        134,
        "the cell in front stands at its mean"
    );
    for y in [135, 136] {
        // Every block of its ground in the row is cave at level 0 ...
        let cave = (0..16).all(|k| {
            let (x, z) = (5460 + k % 4, 3068 + k / 4);
            y > gen.column_top(x, z) || !gen.is_solid(x, y, z)
        });
        assert!(cave, "row {y}: the cave is no longer there to be closed");
        // ... and the cell in front leaves the row open where level 0 has
        // ground at or over it: its columns stand to 139.
        assert!(
            g.unroofs(5460, 3072, 4, y),
            "row {y}: level 0 is open in front too"
        );
        assert_ne!(
            chunk.drawn(lx, (y - origin[1]) as usize, lz),
            BlockId::AIR,
            "row {y} of the cell at (5460, 3068): open toward the viewer, through level 0's ground"
        );
    }
}

/// **No level opens a cave through level 0's ground, over a few square
/// kilometres** around the owner's spot and the spawn: every cave row each
/// level draws (`cell_draws_solid`), every side and the row over it, at
/// levels 1 to 3. Main drew 69,342, 23,711 and 2,598 such openings here (the
/// counts are printed). And **caves still nest**: every cave row of a coarse
/// cell is open in each finer cell inside it, cave or over its top -- what a
/// rule asking only its own level's cells breaks (a finer cell inside or
/// beside a cave can stand lower than the coarser one). The levels must
/// still draw caves, some of them open where level 0 is open too, or the
/// rule would be "no caves".
#[test]
fn no_level_opens_a_cave_through_level_0s_ground() {
    let gen = WorldGen::new(World::new().seed());
    let mut g = Ground::new(gen);
    // Two squares of 1,536 blocks: the owner's slopes and the spawn's hills
    // and the mountain north of them. Chosen, not derived: a few square
    // kilometres, and seconds to sweep.
    let squares: [([i32; 2], i32); 2] = [([5376, 3408], 768), ([0, -512], 768)];
    let mut report = Vec::new();
    let mut total = 0;
    for level in 1..=COARSEST {
        let (w, v) = (1i32 << level, if level < 3 { 1 } else { 4 });
        let (mut caves, mut open, mut through, mut nest_breaks) =
            (0usize, 0usize, Vec::new(), 0usize);
        for &([cx, cz], half) in &squares {
            let (xs, zs) = ((cx - half).div_euclid(w) * w, (cz - half).div_euclid(w) * w);
            for b in 0..2 * half / w {
                for a in 0..2 * half / w {
                    let (x0, z0) = (xs + a * w, zs + b * w);
                    let cell: CellGround = gen.cell_ground(x0, z0, w);
                    // Caves are carved within 24 blocks of a cell's highest
                    // column; a margin over that.
                    let rows: Vec<i32> = (cell.peak - 32..=cell.top)
                        .filter(|&y| !gen.cell_draws_solid(&cell, y, v))
                        .collect();
                    caves += rows.len();
                    // Caves nest: each finer cell inside draws every one of
                    // these rows open too -- cave, or over its top.
                    if level > 1 {
                        let (fw, fv) = (w / 2, if level - 1 < 3 { 1 } else { 4 });
                        for q in 0..4 {
                            let (fx, fz) = (x0 + q % 2 * fw, z0 + q / 2 * fw);
                            let finer = gen.cell_ground(fx, fz, fw);
                            nest_breaks += rows
                                .iter()
                                .filter(|&&r| r <= finer.top && gen.cell_draws_solid(&finer, r, fv))
                                .count();
                        }
                    }
                    // How many are open to the sky at all, at this level.
                    open += rows
                        .iter()
                        .filter(|&&r| {
                            r == cell.top
                                || [(w, 0), (-w, 0), (0, w), (0, -w)]
                                    .iter()
                                    .any(|&(dx, dz)| g.top(x0 + dx, z0 + dz, w) < r)
                        })
                        .count();
                    through.extend(openings(&mut g, level, [x0, z0], &rows));
                }
            }
        }
        report.push(format!(
            "level {level}: {caves} cave rows, {open} open to the sky, {} through level 0's ground, {nest_breaks} where a finer cell draws ground",
            through.len()
        ));
        total += through.len() + nest_breaks;
        if let Some(first) = through.first() {
            report.push(format!("  the first: {first:?}"));
        }
        // Chosen, not derived: half of what is measured at level 3 (20 rows
        // open, 2,728 cave rows), so that a rule that drew no caves at all
        // could not pass for one that keeps them shut.
        assert!(
            caves >= 1_000 && open >= 10,
            "level {level}: {caves} cave rows, {open} of them open -- too few for the sweep to mean anything"
        );
    }
    println!("{}", report.join("\n"));
    assert_eq!(total, 0, "\n{}", report.join("\n"));
}
