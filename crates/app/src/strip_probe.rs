//! Throwaway probe (strip lane; not committed as is): the low flight's strip
//! sliver -- the frames round t 5.49 s drawn as the live pass draws them while
//! the voxels' box has moved and the strip's level-3 nodes are not drawn yet,
//! and as the finished frame draws them; the flight's own `HoleTracker`,
//! `confirm_holes`, `stale_pixels` and `split_stale` judge them.
//!
//! Only what can reach a region of interest round the sliver is meshed and
//! generated: a box (node or far patch) is kept if its screen box from any
//! eye of the sequence overlaps the region grown by a margin
//! (`flight::screen_box`, conservative). `PROBE_WEDGE=0` keeps everything,
//! to check that the cut changes nothing inside the region.

use std::collections::{HashMap, HashSet};

use cubara_render::{CameraPose, Frame, MeshedNode, Renderer};
use cubara_voxel::{ChunkCoord, Mesh};
use cubara_world::far::{self, PatchHeights, PatchKey};
use cubara_world::mesh::{mesh_nodes, BuiltNode};
use cubara_world::node::{covered_box_3d, desired_nodes_3d, NodeKey, DEFAULT_RING_SCHEDULE};
use cubara_world::{TerrainBlocks, World, WorldGen};

use crate::far_streaming::{
    far_materials, far_view_scaled, hole_in_blocks, standin_faces, standin_keys, to_far_faces,
    to_far_params, to_far_patch, FarQuality, VoxelBox, FAR_VIEW_RADIUS,
};
use crate::flight::{confirm_holes, screen_box, split_stale, stale_pixels, HoleTracker};
use crate::streaming::{
    morph_params, to_node_id, tree_model_for, tree_placements, VERTICAL_LOD_SQUASH,
};

const W: u32 = 1920;
const H: u32 = 1080;
const LOOK: [f32; 3] = [1.0, -0.15, 0.3];

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name).map_or(default, |s| s.parse().expect(name))
}

fn eye_at(t: f32) -> [f32; 3] {
    let dir = glam::Vec3::from(LOOK).normalize_or_zero();
    (glam::Vec3::new(8.0, 120.0, 8.0) + dir * 24.0 * t).to_array()
}

fn pose(eye: [f32; 3]) -> CameraPose {
    CameraPose {
        eye: glam::Vec3::from(eye).as_dvec3(),
        look_dir: glam::Vec3::from(LOOK).normalize_or_zero(),
    }
}

fn copy(built: &BuiltNode) -> Option<MeshedNode> {
    let g = built.geometry.as_ref()?;
    Some(MeshedNode {
        id: to_node_id(built.node),
        origin: g.origin,
        scale: g.scale,
        mesh: Mesh {
            vertices: g.mesh.vertices.clone(),
            indices: g.mesh.indices.clone(),
            face_indices: g.mesh.face_indices,
        },
        local_aabb: g.local_aabb,
        trees: if std::env::var("PROBE_NO_TREES").is_ok() {
            Vec::new()
        } else {
            tree_placements(&built.trees)
        },
        tree_model: tree_model_for(built.node.level),
    })
}

fn shows_nothing(p: &[u8]) -> bool {
    p[0] >= 250 && p[1] <= 5 && p[2] >= 250
}

#[derive(Clone, Copy, Debug)]
struct Roi {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
}

fn roi() -> Roi {
    let v: Vec<u32> = std::env::var("PROBE_ROI")
        .unwrap_or_else(|_| "190,340,232,376".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    Roi {
        x0: v[0],
        y0: v[1],
        x1: v[2],
        y1: v[3],
    }
}

fn crop(f: &Frame, r: Roi) -> Frame {
    let mut pixels = Vec::new();
    for y in r.y0..r.y1 {
        let a = ((y * f.width + r.x0) * 4) as usize;
        let b = ((y * f.width + r.x1) * 4) as usize;
        pixels.extend_from_slice(&f.pixels[a..b]);
    }
    Frame {
        width: r.x1 - r.x0,
        height: r.y1 - r.y0,
        pixels,
    }
}

fn ascii(f: &Frame) -> String {
    let mut s = String::new();
    for y in 0..f.height {
        let row: String = (0..f.width)
            .map(|x| {
                let i = ((y * f.width + x) * 4) as usize;
                if shows_nothing(&f.pixels[i..i + 4]) {
                    '.'
                } else {
                    '#'
                }
            })
            .collect();
        s.push_str(&format!("      {row}\n"));
    }
    s
}

/// Whether the box `min..max` (blocks) can reach the region from any of
/// `eyes`.
fn reaches(min: [f32; 3], max: [f32; 3], eyes: &[[f32; 3]], r: Roi, margin: u32) -> bool {
    let aspect = W as f32 / H as f32;
    eyes.iter().any(|&eye| {
        let vp = pose(eye).view_proj(aspect, cubara_render::anchor::ORIGIN);
        screen_box(min, max, vp, W, H).is_some_and(|b| {
            b.x0 < r.x1 + margin
                && r.x0.saturating_sub(margin) < b.x1
                && b.y0 < r.y1 + margin
                && r.y0.saturating_sub(margin) < b.y1
        })
    })
}

fn node_box(n: NodeKey) -> ([f32; 3], [f32; 3]) {
    let o = n.chunk_origin();
    let e = (n.extent_chunks() * 16) as f32;
    let min = [o.x as f32 * 16.0, o.y as f32 * 16.0, o.z as f32 * 16.0];
    (min, [min[0] + e, min[1] + e, min[2] + e])
}

fn patch_reach_box(key: PatchKey) -> ([f32; 3], [f32; 3]) {
    let [ox, oz] = key.origin();
    let size = key.size() as f32;
    // Every height a far face can take, skirts included (generously).
    (
        [ox as f32, -2048.0, oz as f32],
        [ox as f32 + size, 2048.0, oz as f32 + size],
    )
}

/// `far::build`'s loop with a cache, generating only what `keep` lets
/// through (a patch's split reads its own error only, so the selection is
/// exact for every patch `keep` passes).
fn build_cached(
    gen: &WorldGen,
    view: &far::FarView,
    cache: &mut HashMap<PatchKey, PatchHeights>,
    keep: &impl Fn(PatchKey) -> bool,
) -> Vec<PatchKey> {
    loop {
        let keys = far::select(view, |k| cache.get(&k).map(|p| p.error));
        let new: Vec<PatchKey> = keys
            .iter()
            .copied()
            .filter(|k| keep(*k) && !cache.contains_key(k))
            .collect();
        if new.is_empty() {
            return keys.into_iter().filter(|k| keep(*k)).collect();
        }
        for k in new {
            cache.insert(k, far::generate(gen, k));
        }
    }
}

struct Scene<'a> {
    nodes: &'a [NodeKey],
    hole: ([i32; 3], [i32; 3]),
    far: Vec<PatchKey>,
    /// The stand-in's patches, cut for this box.
    standin: Vec<PatchKey>,
    voxels: Option<VoxelBox>,
}

fn draw(renderer: &mut Renderer, eye: [f32; 3]) {
    renderer.render(
        pose(eye),
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

fn render_scene(
    renderer: &mut Renderer,
    built: &HashMap<NodeKey, BuiltNode>,
    cache: &HashMap<PatchKey, PatchHeights>,
    scene: &Scene,
    eye: [f32; 3],
    slots: &mut Vec<cubara_render::FarSlot>,
) -> Frame {
    renderer.clear_nodes();
    for slot in slots.drain(..) {
        renderer.far_remove(slot);
    }
    let meshed: Vec<MeshedNode> = scene
        .nodes
        .iter()
        .filter_map(|n| built.get(n).and_then(copy))
        .collect();
    renderer.apply_node_updates(std::iter::empty(), meshed);
    renderer.set_far_params(to_far_params(hole_in_blocks(scene.hole), far_materials()));
    renderer.set_far_shown(true);
    for k in &scene.far {
        let p = &cache[k];
        let faces = to_far_faces(p);
        slots.push(
            renderer
                .far_insert(to_far_patch(p, &faces))
                .expect("room for the patch"),
        );
    }
    for k in &scene.standin {
        let p = &cache[k];
        let faces = standin_faces(p, scene.voxels.expect("a box to cut for"));
        slots.push(
            renderer
                .far_insert_as(to_far_patch(p, &faces), cubara_render::FarRole::StandIn)
                .expect("room for the patch"),
        );
    }
    for _ in 0..200 {
        draw(renderer, eye);
        if renderer.uploads_pending() == 0 {
            break;
        }
    }
    draw(renderer, eye);
    renderer.read_frame().expect("a frame")
}

#[test]
#[ignore = "a GPU probe for the strip sliver"]
fn strip_sliver_sequence() {
    let scale = env_f32("PROBE_SCALE", 3.0) as f64;
    let t0 = env_f32("PROBE_T0", 5.40);
    let t1 = env_f32("PROBE_T1", 5.56);
    let fps = env_f32("PROBE_FPS", 60.0);
    let wedge = std::env::var("PROBE_WEDGE").map_or(true, |v| v != "0");
    let r = roi();
    let margin = 16;
    let mut ts = Vec::new();
    let mut t = t0;
    while t <= t1 + 1e-6 {
        ts.push(t);
        t += 1.0 / fps;
    }
    let eyes: Vec<[f32; 3]> = ts.iter().map(|&t| eye_at(t)).collect();

    let world = World::new();
    let gen = WorldGen::new(world.seed());
    let (mut renderer, assets) = Renderer::offscreen(W, H, pose(eyes[0])).expect("a GPU");
    renderer.set_clear_color(wgpu::Color {
        r: 1.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    });
    renderer.set_morph(morph_params());
    let layers = assets.layers;
    let registry = assets.registry;
    let layer_of = |name: &str| layers.layer_of(name);
    let blocks = TerrainBlocks::from_registry(&registry)
        .with_oak(&crate::game::load_structure_registry(), &registry)
        .with_ores(&crate::game::load_ore_registry(), &registry);

    let old_c = ChunkCoord::new(7, 6, 2);
    let new_c = ChunkCoord::new(8, 6, 2);
    for e in &eyes {
        assert_eq!(ChunkCoord::from_world_pos(*e), new_c, "eye {e:?}");
    }
    let new_nodes = desired_nodes_3d(new_c, VERTICAL_LOD_SQUASH, DEFAULT_RING_SCHEDULE);
    let want = covered_box_3d(new_c, DEFAULT_RING_SCHEDULE);
    let want_old = covered_box_3d(old_c, DEFAULT_RING_SCHEDULE);
    let kept = (
        [0, 1, 2].map(|i| want_old.0[i].max(want.0[i])),
        [0, 1, 2].map(|i| want_old.1[i].min(want.1[i])),
    );
    println!("want {want:?} kept {kept:?}");
    let in_kept = |n: &NodeKey| {
        let o = n.chunk_origin();
        let e = n.extent_chunks();
        o.x >= kept.0[0] && o.x + e <= kept.1[0]
    };
    let reach_node = |n: &NodeKey| {
        !wedge || {
            let (a, b) = node_box(*n);
            reaches(a, b, &eyes, r, margin)
        }
    };
    let all: Vec<NodeKey> = new_nodes.iter().copied().filter(reach_node).collect();
    let strip_less: Vec<NodeKey> = all.iter().copied().filter(in_kept).collect();
    let started = std::time::Instant::now();
    let built: HashMap<NodeKey, BuiltNode> =
        mesh_nodes(&world, &registry, &layer_of, all.clone(), blocks)
            .into_iter()
            .map(|b| (b.node, b))
            .collect();
    println!(
        "meshed {} of {} nodes ({} in the strip) in {:.1?}",
        built.len(),
        new_nodes.len(),
        all.len() - strip_less.len(),
        started.elapsed()
    );
    let reach_patch = |k: PatchKey| {
        !wedge || {
            let (a, b) = patch_reach_box(k);
            reaches(a, b, &eyes, r, margin)
        }
    };
    let mut cache = HashMap::new();
    let mut slots = Vec::new();
    let view = |eye: [f32; 3], hole| {
        far_view_scaled(
            eye.map(|v| v as f64),
            H,
            Some(hole_in_blocks(hole)),
            FarQuality::High,
            scale,
        )
        .expect("High")
    };

    let (cw, ch) = (r.x1 - r.x0, r.y1 - r.y0);
    let mut tracker = HoleTracker::new(cw as usize, (cw * ch) as usize);
    let mut finished_crops = Vec::new();
    let mut flicker_frames = HashSet::new();
    let started = std::time::Instant::now();
    let today = std::env::var("PROBE_LIVE").is_ok_and(|v| v == "today");
    let voxels = VoxelBox {
        wanted: hole_in_blocks(want),
        drawn: hole_in_blocks(kept),
    };
    let standin_all = standin_keys(voxels).expect("a strip across the ground");
    let standin: Vec<PatchKey> = standin_all.iter().copied().filter(|k| reach_patch(*k)).collect();
    println!(
        "stand-in: {} patches over the strip, {} reach the region: {standin:?}",
        standin_all.len(),
        standin.len()
    );
    for k in &standin {
        cache.entry(*k).or_insert_with(|| far::generate(&gen, *k));
    }
    // `PROBE_STRIP_NODES=all|half|none`: which of the strip's nodes the live
    // frame draws already.
    let strip_nodes = std::env::var("PROBE_STRIP_NODES").unwrap_or_else(|_| "none".into());
    let mut strip_all: Vec<NodeKey> = all.iter().copied().filter(|n| !in_kept(n)).collect();
    strip_all.sort();
    let live_nodes: Vec<NodeKey> = match strip_nodes.as_str() {
        "none" => strip_less.clone(),
        "all" => all.clone(),
        "half" => strip_less
            .iter()
            .copied()
            .chain(strip_all.iter().copied().step_by(2))
            .collect(),
        other => panic!("PROBE_STRIP_NODES {other}"),
    };
    let mut mask_diffs = 0usize;
    for (k, (&t, &eye)) in ts.iter().zip(&eyes).enumerate() {
        let live = if today {
            Scene {
                nodes: &live_nodes,
                hole: kept,
                far: build_cached(&gen, &view(eye, kept), &mut cache, &reach_patch),
                standin: Vec::new(),
                voxels: None,
            }
        } else {
            Scene {
                nodes: &live_nodes,
                hole: want,
                far: build_cached(&gen, &view(eye, want), &mut cache, &reach_patch),
                standin: standin.clone(),
                voxels: Some(voxels),
            }
        };
        let shown = crop(
            &render_scene(&mut renderer, &built, &cache, &live, eye, &mut slots),
            r,
        );
        let fin_far = build_cached(&gen, &view(eye, want), &mut cache, &reach_patch);
        let finished = Scene {
            nodes: &all,
            hole: want,
            far: fin_far,
            standin: Vec::new(),
            voxels: None,
        };
        let fin = crop(
            &render_scene(&mut renderer, &built, &cache, &finished, eye, &mut slots),
            r,
        );
        tracker.push(
            shown
                .pixels
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| shows_nothing(p)),
        );
        let dam = tracker.drawn_and_missing();
        let differ: Vec<(u32, u32)> = shown
            .pixels
            .as_chunks::<4>()
            .0
            .iter()
            .zip(fin.pixels.as_chunks::<4>().0)
            .enumerate()
            .filter(|(_, (a, b))| shows_nothing(&a[..]) != shows_nothing(&b[..]))
            .map(|(i, _)| (r.x0 + i as u32 % cw, r.y0 + i as u32 / cw))
            .collect();
        let colour_at: Vec<((u32, u32), [u8; 4], [u8; 4])> = shown
            .pixels
            .as_chunks::<4>()
            .0
            .iter()
            .zip(fin.pixels.as_chunks::<4>().0)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, (a, b))| ((r.x0 + i as u32 % cw, r.y0 + i as u32 / cw), *a, *b))
            .collect();
        let colour_differ = colour_at.len();
        if colour_differ > 0 && colour_differ < 8 {
            println!("    colour differs at {colour_at:?}");
        }
        if colour_differ > 0 {
            let mean: f64 = colour_at
                .iter()
                .map(|(_, a, b)| (0..3).map(|c| (a[c] as f64 - b[c] as f64).abs()).sum::<f64>() / 3.0)
                .sum::<f64>()
                / colour_differ as f64;
            println!("    mean |difference| over those: {mean:.1} of 255");
        }
        mask_diffs += differ.len();
        if !differ.is_empty() || colour_differ > 0 {
            println!(
                "    frame {k}: land/nothing differs at {} px {:?}; {colour_differ} px differ in colour",
                differ.len(),
                &differ[..differ.len().min(10)]
            );
        }
        let stale = stale_pixels(&shown, &fin);
        let split = split_stale(&stale, &dam);
        let at = |i: u32| (r.x0 + i % cw, r.y0 + i / cw);
        println!(
            "frame {k} t {t:.4} eye {eye:.2?}: {} far patches live, {} stale, flicker {:?}",
            live.far.len(),
            stale.len(),
            split.flicker.iter().map(|&i| at(i)).collect::<Vec<_>>()
        );
        if !split.flicker.is_empty() {
            flicker_frames.insert(k);
        }
        if std::env::var("PROBE_ASCII").is_ok() {
            println!("    live:\n{}    finished:\n{}", ascii(&shown), ascii(&fin));
        }
        finished_crops.push(fin);
    }
    for (f, cands) in tracker.candidates.iter().enumerate() {
        if cands.is_empty() {
            continue;
        }
        let holes = confirm_holes(cands, &finished_crops[f]);
        let at = |i: u32| (r.x0 + i % cw, r.y0 + i / cw);
        println!(
            "frame {f} (t {:.4}): candidates {:?}, holes {:?}",
            ts[f],
            cands.iter().map(|&i| at(i)).collect::<Vec<_>>(),
            holes.iter().map(|&i| at(i)).collect::<Vec<_>>()
        );
        if !holes.is_empty() {
            flicker_frames.insert(f);
        }
    }
    let mut ff: Vec<usize> = flicker_frames.into_iter().collect();
    ff.sort();
    println!(
        "scale {scale} live {}: {} of {} frames flicker: {:?}; land/nothing differs from the finished frame at {mask_diffs} px in all ({:.1?}, {} patches generated)",
        if today { "today" } else { "standin" },
        ff.len(),
        ts.len(),
        ff,
        started.elapsed(),
        cache.len()
    );
}
