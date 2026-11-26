//! **The strip the far terrain stands in for, drawn by the game's renderer**
//! (`docs/PHASE3_ARCHITECTURE.md` §9.11).
//!
//! The gate's low flight (`--bench flight --eye 8,120,8 --look 1,-0.15,0.3`)
//! crosses x = 128 at t 5.27 s: the voxels' box moves a coarsest node on in
//! +x and gains the strip x 1,152 to 1,280, a kilometre ahead, whose nodes
//! take frames to be searched for, meshed and drawn. Meanwhile the far
//! terrain draws what [`far_stand`] says. Every frame from t 5.28 to 5.70 s,
//! sixty a second, is held against the finished frame at the same eye --
//! every node drawn, the far terrain holed at the box the voxels want -- by
//! the flight's own measure ([`HoleTracker`], [`confirm_holes`],
//! [`stale_pixels`], [`split_stale`]). The owner's flicker is land drawn
//! before showing nothing where the finished frame has it inside land.
//!
//! The region is the sliver's: #443's hunt found (211, 358) at t 5.48 s
//! looking through a fifth of a pixel of sky between two far ridges, and
//! both quiet x3 runs of #442 counted it at t 5.49 s. Only what reaches the
//! region is meshed and generated -- a node or a patch whose box lands on
//! the screen within [`MARGIN`] of it from some eye of the flight
//! (`flight::screen_box`): 67 of the selection's 13,663 nodes and 26 far
//! patches, which draw the region exactly as the whole scene does
//! (`the_cut_draws_the_region_as_the_whole_scene_does`, ignored: it meshes
//! everything).

use std::collections::HashMap;

use cubara_render::{CameraPose, Frame, MeshedNode, Renderer};
use cubara_voxel::{ChunkCoord, Mesh};
use cubara_world::far::{self, PatchHeights, PatchKey};
use cubara_world::mesh::{mesh_nodes, BuiltNode};
use cubara_world::node::{covered_box_3d, desired_nodes_3d, NodeKey, DEFAULT_RING_SCHEDULE};
use cubara_world::{TerrainBlocks, World, WorldGen};

use crate::far_streaming::{
    far_materials, far_stand, far_view_scaled, hole_in_blocks, standin_faces, to_far_faces,
    to_far_params, to_far_patch, FarQuality, VoxelBox, FAR_QUAD_SCALE, FAR_VIEW_RADIUS,
};
use crate::flight::{
    confirm_holes, eye_at, inside_land, land_of, pose, screen_box, split_stale, stale_pixels,
    Flight, HoleTracker, FLY_SPEED,
};
use crate::streaming::{
    morph_params, next_far_hole, to_node_id, tree_model_for, tree_placements,
    VERTICAL_LOD_SQUASH,
};

/// The flight's screen.
const SIZE: (u32, u32) = (1920, 1080);

/// The pixels held against the finished frame, `[x0, y0, x1, y1)`: round
/// the sliver, where the strip's skyline meets the far ridges behind it.
const REGION: [u32; 4] = [190, 340, 232, 376];

/// How far beyond the region (grown by a pixel for its four sides) a box
/// may land on the screen and still be drawn. **Chosen, not derived**:
/// anything that can cover a pixel of the region lands on it, so any margin
/// is exact; this one leaves room for the sub-pixel seal.
const MARGIN: u32 = 16;

/// The frames: from just after the box moved (t 5.274 s) past the sliver,
/// sixty a second -- the flight's own pace.
const T0: f32 = 5.28;
const T1: f32 = 5.80;
const FPS: f32 = 60.0;

/// The eye's chunk before the box moved, and after.
const BEFORE: [i32; 3] = [7, 6, 2];
const AFTER: [i32; 3] = [8, 6, 2];

/// The gate's low flight, as `--bench flight` flies it.
fn low_flight() -> Flight {
    Flight {
        start: [8.0, 120.0, 8.0],
        look: [1.0, -0.15, 0.3],
        speed: FLY_SPEED,
        seconds: 20.0,
        fps: 60,
        every: 0.25,
        size: SIZE,
        quality: FarQuality::High,
        far_quad_scale: FAR_QUAD_SCALE,
        morph: true,
    }
}

/// The region grown by a pixel: what a pixel of it is judged against.
fn crop_box() -> [u32; 4] {
    [REGION[0] - 1, REGION[1] - 1, REGION[2] + 1, REGION[3] + 1]
}

fn crop(frame: &Frame) -> Frame {
    let [x0, y0, x1, y1] = crop_box();
    let mut pixels = Vec::with_capacity(((x1 - x0) * (y1 - y0) * 4) as usize);
    for y in y0..y1 {
        let row = |x: u32| ((y * frame.width + x) * 4) as usize;
        pixels.extend_from_slice(&frame.pixels[row(x0)..row(x1)]);
    }
    Frame {
        width: x1 - x0,
        height: y1 - y0,
        pixels,
    }
}

/// A crop's pixel `i` on the whole screen.
fn at(i: usize) -> (u32, u32) {
    let [x0, y0, x1, _] = crop_box();
    let w = (x1 - x0) as usize;
    (x0 + (i % w) as u32, y0 + (i / w) as u32)
}

/// Whether the box `min..max` (blocks) lands on the screen within
/// [`MARGIN`] of the crop from any of `eyes`.
fn reaches(flight: &Flight, eyes: &[[f32; 3]], (min, max): ([f32; 3], [f32; 3])) -> bool {
    let [x0, y0, x1, y1] = crop_box();
    let aspect = SIZE.0 as f32 / SIZE.1 as f32;
    eyes.iter().any(|&eye| {
        let vp = pose(flight, eye).view_proj(aspect, cubara_render::anchor::ORIGIN);
        screen_box(min, max, vp, SIZE.0, SIZE.1).is_some_and(|b| {
            b.x0 < x1 + MARGIN
                && x0.saturating_sub(MARGIN) < b.x1
                && b.y0 < y1 + MARGIN
                && y0.saturating_sub(MARGIN) < b.y1
        })
    })
}

fn node_box(node: NodeKey) -> ([f32; 3], [f32; 3]) {
    let o = node.chunk_origin();
    let min = [o.x, o.y, o.z].map(|c| (c * 16) as f32);
    let e = (node.extent_chunks() * 16) as f32;
    (min, min.map(|c| c + e))
}

/// Everything a far face of `key` can reach: its ground, at any height a
/// top, a wall or a skirt can have.
fn patch_reach(key: PatchKey) -> ([f32; 3], [f32; 3]) {
    let [ox, oz] = key.origin().map(|o| o as f32);
    let size = key.size() as f32;
    ([ox, -4096.0, oz], [ox + size, 4096.0, oz + size])
}

/// A node's mesh as the renderer takes it, again (a renderer takes it by
/// value, and two draw the same node).
fn meshed(built: &BuiltNode) -> Option<MeshedNode> {
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
        trees: tree_placements(&built.trees),
        tree_model: tree_model_for(built.node.level),
    })
}

/// A renderer as the flight's: cleared to magenta (`flight::NOTHING`), the
/// game's morph, and what it draws of the far terrain.
struct View {
    renderer: Renderer,
    slots: Vec<cubara_render::FarSlot>,
}

impl View {
    fn new(flight: &Flight, eye: [f32; 3]) -> Option<(Self, cubara_render::MeshAssets)> {
        let (mut renderer, assets) = Renderer::offscreen(SIZE.0, SIZE.1, pose(flight, eye))?;
        renderer.set_clear_color(wgpu::Color {
            r: 1.0,
            g: 0.0,
            b: 1.0,
            a: 1.0,
        });
        renderer.set_morph(morph_params());
        renderer.set_far_shown(true);
        Some((
            Self {
                renderer,
                slots: Vec::new(),
            },
            assets,
        ))
    }

    /// Draw from now on: the far terrain `far`, holed at `hole`, and the
    /// stand-in `standin` cut for `voxels`.
    fn far(
        &mut self,
        patches: &HashMap<PatchKey, PatchHeights>,
        (hole, far): (far::Hole, &[PatchKey]),
        (standin, voxels): (&[PatchKey], VoxelBox),
    ) {
        for slot in self.slots.drain(..) {
            self.renderer.far_remove(slot);
        }
        self.renderer
            .set_far_params(to_far_params(hole, far_materials()));
        for key in far {
            let p = &patches[key];
            let faces = to_far_faces(p);
            let slot = self.renderer.far_insert(to_far_patch(p, &faces));
            self.slots.push(slot.expect("room for a far patch"));
        }
        for key in standin {
            let p = &patches[key];
            let faces = standin_faces(p, voxels);
            let slot = self
                .renderer
                .far_insert_as(to_far_patch(p, &faces), cubara_render::FarRole::StandIn);
            self.slots.push(slot.expect("room for a stand-in"));
        }
    }

    /// The frame at `eye`, every upload drawn.
    fn frame(&mut self, camera: CameraPose) -> Frame {
        let draw = |r: &mut Renderer| {
            r.render(
                camera,
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
            )
        };
        for _ in 0..200 {
            draw(&mut self.renderer);
            if self.renderer.uploads_pending() == 0 {
                break;
            }
        }
        draw(&mut self.renderer);
        self.renderer.read_frame().expect("a frame")
    }
}

/// The scene round the strip: the nodes and patches that reach the region.
struct Scene {
    flight: Flight,
    gen: WorldGen,
    eyes: Vec<[f32; 3]>,
    voxels: VoxelBox,
    built: HashMap<NodeKey, BuiltNode>,
    /// The selection's nodes inside the box the voxels draw: drawn on every
    /// frame. And the strip's, in key order.
    kept: Vec<NodeKey>,
    strip: Vec<NodeKey>,
    patches: HashMap<PatchKey, PatchHeights>,
    live: View,
    finished: View,
    /// Whether only what reaches the region is meshed and generated.
    cut: bool,
}

impl Scene {
    /// `None` without a GPU.
    fn new(cut: bool) -> Option<Self> {
        let flight = low_flight();
        let mut eyes = Vec::new();
        let mut t = T0;
        while t <= T1 + 1e-4 {
            eyes.push(eye_at(&flight, t));
            t += 1.0 / FPS;
        }
        let (mut live, assets) = View::new(&flight, eyes[0])?;
        let (finished, _) = View::new(&flight, eyes[0])?;
        let world = World::new();
        let gen = WorldGen::new(world.seed());
        let [before, after] = [BEFORE, AFTER].map(|[x, y, z]| ChunkCoord::new(x, y, z));
        for &e in &eyes {
            assert_eq!(ChunkCoord::from_world_pos(e), after, "the eye at {e:?}");
        }
        // The box the voxels want, and what of it they draw while the
        // strip's nodes load: the game's own rule for it.
        let want = covered_box_3d(after, DEFAULT_RING_SCHEDULE);
        let drawn = next_far_hole(covered_box_3d(before, DEFAULT_RING_SCHEDULE), want, |_| false);
        assert_ne!(drawn, want, "the box did not move");
        let voxels = VoxelBox {
            wanted: hole_in_blocks(want),
            drawn: hole_in_blocks(drawn),
        };
        let inside = |n: &NodeKey| {
            let o = n.chunk_origin();
            let o = [o.x, o.y, o.z];
            let e = n.extent_chunks();
            (0..3).all(|a| drawn.0[a] <= o[a] && o[a] + e <= drawn.1[a])
        };
        let nodes: Vec<NodeKey> =
            desired_nodes_3d(after, VERTICAL_LOD_SQUASH, DEFAULT_RING_SCHEDULE)
                .into_iter()
                .filter(|n| !cut || reaches(&flight, &eyes, node_box(*n)))
                .collect();
        let registry = assets.registry;
        let layers = assets.layers;
        let layer_of = |name: &str| layers.layer_of(name);
        let blocks = TerrainBlocks::from_registry(&registry)
            .with_oak(&crate::game::load_structure_registry(), &registry)
            .with_ores(&crate::game::load_ore_registry(), &registry);
        let built: HashMap<NodeKey, BuiltNode> =
            mesh_nodes(&world, &registry, &layer_of, nodes.clone(), blocks)
                .into_iter()
                .map(|b| (b.node, b))
                .collect();
        let (kept, mut strip): (Vec<NodeKey>, Vec<NodeKey>) = nodes.into_iter().partition(inside);
        strip.sort();
        live.renderer.apply_node_updates(
            std::iter::empty(),
            kept.iter().filter_map(|n| meshed(&built[n])),
        );
        let mut scene = Self {
            flight,
            gen,
            eyes,
            voxels,
            built,
            kept,
            strip,
            patches: HashMap::new(),
            live,
            finished,
            cut,
        };
        let all: Vec<MeshedNode> = scene
            .kept
            .iter()
            .chain(&scene.strip)
            .filter_map(|n| meshed(&scene.built[n]))
            .collect();
        scene
            .finished
            .renderer
            .apply_node_updates(std::iter::empty(), all);
        Some(scene)
    }

    /// The far terrain at `eye` holed at `hole`: what the game's plan
    /// draws once it has settled there (`far::build`'s loop), of the
    /// patches that reach the region.
    fn far_at(&mut self, eye: [f32; 3], hole: far::Hole) -> Vec<PatchKey> {
        let view = far_view_scaled(
            eye.map(|v| v as f64),
            SIZE.1,
            Some(hole),
            self.flight.quality,
            self.flight.far_quad_scale,
        )
        .expect("the flight's quality draws the far terrain");
        let (flight, eyes, cut) = (&self.flight, &self.eyes, self.cut);
        far::build_where(&self.gen, &view, &mut self.patches, |k| {
            !cut || reaches(flight, eyes, patch_reach(k))
        })
    }

    /// Fly the frames, the strip's nodes arriving as `arrived(frame)` says
    /// (how many of them, in key order), and hold each against its finished
    /// frame.
    fn fly(&mut self, arrived: impl Fn(usize) -> usize) -> Flown {
        let [x0, y0, x1, y1] = crop_box();
        let (cw, ch) = ((x1 - x0) as usize, (y1 - y0) as usize);
        let mut tracker = HoleTracker::new(cw, cw * ch);
        let mut out = Flown::default();
        let mut finished_crops = Vec::new();
        let mut drawn_strip = 0;
        // What the far terrain draws while the strip waits (the plan's
        // rule), every patch of it generated: the ring round the box had it
        // generated before the box moved
        // (`far_streaming`'s `the_live_selection_is_the_exact_one_on_every_frame`).
        let (hole, standin) = far_stand(self.voxels, |_| true);
        let standin: Vec<PatchKey> = standin
            .into_iter()
            .filter(|&k| !self.cut || reaches(&self.flight, &self.eyes, patch_reach(k)))
            .collect();
        for &k in &standin {
            self.patches
                .entry(k)
                .or_insert_with(|| far::generate(&self.gen, k));
        }
        out.standin = standin.len();
        for (f, eye) in self.eyes.clone().into_iter().enumerate() {
            let camera = pose(&self.flight, eye);
            let n = arrived(f).min(self.strip.len());
            if n > drawn_strip {
                let new: Vec<MeshedNode> = self.strip[drawn_strip..n]
                    .iter()
                    .filter_map(|k| meshed(&self.built[k]))
                    .collect();
                self.live
                    .renderer
                    .apply_node_updates(std::iter::empty(), new);
                drawn_strip = n;
            }
            let far = self.far_at(eye, hole);
            self.live
                .far(&self.patches, (hole, &far), (&standin, self.voxels));
            let shown = crop(&self.live.frame(camera));
            let far = self.far_at(eye, self.voxels.wanted);
            self.finished
                .far(&self.patches, (self.voxels.wanted, &far), (&[], self.voxels));
            let fin = crop(&self.finished.frame(camera));

            // The owner's flicker, as the flight counts it: a kept frame's
            // stale pixels drawn before ...
            tracker.push(land_of(&shown).iter().map(|l| !l));
            let stale = stale_pixels(&shown, &fin);
            let drawn_before = split_stale(&stale, &tracker.drawn_and_missing()).flicker;
            out.flicker.extend(drawn_before.iter().map(|&i| (f, at(i as usize))));
            // ... where the finished frame's own skyline does not pass.
            let (live_land, fin_land) = (land_of(&shown), land_of(&fin));
            let fin_sky: Vec<bool> = fin_land.iter().map(|l| !l).collect();
            for i in 0..live_land.len() {
                let skyline = !inside_land(&fin_land, cw, i) && !inside_land(&fin_sky, cw, i);
                let interior = {
                    let (x, y) = at(i);
                    (REGION[0]..REGION[2]).contains(&x) && (REGION[1]..REGION[3]).contains(&y)
                };
                if !interior {
                    continue;
                }
                out.skyline_px += usize::from(skyline);
                if live_land[i] != fin_land[i] && !skyline {
                    out.unlike.push((f, at(i)));
                }
                // With every node of the strip drawn, the stand-in under them
                // shows nowhere: inside land the colour is the finished
                // frame's.
                let colour = i * 4..i * 4 + 4;
                if n == self.strip.len()
                    && inside_land(&fin_land, cw, i)
                    && shown.pixels[colour.clone()] != fin.pixels[colour]
                {
                    out.recoloured.push((f, at(i)));
                }
            }
            finished_crops.push(fin);
        }
        // ... and a hole: inside land, nothing for a few frames, land again,
        // the finished frame agreeing.
        for (f, cands) in tracker.candidates.iter().enumerate() {
            for i in confirm_holes(cands, &finished_crops[f]) {
                out.flicker.push((f, at(i as usize)));
            }
        }
        out.flicker.sort();
        out.flicker.dedup();
        out
    }
}

/// What a flight past the strip found.
#[derive(Default, Debug)]
struct Flown {
    /// The owner's flicker: (frame, pixel) of land drawn before showing
    /// nothing where the finished frame has it inside land.
    flicker: Vec<(usize, (u32, u32))>,
    /// Pixels showing land where the finished frame shows nothing, or the
    /// other way round, off the finished frame's own skyline.
    unlike: Vec<(usize, (u32, u32))>,
    /// Pixels inside the finished frame's land of another colour, on a
    /// frame with every node of the strip drawn.
    recoloured: Vec<(usize, (u32, u32))>,
    /// Stand-in patches that reach the region.
    standin: usize,
    /// Pixels of the region on the finished frame's skyline, over the
    /// flight: that there is a skyline to judge.
    skyline_px: usize,
}

/// **While the strip's nodes load, the strip is drawn as the finished frame
/// draws it** (§9.11): no flicker by the flight's own measure, every pixel
/// of the region land or nothing as in the finished frame but where the
/// finished frame's own skyline passes through it (sub-pixel), and --
/// once the strip's nodes are drawn over the stand-in -- every pixel inside
/// the finished frame's land of its colour.
///
/// The strip's four nodes in the region arrive as they do in the game,
/// some at a time: none for the first 22 frames (to t 5.63 s), two for the
/// next five, all four for the last five.
///
/// Red before the stand-in, when the far terrain stood in for the strip at
/// its own quads, selected and holed at the box the voxels draw: flicker at
/// t 5.63 s at (209, 357) -- the sliver of t 5.49 s, here at 60 Hz frame
/// times from t 5.28 -- over a thousand pixels of the region unlike the
/// finished frame (its ridges stood up to five pixels above level 3's), and
/// over two thousand of the far terrain's colour in front of drawn nodes.
#[test]
fn the_strip_stands_in_as_the_finished_frame_draws_it() {
    let Some(mut scene) = Scene::new(true) else {
        cubara_render::headless::skip_or_require_gpu(
            "the_strip_stands_in_as_the_finished_frame_draws_it",
        );
        return;
    };
    assert!(
        scene.strip.len() >= 2,
        "{} of the strip's nodes reach the region: none can arrive before another",
        scene.strip.len()
    );
    let found = scene.fly(|f| match f {
        0..22 => 0,
        22..27 => 2,
        _ => usize::MAX,
    });
    println!(
        "{} stand-in patches and {} of the strip's nodes reach the region; {} skyline pixels",
        found.standin,
        scene.strip.len(),
        found.skyline_px
    );
    // Every way it can be wrong, reported together.
    let mut wrong = Vec::new();
    if !found.flicker.is_empty() {
        wrong.push(format!(
            "land drawn before shows nothing inside the finished frame's land at (frame, \
             pixel) {:?}",
            found.flicker
        ));
    }
    if !found.unlike.is_empty() {
        wrong.push(format!(
            "{} pixels are unlike the finished frame off its skyline, the first at (frame, \
             pixel) {:?}",
            found.unlike.len(),
            &found.unlike[..found.unlike.len().min(12)]
        ));
    }
    if !found.recoloured.is_empty() {
        wrong.push(format!(
            "{} pixels inside the finished frame's land show the stand-in over a drawn node, \
             the first at (frame, pixel) {:?}",
            found.recoloured.len(),
            &found.recoloured[..found.recoloured.len().min(12)]
        ));
    }
    // Not vacuous: something stands in, and the region holds a skyline --
    // about a hundred pixels of it a frame.
    if found.standin == 0 {
        wrong.push("no stand-in reaches the region".into());
    }
    if found.skyline_px < scene.eyes.len() * 50 {
        wrong.push(format!(
            "{} pixels of skyline over the flight: the region shows no ridge",
            found.skyline_px
        ));
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// **The cut changes nothing in the region**: the live frames and the
/// finished frames of the flight above, drawn with every node of the
/// selection and every far patch in view, are the cut's pixel for pixel
/// inside the region. Meshes all 13,663 nodes, so minutes:
///
/// ```text
/// cargo test -p cubara --bin cubara strip_standin -- --ignored --nocapture
/// ```
#[test]
#[ignore = "meshes every node of the selection: minutes"]
fn the_cut_draws_the_region_as_the_whole_scene_does() {
    let frames = |cut: bool| -> Option<Vec<(Frame, Frame)>> {
        let mut scene = Scene::new(cut)?;
        let (hole, standin) = far_stand(scene.voxels, |_| true);
        for &k in &standin {
            scene
                .patches
                .entry(k)
                .or_insert_with(|| far::generate(&scene.gen, k));
        }
        let mut out = Vec::new();
        for eye in scene.eyes.clone().into_iter().step_by(5) {
            let camera = pose(&scene.flight, eye);
            let far = scene.far_at(eye, hole);
            scene
                .live
                .far(&scene.patches, (hole, &far), (&standin, scene.voxels));
            let live = crop(&scene.live.frame(camera));
            let far = scene.far_at(eye, scene.voxels.wanted);
            scene
                .finished
                .far(&scene.patches, (scene.voxels.wanted, &far), (&[], scene.voxels));
            out.push((live, crop(&scene.finished.frame(camera))));
        }
        Some(out)
    };
    let (Some(cut), Some(whole)) = (frames(true), frames(false)) else {
        cubara_render::headless::skip_or_require_gpu(
            "the_cut_draws_the_region_as_the_whole_scene_does",
        );
        return;
    };
    for (n, ((a, b), (c, d))) in cut.iter().zip(&whole).enumerate() {
        assert!(a.pixels == c.pixels, "the live frame {n} differs");
        assert!(b.pixels == d.pixels, "the finished frame {n} differs");
    }
}

/// **What the stand-in costs** (`docs/PHASE3_ARCHITECTURE.md` §9.11), in
/// the far terrain's triangles -- what the far pass's cost follows (§9.8) --
/// counted as `FarTerrain::prepare` sends them (`far::runs_of_roles`), with
/// no GPU: at every frame of the low flight's first box move while its strip
/// waits (t 5.28 to 5.70 s), the far terrain as it was (selected and holed
/// at the box the voxels draw, standing in for the strip itself) against
/// the far terrain now (holed at the box they want) and the stand-in. At
/// the gate's standing eyes nothing waits and the far terrain is what it
/// was: no stand-in, the same selection, the same triangles.
///
/// ```text
/// cargo test -p cubara --bin cubara standin_cost_report -- --ignored --nocapture
/// ```
#[test]
#[ignore = "a measurement, not a check: prints the stand-in's triangles"]
fn standin_cost_report() {
    use cubara_render::far::{runs_of_roles, FarCulling, FarRole};
    let flight = low_flight();
    let gen = WorldGen::new(World::new().seed());
    let before = covered_box_3d(ChunkCoord::new(7, 6, 2), DEFAULT_RING_SCHEDULE);
    let want = covered_box_3d(ChunkCoord::new(8, 6, 2), DEFAULT_RING_SCHEDULE);
    let voxels = VoxelBox {
        wanted: hole_in_blocks(want),
        drawn: hole_in_blocks(next_far_hole(before, want, |_| false)),
    };
    let (_, standin) = far_stand(voxels, |_| true);
    let aspect = SIZE.0 as f32 / SIZE.1 as f32;
    for scale in [FAR_QUAD_SCALE, 3.0] {
        let mut have = HashMap::new();
        for &k in &standin {
            have.entry(k).or_insert_with(|| far::generate(&gen, k));
        }
        let (mut was, mut now, mut stood, mut frames) = (0u64, 0u64, 0u64, 0u64);
        let mut t = T0;
        while t <= T1 + 1e-4 {
            let eye = eye_at(&flight, t);
            let camera = pose(&flight, eye);
            let frustum = cubara_render::Frustum::from_view_proj(
                camera.view_proj(aspect, cubara_render::anchor::ORIGIN),
            );
            let eye_v = glam::Vec3::from(eye);
            let view = |hole| {
                far_view_scaled(eye.map(f64::from), SIZE.1, Some(hole), FarQuality::High, scale)
                    .expect("High")
            };
            let count = |hole: far::Hole,
                         keys: &[PatchKey],
                         standin: &[PatchKey],
                         have: &HashMap<PatchKey, PatchHeights>|
             -> (u64, u64) {
                let faces: Vec<Vec<cubara_render::FarFace>> = keys
                    .iter()
                    .map(|k| to_far_faces(&have[k]))
                    .chain(standin.iter().map(|k| standin_faces(&have[k], voxels)))
                    .collect();
                let patches: Vec<(cubara_render::FarPatch<'_>, FarRole)> = keys
                    .iter()
                    .chain(standin)
                    .zip(&faces)
                    .enumerate()
                    .map(|(n, (k, f))| {
                        let role = if n < keys.len() {
                            FarRole::Terrain
                        } else {
                            FarRole::StandIn
                        };
                        (to_far_patch(&have[k], f), role)
                    })
                    .collect();
                let holed = Some((hole.min.map(|v| v as i32), hole.max.map(|v| v as i32)));
                let runs = runs_of_roles(&patches, holed, FarCulling::Tiles, (&frustum, eye_v));
                let (mut all, mut of_standin) = (0, 0);
                for (n, _, count) in runs {
                    all += 2 * count as u64;
                    if n >= keys.len() {
                        of_standin += 2 * count as u64;
                    }
                }
                (all, of_standin)
            };
            let old = far::build_where(&gen, &view(voxels.drawn), &mut have, |_| true);
            let new = far::build_where(&gen, &view(voxels.wanted), &mut have, |_| true);
            was += count(voxels.drawn, &old, &[], &have).0;
            let (all, of_standin) = count(voxels.wanted, &new, &standin, &have);
            now += all;
            stood += of_standin;
            frames += 1;
            t += 1.0 / FPS;
        }
        let per = |v: u64| v / frames;
        println!(
            "STANDIN_COST x{scale}, High, the low flight's strip at t {T0}..{T1} ({frames} frames): \
             far triangles a frame as it was {}, now {} ({:+}), of which the stand-in {} \
             ({} patches of level 3; the ring round the box generated ahead: {} patches)",
            per(was),
            per(now),
            per(now) as i64 - per(was) as i64,
            per(stood),
            standin.len(),
            crate::far_streaming::standin_ring(voxels.wanted).len()
        );
    }
}
