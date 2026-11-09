# Render crate audit (read-only; crates/render/src identical on gui-test and origin/main)

Not found: 16-bit node index overflow (MAX_NODES=16,384 < 65,536; debug_assert in with_node_index), uniform layout mismatches (FrameUniform/far Frame/ParamsGpu/text Screen/Outline all vec4-only and match), per-frame GPU resource leaks, zero-size surface (resize ignores 0; Outdated reconfigures).

## 1. set_lighting respawns a pipeline-build thread EVERY frame while a rebuild is in flight (HIGH, CONFIRMED by trace)
scene.rs:437-454, called each frame from set_camera (render.rs:1144, scene.rs:514). It only compares against `pipeline_lighting`, which is updated only when a rebuild LANDS (scene.rs:466-469). While a rebuild is outstanding, every frame sees lighting != pipeline_lighting, spawns a new thread, and replaces `lighting_rebuild`, dropping the previous receiver. The doc comment says "superseded" assuming the value changed; here it is the same value.
Scenario: render radius goes 0 -> 1024 (fog_range changes) -> rebuild needed; compile takes longer than one frame (doc says ~40 ms first time; frames are 1-16 ms) -> each frame's receiver is replaced before its thread finishes -> the new pipeline (fog) NEVER lands while frames keep coming, and ~compile_time/frame_time threads (each holding a Device clone) pile up. Fog stays off / stale lighting indefinitely. Fix: remember the in-flight target Lighting and skip if equal.

## 2. Camera look direction is quantised far from the origin (MEDIUM, CONFIRMED analytically)
render.rs:225 `look_at_rh(eye, eye + look_dir, Y)`: `eye + look_dir` rounds in f32. At |eye|~1e5 ulp=0.0078, so the direction error is up to ~0.2 deg (~4 px at 1080p/60deg); at 2.6e5 (far terrain reaches 262 km) ~1 deg steps: mouse-look becomes stepped. Fix: `look_to_rh(eye, look_dir, Y)`. Related: the whole renderer is world-space f32 (mesh.wgsl:125, far.wgsl:87-91, figure/outline), view translation = -R*eye in f32, so geometry shimmers ~ulp(coord) from ~1e5 on (PLAUSIBLE; depends on how far players travel). Camera-relative rendering is the fix. figure.wgsl:53 derives the normal from dpdx/dpdy of an interpolated f32 world_pos; at large coordinates the derivatives quantise (cross may be 0 -> normalize(0)=NaN colour) on figures and cracks (PLAUSIBLE, low).

## 3. MAX_DRAWS truncation sheds the coarsest nodes silently, and the capacity check compares nodes, not draws (MEDIUM, CONFIRMED mechanism; trigger PLAUSIBLE)
arena.rs:641 stops at 16,384 DRAWS, but a node emits up to 3 draws (face runs, :660-683), so truncation starts at ~5.5k-8k visible nodes. ArenaUsage::exhausted (:156) compares resident_nodes >= max_draws, so the warning fires far too late; the cap itself never logs. Truncation drops the highest levels = the far ring; the far-terrain hole (far.wgsl:99) covers exactly that box, so the dropped region shows sky/clear colour through the hole. Aspect link: a 3.6 aspect (129 deg horizontal FOV) puts many more nodes in the frustum than a 16:9 window, so this is reached sooner there, and the visible set oscillating around the cap while turning would flicker large coarse patches. Worth a log::warn at the break and a draws-based exhausted().

## 4. Upload queue stalls while the arena is full (MEDIUM, PLAUSIBLE)
render.rs:934-941 + uploads.rs:102. A failed insert is requeued at the BACK and the whole frame's draining stops (`break`). The Remove ops that would free space sit behind all the inserts. With n queued inserts that all fail, one is tried per frame, so the drops are reached only after ~n frames (n=2,400 -> 40 s at 60 fps) and nothing that fits uploads meanwhile. Also render.rs:925 removes the old geometry before the replacement is placed; if placement fails the node is absent until retried (visible gap). Fragmented first-fit (SlabAllocator, no compaction) can fail with free space remaining and no drop coming: permanent livelock of the head item.

## 5. Frustum: far plane degenerates to a NaN plane (LOW, CONFIRMED, benign by accident)
culling.rs:40-46 with FAR_PLANE=1e7: in f32 `far/(near-far)` == -1.0 exactly, so reversed-Z row2 = [0,0,0,0.1]; its normal length is 0, the plane becomes NaN/inf, and `NaN < 0.0` is false, so it never culls. Correct only because the far plane is effectively infinite. Tests use far=100 (culling.rs:82, far.rs:527) and never exercise the production projection with FAR_PLANE; no test would notice if this started rejecting everything. Add a test using CameraUniform::look_view_proj and a far-away box.

## 6. Aspect/size dependencies reviewed (the 3136x860 white triangle: no definite cause found)
- mesh.wgsl radial fog (:190-194): formula is correct for any aspect; viewport/aspect overrides are rebuilt synchronously on resize (scene.rs:389-404). OK.
- Frustum planes, faces_facing, HUD layout: aspect independent. HUD hearts/hotbar are pixel-sized and centred; fine at 3136 wide.
- far_streaming (app) uses pixel = FOV_Y/height (out of scope): at wide FOV, off-axis quads cover MORE pixels than the "16 px" rule assumes, and far.wgsl shades per vertex on the assumption that the view direction barely turns across a quad (far.wgsl:209-212). With 129 deg horizontal FOV both assumptions degrade near the left/right edges.
- far.wgsl masked_top (:129-148): `g_a*v_a/seen` has seen = v_y + g_a*v_a; for terrain above the eye v_y<0 so `seen` crosses 0 on grazing slopes: +inf clamps to 15, the sign flip clamps to 0, and exactly 0 gives NaN into clamp/textureSampleLevel (unspecified). The vertex colour jumps discontinuously across that line, so a whole triangle can flip colour as the camera moves; a wider FOV shows more such above-eye terrain at once. This is the best candidate for a flickering whole-triangle colour in scope (PLAUSIBLE; a guard `max(seen, eps)` is cheap). Also normalize(eye-p) (far.wgsl:214) is NaN if a vertex coincides with the eye.
- Candidate if flicker coincides with coarse patches: finding 3.

## 7. Per-frame CPU allocation (LOW)
far.rs:337-338 allocates two Vecs per frame (<=4096 entries); scene.rs:551 a Vec; render.rs:1030 a String for menus/debug text. Negligible.

## 8. Tests that cannot fail (LOW)
- culling tests only use a finite far plane (see 5).
- render.rs `reverse_z_depth_constants_invert...` tolerates 2e-3 at depths <= 2 km, the far plane is 1e7; the comments still describe a 2,000-block far plane. It proves nothing about the distances the far terrain uses.
- No test covers set_lighting being called repeatedly while a rebuild is pending (finding 1).
