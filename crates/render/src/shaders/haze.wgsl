// Exponential height haze, closed form along the eye->p ray (one evaluation,
// no ray march). The CPU mirror is `haze.rs`'s `Haze::amount`, line for line;
// the unit tests there pin the model.
//
// Extinction is sigma(y) = density * exp(-y / H). The optical depth from the
// eye e to the point p at distance d is
//     tau = density * d * exp(-e.y / H) * g(delta),
//     delta = (p.y - e.y) / H,  g(delta) = (1 - exp(-delta)) / delta,
// with the series g = 1 - delta/2 + delta^2/6 for |delta| < 1e-2 (a nearly horizontal ray). Every exp
// argument is clamped to [-30, 30] so deep caves and very high eyes stay finite.
//
// `haze.x` is the density (blocks^-1), `haze.y` is 1 / scale_height (blocks^-1).
//
// `eye` and `p` are relative to the render anchor (issue #379), and only the
// extinction's *absolute* height needs the anchor back: `anchor_y` is the
// anchor's y (`Frame.anchor.y`), so `eye.y + anchor_y` is the eye's true
// height (decision 4). Distance and the rise `p.y - eye.y` are relative
// already. Near the origin `anchor_y` is 0 and this is the old formula.
fn haze_amount(eye: vec3<f32>, p: vec3<f32>, haze: vec4<f32>, anchor_y: f32) -> f32 {
    let d = distance(p, eye);
    let delta = (p.y - eye.y) * haze.y;
    let g = select(
        (1.0 - exp(clamp(-delta, -30.0, 30.0))) / delta,
        1.0 - 0.5 * delta + delta * delta / 6.0,
        abs(delta) < 1e-2,
    );
    let tau = haze.x * d * exp(clamp(-(eye.y + anchor_y) * haze.y, -30.0, 30.0)) * g;
    return 1.0 - exp(-tau);
}
