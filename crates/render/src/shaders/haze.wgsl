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
fn haze_amount(eye: vec3<f32>, p: vec3<f32>, haze: vec4<f32>) -> f32 {
    let d = distance(p, eye);
    let delta = (p.y - eye.y) * haze.y;
    let g = select(
        (1.0 - exp(clamp(-delta, -30.0, 30.0))) / delta,
        1.0 - 0.5 * delta + delta * delta / 6.0,
        abs(delta) < 1e-2,
    );
    let tau = haze.x * d * exp(clamp(-eye.y * haze.y, -30.0, 30.0)) * g;
    return 1.0 - exp(-tau);
}
