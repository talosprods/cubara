//! Exponential height haze: the CPU reference for `shaders/haze.wgsl`.
//!
//! Extinction is `sigma(y) = density * exp(-y / scale_height)`; the optical
//! depth from the eye `e` to a point `p` at distance `d` is
//! `tau = density * d * exp(-e.y / H) * g(delta)` with `delta = (p.y - e.y) / H`
//! and `g(delta) = (1 - exp(-delta)) / delta`. `amount = 1 - exp(-tau)`.
//!
//! This function mirrors the WGSL line for line; the tests below run on the
//! CPU, so the model is pinned with no GPU. The three shader sources that
//! include the WGSL are defined here, once, as consts.

/// `haze.wgsl` prepended to `mesh.wgsl`.
pub(crate) const MESH_WGSL: &str = concat!(
    include_str!("shaders/haze.wgsl"),
    include_str!("shaders/mesh.wgsl")
);
/// `haze.wgsl` prepended to `far.wgsl`.
pub(crate) const FAR_WGSL: &str = concat!(
    include_str!("shaders/haze.wgsl"),
    include_str!("shaders/far.wgsl")
);
/// `haze.wgsl` prepended to `figure.wgsl`.
pub(crate) const FIGURE_WGSL: &str = concat!(
    include_str!("shaders/haze.wgsl"),
    include_str!("shaders/figure.wgsl")
);
/// `haze.wgsl` prepended to `trees.wgsl` (issue #332, R2).
pub(crate) const TREES_WGSL: &str = concat!(
    include_str!("shaders/haze.wgsl"),
    include_str!("shaders/trees.wgsl")
);

/// The atmosphere: how thick, and how fast it thins with height.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Haze {
    /// Extinction at `y = 0`, in blocks^-1.
    pub density: f32,
    /// Height over which the extinction falls by `e`, in blocks.
    pub scale_height: f32,
}

impl Default for Haze {
    fn default() -> Self {
        Self {
            density: 1.0 / 20_000.0,
            scale_height: 600.0,
        }
    }
}

impl Haze {
    /// Fraction of the haze colour covering `p` as seen from `eye`, `0..=1`.
    pub fn amount(&self, eye: glam::Vec3, p: glam::Vec3) -> f32 {
        let inv_h = 1.0 / self.scale_height;
        let d = p.distance(eye);
        let delta = (p.y - eye.y) * inv_h;
        let g = if delta.abs() < 1e-2 {
            1.0 - 0.5 * delta + delta * delta / 6.0
        } else {
            (1.0 - (-delta).clamp(-30.0, 30.0).exp()) / delta
        };
        let tau = self.density * d * (-eye.y * inv_h).clamp(-30.0, 30.0).exp() * g;
        1.0 - (-tau).exp()
    }

    /// The `FrameUniform::haze` packing: `[density, 1 / scale_height, 0, 0]`.
    pub fn uniform(&self) -> [f32; 4] {
        [self.density, 1.0 / self.scale_height, 0.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::vec3;

    const GROUND: glam::Vec3 = vec3(8.0, 40.0, 8.0);

    #[test]
    fn zero_distance_is_clear_bounded_and_monotonic_along_a_ray() {
        let h = Haze::default();
        assert_eq!(h.amount(GROUND, GROUND), 0.0);
        for eye_y in [-500.0, 40.0, 3000.0] {
            for slope in [-0.5f32, 0.0, 0.3, 2.0] {
                let eye = vec3(0.0, eye_y, 0.0);
                let dir = vec3(1.0, slope, 0.0).normalize();
                let mut prev = 0.0;
                for i in 0..200 {
                    let a = h.amount(eye, eye + dir * (i as f32 * 1500.0));
                    assert!((0.0..=1.0).contains(&a), "{a} out of range");
                    assert!(a >= prev - 1e-6, "{a} < {prev}: not monotonic");
                    prev = a;
                }
            }
        }
    }

    #[test]
    fn it_is_continuous_through_the_horizontal_and_finite_everywhere() {
        let h = Haze::default();
        let eye = vec3(0.0, 0.0, 0.0);
        // About one optical depth, where haze is most sensitive to g.
        let d = 1.0 / h.density;
        let rise = |delta: f32| delta * h.scale_height;
        let flat = h.amount(eye, eye + vec3(d, 0.0, 0.0));
        let almost = h.amount(eye, eye + vec3(d, rise(1e-6), 0.0));
        assert!((almost - flat).abs() < 1e-5, "{almost} vs {flat}");
        // Both sides of the series threshold (delta = 1e-2).
        let below = h.amount(eye, eye + vec3(d, rise(0.9998e-2), 0.0));
        let above = h.amount(eye, eye + vec3(d, rise(1.0002e-2), 0.0));
        assert!((above - below).abs() < 2e-6, "{above} vs {below}");

        for ey in [-1e6f32, 0.0, 1e7] {
            for py in [-1e6f32, 0.0, 1e7] {
                for d in [0.0f32, 1e9] {
                    let a = h.amount(vec3(0.0, ey, 0.0), vec3(d, py, 0.0));
                    assert!(
                        a.is_finite() && (0.0..=1.0).contains(&a),
                        "{ey} {py} {d}: {a}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_near_world_stays_clear() {
        let a = Haze::default().amount(GROUND, GROUND + vec3(1024.0, 0.0, 0.0));
        assert!(a < 0.05, "{a} at 1024 blocks");
    }

    #[test]
    fn it_is_an_atmosphere_not_a_wall() {
        let a = Haze::default().amount(GROUND, GROUND + vec3(50_000.0, 0.0, 0.0));
        assert!((0.5..0.95).contains(&a), "{a} at 50 km");
    }

    #[test]
    fn higher_sees_farther() {
        let h = Haze::default();
        let far = vec3(100_000.0, 0.0, 0.0);
        let low = h.amount(vec3(0.0, 40.0, 0.0), vec3(0.0, 40.0, 0.0) + far);
        let high = h.amount(vec3(0.0, 3000.0, 0.0), vec3(0.0, 3000.0, 0.0) + far);
        assert!(high < low, "{high} !< {low}");
    }

    #[test]
    fn the_distance_is_the_straight_line_not_the_horizontal_one() {
        // Straight down 20 km: no horizontal travel at all, through air that
        // thickens by e^10 over the way.
        let a = Haze::default().amount(GROUND, GROUND - vec3(0.0, 20_000.0, 0.0));
        assert!(a > 0.99, "{a}");
    }

    #[test]
    fn the_uniform_packs_density_and_inverse_scale_height() {
        let u = Haze::default().uniform();
        assert_eq!(u, [1.0 / 20_000.0, 1.0 / 600.0, 0.0, 0.0]);
    }
}
