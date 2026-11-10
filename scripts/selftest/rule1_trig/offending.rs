// ARCHITECTURE.md Rule 1: no floating-point trigonometry in a simulation
// crate. Trips check-architecture.sh's sin/cos/tan/... grep.
fn offending(yaw: f32) -> f32 {
    yaw.sin()
}
