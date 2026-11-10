// ARCHITECTURE.md Rule 3/4: no GPU/windowing types in a data or simulation
// crate. Trips check-architecture.sh's `\bwgpu::|\bwinit::` grep. (Never
// compiled -- the self-test only greps source text.)
fn offending(_device: wgpu::Device) {}
