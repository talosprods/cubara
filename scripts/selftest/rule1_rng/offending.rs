// ARCHITECTURE.md Rule 1: no unseeded RNG in a simulation crate.
// Trips check-architecture.sh's `thread_rng|random\(\)` grep.
fn offending() {
    let _ = rand::thread_rng();
}
