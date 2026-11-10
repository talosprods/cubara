// ARCHITECTURE.md Rule 1: no wall-clock time in a simulation crate.
// Trips check-architecture.sh's `Instant::now|SystemTime::now` grep.
fn offending() {
    let _ = std::time::Instant::now();
}
