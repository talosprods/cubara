// ARCHITECTURE.md Rule 8: no process-wide World. Trips check-architecture.sh's
// `static ...: ...World|thread_local!.*World` grep.
static WORLD: std::sync::Mutex<cubara_sim::World> = std::sync::Mutex::new(cubara_sim::World::new());
