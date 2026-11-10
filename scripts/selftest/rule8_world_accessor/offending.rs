// ARCHITECTURE.md Rule 8: no implicit "the world" accessor. Trips
// check-architecture.sh's `fn (the_world|current_world|global_world|world_instance)` grep.
fn current_world() {}
