// ARCHITECTURE.md Rule 2: no ambient state.
// Trips check-architecture.sh's `static ...: (OnceLock|Mutex|RwLock|LazyLock)`
// grep -- which matches the bare type name right after the colon, the way it
// is written once `use std::sync::OnceLock;` is in scope, same as the real
// offending shape would be.
use std::sync::OnceLock;
static CACHE: OnceLock<u32> = OnceLock::new();
