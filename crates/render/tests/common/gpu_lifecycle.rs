//! One lock around every `wgpu::Instance`/adapter/device creation and
//! teardown in a test binary -- shared by every GPU integration test in this
//! crate (`#[path = "common/gpu_lifecycle.rs"] mod gpu_lifecycle;`), so there
//! is one copy of it (`ARCHITECTURE.md` Rule 5), not one per test file that
//! happens to have remembered the bug.
//!
//! Cargo runs a binary's `#[test]` fns on several threads by default, and each
//! GPU test independently opens and drops a Vulkan instance+device. That races
//! against known thread-safety bugs in the Khronos Vulkan Loader's own
//! instance/device bookkeeping -- see
//! <https://github.com/KhronosGroup/Vulkan-Loader/issues/200> ("Race condition
//! in vkCreateDevice()+vkDestroyDevice()+vkGetDeviceProcAddr()"), fixed
//! upstream the same way this does: a lock around create/destroy.
//!
//! Seen twice in this repo, both times as an intermittent SIGSEGV with no
//! test logic involved:
//!
//! - Issue #265, `golden.rs` on an NVIDIA driver: two independent crashes
//!   faulting at the identical code offset inside `libvulkan.so.1.4.341`
//!   (not wgpu, not the driver, not this crate), from different tests --
//!   the crashing thread was incidental. `--test-threads=1`, `--release`
//!   and `gdb` all pass every time; the race needs the debug build's timing
//!   and a free-running scheduler. That is why this cannot be pinned down
//!   with a debugger, and why the lock went in.
//! - Issue #412, `tree_culling.rs` on CI's lavapipe: SIGSEGV 0.76 s after
//!   `running 3 tests`, before any of the three had finished -- three
//!   threads inside instance/device creation at once. That binary's tests
//!   never issue a draw (`TreeArena::prepare` only uploads instance
//!   buffers), so the suspected shader read (`node_origins[node_index]`)
//!   never executed. It had the same three-threads-three-devices shape
//!   `golden.rs` had before #265, without the lock. Not reproduced locally
//!   in 400 plain runs, 150 under gdb and 3,600 runs under scheduling
//!   pressure (4-6 copies pinned to one or two CPUs) -- consistent with
//!   #265's own history.
//!
//! This is a test-harness lock, not simulation state: `ARCHITECTURE.md` Rule
//! 2 ("no ambient state") governs the engine and render crates, not a fixture
//! working around a driver bug, and `scripts/check-architecture.sh` excludes
//! this one file by name for exactly that reason. Hold the guard for the
//! whole lifetime of the device it protects -- declare it *first* in the
//! test, so it is dropped *last*, after the device and queue -- and keep the
//! CPU-bound work (meshing) outside it where that is possible.

use std::sync::{Mutex, MutexGuard};

static GPU_LIFECYCLE: Mutex<()> = Mutex::new(());

/// Take the lifecycle lock. A panic in another test while it held the lock
/// poisons it; that test has already failed, so the lock is still usable.
pub fn lock() -> MutexGuard<'static, ()> {
    GPU_LIFECYCLE.lock().unwrap_or_else(|e| e.into_inner())
}
