#!/usr/bin/env bash
# Proves `check-architecture.sh` and `check-single-render-path.sh` still catch
# what they were written for.
#
# They are greps. A regex edit that silently stops matching -- a typo, a
# refactor that renames the thing being matched, an exclusion that grew too
# wide -- turns the `architecture rules` CI job green forever, with nobody
# finding out until the violation it existed to stop lands for real. CLAUDE.md
# names this directly: "a regex edit that silently stops matching turns the
# `architecture rules` job green forever."
#
# HOW IT WORKS
#
# For every rule block in both scripts there is a fixture under
# `scripts/selftest/<rule>/` -- the smallest snippet that violates it. For each
# one, this builds a synthetic tree shaped like the crate layout the greps
# expect (`crates/<crate>/src/...`, a `Cargo.toml` where a rule reads one),
# splices the fixture into the one file the rule under test reads, and runs
# the real check script against that tree via `CUBARA_CHECK_ROOT` -- no
# copying the scripts themselves, which is what that variable is for. The
# fixture must make the check fail, naming the rule it exists to test; if it
# doesn't, the rule's regex has drifted and nothing else would have noticed.
#
# Every fixture tree is otherwise clean -- nothing else in it should trip any
# other rule -- so a failure that isn't the expected one is itself a finding
# (the skeleton drifted from what the real scripts expect, which is worth
# knowing too).
#
# Once every fixture has been shown to trip its rule, both scripts are run
# against the real repository tree and must pass: a self-test that only ever
# runs against synthetic trees could not tell you the real tree is clean.
#
# NOT exhaustive: a rule's regex could still accept a shape distinct from its
# fixture while retaining the fixture's own shape. One fixture per rule is the
# contract (CLAUDE.md "the owner's call is to add a fixture when a rule's
# regex changes"), not a proof the regex is complete.
set -uo pipefail
cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
ARCH_SCRIPT="$REPO_ROOT/scripts/check-architecture.sh"
SRP_SCRIPT="$REPO_ROOT/scripts/check-single-render-path.sh"
FIXTURES="$REPO_ROOT/scripts/selftest"

tripped=0
missed=0
missed_names=""

# A synthetic tree where every rule in both scripts passes. Shaped exactly
# like the crates the real scripts read, so splicing one fixture in is the
# only difference from a tree that passes clean.
build_skeleton() {
    local root="$1"
    for c in voxel world sim server; do
        mkdir -p "$root/crates/$c/src"
        printf '// selftest: nothing here trips any rule.\npub fn nothing() {}\n' \
            >"$root/crates/$c/src/lib.rs"
        printf '[package]\nname = "cubara-%s"\nversion = "0.0.1"\n\n[dependencies]\n' "$c" \
            >"$root/crates/$c/Cargo.toml"
    done
    mkdir -p "$root/crates/render/src/shaders"
    printf '// selftest: no input/gameplay functions here.\npub fn draw() {}\n' \
        >"$root/crates/render/src/render.rs"
    printf '// selftest: the one legitimate wgpu::Instance::new call site.\npub fn new_instance() {\n    let _ = wgpu::Instance::new(Default::default());\n}\n' \
        >"$root/crates/render/src/instance.rs"
    printf '// selftest: the one legitimate encode_scene/build_pipeline/begin_render_pass site.\npub fn encode_scene() {}\npub fn build_pipeline() {}\nfn use_it() {\n    build_pipeline();\n}\n' \
        >"$root/crates/render/src/scene.rs"
    printf '// selftest: arena.rs is the one file allowed an indirect draw.\nfn offending(pass: &mut wgpu::RenderPass, buf: &wgpu::Buffer) {\n    pass.draw_indirect(buf, 0);\n}\n' \
        >"$root/crates/render/src/arena.rs"
    printf '// selftest: a clean shader, no builtin reads.\n' \
        >"$root/crates/render/src/shaders/mesh.wgsl"
    printf '[package]\nname = "cubara-render"\nversion = "0.0.1"\n\n[dependencies]\nwgpu = "0"\nwinit = "0"\n' \
        >"$root/crates/render/Cargo.toml"
}

# Append $2 (a fixture file's content) to $1 (a path inside the skeleton),
# stripping `//`-comment lines first.
#
# A fixture's own header comment documents which grep it trips, which means
# it names the pattern -- e.g. `Instant::now` -- in plain text. Left in, that
# comment line would itself satisfy a *broken* version of the same grep (one
# that no longer matches the real offending line below it) just as well as
# the code does, which would make this self-test blind to exactly the
# regression it exists to catch. Stripping comments means only the code a
# rule is meant to catch can make it trip.
splice() {
    grep -v '^[[:space:]]*//' "$2" >>"$1"
}

# $1 fixture dir name, $2 script var (ARCH_SCRIPT or SRP_SCRIPT), $3 target
# path relative to the skeleton root, $4 fixture file name, $5 expected
# substring of the FAIL message.
check_fixture() {
    local name="$1" script="$2" target="$3" fixture="$4" expect="$5"
    local root out status
    root="$(mktemp -d)"
    build_skeleton "$root"
    splice "$root/$target" "$FIXTURES/$name/$fixture"

    out=$(CUBARA_CHECK_ROOT="$root" "$script" 2>&1)
    status=$?
    rm -rf "$root"

    if [ "$status" -eq 0 ]; then
        echo "MISSED   $name -- the check passed on a tree containing its own fixture"
        missed=$((missed + 1))
        missed_names="$missed_names $name(no-fail)"
        return
    fi
    if ! printf '%s' "$out" | grep -qF "$expect"; then
        echo "MISSED   $name -- failed, but not with the expected message (\"$expect\")"
        printf '%s\n' "$out" | sed 's/^/           /'
        missed=$((missed + 1))
        missed_names="$missed_names $name(wrong-message)"
        return
    fi
    echo "tripped  $name"
    tripped=$((tripped + 1))
}

echo "Architecture self-test -------------------------------------------------"
echo

check_fixture rule1_wall_clock "$ARCH_SCRIPT" crates/sim/src/lib.rs offending.rs "wall-clock time in a simulation crate"
check_fixture rule1_rng "$ARCH_SCRIPT" crates/sim/src/lib.rs offending.rs "unseeded RNG in a simulation crate"
check_fixture rule1_trig "$ARCH_SCRIPT" crates/sim/src/lib.rs offending.rs "floating-point trigonometry in a simulation crate"
check_fixture rule2_static "$ARCH_SCRIPT" crates/server/src/lib.rs offending.rs "global mutable state"
check_fixture rule3_4_cargo_dep "$ARCH_SCRIPT" crates/sim/Cargo.toml offending.toml "GPU/windowing dependency in a data or simulation crate"
check_fixture rule3_4_type_use "$ARCH_SCRIPT" crates/sim/src/lib.rs offending.rs "GPU/windowing types in a data or simulation crate"
check_fixture rule3_input_on_renderer "$ARCH_SCRIPT" crates/render/src/render.rs offending.rs "input/gameplay on the renderer"
check_fixture rule3_render_world_dep "$ARCH_SCRIPT" crates/render/Cargo.toml offending.toml "cubara-render depends on cubara-world"
check_fixture rule8_static_world "$ARCH_SCRIPT" crates/server/src/lib.rs offending.rs "a process-wide World"
check_fixture rule8_world_accessor "$ARCH_SCRIPT" crates/server/src/lib.rs offending.rs "implicit 'the world' accessor"
check_fixture rule5_instance "$ARCH_SCRIPT" crates/render/src/scene.rs offending.rs "made outside cubara_render::new_instance"
check_fixture rule5_indirect_draw "$ARCH_SCRIPT" crates/render/src/scene.rs offending.rs "indirect draw outside arena.rs"
check_fixture rule5_builtin_read "$ARCH_SCRIPT" crates/render/src/shaders/mesh.wgsl offending.wgsl "reads vertex_index/instance_index"
check_fixture single_render_path_build_pipeline "$SRP_SCRIPT" crates/render/src/render.rs offending.rs "build_pipeline"
check_fixture single_render_path_begin_render_pass "$SRP_SCRIPT" crates/render/src/render.rs offending.rs "begin_render_pass"

echo
echo "The real tree ----------------------------------------------------------"
echo

real_fail=0
if ! "$ARCH_SCRIPT"; then
    echo "the real tree fails check-architecture.sh -- fix that before trusting this self-test"
    real_fail=1
fi
if ! "$SRP_SCRIPT"; then
    echo "the real tree fails check-single-render-path.sh -- fix that before trusting this self-test"
    real_fail=1
fi

echo
echo "$tripped/$((tripped + missed)) fixtures tripped their rule."
if [ "$missed" -gt 0 ]; then
    echo "Missed:$missed_names"
    echo "A rule's grep no longer catches the shape it exists to catch."
fi

if [ "$missed" -gt 0 ] || [ "$real_fail" -ne 0 ]; then
    exit 1
fi
echo "OK: every rule's fixture trips it, and the real tree is clean."
