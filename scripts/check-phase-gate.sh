#!/usr/bin/env bash
# ROADMAP.md's exit gate, answered by a script instead of a judgement call.
#
# One line per criterion, PASS or FAIL, exactly the list under "Exit gate" for
# the given phase in ROADMAP.md -- no more, no fewer. A criterion whose test
# does not exist yet prints FAIL (not implemented); it never prints PASS and is
# never skipped. As each later phase-1 block lands its own test (determinism
# harness, golden images, save round-trip, ...), it should wire its criterion in
# here rather than leaving it a permanent "not implemented".
#
# Usage:  ./scripts/check-phase-gate.sh <phase>
set -uo pipefail
cd "$(dirname "$0")/.."

# What CI sets for the entire workflow (.github/workflows/ci.yml). Exported
# rather than passed per-command so that every cargo invocation below is the one
# CI runs -- including the ones added later by someone not reading this comment.
export RUSTFLAGS="${RUSTFLAGS:--D warnings}"

phase="${1:-1}"

pass=0
fail=0
out=$(mktemp)
trap 'rm -f "$out"' EXIT

# run <label> <cmd...> -- PASS if <cmd> exits 0, FAIL (with output) otherwise.
run() {
    local label="$1"
    shift
    # GATE_ONLY_TESTS=1 skips the whole-workspace, lint and perf lines and runs
    # only the named-test lines (run_n / golden_run) -- the fast way to check a
    # rename did not orphan a gated test.
    if [ -n "${GATE_ONLY_TESTS:-}" ]; then
        return
    fi
    if "$@" >"$out" 2>&1; then
        echo "PASS  $label"
        pass=$((pass + 1))
    else
        echo "FAIL  $label"
        sed 's/^/      /' "$out" | tail -20
        fail=$((fail + 1))
    fi
}

# run_n <label> <count> <cmd...> -- a *named* test, or a named test file: PASS
# only if <cmd> exits 0 AND exactly <count> tests passed. `cargo test <filter>`
# exits 0 on "running 0 tests", so a gated test that was renamed or deleted made
# its criterion print PASS having run nothing -- the test audit of 2026-11 found
# every gated line had that hole. Now a rename fails the gate, loudly, with the
# count it saw. Raise <count> in the same change that adds a test to a gated file.
#
# The count is summed over every "test result:" line, because a package with a
# lib, bins and integration tests prints one per target.
sum_field() {
    # sum_field <word>: the total of "N <word>" over the "test result:" lines.
    grep -E '^test result:' "$out" \
        | sed -E "s/.* ([0-9]+) $1.*/\1/" \
        | awk '{ n += $1 } END { print n + 0 }'
}

run_n() {
    local label="$1" want="$2"
    shift 2
    local status=0 got failed
    "$@" >"$out" 2>&1 || status=$?
    got=$(sum_field passed)
    failed=$(sum_field failed)
    if [ "$status" -eq 0 ] && [ "$got" -eq "$want" ] && [ "$failed" -eq 0 ]; then
        echo "PASS  $label ($got passed)"
        pass=$((pass + 1))
    else
        if [ "$status" -eq 0 ] && [ "$got" -ne "$want" ]; then
            echo "FAIL  $label (expected $want tests to pass, $got did -- renamed or deleted?)"
        else
            echo "FAIL  $label"
        fi
        sed 's/^/      /' "$out" | tail -20
        fail=$((fail + 1))
    fi
}

# golden_run <label> <count> <cmd...> -- run_n, plus the line the golden tests
# cannot give on their own. They SKIP (and pass) on a machine with no GPU
# adapter, which is right for a CI runner and wrong for the person deciding a
# phase is finished: a gate that compared zero images is not a gate. So this
# counts the "golden <name>: N% differ" lines the tests print when they really
# compared, prints "golden: compared N images", and FAILS if any were skipped.
golden_run() {
    local label="$1" want="$2"
    shift 2
    local compared skipped before=$fail
    run_n "$label" "$want" "$@"
    compared=$(grep -cE '^golden [^:]+: [0-9.]+% differ' "$out")
    skipped=$(grep -cE '^SKIPPED ' "$out")
    echo "      golden: compared $compared images, $skipped skipped"
    if [ "$fail" -eq "$before" ] && { [ "$compared" -lt "$want" ] || [ "$skipped" -gt 0 ]; }; then
        echo "FAIL  $label (the goldens did not all actually run: no GPU adapter on this machine?)"
        pass=$((pass - 1))
        fail=$((fail + 1))
    fi
}

# perf_gate <label> -- the 1000-FPS criterion. Since 2026-09-26 it measures
# what the player sees: the owner's three first-person eyes (ground, hill,
# flight) at the game's own view distance, each at 1000 FPS or more --
# `--bench gate`, whose GATE line is parsed rather than its exit code trusted,
# so a crash with no GATE line is a FAIL with the output shown, not a PASS.
perf_gate() {
    local label="$1" bench_out line
    if [ -n "${GATE_ONLY_TESTS:-}" ]; then
        return
    fi
    bench_out=$(cargo run --release -- --bench gate 2>&1 || true)
    line=$(printf '%s\n' "$bench_out" | grep -oE 'GATE: .*' | tail -1)
    case "$line" in
        "GATE: MET "*)
            echo "PASS  $label (${line#GATE: MET | })"
            pass=$((pass + 1))
            ;;
        "GATE: NOT MET "*)
            echo "FAIL  $label (${line#GATE: NOT MET | })"
            fail=$((fail + 1))
            ;;
        *)
            echo "FAIL  $label (no GATE line -- see output below)"
            printf '%s\n' "$bench_out" | tail -20 | sed 's/^/      /'
            fail=$((fail + 1))
            ;;
    esac
}

# not_implemented <label> -- always FAIL, explicitly, for a criterion whose test
# does not exist in the repo yet.
not_implemented() {
    echo "FAIL  $1 (not implemented)"
    fail=$((fail + 1))
}

if [ "$phase" != "1" ] && [ "$phase" != "2" ]; then
    echo "phase $phase's gate is not wired up yet -- only phases 1 and 2 are." >&2
    echo "(ROADMAP.md's phase $phase exit gate exists on paper; this script doesn't answer it yet.)" >&2
    exit 2
fi

if [ "$phase" = "2" ]; then
    echo "Phase 2 exit gate (ROADMAP.md) ------------------------------------------"
    echo
    echo "  Wired up while block 2.7a was landing, deliberately RED: the point of a"
    echo "  gate is to say what is missing before the work is done, which is what"
    echo "  block 1.0 established when it shipped phase 1's gate failing."
    echo

    # "Everything phase 1's gate checks, still passing -- a perf regression
    # blocks the phase (Rule 7), it is not noted and forgotten."
    run "cargo test --all" cargo test --all
    run "cargo clippy --workspace --all-targets --all-features" cargo clippy --workspace --all-targets --all-features
    run "cargo fmt --all --check" cargo fmt --all --check
    run "architecture rules (check-architecture.sh)" ./scripts/check-architecture.sh
    run "single render path (check-single-render-path.sh)" ./scripts/check-single-render-path.sh

    perf_gate "phase 1 perf holds"

    run_n "determinism replay still passes single- vs multi-threaded (the hash reducer at 1 and 6 workers; the tick itself has no threaded path)" 1 \
        cargo test -p cubara-sim --test determinism the_fixture_reaches_a_known_hash_regardless_of_worker_count

    # "The dormant test: a chunk left dormant for N ticks and then activated
    # ends in the same state as one simulated continuously for N ticks."
    # Landed in block 2.6 (#47); the bounded path from 2.7a (#58) is checked
    # against its own one-tick reference.
    run_n "dormant test: a chunk dormant for N ticks == one ticked continuously" 1 \
        cargo test -p cubara --bin cubara a_dormant_chunk_ends_where_a_continuously_ticked_one_would
    run_n "catch-up matches a tick-by-tick reference (bounded path, 2.7a)" 1 \
        cargo test -p cubara-world --lib bounded_catch_up_agrees_with_the_one_tick_reference

    # "The survival replay test: a fixed, scripted input sequence runs headlessly
    # and completes the loop -- chop a tree, craft a tool, mine iron ore, smelt
    # it, take damage -- then asserts a world-state hash. It runs single-threaded
    # and multi-threaded and must agree."
    #
    # "Eat" is not in that list any more: on 2026-09-05 the project owner deferred
    # hunger, food and hostile mobs (block 2.9b) out of phase 2, and ROADMAP.md's
    # gate was amended in the open in the same change -- moving a gate is the
    # owner's call, and this is the record of them making it. See
    # docs/PHASE2_ARCHITECTURE.md section 9. Damage is a scripted fall, which
    # since block 2.9a is the one thing in this world that hurts.
    #
    # Three tests rather than one: the loop and its pinned hash at one worker and
    # at six, the same script run twice, and the register of stations that stops
    # an empty run passing by hashing a world in which nothing happened.
    run_n "survival replay test (scripted headless loop, asserts a world-state hash)" 3         cargo test -p cubara-server --test survival_replay

    # "The round-trip test, extended from phase 1 to cover phase 2's state: save
    # the world mid-script, reload it, run the rest of the script, and land on
    # the same hash as the uninterrupted run."
    #
    # Landed in block 2.8 (#170). The test asserts the reloaded state is
    # non-empty before comparing hashes, so it cannot pass by both runs being
    # equally empty.
    run_n "round-trip covering phase 2 state (block entities, entities, inventory)" 1 \
        cargo test -p cubara-sim --test save_load a_world_saved_mid_script_finishes_where_an_uninterrupted_one_does

    # Multiplayer, moved into phase 2 by the owner on 2026-09-05 (ROADMAP.md).
    # Red on landing, exactly as this file's survival criterion was: a gate says
    # what is missing before the work is done. Blocks 2.10-2.17, designed in
    # docs/PHASE2_MULTIPLAYER.md.
    run_n "two clients, one world, one hash (block 2.11)" 1         cargo test -p cubara-server --test client_view two_clients_in_one_world_leave_one_authoritative_state
    run_n "the server never sends terrain (block 2.11)" 1         cargo test -p cubara-server --test client_view the_join_handshake_carries_no_terrain
    run_n "bandwidth per client does not grow with player count (block 2.11)" 1         cargo test -p cubara-server --test client_view bytes_to_one_client_do_not_grow_with_the_player_count
    run_n "a real socket, two processes (block 2.12)" 6         cargo test -p cubara-server --test two_processes
    run_n "a region simulated elsewhere lands where it would have locally (block 2.16)" 1 \
        cargo test -p cubara-server --test sharding a_region_simulated_elsewhere_lands_where_it_would_have_locally

    echo
    echo "$pass passed, $fail failed."
    if [ "$fail" -gt 0 ]; then
        echo
        echo "Phase 2 is NOT finished. See the FAIL lines above."
        exit 1
    fi
    echo "OK: phase 2 exit gate met."
    exit 0
fi

echo "Phase 1 exit gate (ROADMAP.md) ------------------------------------------"
echo

run "cargo test --all" cargo test --all
# `--workspace`, matching .github/workflows exactly. It did not, until a PR
# went red on a lint the gate had just called clean: the toolchain pin makes
# local and CI agree about which lints *exist*, and it cannot help when the two
# are not running the same command. A gate that checks less than CI is a gate
# that lies.
#
# And it went on lying, because matching the *command* is not enough: CI sets
# `RUSTFLAGS: -D warnings` for the whole workflow, so a lint that is a warning
# here is an error there. A second PR went red on `needless_borrows_for_generic_args`
# that this script had just called clean. `RUSTFLAGS` is exported at the top of
# this file now, so local and CI disagree about nothing.
run "cargo clippy --workspace --all-targets --all-features" cargo clippy --workspace --all-targets --all-features
run "cargo fmt --all --check" cargo fmt --all --check
run "architecture rules (check-architecture.sh)" ./scripts/check-architecture.sh
run "single render path (check-single-render-path.sh)" ./scripts/check-single-render-path.sh

perf_gate "1000 FPS at every gate eye"

run_n "determinism replay test: single- vs multi-threaded, identical world-state hash (the hash reducer at 1 and 6 workers; the tick itself has no threaded path)" 1 \
    cargo test -p cubara-sim --test determinism the_fixture_reaches_a_known_hash_regardless_of_worker_count
golden_run "golden images: all three block types textured, a cave mouth, an LOD boundary" 3 \
    cargo test -p cubara-render --test golden -- distinct_materials_render_with_distinct_textures a_cave_mouth_is_visible no_crack_at_a_real_lod_boundary --nocapture
run_n "unit test: a player AABB never passes through a solid voxel" 1 \
    cargo test -p cubara-sim --lib never_tunnels_through_the_floor_across_a_spread_of_velocities
run_n "unit test: the same seed produces a bit-identical chunk on both platforms" 1 \
    cargo test -p cubara-world --lib cross_platform_block_array_is_bit_identical
run_n "isolation test: a chunk generated alone == the same chunk after shuffled neighbours" 1 \
    cargo test -p cubara-world --lib generation_is_isolated_from_neighbours
# The "identically on Windows and macOS" half isn't this one command's job --
# it's the same fixture-hash test passing in CI on both windows-latest and
# macos-latest runners (every merged PR's required checks), not something a
# single local run can prove either way.
run_n "save round-trip test, plus a fixture world hashing identically on Windows and macOS" 1 \
    cargo test -p cubara-sim --test save_load the_committed_fixture_loads_to_a_known_hash

echo
echo "$pass passed, $fail failed."
if [ "$fail" -gt 0 ]; then
    exit 1
fi
echo "OK: phase 1 exit gate met."
