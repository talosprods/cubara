#!/usr/bin/env bash
# What to work on next.
#
# An autonomous session should never have to decide which piece of work comes
# next -- that is a judgement, and judgements drift. Ordering lives in the issue
# titles, which carry their ROADMAP.md block number, and this script reads it.
#
# Usage:  ./scripts/next-block.sh [phase]      (default: the active phase)
#
# With no argument the phase is the ACTIVE one. CLAUDE.md's "go ahead" procedure
# calls this with no argument, so the default must not be a fixed phase: when it
# was phase 1, a session got "No open blocks left in Phase 1" while phase 3 was
# active and full of open issues. The active phase is found, in order, by:
#   1. ROADMAP.md's phase table: exactly one row whose status reads **Active.
#   2. Otherwise the lowest phase whose GitHub milestone has open issues.
# The script prints one line saying which rule chose the phase.
set -euo pipefail
cd "$(dirname "$0")/.."

milestone_for() {
    case "$1" in
        1) echo "Phase 1 — The engine stands" ;;
        2) echo "Phase 2 — The first survival world" ;;
        3) echo "Phase 3 — The world" ;;
    esac
}

if ! command -v gh >/dev/null 2>&1; then
    echo "gh (GitHub CLI) is not installed -- it is how work is tracked here." >&2
    exit 2
fi

chosen=""
if [ $# -ge 1 ]; then
    phase="$1"
    case "$phase" in
        1|2|3) ;;
        *) echo "unknown phase: $phase (expected 1, 2 or 3)" >&2; exit 2 ;;
    esac
else
    active_rows=$(grep -E '^\| \*\*[0-9]+\*\* \|' ROADMAP.md | grep -F '**Active' || true)
    active_count=$(grep -c . <<<"$active_rows" || true)
    if [ "$active_count" -eq 1 ]; then
        phase=$(sed -E 's/^\| \*\*([0-9]+)\*\*.*/\1/' <<<"$active_rows")
        chosen="(active phase $phase, from ROADMAP.md)"
    else
        phase=""
        for p in 1 2 3; do
            n=$(gh issue list --milestone "$(milestone_for "$p")" --state open \
                    --limit 100 --json number --jq 'length')
            if [ "$n" -gt 0 ]; then
                phase="$p"
                break
            fi
        done
        if [ -n "$phase" ]; then
            chosen="(active phase $phase, lowest phase with open issues; ROADMAP.md has no single Active row)"
        else
            phase=3
            chosen="(no phase has open issues; ROADMAP.md has no single Active row; defaulting to phase 3)"
        fi
    fi
fi

milestone=$(milestone_for "$phase")
if [ -n "$chosen" ]; then
    echo "$chosen"
fi

# A milestone name that does not exist on GitHub must be a loud error, not
# "no open blocks left" -- found while planning phase 3 (#339): the milestone
# was renamed from "Phase 3 — Modern depth, with synergy" to "Phase 3 — The
# world", this script still had the old name, and `gh issue list --milestone
# <a name that does not exist>` silently returns nothing and exits 0. An
# autonomous session told to "go ahead" would have concluded the phase was
# finished before block 3.1 had even started.
#
# `state=all` (not just open): a *finished* phase's milestone is closed, and
# that must still count as existing -- only a genuinely wrong name (a typo, or
# a rename this script was not updated for) is the error this guards against.
all_milestones=$(gh api 'repos/{owner}/{repo}/milestones?state=all' --paginate --jq '.[].title')
if ! grep -qxF "$milestone" <<<"$all_milestones"; then
    cat <<EOF >&2
error: no milestone named "$milestone" (phase $phase) exists on GitHub.
Milestones that do exist:
EOF
    sed 's/^/  - /' <<<"$all_milestones" >&2
    exit 2
fi

# One row per open issue: title, "#N", and whether it carries the `blocked`
# label (an issue waiting on the project owner's answer -- CLAUDE.md: "Nobody
# waits on a question"). Sorted by title so the block number in the title
# drives the order, as it always has.
issues=$(gh issue list --milestone "$milestone" --state open --limit 100 \
        --json number,title,labels \
        --jq '.[] | [.title, "#\(.number)", (if (.labels | any(.name == "blocked")) then "blocked" else "unblocked" end)] | @tsv' \
        | sort -V)

if [ -z "$issues" ]; then
    cat <<EOF
No open blocks left in "$milestone".

That is not the same as the phase being finished. Run the gate:

    ./scripts/check-phase-gate.sh $phase

on **both** machines, record the BENCHMARKS.md rows, and report to the project
owner. Do not start the next phase.
EOF
    exit 0
fi

unblocked=$(awk -F'\t' '$3 == "unblocked" { print $1 "\t" $2 }' <<<"$issues")
blocked=$(awk -F'\t' '$3 == "blocked" { print $1 "\t" $2 }' <<<"$issues")

if [ -z "$unblocked" ]; then
    cat <<EOF
No unblocked issues left in "$milestone" -- every remaining one is waiting for
the owner:

$(sed 's/^/    /' <<<"$blocked")

That is not the same as the phase being finished, and it is not a reason to
sit idle: send the question to the contact session (CLAUDE.md) and pick up
other work. Re-run this once an answer unblocks one of the above.
EOF
    exit 0
fi

next=$(head -1 <<<"$unblocked")
rest=$(tail -n +2 <<<"$unblocked")

cat <<EOF
Phase $phase — next block:

    ${next}

Before writing code:
  - ROADMAP.md, the phase $phase block table -- what this block is for
  - docs/PHASE${phase}_ARCHITECTURE.md -- the decisions it implements, already made
  - the issue itself: its Design decisions section is binding, not advisory

Then: branch, implement, test, PR, wait for CI, merge. One block, one arc.

Remaining after this one:
EOF
sed 's/^/    /' <<<"$rest"

if [ -n "$blocked" ]; then
    cat <<EOF

Waiting for the owner (not offered as next -- CLAUDE.md: "Nobody waits on a
question"):
EOF
    sed 's/^/    /' <<<"$blocked"
fi
