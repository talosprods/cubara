<!--
For the lead, not part of the comment. Delete this block before posting.

- A and B are engineering readings, confirmed by the second-model review.
- C, D, E and F are YOUR decisions. The text below is what the plan proposes
  ("lead decision, proposed" in plans/380.md §3). Change the wording here if you
  decide otherwise; the executor follows what is on #380, not the plan.
- The one sentence most likely to need your edit is D's "If the flips drop" rule.
- Post with: gh issue comment 380 --body-file <this file, block removed>
-->

## Amended 2026-11-11, before implementation: two readings made exact, trees, and what the pop test measures

Nothing above is deleted. This follows the plan and its review of 2026-11-11.
"S2" (review finding S2 of #407: the lift lives in `NodeSlot` and goes through one arena
method that `insert` and `rebase` both apply, `docs/PHASE3_ARCHITECTURE.md` §7) binds and
is followed.

**A. Decision 3, made per node.** *"Only batches that carry Removes fade, and only their
new ids rise"* is replaced by:

- The streaming caller states, per node, whether it is part of a swap and the swap's Δ.
- Each arriving mesh carries `rise`: Δ when it replaces resident nodes (its parent or its
  children), else 0.
- Each leaving id carries `sink`: Δ when something visible replaces it (its children or
  its parent), else 0.
- Δ is `cell_height_blocks()` of the coarser level, computed in `cubara-app`.
- An Insert for an id that is already resident and not fading never fades, whatever
  `rise` says. That is an edit (`NodeStreaming::invalidate`).

*Why:* a batch is not a swap. The children of one split arrive over several frames,
mostly in batches with no Remove. The parent's Remove rides the batch of whichever child
completes the set. A batch's Inserts also include new edge nodes and edit re-meshes. Read
literally, the rule makes an arbitrary subset rise, including edge nodes that replace
nothing, and leaves the rest to pop. The renderer cannot compute Δ either: the level rule
lives in `cubara-world` (Rule 3, and Rule 5 against a copy). What decision 3 protects is
kept exactly: edits never fade, only swaps do.

**B. Decision 5, "sinks from where it is", made exact.** A node whose Remove becomes
ready while it is still rising keeps rising through its hold half. It is capped 0.01
below full height, as every leaving node is. It then sinks from where it is, which is
full height. It is never frozen mid-rise.

*Why:* freezing breaks this decision's own property. Parent O is replaced by child N, so
N rises while O holds until t_O + H (H = 125 ms). N's children N′ land at t_N, with
t_O < t_N < t_O + H. If N froze below full height at t_N, then on [t_O + H, t_N + H) O is
sinking, N is low, and N′ may still be rising, so nothing is at full height over N's
footprint. Continuing the rise closes the gap. Nothing visible changes, because N is
under O while it finishes. Decision 9's property test pins this interleaving by name, and
also checks that no node's height jumps between frames and that a leaver is removed only
once it is fully sunk.

**C. Trees (not covered above).** A node's tree instances (`trees.wgsl`) and its voxel
trees (level 0) slide with it. At 0↔1 and 1↔2 the outgoing and incoming nodes draw the
same oak at the same site (`WorldGen::node_trees`), so today's tree hand-over is
invisible. A plain slide would make it visible: trees doubled for 250 ms, and a band Δ
tall appearing at insert and vanishing at removal.

Rule:

- an incoming node's instances are inserted when its rise completes;
- an outgoing node's instances are removed when its hold ends.

Both moments have old and new at full height, so identical trees hand over invisibly.
The same rule applies at 2↔3 (oak ↔ boxes), which is visible today as well. Level-0
voxel trees cannot be separated from their mesh without a shader change, which decision 2
excludes. At the ring-0 edge they still show a Δ band for one frame. The pop test reports
it; it is not worked around here.

**D. What the pop test measures (decision 7, made exact).**

- **Setup:** camera fixed at the low flight's start, (8, 120, 8) looking (1, −0.15, 0.3);
  the streaming eye moved +16 in x eight times, each step drawn until settled.
- **Frame step and size:** a fixed 1/60 s handed to the renderer (the flights' display
  rate; never shorter, never changed to pass), at 1920×1080.
- **Measure:** `changed_pixels` at `POP_TOLERANCE` between consecutive frames, inside the
  union of the screen boxes of every node that left the arena during the step. The same
  union is used for both arms. The number is the maximum over all frames of the eight
  steps.
- **Pass:** median over three runs of (fade-on maximum) × 4 ≤ median of (fade-off
  maximum). All six maxima are in the PR, and the final frame of each step is byte-equal
  on vs off. (Median of three is chosen, not derived.)
- **Diagnostics, never deciding:** land↔nothing flips, `silhouette_shift`, and the
  land→land share of the maximum frame are printed beside it.

*Known limit:* coarse tops tile their texture once per cell (2^L blocks) and scale AO per
cell. Where fine and coarse tops coincide, the hand-over is therefore a one-frame
texture-scale and AO switch that no height slide spreads. **If the flips drop ≥ 4× and
`changed_pixels` does not**, the PR merges as an improvement only if nothing got worse
(the fade-on maximum is not above the fade-off maximum, and HOLES and STALE hold per F),
with both numbers and the land→land share in it. This issue then stays open: done-when is
not met. The owner's eye decides whether a texture cross-fade (a shader change, outside
decision 2) is worth its own issue. If the measure is worse with the fade on, the PR
stays a draft.

The CI test of the mechanism (one synthetic swap, a 1-block step against the sky) uses the
same fixed 1/60 s and counts land↔nothing flips; if it is marginal the resolution is
raised, never the frame rate and never the 4.

**E. Decision 8's "routinely", derived from D.** Per cell, with d = fine − coarse,
min(|d|, Δ) slides and max(|d| − Δ, 0) pops: at insert where fine is higher, at removal
where fine is lower. At 60 Hz a half-fade is 7.5 frames. For 0↔1 and 1↔2, over the four
flights, implementation goes ahead only if

max(U↑ + C↑/7.5, U↓ + C↓/7.5) ≤ (U↑ + C↑ + U↓ + C↓)/4,

where U is the uncovered and C the covered sum on each side. That is, the 4× must be
reachable on height alone: a necessary condition, not a sufficient one (texture, AO and
trees come on top), with block heights standing in for pixels. It needs roughly 58 % of
the height covered when the two sides are even and 87 % when one-sided. If it is not
reachable, decision 2 is revisited here with the table. The table also prints the literal
reading (the share of swaps whose largest |d| exceeds Δ). The 2↔3 row is reported and not
acted on.

**F. STALE.** The incoming node starts Δ low, so where it is higher than the outgoing
one, the live frame lags the finished frame by up to 125 ms. The STALE term counts that.
The PR reports STALE on the low flight, fade on vs `--swap-fade-ms 0`, two runs each from
the same binary. More STALE kept frames with the fade on than in both fade-off runs is a
regression (decision 6's "must not regress", extended to STALE), and the PR stays a
draft. Note for reading the numbers: #406's attribution files these pixels under
"neither"; with the fade on, those whose rays land inside the voxels' reach are R6's
rise, not far terrain.

**Cost (decision 6, made measurable).** The lifts are flushed in one `write_buffer` per
frame from the arena's own mirror, not one per node. Budget, chosen: R6's CPU per frame
≤ 0.05 ms at 2,000 fading nodes on the M3, measured by an ignored `fade_cost_report`. On
a full arena a swap's Remove applies at once (space beats smoothness).

**Delivery.** Three PRs, each "Part of #380": (1) decision 8's table and today's value
of D's measure, no behaviour change; (2a) the render crate, game unchanged; (2b) streaming
passes Δ, with the measurements. The issue is not closed by a merge: the owner's eye is
part of done-when.
