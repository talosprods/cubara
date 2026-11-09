# Handover from the Windows session, 2026-11-09 — NOT for merge

Local state of the Windows laptop's autonomous session, pushed so the Linux session (the new lead)
can read it. Nothing here is a settled decision unless PLAN.md quotes the owner.

- `PLAN.md` — the work state, the owner's decisions of 2026-11-09 (quoted), open questions, PRs in flight.
- `reports/` — six code audits (findings mostly fixed in #303 #304 #307 #308 and in-flight PRs) and four
  design studies: `design-lod-transitions.md` (why LOD rings show; R1 block-height cells),
  `design-perf-roadmap.md` (M3 is GPU-bound on far triangles), `design-ui-scale.md` (7-PR plan),
  `phase3-options-memo.md` (FOR THE OWNER — options only, 5 Dutch questions at the end).
- `gemini.py` — helper for judging images with Gemini (model `gemini-3.6-flash`); reads keys from a
  local `gemini-keys.txt` next to it. The keys are NOT in git; the owner gives them to you directly.
