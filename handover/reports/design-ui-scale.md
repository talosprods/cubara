# Design: UI scale and a minimal layout layer

Status: design study, 2026-11-09. Nothing implemented. Written read-only against `248fc03`.
Trigger: the 2026-11-09 play-test. On a maximised ~3136×860 window the pause menu was a few tiny lines in the top-left with no dimming, F3 was too small to read, and the HUD is fixed-size.

Yardstick: performance, efficiency, clean code that is easy to extend. Rules: 2 (no globals), 3 (the renderer doesn't know gameplay), 5 (one implementation per concern), 6 (pin behaviour before rewriting it).

## 0. What exists today (reuse, don't rebuild)

| Existing piece | Where | Reused as |
|---|---|---|
| One 2D pipeline for glyphs, solid rectangles, alpha rectangles and icons; `queue(text, x, y, scale: f32, ..)` already takes a per-call glyph scale | `crates/render/src/text/mod.rs` (`queue` l.282, `queue_rect_alpha` l.313, `push_rect_mode` l.352) | All UI drawing stays here. No new pipeline, pass or shader (Rule 5). |
| Dimmed backdrop and panel frame | `crates/render/src/scene.rs` `queue_panel` l.735-738 (`DIM` at alpha 0.72, `BACK`) | Pulled out into `queue_backdrop` / `queue_frame`, shared by the inventory and the menus. |
| "One layout, two consumers": a pure layout plus `hit(x, y)`, tested without a GPU | `crates/render/src/panel.rs` (`InventoryPanel::layout` l.75, `hit` l.214, tests l.222-416) | Generalised to `Panel<T>`. Menus are `Panel<MenuTarget>`. No second hit-test. |
| Click routing through that layout | `crates/app/src/game.rs` `click_panel` l.1324, `hovered_item_name` l.1484, `panel_stacks` l.1506 | Same calls, now given a `Screen` instead of `(w, h)`. |
| Pure screen-state rules (mouse capture, Escape) | `crates/app/src/capture.rs`, `escape_action` in `main.rs` l.167 | Unchanged. Menus already free the mouse. |
| Menu and options text as pure, tested functions | `main.rs` `menu_text` l.187, `crates/app/src/options.rs` `options_text` / `options_key` / `apply` | Become row models (section 2.3). Their tests carry over. |
| `CursorMoved` tracks the cursor in physical pixels at all times | `main.rs` l.656 | Hover and menu clicks use it as is. |
| Headless `Shot` with static `menu` text, `--screenshot --menu` | `crates/render/src/headless.rs` l.59-130, `main.rs` l.1044 | `Shot` gains `scale_factor` and `debug_text`; `menu` becomes a `MenuView`. |
| Goldens with pixel probes for elements below the 0.5% threshold | `crates/render/tests/golden.rs` crosshair l.956, furnace gauges l.1110 | Same pattern for the backdrop, glyph height and button rectangles. |

**Problems found along the way** (each fixed by a step below):
1. **Hearts duplicate the hotbar's geometry.** `scene.rs` l.892 and l.905 hard-code `HOTBAR = 48.0` and `9.0 * 48.0 + 8.0 * 4.0`, and l.908 computes `total` and throws it away (`let _ = total;`). A hotbar that isn't 9 slots wide puts the hearts out of line. This is a Rule 5 problem.
2. **Two things aren't crisp today.**
   - Stack counts use glyph scale 1.5 (l.867), so font pixels are 1.5 px and columns come out uneven.
   - Inventory icons are 40 − 2×6 = 28 px for 16-px art, which is 1.75×.
3. **Centring can land on half pixels.** `(w − total) * 0.5` gives a .5 origin for odd widths, which puts glyphs off the pixel grid.
4. **A stale comment.** The hotbar's reason for fixed pixels ("same physical size on 1080p and 4K", l.930, repeated in `panel.rs` l.44) is wrong: fixed pixels are physically smaller on a dense screen. This design delivers what that comment intended.
5. **Menu keys live in two places.** The `[C]` / `[N]` / `[O]` labels are in `menu_text`, and the keys are in `main.rs`'s if-chain (l.592-630). They can drift apart.
6. **Silent truncation.** `push_rect_mode` drops quads past `MAX_CHARS` without a word (l.363). It isn't reachable at these sizes, but it deserves a one-time `log::warn`.
7. **Duplicate work.** `panel_stacks` runs twice per frame while the inventory is open (`panel_view` and `hovered_item_name`). It's minor but can be computed once.
8. **Nothing reads DPI.** `scale_factor` and `ScaleFactorChanged` appear nowhere in `crates/`.

## 1. The UI scale

### 1.1 Units (all physical pixels)
- **Physical pixel.** The surface size (`Renderer::size()`), winit's `CursorMoved` and the text pipeline all use it. There is no logical-pixel conversion anywhere except inside the scale function.
- **Design pixel.** The numbers already in the code: 48-px hotbar slots, 18-px hearts, glyph scale 2. Each is a physical pixel at the **reference scale 2**, which is what the UI looks like today at 1280×720 and 100%. Constants keep their current values.
- **`UiScale(s)`.** An integer of at least 1: physical pixels per font or art pixel. `px(d) = d·s/2`. Text always draws at `s` px per font pixel. Hotbar icons draw at `s` px per art pixel (32 design px for 16-px art), so they are crisp at every step.

### 1.2 The function (pure, in `crates/render/src/ui.rs`, GPU-free like `panel.rs`)
```rust
pub const REFERENCE: u32 = 2;          // the scale every design constant is written at
const ROWS_PER_STEP: u32 = 360;        // physical rows per scale step
const CANVAS: (u32, u32) = (480, 360); // design px every screen must fit in (bench inventory 424x360, hotbar 470 wide)

pub fn ui_scale(width: u32, height: u32, scale_factor: f64) -> UiScale {
    let dpi = if scale_factor.is_finite() && scale_factor > 0.0 { scale_factor } else { 1.0 };
    let by_height = height / ROWS_PER_STEP;
    let by_dpi = (dpi * REFERENCE as f64).floor() as u32;          // 100%→2, 125%→2, 150%→3, 200%→4
    let fit = (width * REFERENCE / CANVAS.0).min(height * REFERENCE / CANVAS.1);
    UiScale(by_height.max(by_dpi).min(fit).max(1))
}
```
- **Height sets the size.** At 16:9 the hotbar is about 36% of the window width at 720p, 1080p, 1440p and 2160p.
- **DPI sets a floor**, so a small window on a dense screen still follows the operating system's size setting.
- **Width only caps the scale** (via `fit`), so an ultrawide stays at the scale its height gives. A window narrower than 480 design px steps down instead of clipping.
- **Never 0, never a panic.** A minimised (0×0) window gives 1, and NaN DPI counts as 100%.

| Window (physical) @ DPI | by height | by DPI | fit | **s** |
|---|---|---|---|---|
| 960×540 @1.0 (every existing golden) | 1 | 2 | 3 | **2** (unchanged) |
| 1280×720 @1.0 | 2 | 2 | 4 | **2** |
| 1920×1080 @1.0 | 3 | 2 | 6 | **3** |
| 2560×1440 @1.0 | 4 | 2 | 8 | **4** |
| 3840×2160 @1.0 | 6 | 2 | 12 | **6** |
| 3136×860 @1.0 (play-test, if physical) | 2 | 2 | 4 | **2** |
| 3136×860 @1.5 | 2 | 3 | 4 | **3** |
| 4704×1290 @1.5 (play-test, if 3136×860 was logical) | 3 | 3 | 7 | **3** |
| 2560×1600 @2.0 (Mac, 1280×800 logical) | 4 | 4 | 8 | **4** |
| 1280×720 @2.0 (small window, dense screen) | 2 | 4 | 4 | **4** |
| 400×300 @1.0 | 0 | 2 | 1 | **1** |
| 0×0 | 0 | 2 | 0 | **1** |

### 1.3 One place computes it, and it's passed in
- `pub struct Screen { pub width: u32, pub height: u32, pub scale: UiScale }`, built only by `Screen::new(w, h, scale_factor)`, which calls `ui_scale`.
- **The window:** `Renderer::screen(&self) -> Screen` reads `self.size()` and `window.scale_factor()` (1.0 for `Target::Offscreen`).
  - It's computed on demand rather than stored, so it can't go stale. No `ScaleFactorChanged` handler is needed: winit follows a DPI change with a `Resized`, and the factor is read each frame. The cost is one mutex-guarded field read per frame.
  - `Renderer` stores the last scale it logged, as an ordinary field (not a global), and logs `UI scale N (window WxH, scale factor F)` when it changes. **This gives the next play-test exact numbers.**
- **Drawing:** `Renderer::render` passes `self.screen()` into `SceneFrame { screen, .. }`. `encode_scene` `debug_assert!`s that the size matches its own.
- **Clicks:** `main.rs` calls `renderer.screen()` and passes it to `Game::click_panel` / `panel_view` / `hovered_item_name` and to menu clicks, instead of `(w, h)`.
- **Headless:** `Shot.scale_factor: f64` (default 1.0), and `render_arena` builds `Screen::new(width, height, scale_factor)`. `bench.rs` and `flight.rs` use `Screen::new(w, h, 1.0)`.
- **Mechanical guard** (one line each in `scripts/check-architecture.sh`, Rule 5): `scale_factor()` may appear only in `crates/render/src/render.rs`, and `ui_scale(` only in `ui.rs`. Answer to "what fails when someone breaks it": the check.

### 1.4 Crispness, enforced in one place
- `push_rect_mode` rounds the four edges of every quad to whole pixels. That is the single point every glyph, rectangle and icon goes through. At 960×540 every current edge is already whole, so the goldens prove this changes nothing.
- Layout centring uses integer division.
- Half-pixel design values (hotbar border 3, heart gap 3, crosshair arm 9, tooltip padding 5) get floored at odd `s`. They are solid fills, so this is deterministic and not blurry.

## 2. A minimal layout layer

### 2.1 One generic hit-test (extracted from `panel.rs`, not added beside it)
```rust
pub struct Rect { pub x: f32, pub y: f32, pub w: f32, pub h: f32 }       // whole physical px
impl Rect { pub fn contains(&self, x: f32, y: f32) -> bool; pub fn centre(&self) -> (f32, f32) }
pub struct Panel<T> { pub frame: Rect, targets: Vec<(Rect, T)> }
impl<T: Copy> Panel<T> {
    pub fn targets(&self) -> &[(Rect, T)];
    pub fn hit(&self, x: f32, y: f32) -> Option<T>;          // first containing target; gaps hit nothing
    pub fn hit_index(&self, x: f32, y: f32) -> Option<usize>; // lets hovered_item_name skip its position() search
}
```
- `InventoryPanel` becomes `Panel<SlotId>` (`SlotId { kind: PanelSlotKind, index }`). `slots()` becomes `targets()`, and `x/y/width/height` become `frame`.
- The current `contains` and `hit` semantics move over unchanged, including "a gap hits nothing; never guess the nearest" (`panel.rs` l.208-213).
- Test helper `assert_panel_sound(&Panel<T>, &Screen)`, used for every layout:
  - every target's centre hits that target;
  - no two targets overlap;
  - a point in a gap hits nothing, and a point outside hits nothing;
  - the frame lies inside the screen;
  - left and right margins differ by at most 1 px;
  - every edge is a whole number.

### 2.2 Layouts on top of it
- **`HudLayout::of(screen, hotbar_len)`**: the hotbar slot rectangles plus the hearts' origin, computed once. `queue_hotbar` and `queue_hearts` both read it (fixes problem 1). Not hit-tested; it's layout only.
- **`InventoryPanel::layout(screen, grid)` / `layout_furnace(screen)`**: the same algorithm with `SLOT`, `GAP` and `PAD` through `screen.scale.px(..)`. The slot size becomes a field (`scene.rs` l.757 reads `panel::SLOT` today).
- **`menu_panel(screen, &MenuView) -> Panel<MenuTarget>`** (`MenuTarget { row, cell }`):
  - a vertical list of rows, each with 1 to n cells split evenly (so Options' four quality choices are four buttons in one row);
  - width comes from the content (a `measure(text) -> (cols, lines)` helper next to the font, 8·s px per column) with a minimum, capped at `screen.width` minus margins;
  - centred on both axes, so on an ultrawide it sits mid-screen and does not stretch;
  - text-only rows (title, the console's input line) are drawn but are not targets.
- **The console's input row** has a fixed minimum width (about 60 columns, so the panel doesn't twitch while typing) and shows the tail of long input through a pure `visible_tail(text, cols)`.

### 2.3 The renderer draws labels; the app owns actions (Rule 3)
- **Renderer** (`crates/render/src/menu.rs`):
  - `MenuView { title: Option<String>, rows: Vec<MenuRowView>, hovered: Option<MenuTarget> }` and `MenuRowView { cells: Vec<String>, style: Button | Disabled | Text | Input }`.
  - Owned strings are allocated only while a menu is open and remove the per-frame `text.to_string()` in `render.rs` l.1030.
  - `Hud.menu: Option<&MenuView>` replaces `Option<&str>`.
- **App** (`crates/app/src/menu.rs`, pure): `Menu<A> { title, rows: Vec<Row<A>> }`, with `Row<A> { cells: Vec<Cell<A>> }` and `Cell<A> { label, key: Option<char>, action: Option<A> }`.
  - `to_view(hovered)` generates the `[C]` label prefix from `key`, so the label and the key can't drift (fixes problem 5).
  - `key(c) -> Option<A>` and `click(screen, x, y) -> Option<A>` (builds `menu_panel` and calls `hit`).
  - `pause_menu(&Game) -> Menu<PauseAction>` replaces `menu_text`.
  - `options_menu(&Settings, fps, benchmarking) -> Menu<OptionsKey>` reuses `OptionsKey`; `options_key` becomes a row lookup.
  - `console_menu(&Game) -> Menu<ConsoleAction>`.
- **Routing in `main.rs`:**
  - The C/O/N if-chain becomes `pause_menu(&game).key(c)` followed by `apply_pause(action)`.
  - `MouseInput` (press, the same as the inventory) with a menu open does `menu.click(renderer.screen(), cursor)` and then the same `apply_*`.
  - Hover: `panel.hit(cursor)` each frame into `MenuView.hovered`.
  - Escape, capture and `follow_screen` don't change. `New World`'s two-press confirmation works for clicks without changes because both paths call `press_new_world`.
  - Keys come from `event.text` (layout-aware, as Options and the console already do), not `KeyCode`.
- **Drawing order** in `encode_scene`: world → HUD → `queue_backdrop` (full window, alpha) → `queue_frame(panel.frame)` → rows (button wells, hover highlight, labels at `s`). The inventory reuses `queue_backdrop` and `queue_frame`. `Renderer::render` keeps hiding F3 while a menu is open, but they become separate `SceneFrame` fields (`debug_text`, `menu`).

## 3. Debug overlay and HUD use the same scale
- **F3:** origin `px(8)`, shadow offset `px(2)` (= `s`), glyph scale `s`. `Shot.debug_text: Option<String>` lets a golden pin how it draws without the live FPS number. A translucent dark panel behind the text helps legibility on bright sky more than size does (owner's call, PR G).
- **Hotbar, hearts, crosshair, tooltip, stack counts, furnace gauges:** every constant goes through `px()`, hotbar and hearts through `HudLayout`. Because the reference scale is 2, nothing changes at 960×540, 1280×720 or 3136×860 at 100%. Sizes change from 1080 rows up and at 150% DPI or more.

## 4. How it's pinned

**Pure unit tests (no GPU):**
- `ui_scale`: the table in section 1.2, row for row.
- Properties:
  - for all w ≥ 2·`CANVAS.0`, `s(w, h) == s(2w, h)` (ultrawide never grows);
  - the scale never decreases as h or DPI grow;
  - s ≥ 1;
  - `px(d)` at the reference scale returns d;
  - **every real layout fits the canvas**: inventory grids 2 and 3, furnace, pause, options and console, with typical content, measured at s = 1. This test fails if a panel outgrows `CANVAS`, which keeps the constant honest.
- `assert_panel_sound` for every layout at 1280×720@1, 3136×860@1, 3136×860@1.5 and 2560×1600@2.
- `HudLayout`:
  - the hearts' left edge equals the hotbar's left edge at every scale and slot count;
  - the hotbar is centred;
  - nothing goes off-screen.
- **App:**
  - for each menu, every cell with a key returns the same action from `key(c)` as from `click` at its centre;
  - the existing `menu_text_*` tests become row tests (Survival/Creative, the armed New World text, console over pause, the reply);
  - `click_panel` at s = 3 (1920×1080) moves the item. This is the #151 "clicks land on the right slot" property at a scale other than the reference;
  - `hovered_item_name` at s = 4.

**Goldens (`crates/render/tests/golden.rs`):**
- All 18 existing 960×540 goldens **must stay unchanged** through PRs A to F. That is the proof the refactor is neutral.
- The new set is drawn over plain sky with an explicit camera (no meshing), so the PNGs are tens of KB, the tests are fast, and integer-aligned quads should match exactly across backends:
  - `hud_1280x720`, `hud_3136x860`, `hud_2560x1600_hidpi`: hotbar with items, hearts at 13/20, crosshair, static F3 text;
  - `pause_menu_1280x720`, `pause_menu_3136x860`, `pause_menu_2560x1600_hidpi`: backdrop and panel over the HUD, with one row hovered.
- **Probes:** a menu panel easily clears 0.5% of pixels, but these catch the cheap mistakes the image diff can miss:
  - a corner pixel is the sky colour dimmed by the backdrop alpha;
  - glyph cell height equals 8·s (measured on the title);
  - the frame's left and right margins are equal;
  - **each button's centre (from `menu_panel`) has the button or hover colour, and the gap between rows has the frame colour**. This ties the hit geometry to the drawn pixels.
- **Mutations for each PR's table:**
  - drop DPI from `ui_scale`: the table and the `hidpi` goldens go red;
  - use width instead of height: the 3136×860 rows go red;
  - drop `queue_backdrop`: the corner probe goes red;
  - hit-test against an unscaled layout: the s = 3 click test goes red;
  - put the old hotbar constant back into the hearts: the `HudLayout` test goes red.

## 5. Migration (each PR shippable)

| PR | What | Visible change | Tests |
|---|---|---|---|
| **A** | `ui.rs`: `UiScale`, `ui_scale`, `Screen`, `Renderer::screen()`, log on change. `Shot.scale_factor`, `SceneFrame.screen`. The two architecture-check greps. Optionally convert the 17 explicit `Shot { .. }` literals in `golden.rs` to `..Shot::default()`, since B, C and F each add fields. | None (log line only) | Table and property tests. All goldens unchanged. |
| **B** | HUD and F3 through `px()`. `HudLayout` (fixes the hearts duplication). Edge snapping in `push_rect_mode`. Integer centring. `Shot.debug_text` split from `menu`. | Only from 1080 rows or 150% DPI up | All 960×540 goldens unchanged. `HudLayout` tests. New `hud_*` goldens ×3 with probes. |
| **C** | Extract `Rect` / `Panel<T>` from `panel.rs`. `InventoryPanel` builds on it. `assert_panel_sound`. | None | Existing panel and game tests unchanged except for renames. Goldens unchanged. |
| **D** | Inventory and furnace take `Screen`. `Game::{click_panel, panel_view, hovered_item_name}` take `Screen`. `PanelLayout::build(screen)`. Rewrite `the_layout_moves_with_the_window_but_keeps_its_slots` (`panel.rs` l.361): slots now scale by s/2, and the PR says it reverses the #145 fixed-pixel rule and why. | Inventory scales at 1080 rows or more | Panel soundness at 4 screens. Click test at s = 3, hover at s = 4. Goldens unchanged. |
| **E** | App `menu.rs`: `Menu<A>`, `pause_menu` / `options_menu` / `console_menu`. Key chain replaced by `Menu::key`. Still drawn as the same text string. | None | Key/action table tests. Ported `menu_text` tests. |
| **F** | Render `menu.rs`: `MenuView`, `menu_panel`. Shared `queue_backdrop` / `queue_frame`. Menus drawn as centred panels with hover. Mouse clicks routed to the same `apply_*`. `Shot.menu: Option<MenuView>`; `--screenshot --menu` turns its lines into rows. One `--bench 64 --overlay` before/after row in `BENCHMARKS.md`. | The fix the play-test asked for | Soundness tests for the menu layouts. Click and action tests. `pause_menu_*` goldens ×3 with probes. Optionally `options_1280x720`, `console_1280x720`. |
| **G** | The owner's-call items below, with before/after images and an explicit re-bless list (hotbar, inventory, furnace, item_icons, item_tooltip goldens). | Yes, by choice | Re-blessed goldens, explained in the PR. |

**Performance:**
- `ui_scale` is a few integer operations per frame.
- Menu layout is about ten rectangles, built only while a menu is open.
- The quad count doesn't depend on `s`.
- The backdrop is one alpha-blended quad, only while a screen is open (2.7 Mpx at 3136×860, trivial).
- No new pipeline, pass, bind group or shader.
- The bench scene draws no UI except with `--overlay`, so only PR F needs a bench row.

## 6. Owner's call (decided by the owner, not by us)
- **What the menus contain:** rows, wording, order, the title, and whether "Commands" is a row or only `/`. The 2026-09-18 list stands. This design only arranges it.
- **Colours and look:** backdrop alpha (0.72 today), panel, button, hover and disabled colours. The default is the inventory's existing palette. The font stays font8x8; a different one is his decision.
- **How big the UI is.** `ROWS_PER_STEP = 360` makes the hotbar about 36% of a 16:9 width. 125% DPI rounds down to the same scale as 100%.
- **A manual "UI scale" choice in Options.** It would add one parameter to `ui_scale`; it isn't built in advance (ARCHITECTURE.md: no pre-built slots).
- **Console placement** (centred panel or bottom bar) and where the command reply appears.
- **The visual crispness fixes in G:** stack-count digits 1.5 → 2, inventory icon padding 6 → 4 (32-px icons), a dark panel behind F3.

## 7. Open question
At 860 physical rows and 100% DPI this formula gives s = 2, the current size, so F3 and the HUD would **not** grow on the play-test window. If that window was really 100% DPI, the only ways to make it bigger are a smaller `ROWS_PER_STEP` (about 286 gives 3 at 860 rows but also 3 at 1080 and 5 at 1440) or the manual option. Both are the owner's call. PR A's log line will say which case it was.

### Critical files for implementation
- `crates\render\src\scene.rs`
- `crates\render\src\panel.rs`
- `crates\render\src\text\mod.rs`
- `crates\app\src\main.rs`
- `crates\render\tests\golden.rs`

Also touched: `crates\render\src\render.rs`, `crates\render\src\headless.rs`, `crates\app\src\game.rs`, `crates\app\src\options.rs`.
