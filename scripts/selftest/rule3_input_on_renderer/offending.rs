// ARCHITECTURE.md Rule 3: the renderer does not own input or gameplay.
// Trips check-architecture.sh's grep on crates/render/src/render.rs for
// `pub fn (key_input|mouse_look|set_cursor_captured|edit_block)`.
pub fn key_input() {}
