//! The renderer's form of the world's far-terrain faces, for this crate's
//! own tests: `golden.rs` draws them and `lod_consistency` measures them
//! (`#[path = "common/far_faces.rs"] mod far_faces;`), from one copy
//! (`ARCHITECTURE.md` Rule 5).
//!
//! The render crate does not know the world crate (Rule 3), so the game
//! converts in `cubara-app` (`far_streaming::to_far_faces`), which a test of
//! this crate cannot call; `the_renderer_and_the_world_agree_on_the_patch_size`
//! there holds that one field for field.

use cubara_render::{FarDir, FarFace};
use cubara_world::far::{Face, FaceDir};

/// `faces` as the renderer takes them, field for field.
pub fn far_faces(faces: &[Face]) -> Vec<FarFace> {
    faces
        .iter()
        .map(|f| FarFace {
            dir: match f.dir {
                FaceDir::Top => FarDir::Top,
                FaceDir::PosX => FarDir::PosX,
                FaceDir::NegX => FarDir::NegX,
                FaceDir::PosZ => FarDir::PosZ,
                FaceDir::NegZ => FarDir::NegZ,
            },
            i0: f.i0,
            j0: f.j0,
            i1: f.i1,
            j1: f.j1,
            top: f.top,
            rise: f.rise,
            border: f.border,
            mix: f.mix,
            open: f.open,
        })
        .collect()
}
