//! How a character looks: skin, hair and clothes, drawn voxel by voxel into
//! the eleven parts it is made of.
//!
//! Each part is a small voxel grid (a `BodyShape`, so it can also become a
//! rigid body) with the voxel its joint sits at. Parts are built once per
//! look and shared.

use crate::skeleton::{BONES, Bone, SHIN_VOXELS, THIGH_VOXELS};
use glam::{IVec3, Vec3};
use mc2_physics::shape::BodyShape;
use mc2_voxel::material::{MaterialId, ids};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Look {
    pub skin: MaterialId,
    pub hair: MaterialId,
    pub shirt: MaterialId,
    pub trousers: MaterialId,
    pub shoes: MaterialId,
    /// Sleeves to the wrist, or to above the elbow.
    pub long_sleeves: bool,
    /// Hair to the collar, or cropped.
    pub long_hair: bool,
    pub beard: bool,
    /// A belt at the waist.
    pub belt: bool,
}

const SKINS: [MaterialId; 4] = [
    ids::SKIN_PALE,
    ids::SKIN_TAN,
    ids::SKIN_BROWN,
    ids::SKIN_DARK,
];
const HAIRS: [MaterialId; 4] = [
    ids::HAIR_BLACK,
    ids::HAIR_BROWN,
    ids::HAIR_BLOND,
    ids::HAIR_RED,
];
const SHIRTS: [MaterialId; 6] = [
    ids::CLOTH_RED,
    ids::CLOTH_BLUE,
    ids::CLOTH_GREEN,
    ids::CLOTH_OCHRE,
    ids::CLOTH_WHITE,
    ids::CLOTH_GREY,
];
const TROUSERS: [MaterialId; 4] = [ids::DENIM, ids::CLOTH_GREY, ids::LEATHER, ids::CLOTH_OCHRE];

impl Look {
    /// A look picked by `seed`: any skin, any hair, a shirt and trousers
    /// that are not the same.
    pub fn from_seed(seed: u64) -> Look {
        let mut h = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x2545_f491_4f6c_dd1d;
        let mut next = |n: usize| {
            h ^= h >> 33;
            h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
            h ^= h >> 29;
            (h % n as u64) as usize
        };
        let skin = SKINS[next(SKINS.len())];
        let hair = HAIRS[next(HAIRS.len())];
        let shirt = SHIRTS[next(SHIRTS.len())];
        let mut trousers = TROUSERS[next(TROUSERS.len())];
        if trousers == shirt {
            trousers = ids::DENIM;
        }
        Look {
            skin,
            hair,
            shirt,
            trousers,
            shoes: if next(3) == 0 {
                ids::CLOTH_GREY
            } else {
                ids::LEATHER
            },
            long_sleeves: next(2) == 0,
            long_hair: next(3) == 0,
            beard: next(4) == 0,
            belt: next(2) == 0,
        }
    }
}

/// One part: its voxels and the voxel (continuous, grid coordinates) its
/// joint sits at.
#[derive(Clone, Debug)]
pub struct Part {
    pub shape: Arc<BodyShape>,
    pub pivot: Vec3,
}

/// The eleven parts of a look, in `Bone::ALL` order.
pub fn parts(look: &Look) -> [Part; BONES] {
    Bone::ALL.map(|b| part(look, b))
}

fn part(look: &Look, bone: Bone) -> Part {
    use Bone::*;
    // Grid size, where the joint is in it, and what each voxel is.
    let (size, pivot): (IVec3, Vec3) = match bone {
        Pelvis => (IVec3::new(6, 2, 4), Vec3::new(3.0, 0.0, 2.0)),
        Torso => (IVec3::new(8, 9, 4), Vec3::new(4.0, 0.0, 2.0)),
        Head => (IVec3::new(5, 5, 5), Vec3::new(2.5, 0.0, 2.5)),
        UpperArmL | UpperArmR | ForeArmL | ForeArmR => {
            (IVec3::new(2, 5, 2), Vec3::new(1.0, 5.0, 1.0))
        }
        ThighL | ThighR => (
            IVec3::new(3, THIGH_VOXELS as i32, 3),
            Vec3::new(1.5, THIGH_VOXELS, 1.5),
        ),
        // The shoe reaches a voxel forward.
        ShinL | ShinR => (
            IVec3::new(3, SHIN_VOXELS as i32, 4),
            Vec3::new(1.5, SHIN_VOXELS, 1.5),
        ),
    };
    let mut voxels = vec![MaterialId(0); (size.x * size.y * size.z) as usize];
    for z in 0..size.z {
        for y in 0..size.y {
            for x in 0..size.x {
                let m = paint(look, bone, IVec3::new(x, y, z), size);
                voxels[(x + size.x * (y + size.y * z)) as usize] = m;
            }
        }
    }
    let shape = BodyShape::from_voxels(size, voxels).expect("a character part has voxels");
    Part {
        shape: Arc::new(shape),
        pivot,
    }
}

/// The material of voxel `v` of `bone`'s part (air where there is none).
fn paint(look: &Look, bone: Bone, v: IVec3, size: IVec3) -> MaterialId {
    use Bone::*;
    let top = size.y - 1;
    // Front is +Z.
    let front = v.z == size.z - 1;
    match bone {
        Head => {
            let (x, y) = (v.x, v.y);
            // Hair over the crown and back, down the sides at the top.
            let back = v.z == 0;
            let side = x == 0 || x == size.x - 1;
            if y == top || (back && (y >= 2 || look.long_hair)) || (side && y >= 3 && v.z < 3) {
                return look.hair;
            }
            if front && y == 2 && (x == 1 || x == 3) {
                return ids::EYE;
            }
            if look.beard && front && y == 0 && (1..=3).contains(&x) {
                return look.hair;
            }
            look.skin
        }
        Torso => {
            // The neck is skin, and a collar of it shows at the front.
            if v.y == top && (2..=5).contains(&v.x) && v.z >= 1 && v.z <= 2 {
                return look.skin;
            }
            if v.y == top && (v.x == 0 || v.x == size.x - 1) {
                return MaterialId(0);
            }
            if look.belt && v.y == 0 {
                return ids::LEATHER;
            }
            look.shirt
        }
        Pelvis => look.trousers,
        UpperArmL | UpperArmR => {
            // Short sleeves stop two voxels below the shoulder.
            if look.long_sleeves || v.y >= 3 {
                look.shirt
            } else {
                look.skin
            }
        }
        ForeArmL | ForeArmR => {
            if v.y == 0 || !look.long_sleeves {
                look.skin
            } else {
                look.shirt
            }
        }
        ThighL | ThighR => look.trousers,
        ShinL | ShinR => {
            // Shoes over the bottom two voxels, reaching forward; the rest of
            // the front row is empty above them.
            if v.y <= 1 {
                look.shoes
            } else if front {
                MaterialId(0)
            } else {
                look.trousers
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn looks_vary_and_parts_are_whole() {
        let a = Look::from_seed(1);
        let b = (2..40).map(Look::from_seed).find(|l| *l != a);
        assert!(b.is_some(), "every seed looks the same");
        for seed in 0..20 {
            let look = Look::from_seed(seed);
            assert_ne!(look.shirt, look.trousers);
            let parts = parts(&look);
            for (b, p) in Bone::ALL.iter().zip(&parts) {
                assert!(p.shape.solid_count > 0, "{b:?}");
                assert!(p.shape.mass > 0.1, "{b:?} {}", p.shape.mass);
            }
            // A head has eyes.
            let head = &parts[Bone::Head.index()].shape;
            assert!(head.voxels.contains(&ids::EYE));
        }
    }

    #[test]
    fn a_body_weighs_about_what_a_person_does() {
        let look = Look::from_seed(7);
        let total: f32 = parts(&look).iter().map(|p| p.shape.mass).sum();
        assert!((25.0..120.0).contains(&total), "{total} kg");
    }
}
