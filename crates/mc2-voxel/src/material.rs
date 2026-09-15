//! Global material table. A voxel stores a palette index that resolves to a
//! `MaterialId`; everything physical about the voxel comes from here.
//!
//! Units: density kg/m^3, strengths MPa, temperatures degrees C, conductivity
//! W/(m K), specific heat J/(kg K). Emission is linear radiance relative to a
//! sun disc of roughly 100 000.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct MaterialId(pub u16);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Air,
    Solid,
    /// Alpha tested in shading, e.g. leaves.
    Foliage,
    /// Refracting, e.g. glass and ice.
    Transparent,
    /// Rendered and simulated by the fluid solver, never stored in bricks.
    Liquid,
}

#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub name: &'static str,
    pub kind: Kind,
    pub albedo: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
    pub emission: [f32; 3],
    pub ior: f32,
    pub density: f32,
    pub compressive: f32,
    pub tensile: f32,
    /// Relative mining effort; tools divide it by their tier.
    pub hardness: f32,
    pub conductivity: f32,
    pub specific_heat: f32,
    /// Chance per second of catching from a burning neighbour, 0..1.
    pub flammability: f32,
    pub ignition_c: f32,
    pub melt_c: f32,
    /// Fraction of incident sound energy absorbed per reflection.
    pub absorption: f32,
}

const BASE: Material = Material {
    name: "",
    kind: Kind::Solid,
    albedo: [0.5, 0.5, 0.5],
    roughness: 0.8,
    metallic: 0.0,
    emission: [0.0; 3],
    ior: 1.5,
    density: 2500.0,
    compressive: 100.0,
    tensile: 5.0,
    hardness: 3.0,
    conductivity: 2.5,
    specific_heat: 800.0,
    flammability: 0.0,
    ignition_c: f32::INFINITY,
    melt_c: f32::INFINITY,
    absorption: 0.03,
};

macro_rules! materials {
    ($($id:ident = $n:expr => $m:expr,)*) => {
        pub mod ids {
            use super::MaterialId;
            $(pub const $id: MaterialId = MaterialId($n);)*
        }
        pub static MATERIALS: &[Material] = &[$($m,)*];
    };
}

materials! {
    AIR = 0 => Material { name: "air", kind: Kind::Air, density: 1.2, compressive: 0.0, tensile: 0.0, hardness: 0.0, conductivity: 0.026, absorption: 0.0, ..BASE },
    GRANITE = 1 => Material { name: "granite", albedo: [0.42, 0.39, 0.37], roughness: 0.75, density: 2700.0, compressive: 200.0, tensile: 12.0, hardness: 6.0, melt_c: 1250.0, ..BASE },
    BASALT = 2 => Material { name: "basalt", albedo: [0.13, 0.13, 0.14], roughness: 0.85, density: 3000.0, compressive: 250.0, tensile: 14.0, hardness: 6.5, melt_c: 1100.0, ..BASE },
    LIMESTONE = 3 => Material { name: "limestone", albedo: [0.62, 0.58, 0.50], density: 2500.0, compressive: 80.0, tensile: 5.0, hardness: 3.5, ..BASE },
    SANDSTONE = 4 => Material { name: "sandstone", albedo: [0.66, 0.47, 0.30], roughness: 0.9, density: 2300.0, compressive: 60.0, tensile: 3.0, hardness: 2.5, ..BASE },
    SHALE = 5 => Material { name: "shale", albedo: [0.25, 0.24, 0.24], density: 2400.0, compressive: 50.0, tensile: 2.0, hardness: 2.0, ..BASE },
    MARBLE = 6 => Material { name: "marble", albedo: [0.80, 0.79, 0.76], roughness: 0.35, density: 2700.0, compressive: 120.0, tensile: 7.0, hardness: 4.0, ..BASE },
    SLATE = 7 => Material { name: "slate", albedo: [0.20, 0.22, 0.25], roughness: 0.6, density: 2750.0, compressive: 150.0, tensile: 9.0, hardness: 4.5, ..BASE },
    DIRT = 8 => Material { name: "dirt", albedo: [0.28, 0.19, 0.12], roughness: 0.95, density: 1500.0, compressive: 0.4, tensile: 0.02, hardness: 0.6, conductivity: 1.0, absorption: 0.15, ..BASE },
    GRASS = 9 => Material { name: "grass", albedo: [0.20, 0.34, 0.09], roughness: 0.9, density: 1400.0, compressive: 0.4, tensile: 0.05, hardness: 0.6, conductivity: 0.8, flammability: 0.02, ignition_c: 260.0, absorption: 0.3, ..BASE },
    SAND = 10 => Material { name: "sand", albedo: [0.76, 0.66, 0.47], roughness: 0.95, density: 1600.0, compressive: 0.1, tensile: 0.0, hardness: 0.5, conductivity: 0.3, melt_c: 1700.0, absorption: 0.2, ..BASE },
    GRAVEL = 11 => Material { name: "gravel", albedo: [0.40, 0.38, 0.36], roughness: 0.95, density: 1800.0, compressive: 0.2, tensile: 0.0, hardness: 0.8, absorption: 0.25, ..BASE },
    CLAY = 12 => Material { name: "clay", albedo: [0.55, 0.42, 0.34], density: 1800.0, compressive: 1.0, tensile: 0.1, hardness: 0.8, ..BASE },
    SNOW = 13 => Material { name: "snow", albedo: [0.92, 0.94, 0.97], roughness: 0.7, density: 300.0, compressive: 0.05, tensile: 0.01, hardness: 0.2, conductivity: 0.2, specific_heat: 2100.0, melt_c: 0.0, absorption: 0.45, ..BASE },
    ICE = 14 => Material { name: "ice", kind: Kind::Transparent, albedo: [0.80, 0.90, 0.97], roughness: 0.1, ior: 1.31, density: 917.0, compressive: 5.0, tensile: 1.0, hardness: 1.0, conductivity: 2.2, specific_heat: 2100.0, melt_c: 0.0, ..BASE },
    WATER = 15 => Material { name: "water", kind: Kind::Liquid, albedo: [0.02, 0.05, 0.06], roughness: 0.02, ior: 1.33, density: 1000.0, compressive: 0.0, tensile: 0.0, hardness: 0.0, conductivity: 0.6, specific_heat: 4186.0, absorption: 0.01, ..BASE },
    LAVA = 16 => Material { name: "lava", kind: Kind::Liquid, albedo: [0.30, 0.05, 0.01], roughness: 0.6, emission: [2400.0, 600.0, 80.0], density: 2600.0, compressive: 0.0, tensile: 0.0, hardness: 0.0, conductivity: 1.5, specific_heat: 1000.0, ..BASE },
    OBSIDIAN = 17 => Material { name: "obsidian", albedo: [0.03, 0.02, 0.04], roughness: 0.08, density: 2400.0, compressive: 180.0, tensile: 10.0, hardness: 7.0, melt_c: 1000.0, ..BASE },
    GLASS = 18 => Material { name: "glass", kind: Kind::Transparent, albedo: [0.95, 0.97, 0.96], roughness: 0.02, density: 2500.0, compressive: 50.0, tensile: 3.0, hardness: 1.5, conductivity: 1.0, melt_c: 1400.0, ..BASE },
    OAK_LOG = 19 => Material { name: "oak log", albedo: [0.30, 0.21, 0.13], roughness: 0.85, density: 750.0, compressive: 40.0, tensile: 60.0, hardness: 1.8, conductivity: 0.17, specific_heat: 1700.0, flammability: 0.35, ignition_c: 300.0, absorption: 0.1, ..BASE },
    PINE_LOG = 20 => Material { name: "pine log", albedo: [0.25, 0.17, 0.11], roughness: 0.85, density: 550.0, compressive: 35.0, tensile: 50.0, hardness: 1.5, conductivity: 0.12, specific_heat: 1700.0, flammability: 0.45, ignition_c: 280.0, absorption: 0.1, ..BASE },
    PLANKS = 21 => Material { name: "planks", albedo: [0.52, 0.38, 0.22], roughness: 0.7, density: 600.0, compressive: 30.0, tensile: 40.0, hardness: 1.2, conductivity: 0.12, specific_heat: 1700.0, flammability: 0.5, ignition_c: 280.0, absorption: 0.12, ..BASE },
    LEAVES = 22 => Material { name: "leaves", kind: Kind::Foliage, albedo: [0.10, 0.25, 0.05], roughness: 0.8, density: 200.0, compressive: 0.01, tensile: 0.05, hardness: 0.1, conductivity: 0.1, specific_heat: 2500.0, flammability: 0.6, ignition_c: 250.0, absorption: 0.5, ..BASE },
    PINE_NEEDLES = 23 => Material { name: "pine needles", kind: Kind::Foliage, albedo: [0.05, 0.16, 0.08], roughness: 0.8, density: 200.0, compressive: 0.01, tensile: 0.05, hardness: 0.1, conductivity: 0.1, specific_heat: 2500.0, flammability: 0.7, ignition_c: 240.0, absorption: 0.5, ..BASE },
    CHARCOAL = 24 => Material { name: "charcoal", albedo: [0.04, 0.04, 0.04], roughness: 0.95, density: 300.0, compressive: 2.0, tensile: 1.0, hardness: 0.5, conductivity: 0.08, flammability: 0.2, ignition_c: 350.0, absorption: 0.2, ..BASE },
    ASH = 25 => Material { name: "ash", albedo: [0.35, 0.34, 0.33], roughness: 1.0, density: 400.0, compressive: 0.01, tensile: 0.0, hardness: 0.1, conductivity: 0.1, absorption: 0.4, ..BASE },
    COBBLESTONE = 26 => Material { name: "cobblestone", albedo: [0.36, 0.35, 0.34], roughness: 0.85, density: 2400.0, compressive: 60.0, tensile: 1.5, hardness: 4.0, ..BASE },
    STONE_BRICK = 27 => Material { name: "stone brick", albedo: [0.45, 0.43, 0.41], roughness: 0.7, density: 2300.0, compressive: 80.0, tensile: 2.5, hardness: 4.0, ..BASE },
    COAL_ORE = 28 => Material { name: "coal ore", albedo: [0.10, 0.10, 0.10], roughness: 0.7, density: 1400.0, compressive: 30.0, tensile: 2.0, hardness: 3.0, flammability: 0.05, ignition_c: 450.0, ..BASE },
    IRON_ORE = 29 => Material { name: "iron ore", albedo: [0.45, 0.30, 0.22], roughness: 0.6, metallic: 0.2, density: 4500.0, compressive: 150.0, tensile: 10.0, hardness: 5.0, melt_c: 1500.0, ..BASE },
    COPPER_ORE = 30 => Material { name: "copper ore", albedo: [0.30, 0.45, 0.35], roughness: 0.6, metallic: 0.3, density: 4000.0, compressive: 120.0, tensile: 8.0, hardness: 4.0, melt_c: 1085.0, ..BASE },
    GOLD_ORE = 31 => Material { name: "gold ore", albedo: [0.70, 0.55, 0.20], roughness: 0.4, metallic: 0.6, density: 5000.0, compressive: 100.0, tensile: 6.0, hardness: 4.5, melt_c: 1064.0, ..BASE },
    TORCH_FLAME = 32 => Material { name: "torch flame", kind: Kind::Foliage, albedo: [1.0, 0.7, 0.3], roughness: 1.0, emission: [900.0, 420.0, 110.0], density: 1.0, compressive: 0.0, tensile: 0.0, hardness: 0.0, conductivity: 0.1, absorption: 0.0, ..BASE },
    GLOWSTONE = 33 => Material { name: "lantern glass", kind: Kind::Solid, albedo: [1.0, 0.85, 0.6], roughness: 0.3, emission: [600.0, 450.0, 250.0], density: 2500.0, compressive: 20.0, tensile: 1.0, hardness: 1.0, ..BASE },
    IRON = 34 => Material { name: "iron", albedo: [0.56, 0.57, 0.58], roughness: 0.35, metallic: 1.0, density: 7870.0, compressive: 250.0, tensile: 400.0, hardness: 8.0, conductivity: 80.0, specific_heat: 450.0, melt_c: 1538.0, absorption: 0.02, ..BASE },
    TNT = 35 => Material { name: "explosive", albedo: [0.70, 0.12, 0.08], roughness: 0.8, density: 1600.0, compressive: 1.0, tensile: 0.5, hardness: 0.3, flammability: 1.0, ignition_c: 200.0, ..BASE },
    FIRE = 36 => Material { name: "fire", kind: Kind::Foliage, albedo: [1.0, 0.5, 0.1], roughness: 1.0, emission: [1800.0, 700.0, 120.0], density: 1.0, compressive: 0.0, tensile: 0.0, hardness: 0.0, conductivity: 0.1, absorption: 0.0, ..BASE },
    EMBER = 37 => Material { name: "embers", albedo: [0.1, 0.03, 0.01], roughness: 0.9, emission: [180.0, 45.0, 8.0], density: 300.0, compressive: 0.5, tensile: 0.1, hardness: 0.2, conductivity: 0.1, flammability: 0.3, ignition_c: 300.0, ..BASE },
    WET_DIRT = 38 => Material { name: "mud", albedo: [0.16, 0.11, 0.07], roughness: 0.5, density: 1700.0, compressive: 0.2, tensile: 0.02, hardness: 0.5, conductivity: 1.5, absorption: 0.2, ..BASE },
    ROOF_TILE = 39 => Material { name: "roof tile", albedo: [0.45, 0.16, 0.10], roughness: 0.6, density: 1900.0, compressive: 40.0, tensile: 2.0, hardness: 2.0, ..BASE },
    WHITEWASH = 40 => Material { name: "whitewash plaster", albedo: [0.82, 0.80, 0.74], roughness: 0.9, density: 1700.0, compressive: 5.0, tensile: 0.5, hardness: 1.0, ..BASE },
    WOOL = 41 => Material { name: "wool", albedo: [0.75, 0.72, 0.66], roughness: 1.0, density: 150.0, compressive: 0.01, tensile: 0.5, hardness: 0.2, conductivity: 0.04, flammability: 0.5, ignition_c: 230.0, absorption: 0.7, ..BASE },
    STEEL = 42 => Material { name: "steel", albedo: [0.62, 0.63, 0.65], roughness: 0.25, metallic: 1.0, density: 7850.0, compressive: 400.0, tensile: 500.0, hardness: 9.0, conductivity: 50.0, specific_heat: 490.0, melt_c: 1450.0, absorption: 0.02, ..BASE },
    BEDROCK = 43 => Material { name: "bedrock", albedo: [0.08, 0.08, 0.09], roughness: 0.9, density: 3300.0, compressive: 1.0e6, tensile: 1.0e6, hardness: f32::INFINITY, ..BASE },
    RHYOLITE = 44 => Material { name: "rhyolite", albedo: [0.55, 0.45, 0.42], roughness: 0.8, density: 2500.0, compressive: 150.0, tensile: 9.0, hardness: 5.0, melt_c: 800.0, ..BASE },
    CONGLOMERATE = 45 => Material { name: "conglomerate", albedo: [0.48, 0.40, 0.33], roughness: 0.95, density: 2400.0, compressive: 40.0, tensile: 2.0, hardness: 2.5, ..BASE },
    CHALK = 46 => Material { name: "chalk", albedo: [0.88, 0.87, 0.82], roughness: 0.95, density: 2000.0, compressive: 20.0, tensile: 1.0, hardness: 1.2, ..BASE },
    GNEISS = 47 => Material { name: "gneiss", albedo: [0.38, 0.36, 0.36], roughness: 0.7, density: 2750.0, compressive: 180.0, tensile: 10.0, hardness: 6.0, ..BASE },
    FARMLAND = 48 => Material { name: "farmland", albedo: [0.22, 0.14, 0.08], roughness: 0.95, density: 1300.0, compressive: 0.3, tensile: 0.02, hardness: 0.5, absorption: 0.2, ..BASE },
    WHEAT = 49 => Material { name: "wheat", kind: Kind::Foliage, albedo: [0.72, 0.62, 0.30], roughness: 0.9, density: 100.0, compressive: 0.0, tensile: 0.01, hardness: 0.05, flammability: 0.8, ignition_c: 220.0, absorption: 0.5, ..BASE },
    GLOWING_ROCK = 50 => Material { name: "cooling rock", albedo: [0.15, 0.12, 0.11], roughness: 0.8, emission: [60.0, 12.0, 2.0], density: 2900.0, compressive: 150.0, tensile: 9.0, hardness: 5.0, conductivity: 1.8, ..BASE },
}

impl MaterialId {
    pub fn get(self) -> &'static Material {
        MATERIALS.get(self.0 as usize).unwrap_or(&MATERIALS[0])
    }

    pub fn is_air(self) -> bool {
        self.0 == 0
    }

    pub fn is_solid(self) -> bool {
        matches!(self.get().kind, Kind::Solid | Kind::Transparent)
    }

    pub fn is_emissive(self) -> bool {
        self.get().emission.iter().any(|&e| e > 0.0)
    }
}

pub fn count() -> usize {
    MATERIALS.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_index_their_entries() {
        assert_eq!(ids::AIR.get().name, "air");
        assert_eq!(ids::GRANITE.get().name, "granite");
        assert_eq!(ids::GLOWING_ROCK.get().name, "cooling rock");
        assert_eq!(count(), ids::GLOWING_ROCK.0 as usize + 1);
        assert!(ids::LAVA.is_emissive());
        assert!(!ids::AIR.is_solid());
    }
}
