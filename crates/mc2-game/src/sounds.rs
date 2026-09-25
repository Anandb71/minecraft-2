//! What the world sounds like this frame, for whoever plays it.
//!
//! Systems here and elsewhere leave `Heard` events (a sound, where, how
//! loud) and set the beds (rain, fire, wind, an engine); the app turns them
//! into audio through the place's acoustics. Nothing here makes a sound.

use crate::Voxels;
use crate::fire::Fire;
use crate::interact::Interaction;
use crate::physics::Physics;
use crate::player::{Body, Player};
use crate::vehicles::Garage;
use crate::weather::Weather;
use bevy_ecs::prelude::*;
use glam::DVec3;
use mc2_audio::{Beds, Ground, Sound};
use mc2_voxel::material::{Kind, MaterialId, ids};

/// A walker's stride, metres: one step each.
const STRIDE: f64 = 0.75;
/// Burning blocks within this many metres make the fire bed louder, up to
/// `LOUDEST_FIRE` of them.
const FIRE_NEAR: f64 = 24.0;
const LOUDEST_FIRE: f32 = 40.0;

/// A sound somewhere in the world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Heard {
    pub sound: Sound,
    pub at: DVec3,
    pub gain: f32,
}

/// This frame's sounds and the beds' levels.
#[derive(Resource, Default)]
pub struct Sounds {
    pub events: Vec<Heard>,
    pub beds: Beds,
}

impl Sounds {
    pub fn hear(&mut self, sound: Sound, at: DVec3, gain: f32) {
        self.events.push(Heard { sound, at, gain });
    }

    /// Hands over this frame's events.
    pub fn take(&mut self) -> Vec<Heard> {
        std::mem::take(&mut self.events)
    }
}

/// Distance walked since the last footstep.
#[derive(Component, Default, Debug, Clone, Copy)]
pub struct Stride(pub f64);

/// How a foot on `m` sounds.
pub fn ground_of(m: MaterialId) -> Ground {
    match m {
        ids::GRAVEL => Ground::Gravel,
        ids::PLANKS | ids::DARK_PLANKS | ids::OAK_LOG | ids::PINE_LOG | ids::BIRCH_LOG => {
            Ground::Wood
        }
        ids::GRASS | ids::DIRT | ids::SAND | ids::SNOW | ids::CLAY | ids::MOSS => Ground::Soft,
        _ => match m.get().kind {
            Kind::Liquid => Ground::Water,
            Kind::Foliage => Ground::Soft,
            _ => Ground::Hard,
        },
    }
}

/// How hard something breaking sounds, from its material (0..10).
pub fn hardness_of(m: MaterialId) -> f32 {
    m.get().hardness.clamp(0.0, 10.0)
}

/// Every fixed tick: a footstep for every stride a walker covers on the
/// ground, sounding of what is underfoot (or the water it wades in).
pub fn footsteps(
    voxels: Res<Voxels>,
    mut sounds: ResMut<Sounds>,
    mut walkers: Query<(&Body, &mut Stride, Option<&Player>)>,
) {
    for (b, mut stride, player) in &mut walkers {
        if player.is_some_and(|p| !p.on_ground || p.seated) {
            stride.0 = 0.0;
            continue;
        }
        let moved = b.feet - b.prev_feet;
        stride.0 += moved.x.hypot(moved.z);
        if stride.0 < STRIDE {
            continue;
        }
        stride.0 -= STRIDE;
        let at = |dy: f64| {
            voxels
                .0
                .voxel(((b.feet + DVec3::Y * dy) * 16.0).floor().as_ivec3())
        };
        let wading = at(0.05).get().kind == Kind::Liquid;
        let ground = if wading {
            Ground::Water
        } else {
            ground_of(at(-0.03))
        };
        // Someone else's steps are a little quieter than your own.
        let gain = if player.is_some() { 0.5 } else { 0.35 };
        sounds.hear(Sound::Step(ground), b.feet, gain);
    }
}

/// Every frame: blasts, thunder, blocks broken and set down, and the
/// beds from the weather, the fires near and the car being driven. Runs
/// before the fire takes the blasts.
#[allow(clippy::too_many_arguments)]
pub fn listen(
    mut sounds: ResMut<Sounds>,
    mut interaction: ResMut<Interaction>,
    physics: Res<Physics>,
    weather: Res<Weather>,
    fire: Res<Fire>,
    garage: Res<Garage>,
    players: Query<&Body, With<Player>>,
) {
    for &(centre, radius) in &physics.blasts_out {
        sounds.hear(Sound::Blast { radius }, centre, 1.0);
    }
    let ear = players.iter().next().map(|b| b.feet);
    for &strike in &weather.strikes {
        let distance = ear.map_or(1_000.0, |e| e.distance(strike) as f32);
        sounds.hear(Sound::Thunder { distance }, strike, 1.0);
    }
    for (sound, at) in interaction.noises.drain(..) {
        sounds.hear(sound, at, 0.6);
    }
    let near_fire = ear.map_or(0, |e| fire.burning_near(e, FIRE_NEAR));
    let engine = garage.driven().and_then(|car| {
        physics
            .host
            .body(car.body)
            .map(|b| (b.vel.length(), garage.throttle))
    });
    sounds.beds = Beds {
        rain: weather.rain,
        fire: (near_fire as f32 / LOUDEST_FIRE).min(1.0),
        wind: (weather.wind.length() / 12.0).min(1.0),
        engine,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::IVec3;
    use mc2_voxel::world::VoxelWorld;

    #[test]
    fn grounds_sound_of_what_they_are() {
        assert_eq!(ground_of(ids::GRASS), Ground::Soft);
        assert_eq!(ground_of(ids::GRANITE), Ground::Hard);
        assert_eq!(ground_of(ids::PLANKS), Ground::Wood);
        assert_eq!(ground_of(ids::GRAVEL), Ground::Gravel);
        assert_eq!(ground_of(ids::WATER), Ground::Water);
        assert_eq!(ground_of(ids::LEAVES), Ground::Soft);
    }

    #[test]
    fn walking_makes_a_step_a_stride_on_what_is_underfoot() {
        let mut game = crate::Game::new();
        let mut w = VoxelWorld::new();
        w.fill_box(IVec3::ZERO, IVec3::new(511, 31, 511), ids::PLANKS);
        game.world.resource_mut::<Voxels>().0 = w;
        game.spawn_player(DVec3::new(8.0, 2.0, 8.0), 0.0, 0.0);
        for _ in 0..30 {
            game.update(1.0 / 60.0);
        }
        game.world.resource_mut::<Sounds>().take();
        game.input().captured = true;
        game.input().key_down(crate::input::Key::Forward);
        let mut steps = Vec::new();
        for _ in 0..120 {
            game.update(1.0 / 60.0);
            steps.extend(game.world.resource_mut::<Sounds>().take());
        }
        let walked = {
            let mut q = game.world.query::<(&Body, &Player)>();
            q.iter(&game.world).next().unwrap().0.feet.z - 8.0
        };
        let expected = (walked / STRIDE) as usize;
        assert!(
            steps.len() + 1 >= expected && steps.len() <= expected + 1,
            "{} steps over {walked} m",
            steps.len()
        );
        assert!(steps.iter().all(|h| h.sound == Sound::Step(Ground::Wood)));
    }
}
