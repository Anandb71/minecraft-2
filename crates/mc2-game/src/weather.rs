//! Weather: clouds, rain, storms and wind, drifting from one to the next.
//!
//! The sky moves between clear, cloudy, rain and storm, each lasting a few
//! minutes; cloud cover, rain and wind ease toward what the sky calls for.
//! Storms throw lightning near the player now and then: a flash, and what
//! the bolt strikes may catch fire. Fire feels the wind and the rain.

use crate::Voxels;
use crate::fire::Fire;
use crate::input::Time;
use crate::player::Body;
use bevy_ecs::prelude::*;
use glam::{DVec3, IVec3, Vec2};
use mc2_voxel::coords::{BlockPos, VOXELS_PER_BLOCK};
use mc2_voxel::world::VoxelWorld;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sky {
    Clear,
    Cloudy,
    Rain,
    Storm,
}

impl Sky {
    pub fn parse(s: &str) -> Option<Sky> {
        Some(match s {
            "clear" => Sky::Clear,
            "cloudy" => Sky::Cloudy,
            "rain" => Sky::Rain,
            "storm" => Sky::Storm,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Sky::Clear => "clear",
            Sky::Cloudy => "cloudy",
            Sky::Rain => "rain",
            Sky::Storm => "storm",
        }
    }

    /// Cloud cover, rain, and the least and most wind (m/s) it brings.
    fn calls_for(self) -> (f32, f32, f32, f32) {
        match self {
            Sky::Clear => (0.3, 0.0, 1.0, 4.0),
            Sky::Cloudy => (0.65, 0.0, 3.0, 7.0),
            Sky::Rain => (0.85, 0.6, 4.0, 9.0),
            Sky::Storm => (0.95, 1.0, 8.0, 15.0),
        }
    }

    /// How long it lasts, seconds, least and most.
    fn lasts(self) -> (f32, f32) {
        match self {
            Sky::Clear => (120.0, 360.0),
            Sky::Cloudy => (90.0, 240.0),
            Sky::Rain => (60.0, 180.0),
            Sky::Storm => (40.0, 120.0),
        }
    }

    /// What follows it, given a roll in `0..1`.
    fn next(self, roll: f32) -> Sky {
        match self {
            Sky::Clear if roll < 0.7 => Sky::Cloudy,
            Sky::Clear => Sky::Clear,
            Sky::Cloudy if roll < 0.45 => Sky::Rain,
            Sky::Cloudy if roll < 0.8 => Sky::Clear,
            Sky::Cloudy => Sky::Cloudy,
            Sky::Rain if roll < 0.3 => Sky::Storm,
            Sky::Rain if roll < 0.8 => Sky::Cloudy,
            Sky::Rain => Sky::Rain,
            Sky::Storm if roll < 0.7 => Sky::Rain,
            Sky::Storm => Sky::Cloudy,
        }
    }
}

#[derive(Resource)]
pub struct Weather {
    pub sky: Sky,
    /// Seconds until the sky changes.
    left: f32,
    /// Cloud cover and rain, 0..1, eased toward what the sky calls for.
    pub cover: f32,
    pub rain: f32,
    /// How wet open ground is, 0..1: it soaks while it rains and dries
    /// slowly after.
    pub wetness: f32,
    /// Wind at the ground, metres a second (x, z).
    pub wind: Vec2,
    wind_goal: Vec2,
    /// Brightness of the last lightning flash, fading.
    pub flash: f32,
    /// Where lightning struck since the last frame, metres.
    pub strikes: Vec<DVec3>,
    next_strike: f32,
    rng: u64,
    /// Holds the sky as it is (captures, photo mode).
    pub frozen: bool,
}

impl Default for Weather {
    fn default() -> Self {
        Self {
            sky: Sky::Clear,
            left: 240.0,
            cover: 0.3,
            rain: 0.0,
            wetness: 0.0,
            wind: Vec2::new(2.0, 1.0),
            wind_goal: Vec2::new(2.0, 1.0),
            flash: 0.0,
            strikes: Vec::new(),
            next_strike: 8.0,
            rng: 0x9e37_79b9_97f4_a7c1,
            frozen: false,
        }
    }
}

/// Seconds of full rain to soak open ground, and to dry it after.
const SOAK_S: f32 = 45.0;
const DRY_S: f32 = 240.0;
/// Cover, rain and wind ease toward their goals over about this long.
const EASE_S: f32 = 20.0;
const WIND_EASE_S: f32 = 8.0;
/// Seconds between lightning strikes in a storm, least and most.
const STRIKE_EVERY: (f32, f32) = (4.0, 15.0);
/// Lightning strikes this far from the player, metres, least and most.
const STRIKE_RANGE: (f32, f32) = (25.0, 90.0);

impl Weather {
    fn roll(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 40) as f32 / (1u64 << 24) as f32
    }

    fn between(&mut self, (lo, hi): (f32, f32)) -> f32 {
        lo + (hi - lo) * self.roll()
    }

    /// Sets the sky at once, cover, rain and wind already there.
    pub fn set(&mut self, sky: Sky) {
        self.sky = sky;
        let (cover, rain, lo, hi) = sky.calls_for();
        self.cover = cover;
        self.rain = rain;
        self.wetness = if rain > 0.0 { 1.0 } else { 0.0 };
        let dir = Vec2::from_angle(self.roll() * std::f32::consts::TAU);
        self.wind_goal = dir * (lo + hi) * 0.5;
        self.wind = self.wind_goal;
        self.left = self.between(sky.lasts());
    }

    /// Advances the weather `dt` seconds.
    pub fn advance(&mut self, dt: f32) {
        self.flash = (self.flash - dt * 4.0).max(0.0);
        self.wetness = if self.rain > 0.05 {
            (self.wetness + dt * self.rain / SOAK_S).min(1.0)
        } else {
            (self.wetness - dt / DRY_S).max(0.0)
        };
        if self.frozen {
            return;
        }
        self.left -= dt;
        if self.left <= 0.0 {
            let roll = self.roll();
            self.sky = self.sky.next(roll);
            self.left = self.between(self.sky.lasts());
            let (_, _, lo, hi) = self.sky.calls_for();
            let turn = (self.roll() - 0.5) * 1.5;
            let dir = if self.wind_goal.length() > 0.1 {
                Vec2::from_angle(turn).rotate(self.wind_goal.normalize())
            } else {
                Vec2::X
            };
            self.wind_goal = dir * self.between((lo, hi));
        }
        let (cover, rain, _, _) = self.sky.calls_for();
        let k = (dt / EASE_S).min(1.0);
        self.cover += (cover - self.cover) * k;
        self.rain += (rain - self.rain) * k;
        self.wind += (self.wind_goal - self.wind) * (dt / WIND_EASE_S).min(1.0);
    }

    /// In a storm, now and then, a bolt somewhere around `around`: returns
    /// where it came down, the top of what it hit.
    fn lightning(&mut self, world: &VoxelWorld, around: DVec3, dt: f32) -> Option<DVec3> {
        if self.sky != Sky::Storm || self.rain < 0.7 {
            return None;
        }
        self.next_strike -= dt;
        if self.next_strike > 0.0 {
            return None;
        }
        self.next_strike = self.between(STRIKE_EVERY);
        let angle = self.roll() * std::f32::consts::TAU;
        let dist = self.between(STRIKE_RANGE);
        let x = around.x + f64::from(angle.cos() * dist);
        let z = around.z + f64::from(angle.sin() * dist);
        // Down from well above the player to the first solid thing.
        let top = ((around.y + 80.0) * f64::from(VOXELS_PER_BLOCK)) as i32;
        let (vx, vz) = (
            (x * f64::from(VOXELS_PER_BLOCK)) as i32,
            (z * f64::from(VOXELS_PER_BLOCK)) as i32,
        );
        for step in 0..(160 * 4) {
            let v = IVec3::new(vx, top - step * 4, vz);
            if world.voxel(v).is_solid()
                || world.voxel(v).get().kind == mc2_voxel::material::Kind::Foliage
            {
                self.flash = 1.0;
                return Some(v.as_dvec3() / f64::from(VOXELS_PER_BLOCK));
            }
        }
        None
    }
}

/// Every frame: the sky moves on, lightning strikes, fire hears of it.
pub fn update_weather(
    voxels: Res<Voxels>,
    time: Res<Time>,
    mut weather: ResMut<Weather>,
    mut fire: ResMut<Fire>,
    players: Query<&Body>,
) {
    let dt = time.dt;
    weather.advance(dt);
    weather.strikes.clear();
    if let Some(body) = players.iter().next()
        && let Some(at) = weather.lightning(&voxels.0, body.feet, dt)
    {
        weather.strikes.push(at);
        let b = BlockPos((at * f64::from(VOXELS_PER_BLOCK)).floor().as_ivec3() >> 4);
        fire.ignite.push((b, false));
        fire.ignite.push((BlockPos(b.0 - IVec3::Y), false));
    }
    fire.wind = weather.wind;
    fire.rain = weather.rain;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sky_moves_through_every_state_and_eases_between_them() {
        let mut w = Weather::default();
        let mut seen = [false; 4];
        let mut rained = 0.0f32;
        for _ in 0..(3600 * 10) {
            w.advance(0.5);
            seen[w.sky as usize] = true;
            rained = rained.max(w.rain);
            assert!((0.0..=1.0).contains(&w.cover) && (0.0..=1.0).contains(&w.rain));
            assert!(w.wind.length() < 16.0);
        }
        assert!(seen.iter().all(|&s| s), "{seen:?}");
        assert!(rained > 0.5);
    }

    #[test]
    fn frozen_weather_holds_and_storms_strike_the_ground() {
        let mut w = Weather::default();
        w.set(Sky::Storm);
        w.frozen = true;
        w.advance(1000.0);
        assert_eq!(w.sky, Sky::Storm);
        assert_eq!(w.rain, 1.0);
        let mut world = VoxelWorld::new();
        world.fill_box(
            IVec3::ZERO,
            IVec3::new(4095, 63, 4095),
            mc2_voxel::material::ids::GRANITE,
        );
        let mut struck = None;
        for _ in 0..200 {
            if let Some(at) = w.lightning(&world, DVec3::new(128.0, 5.0, 128.0), 0.1) {
                struck = Some(at);
                break;
            }
        }
        let at = struck.expect("a storm strikes within 20 s");
        assert!((at.y - 4.0).abs() < 0.3, "struck at {at}");
        assert_eq!(w.flash, 1.0);
    }
}
