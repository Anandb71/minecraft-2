//! World clock and the sky it implies: sun and moon directions from time of
//! day, day of year and latitude, and their illuminance at the top of the
//! atmosphere (the renderer applies atmospheric transmittance).
//!
//! Positions use the standard low-precision formulas: solar declination
//! from the day of year, hour angle from local solar time, and a moon that
//! trails the sun by its synodic phase along a path inclined 5.1 degrees to
//! the ecliptic. World axes: +X east, +Y up, -Z north.

use bevy_ecs::prelude::*;
use glam::{DVec3, Vec3};
use std::f64::consts::{PI, TAU};

/// Extraterrestrial solar illuminance, lux.
pub const SOLAR_ILLUMINANCE: f32 = 127_500.0;
/// Full moon illuminance above the atmosphere, lux.
pub const FULL_MOON_ILLUMINANCE: f32 = 0.32;
const SYNODIC_MONTH_DAYS: f64 = 29.530_588;
const OBLIQUITY: f64 = 23.44 * PI / 180.0;
const MOON_INCLINATION: f64 = 5.14 * PI / 180.0;

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct WorldClock {
    /// Days since the world began; the fraction is the time of day.
    pub days: f64,
    /// Game seconds per real second: 72 makes a day last 20 minutes.
    pub speed: f64,
    pub paused: bool,
    /// Observer latitude, degrees north.
    pub latitude: f64,
    /// Day of year at world start, for the seasons.
    pub start_day_of_year: f64,
}

impl Default for WorldClock {
    fn default() -> Self {
        Self {
            days: 0.4,
            speed: 72.0,
            paused: false,
            latitude: 47.0,
            start_day_of_year: 172.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sky {
    /// Unit vectors toward the sun and moon.
    pub sun_dir: Vec3,
    pub moon_dir: Vec3,
    pub sun_illuminance: f32,
    /// Scaled by lunar phase.
    pub moon_illuminance: f32,
    /// 0 new moon, 0.5 full moon.
    pub moon_phase: f32,
    /// Rotation of the star field about the celestial pole, radians.
    pub sidereal_angle: f32,
}

impl WorldClock {
    pub fn advance(&mut self, dt: f64) {
        if !self.paused {
            self.days += dt * self.speed / 86_400.0;
        }
    }

    pub fn hour(&self) -> f64 {
        self.days.rem_euclid(1.0) * 24.0
    }

    /// Jumps to an hour of the current day, keeping the date.
    pub fn set_hour(&mut self, hour: f64) {
        self.days = self.days.floor() + hour.rem_euclid(24.0) / 24.0;
    }

    pub fn sky(&self) -> Sky {
        let lat = self.latitude.to_radians();
        let day_of_year = self.start_day_of_year + self.days;
        // Sun: declination over the year, hour angle over the day.
        let sun_ecliptic_longitude = TAU * (day_of_year - 80.0) / 365.25;
        let sun_declination = (OBLIQUITY.sin() * sun_ecliptic_longitude.sin()).asin();
        let hour_angle = TAU * (self.days.rem_euclid(1.0) - 0.5);
        let sun = horizon_vector(lat, sun_declination, hour_angle);

        // Moon: the phase advances the moon eastward from the sun, so it
        // rises later each day; its declination follows the ecliptic at its
        // own longitude plus the orbital inclination.
        let phase = (self.days / SYNODIC_MONTH_DAYS + 0.5).rem_euclid(1.0);
        let moon_longitude = sun_ecliptic_longitude + TAU * phase;
        let node_angle = TAU * self.days / 27.212;
        let moon_declination = (OBLIQUITY.sin() * moon_longitude.sin()).asin()
            + MOON_INCLINATION * (moon_longitude - node_angle).sin();
        let moon = horizon_vector(lat, moon_declination, hour_angle - TAU * phase);

        // Illuminated fraction, sharpened by the opposition surge so a half
        // moon gives about a tenth of full moon light, as observed.
        let lit_fraction = 0.5 * (1.0 - (TAU * phase).cos());
        let moon_illuminance = FULL_MOON_ILLUMINANCE * lit_fraction.powi(3) as f32;
        Sky {
            sun_dir: sun.as_vec3(),
            moon_dir: moon.as_vec3(),
            sun_illuminance: SOLAR_ILLUMINANCE,
            moon_illuminance,
            moon_phase: phase as f32,
            sidereal_angle: (hour_angle + sun_ecliptic_longitude).rem_euclid(TAU) as f32,
        }
    }
}

/// Direction of a body at `declination` and `hour_angle` for an observer at
/// `latitude` (radians), in world axes.
fn horizon_vector(latitude: f64, declination: f64, hour_angle: f64) -> DVec3 {
    let up =
        latitude.sin() * declination.sin() + latitude.cos() * declination.cos() * hour_angle.cos();
    let east = -declination.cos() * hour_angle.sin();
    let north =
        latitude.cos() * declination.sin() - latitude.sin() * declination.cos() * hour_angle.cos();
    DVec3::new(east, up, -north).normalize()
}

pub fn advance_clock(time: Res<crate::input::Time>, mut clock: ResMut<WorldClock>) {
    clock.advance(f64::from(time.dt));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clock(hour: f64, day_of_year: f64, latitude: f64) -> WorldClock {
        let mut c = WorldClock {
            days: 0.0,
            latitude,
            start_day_of_year: day_of_year,
            ..Default::default()
        };
        c.set_hour(hour);
        c
    }

    #[test]
    fn equinox_sun_rises_east_peaks_south_sets_west() {
        let lat = 47.0;
        let dawn = clock(6.0, 80.0, lat).sky().sun_dir;
        assert!(dawn.y.abs() < 0.03 && dawn.x > 0.99, "{dawn}");
        let noon = clock(12.0, 80.0, lat).sky().sun_dir;
        let elevation = noon.y.asin().to_degrees();
        assert!(
            (elevation - (90.0 - 47.0) as f32).abs() < 0.5,
            "{elevation}"
        );
        // South is +Z.
        assert!(noon.z > 0.0 && noon.x.abs() < 1e-3);
        let dusk = clock(18.0, 80.0, lat).sky().sun_dir;
        assert!(dusk.y.abs() < 0.03 && dusk.x < -0.99);
    }

    #[test]
    fn solstice_noon_elevation_follows_declination() {
        let summer = clock(12.0, 172.0, 47.0).sky().sun_dir.y.asin().to_degrees();
        let winter = clock(12.0, 355.0, 47.0).sky().sun_dir.y.asin().to_degrees();
        assert!((summer - 66.4).abs() < 0.7, "summer {summer}");
        assert!((winter - 19.6).abs() < 0.7, "winter {winter}");
    }

    #[test]
    fn full_moon_opposes_the_sun_and_new_moon_is_dark() {
        let mut c = clock(0.0, 80.0, 0.0);
        // Find a full moon: phase 0.5.
        c.days = SYNODIC_MONTH_DAYS * 1.0;
        let full = c.sky();
        assert!((full.moon_phase - 0.5).abs() < 1e-3);
        assert!(full.sun_dir.dot(full.moon_dir) < -0.95);
        assert!((full.moon_illuminance - FULL_MOON_ILLUMINANCE).abs() < 1e-3);
        c.days = SYNODIC_MONTH_DAYS * 1.5;
        let new = c.sky();
        assert!(new.moon_illuminance < 1e-4);
        assert!(new.sun_dir.dot(new.moon_dir) > 0.95);
    }

    #[test]
    fn set_hour_keeps_the_date_and_advance_respects_pause() {
        let mut c = WorldClock {
            days: 3.25,
            ..Default::default()
        };
        c.set_hour(18.0);
        assert!((c.days - 3.75).abs() < 1e-12);
        c.paused = true;
        c.advance(100.0);
        assert!((c.days - 3.75).abs() < 1e-12);
        c.paused = false;
        // 1200 real seconds at 72x is one game day.
        c.advance(1200.0);
        assert!((c.days - 4.75).abs() < 1e-9);
        assert!((c.hour() - 18.0).abs() < 1e-6);
    }
}
