//! Villagers: the people who live in the villages.
//!
//! Each lives in one of its village's houses. By day they go out to the
//! square, to one another's doors and back to their own, standing a while
//! at each; at dusk they walk home and go in. Villages near the player are
//! peopled as it comes and emptied again once it has gone.

use crate::character::{Character, Look};
use crate::clock::WorldClock;
use crate::input::Time;
use crate::nav;
use crate::player::{Body, EYE, Player};
use crate::{Streaming, Voxels};
use bevy_ecs::prelude::*;
use glam::{DVec3, Vec3};
use mc2_core::FxHashMap;
use mc2_worldgen::settlement::Home;
use std::sync::Arc;

/// Villages whose square is this near the player are peopled, metres.
pub const PEOPLE_M: f64 = 112.0;
/// And emptied again beyond this.
pub const EMPTY_M: f64 = 176.0;
/// At most this many live in one village.
const MAX_PER_VILLAGE: usize = 8;
/// Walking pace, metres a second.
const PACE: f64 = 1.25;
/// Turning speed, radians a second.
const TURN: f32 = 5.0;
/// The hours people are out.
const RISE: f64 = 6.5;
const DUSK: f64 = 20.25;
/// A villager turns its head to someone this near, metres.
const NOTICE_M: f64 = 5.0;
/// Path searches a tick, across every villager.
const SEARCHES: usize = 2;

#[derive(Clone, Debug, PartialEq)]
pub enum Doing {
    /// Standing where it is until the game has run to `until` seconds.
    Idle { until: f64 },
    /// Walking a path, going in at the end if `inside`.
    Walking {
        path: Vec<DVec3>,
        next: usize,
        inside: bool,
    },
    /// In at home.
    Indoors,
}

#[derive(Component, Debug)]
pub struct Villager {
    pub village: (i32, i32),
    pub home: Home,
    /// The square, and every home in the village, to walk to.
    plaza: DVec3,
    homes: Arc<[Home]>,
    pub doing: Doing,
    rng: u64,
}

impl Villager {
    fn roll(&mut self) -> f64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Somewhere to go next: the square most often, else a door.
    fn errand(&mut self) -> DVec3 {
        let r = self.roll();
        if r < 0.45 {
            let a = self.roll() * std::f64::consts::TAU;
            let d = 2.0 + self.roll() * 4.0;
            self.plaza + DVec3::new(a.cos() * d, 0.0, a.sin() * d)
        } else if r < 0.8 {
            let i = (self.roll() * self.homes.len() as f64) as usize;
            self.homes[i.min(self.homes.len() - 1)].door.as_dvec3()
        } else {
            self.home.door.as_dvec3()
        }
    }
}

fn is_day(hour: f64) -> bool {
    (RISE..DUSK).contains(&hour)
}

fn key_of(centre: Vec3) -> (i32, i32) {
    (centre.x.round() as i32, centre.z.round() as i32)
}

fn level_distance(a: DVec3, b: DVec3) -> f64 {
    (a.x - b.x).hypot(a.z - b.z)
}

/// Who lives in each peopled village.
#[derive(Resource, Default)]
pub struct Population {
    pub villages: FxHashMap<(i32, i32), Vec<Entity>>,
    /// Seconds until the next look round.
    wait: f32,
}

/// Twice a second: people the villages the player has come near, and
/// empty the ones it has left.
pub fn people_villages(
    mut commands: Commands,
    time: Res<Time>,
    streaming: Res<Streaming>,
    voxels: Res<Voxels>,
    clock: Res<WorldClock>,
    mut population: ResMut<Population>,
    players: Query<&Body, With<Player>>,
) {
    population.wait -= time.dt;
    if population.wait > 0.0 {
        return;
    }
    population.wait = 0.5;
    let Some(feet) = players.iter().next().map(|b| b.feet) else {
        return;
    };
    let Some(generator) = streaming.0.as_ref().map(|s| s.generator().clone()) else {
        return;
    };
    population.villages.retain(|&(x, z), people| {
        let square = DVec3::new(f64::from(x), feet.y, f64::from(z));
        let keep = level_distance(square, feet) < EMPTY_M;
        if !keep {
            for &e in people.iter() {
                if let Ok(mut e) = commands.get_entity(e) {
                    e.try_despawn();
                }
            }
        }
        keep
    });
    let r = PEOPLE_M as f32;
    let lo = Vec3::new(feet.x as f32 - r, -1000.0, feet.z as f32 - r);
    let hi = Vec3::new(feet.x as f32 + r, 4000.0, feet.z as f32 + r);
    let day = is_day(clock.hour());
    for v in generator.settlements.near(&generator.surface, lo, hi) {
        let key = key_of(v.centre);
        if v.homes.is_empty()
            || population.villages.contains_key(&key)
            || level_distance(v.centre.as_dvec3(), feet) > PEOPLE_M
        {
            continue;
        }
        let homes: Arc<[Home]> = v.homes.clone().into();
        let mut people = Vec::new();
        for (i, home) in homes.iter().take(MAX_PER_VILLAGE).enumerate() {
            // Out at their doors by day, in at home by night, once the
            // ground there has streamed in.
            let at = if day { home.door } else { home.inside }.as_dvec3();
            let Some(y) = nav::stand(&voxels.0, at.x, at.z, at.y) else {
                continue;
            };
            let seed =
                (key.0 as u64).wrapping_mul(0x9e37_79b9) ^ ((key.1 as u64) << 20) ^ (i as u64);
            let mut villager = Villager {
                village: key,
                home: *home,
                plaza: v.centre.as_dvec3(),
                homes: homes.clone(),
                doing: if day {
                    Doing::Idle { until: 0.0 }
                } else {
                    Doing::Indoors
                },
                rng: seed | 1,
            };
            let mut c = Character::new(Look::from_seed(seed));
            c.facing = (villager.roll() * std::f64::consts::TAU) as f32;
            let e = commands
                .spawn((Body::at(DVec3::new(at.x, y, at.z)), c, villager))
                .id();
            people.push(e);
        }
        // Try again later if none of the ground has streamed in yet.
        if !people.is_empty() {
            population.villages.insert(key, people);
        }
    }
}

/// Turns `from` toward `to` by at most `step` radians.
fn turn(from: f32, to: f32, step: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    let d = (to - from + PI).rem_euclid(TAU) - PI;
    from + d.clamp(-step, step)
}

/// Every fixed tick: decide where each villager is going, walk it there
/// along the ground as it is now, face the way it walks, and look at the
/// player when it is near.
pub fn walk_villagers(
    time: Res<Time>,
    clock: Res<WorldClock>,
    voxels: Res<Voxels>,
    players: Query<&Body, (With<Player>, Without<Villager>)>,
    mut villagers: Query<(&mut Villager, &mut Body, &mut Character), Without<Player>>,
) {
    let dt = time.fixed_dt;
    let now = time.elapsed;
    let day = is_day(clock.hour());
    let eye = players.iter().next().map(|b| b.feet + DVec3::Y * EYE);
    let world = &voxels.0;
    let mut searches = SEARCHES;
    let mut search = |from: DVec3, to: DVec3| {
        if searches == 0 {
            return None;
        }
        searches -= 1;
        nav::find(world, from, to)
    };
    for (mut v, mut body, mut c) in &mut villagers {
        body.prev_feet = body.feet;
        let v = &mut *v;
        // Decide.
        let going_home = matches!(v.doing, Doing::Walking { inside: true, .. });
        match v.doing {
            Doing::Indoors if day => {
                if let Some(path) = search(body.feet, v.home.door.as_dvec3()) {
                    v.doing = Doing::Walking {
                        path,
                        next: 0,
                        inside: false,
                    };
                }
            }
            Doing::Idle { .. } | Doing::Walking { .. } if !day && !going_home => {
                if let Some(path) = search(body.feet, v.home.inside.as_dvec3()) {
                    v.doing = Doing::Walking {
                        path,
                        next: 0,
                        inside: true,
                    };
                }
            }
            Doing::Idle { until } if day && now >= until => {
                let goal = v.errand();
                v.doing = match search(body.feet, goal) {
                    Some(path) => Doing::Walking {
                        path,
                        next: 0,
                        inside: false,
                    },
                    None => Doing::Idle { until: now + 2.0 },
                };
            }
            _ => {}
        }
        // Walk.
        if let Doing::Walking { path, next, inside } = &mut v.doing {
            let mut left = PACE * dt;
            while left > 0.0 && *next < path.len() {
                let target = path[*next];
                let d = DVec3::new(target.x - body.feet.x, 0.0, target.z - body.feet.z);
                let dist = d.length();
                if dist <= left {
                    body.feet.x = target.x;
                    body.feet.z = target.z;
                    left -= dist;
                    *next += 1;
                } else {
                    body.feet += d * (left / dist);
                    left = 0.0;
                }
            }
            let done = *next >= path.len();
            let indoors = *inside;
            // Keep to the ground as it is now; if the way has gone (a crater,
            // a new wall), stop and think again.
            match nav::stand(world, body.feet.x, body.feet.z, body.feet.y) {
                Some(y) => body.feet.y = y,
                None => {
                    body.feet = body.prev_feet;
                    v.doing = Doing::Idle { until: now + 1.0 };
                }
            }
            if done && matches!(v.doing, Doing::Walking { .. }) {
                let wait = 4.0 + v.roll() * 10.0;
                v.doing = if indoors {
                    Doing::Indoors
                } else {
                    Doing::Idle { until: now + wait }
                };
            }
        }
        let moved = body.feet - body.prev_feet;
        body.velocity = moved / dt;
        if moved.x.hypot(moved.z) > 1e-4 {
            let want = moved.x.atan2(moved.z) as f32;
            c.facing = turn(c.facing, want, TURN * dt as f32);
        }
        c.look_at = eye.filter(|e| e.distance(body.feet) < NOTICE_M);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::IVec3;
    use mc2_voxel::material::ids;
    use mc2_voxel::world::VoxelWorld;

    /// A 16 m floor, its top at 2 m, with a hut: walls a quarter metre thick
    /// round x 9..13, z 9..13, a doorway on its -z side.
    fn hamlet() -> VoxelWorld {
        let mut w = VoxelWorld::new();
        w.fill_box(IVec3::ZERO, IVec3::new(255, 31, 255), ids::GRANITE);
        w.fill_box(
            IVec3::new(144, 32, 144),
            IVec3::new(207, 79, 207),
            ids::PLANKS,
        );
        w.fill_box(IVec3::new(148, 32, 148), IVec3::new(203, 79, 203), ids::AIR);
        w.fill_box(IVec3::new(168, 32, 144), IVec3::new(183, 69, 147), ids::AIR);
        w
    }

    fn villager(home: Home) -> Villager {
        Villager {
            village: (0, 0),
            home,
            plaza: DVec3::new(4.0, 2.0, 4.0),
            homes: vec![home].into(),
            doing: Doing::Idle { until: 0.0 },
            rng: 7,
        }
    }

    fn run(game: &mut crate::Game, hour: f64, seconds: f64) {
        game.clock().set_hour(hour);
        game.clock().paused = true;
        let ticks = (seconds * 60.0) as u32;
        for _ in 0..ticks {
            game.update(1.0 / 60.0);
        }
    }

    #[test]
    fn villagers_go_out_by_day_and_home_at_night() {
        let mut game = crate::Game::new();
        game.world.resource_mut::<Voxels>().0 = hamlet();
        let home = Home {
            door: Vec3::new(11.0, 2.0, 7.5),
            inside: Vec3::new(11.0, 2.0, 11.0),
        };
        let e = game
            .world
            .spawn((
                Body::at(DVec3::new(11.0, 2.0, 11.0)),
                Character::new(Look::from_seed(1)),
                Villager {
                    doing: Doing::Indoors,
                    ..villager(home)
                },
            ))
            .id();
        // Morning: out of the door, and about the place.
        run(&mut game, 9.0, 20.0);
        let out = game.world.get::<Body>(e).unwrap().feet;
        let in_hut = |p: DVec3| p.x > 9.0 && p.x < 13.0 && p.z > 9.0 && p.z < 13.0;
        assert!(!in_hut(out), "still indoors at {out}");
        // Evening: back through the door and in.
        run(&mut game, 21.0, 40.0);
        let v = game.world.get::<Villager>(e).unwrap();
        assert_eq!(v.doing, Doing::Indoors);
        let back = game.world.get::<Body>(e).unwrap().feet;
        assert!(in_hut(back), "not home: {back}");
        assert!((back.y - 2.0).abs() < 0.01, "{back}");
    }

    #[test]
    fn turning_takes_the_short_way_round() {
        let a = turn(3.0, -3.0, 0.1);
        assert!(a > 3.0, "{a}");
        assert!((turn(0.0, 0.05, 0.1) - 0.05).abs() < 1e-6);
    }
}
