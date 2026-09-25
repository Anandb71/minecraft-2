//! Villagers: the people who live in the villages.
//!
//! Each lives in one of its village's houses. By day they go out to the
//! square, to one another's doors and back to their own, standing a while
//! at each; at dusk they walk home and go in. Villages near the player are
//! peopled as it comes and emptied again once it has gone. Each has a
//! trade; press E on one to see what it buys and sells, and it stops to
//! face you while you deal.

use crate::character::{Character, Look};
use crate::clock::WorldClock;
use crate::crafting::{Profession, Station};
use crate::input::{Input, Key, Time};
use crate::interact::Interaction;
use crate::nav;
use crate::physics::Physics;
use crate::player::{Body, EYE, Player};
use crate::vehicles::Garage;
use crate::view::ViewCamera;
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
/// Someone this near the spot half a metre ahead is in the way, metres;
/// after waiting this long for them, go somewhere else, seconds.
const PERSONAL_M: f64 = 0.8;
const GIVE_WAY_S: f32 = 3.0;
/// Columns of path search a tick, across every villager: a long search
/// runs over several ticks rather than holding one up.
const COLUMNS_PER_TICK: usize = 300;
/// How near a villager must be to trade with, metres.
pub const TRADE_REACH_M: f64 = 4.0;
/// A standing person, for aiming at: an upright cylinder on its feet.
const PERSON_RADIUS: f64 = 0.35;
const PERSON_HEIGHT: f64 = 1.85;

#[derive(Debug)]
pub enum Doing {
    /// Standing where it is until the game has run to `until` seconds.
    Idle { until: f64 },
    /// Working out the way, a little every tick; going in at the end if
    /// `inside`.
    Planning {
        search: Box<nav::Search>,
        inside: bool,
    },
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
    pub profession: Profession,
    /// The square, and every home in the village, to walk to.
    plaza: DVec3,
    homes: Arc<[Home]>,
    pub doing: Doing,
    rng: u64,
    /// Seconds spent waiting for someone in the way.
    blocked: f32,
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
#[allow(clippy::too_many_arguments)]
pub fn people_villages(
    mut commands: Commands,
    time: Res<Time>,
    streaming: Res<Streaming>,
    voxels: Res<Voxels>,
    clock: Res<WorldClock>,
    mut population: ResMut<Population>,
    mut garage: ResMut<Garage>,
    mut physics: ResMut<Physics>,
    players: Query<&Body, With<Player>>,
) {
    population.wait -= time.dt;
    if population.wait > 0.0 {
        return;
    }
    mc2_core::scope!("villagers.people");
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
                profession: Profession::ALL[(seed >> 7) as usize % Profession::ALL.len()],
                plaza: v.centre.as_dvec3(),
                homes: homes.clone(),
                doing: if day {
                    Doing::Idle { until: 0.0 }
                } else {
                    Doing::Indoors
                },
                rng: seed | 1,
                blocked: 0.0,
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
            // Someone in a village owns a car, parked by the square; a town
            // has three.
            if garage.parked.insert(key) {
                let seed = (key.0 as u64).wrapping_mul(31) ^ (key.1 as u64);
                for n in 0..if v.town { 3 } else { 1 } {
                    let seed = seed.wrapping_add(n * 97);
                    garage.park_by(&mut physics, &voxels.0, v.centre.as_dvec3(), seed);
                }
            }
        }
    }
}

/// Who is being traded with, and who the crosshair is on.
#[derive(Resource, Default)]
pub struct Trading {
    /// The villager dealt with now, and its trade.
    pub with: Option<(Entity, Profession)>,
    /// The villager in reach under the crosshair, its trade and distance.
    pub aimed: Option<(Entity, Profession, f64)>,
}

/// Where a ray from `eye` along unit `dir` first meets a person standing
/// at `feet`, metres along it.
pub fn ray_person(eye: DVec3, dir: DVec3, feet: DVec3) -> Option<f64> {
    // The side of the cylinder, solved in the level plane.
    let o = eye - feet;
    let a = dir.x * dir.x + dir.z * dir.z;
    let b = 2.0 * (o.x * dir.x + o.z * dir.z);
    let c = o.x * o.x + o.z * o.z - PERSON_RADIUS * PERSON_RADIUS;
    let inside = |t: f64| (0.0..=PERSON_HEIGHT).contains(&(o.y + dir.y * t));
    if a > 1e-12 {
        let disc = b * b - 4.0 * a * c;
        if disc >= 0.0 {
            let t = (-b - disc.sqrt()) / (2.0 * a);
            if t >= 0.0 && inside(t) {
                return Some(t);
            }
        }
    }
    // Or its top or foot, looking down or up at it.
    [PERSON_HEIGHT, 0.0].iter().find_map(|&y| {
        if dir.y.abs() < 1e-9 {
            return None;
        }
        let t = (y - o.y) / dir.y;
        let p = o + dir * t;
        (t >= 0.0 && p.x * p.x + p.z * p.z <= PERSON_RADIUS * PERSON_RADIUS).then_some(t)
    })
}

/// Every frame: which villager the crosshair is on, if one is in reach and
/// nearer than the block behind it. E on it opens its trades. Walking away
/// ends a deal.
pub fn aim_at_villagers(
    input: Res<Input>,
    view: Res<ViewCamera>,
    mut interaction: ResMut<Interaction>,
    mut trading: ResMut<Trading>,
    players: Query<&Body, With<Player>>,
    villagers: Query<(Entity, &Body, &Villager), Without<Player>>,
) {
    let eye = view.position;
    let (yaw, pitch) = (f64::from(view.yaw), f64::from(view.pitch));
    let dir = DVec3::new(
        yaw.sin() * pitch.cos(),
        pitch.sin(),
        yaw.cos() * pitch.cos(),
    );
    let wall = interaction.target.map_or(f64::MAX, |t| t.hit.t);
    trading.aimed = villagers
        .iter()
        .filter_map(|(e, b, v)| {
            ray_person(eye, dir, b.feet)
                .filter(|&t| t <= TRADE_REACH_M && t < wall)
                .map(|t| (e, v.profession, t))
        })
        .min_by(|a, b| a.2.total_cmp(&b.2));
    let feet = players.iter().next().map(|b| b.feet);
    if let Some((e, _)) = trading.with {
        let near = villagers
            .get(e)
            .ok()
            .zip(feet)
            .is_some_and(|((_, b, _), f)| level_distance(b.feet, f) <= TRADE_REACH_M + 1.0);
        if !near {
            trading.with = None;
        }
    }
    if input.captured
        && input.pressed(Key::Interact)
        && let Some((e, p, _)) = trading.aimed
    {
        trading.with = Some((e, p));
        interaction.opened = Some(Station::Trade(p));
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
    trading: Res<Trading>,
    players: Query<&Body, (With<Player>, Without<Villager>)>,
    mut villagers: Query<(Entity, &mut Villager, &mut Body, &mut Character), Without<Player>>,
) {
    mc2_core::scope!("villagers.walk");
    let dt = time.fixed_dt;
    let now = time.elapsed;
    let day = is_day(clock.hour());
    let eye = players.iter().next().map(|b| b.feet + DVec3::Y * EYE);
    let world = &voxels.0;
    let mut columns = COLUMNS_PER_TICK;
    // Start working out a way somewhere, or wait a little if there is none.
    let plan = |from: DVec3, to: DVec3, inside: bool| match nav::Search::new(world, from, to) {
        Some(search) => Doing::Planning {
            search: Box::new(search),
            inside,
        },
        None => Doing::Idle { until: now + 2.0 },
    };
    // Where everyone stands, to give way to one another.
    let everyone: Vec<(Entity, DVec3)> = villagers.iter().map(|(e, _, b, _)| (e, b.feet)).collect();
    for (e, mut v, mut body, mut c) in &mut villagers {
        body.prev_feet = body.feet;
        let v = &mut *v;
        // Dealing with someone: stand still and face them.
        if trading.with.is_some_and(|(t, _)| t == e)
            && let Some(eye) = eye
        {
            if !matches!(v.doing, Doing::Indoors) {
                v.doing = Doing::Idle { until: now + 2.0 };
            }
            body.velocity = DVec3::ZERO;
            let to = eye - body.feet;
            c.facing = turn(c.facing, to.x.atan2(to.z) as f32, TURN * dt as f32);
            c.look_at = Some(eye);
            continue;
        }
        // Decide.
        let going_home = matches!(
            v.doing,
            Doing::Walking { inside: true, .. } | Doing::Planning { inside: true, .. }
        );
        match v.doing {
            Doing::Indoors if day => v.doing = plan(body.feet, v.home.door.as_dvec3(), false),
            Doing::Idle { .. } | Doing::Walking { .. } | Doing::Planning { .. }
                if !day && !going_home =>
            {
                v.doing = plan(body.feet, v.home.inside.as_dvec3(), true);
            }
            Doing::Idle { until } if day && now >= until => {
                let goal = v.errand();
                v.doing = plan(body.feet, goal, false);
            }
            _ => {}
        }
        // Think: a share of this tick's search.
        let found = match &mut v.doing {
            Doing::Planning { search, inside } if columns > 0 => {
                let (progress, used) = search.advance(world, columns);
                columns -= used;
                match progress {
                    nav::Progress::Found(path) => Some(Doing::Walking {
                        path,
                        next: 0,
                        inside: *inside,
                    }),
                    nav::Progress::NoWay => Some(Doing::Idle { until: now + 2.0 }),
                    nav::Progress::Searching => None,
                }
            }
            _ => None,
        };
        if let Some(doing) = found {
            v.doing = doing;
        }
        // Give way: someone just ahead, stand until they pass; if they do
        // not, go somewhere else.
        if let Doing::Walking { path, next, .. } = &v.doing
            && let Some(target) = path.get(*next)
        {
            let d = DVec3::new(target.x - body.feet.x, 0.0, target.z - body.feet.z);
            let ahead = body.feet + d.normalize_or_zero() * 0.5;
            let in_way = everyone.iter().any(|&(o, at)| {
                o != e && level_distance(at, ahead) < PERSONAL_M && (at.y - ahead.y).abs() < 1.5
            });
            if in_way {
                v.blocked += dt as f32;
                if v.blocked > GIVE_WAY_S {
                    v.blocked = 0.0;
                    v.doing = Doing::Idle {
                        until: now + 1.0 + v.roll() * 2.0,
                    };
                }
                body.velocity = DVec3::ZERO;
                c.look_at = eye.filter(|e| e.distance(body.feet) < NOTICE_M);
                continue;
            }
            v.blocked = 0.0;
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
            profession: Profession::Smith,
            plaza: DVec3::new(4.0, 2.0, 4.0),
            homes: vec![home].into(),
            doing: Doing::Idle { until: 0.0 },
            rng: 7,
            blocked: 0.0,
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
        assert!(matches!(v.doing, Doing::Indoors), "{:?}", v.doing);
        let back = game.world.get::<Body>(e).unwrap().feet;
        assert!(in_hut(back), "not home: {back}");
        assert!((back.y - 2.0).abs() < 0.01, "{back}");
    }

    #[test]
    fn a_ray_finds_a_person_and_misses_beside_them() {
        let feet = DVec3::new(10.0, 2.0, 10.0);
        let eye = DVec3::new(10.0, 3.6, 6.0);
        let t = ray_person(eye, DVec3::Z, feet).expect("hit");
        assert!((t - (4.0 - PERSON_RADIUS)).abs() < 1e-9, "{t}");
        assert!(ray_person(eye + DVec3::X * 0.5, DVec3::Z, feet).is_none());
        // Over their head, and down onto it from above.
        assert!(ray_person(DVec3::new(10.0, 4.5, 6.0), DVec3::Z, feet).is_none());
        let down = ray_person(DVec3::new(10.0, 6.0, 10.0), -DVec3::Y, feet).unwrap();
        assert!((down - (6.0 - 3.85)).abs() < 1e-9, "{down}");
    }

    #[test]
    fn turning_takes_the_short_way_round() {
        let a = turn(3.0, -3.0, 0.1);
        assert!(a > 3.0, "{a}");
        assert!((turn(0.0, 0.05, 0.1) - 0.05).abs() < 1e-6);
    }

    #[test]
    fn villagers_give_way_rather_than_walk_through_each_other() {
        let mut game = crate::Game::new();
        game.world.resource_mut::<Voxels>().0 = hamlet();
        game.clock().set_hour(9.0);
        game.clock().paused = true;
        let home = Home {
            door: Vec3::new(11.0, 2.0, 7.5),
            inside: Vec3::new(11.0, 2.0, 11.0),
        };
        // Two walkers on one line, heading for each other's start.
        let (a, b) = (DVec3::new(2.0, 2.0, 4.0), DVec3::new(7.0, 2.0, 4.0));
        let walker = |from: DVec3, to: DVec3| Villager {
            doing: Doing::Walking {
                path: vec![from, to],
                next: 1,
                inside: false,
            },
            ..villager(home)
        };
        let ea = game
            .world
            .spawn((
                Body::at(a),
                Character::new(Look::from_seed(1)),
                walker(a, b),
            ))
            .id();
        let eb = game
            .world
            .spawn((
                Body::at(b),
                Character::new(Look::from_seed(2)),
                walker(b, a),
            ))
            .id();
        let mut closest = f64::MAX;
        for _ in 0..240 {
            game.update(1.0 / 60.0);
            let pa = game.world.get::<Body>(ea).unwrap().feet;
            let pb = game.world.get::<Body>(eb).unwrap().feet;
            closest = closest.min(level_distance(pa, pb));
        }
        assert!(closest > 0.5, "walked into each other: {closest}");
    }
}
