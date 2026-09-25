//! Health and hunger.
//!
//! Twenty points of each. Falling more than a few metres hurts, as do
//! blasts (less behind cover, by the same trace the ears use), fire and
//! lava, and staying under water once the breath runs out. Everything done
//! makes hungry, walking more than standing and sprinting more again; a
//! full stomach heals and an empty one hurts. At nothing the player wakes
//! at the spawn, pack and all. Creative play feels none of it.

use crate::Voxels;
use crate::input::{Button, Input, Time};
use crate::interact::Interaction;
use crate::physics::Physics;
use crate::player::{Body, EYE, MoveMode, Player};
use bevy_ecs::prelude::*;
use glam::DVec3;
use mc2_voxel::material::{Kind, ids};
use mc2_voxel::world::VoxelWorld;

pub const MAX_HEALTH: f32 = 20.0;
pub const MAX_FOOD: f32 = 20.0;
/// Seconds of breath.
pub const MAX_AIR: f32 = 10.0;
/// A fall hurts a point a metre beyond this, metres.
const SAFE_FALL: f64 = 3.5;
/// Hunger a second, standing; a metre walked; and how much more for a
/// metre sprinted.
const HUNGER_IDLE: f32 = 0.01;
const HUNGER_PER_M: f32 = 0.012;
const SPRINT_HUNGER: f32 = 2.0;
/// Healing a second on a full enough stomach, and the hunger it costs.
const HEAL: f32 = 0.25;
const HEAL_HUNGER: f32 = 0.15;
const WELL_FED: f32 = 16.0;
/// Hurt a second from starving, drowning, fire and lava.
const STARVING: f32 = 0.25;
const DROWNING: f32 = 2.0;
const BURNING: f32 = 2.0;
const LAVA: f32 = 4.0;
/// A blast at its heart does this much; it falls to nothing at its reach.
const BLAST: f32 = 22.0;
const BLAST_REACH: f64 = 2.5;

#[derive(Component, Debug, Clone, Copy)]
pub struct Vitals {
    pub health: f32,
    pub food: f32,
    pub air: f32,
    /// When last hurt, game seconds, for the HUD's flash.
    pub hurt_at: f64,
    /// The highest point of the fall in progress.
    fall_from: Option<f64>,
}

impl Default for Vitals {
    fn default() -> Self {
        Vitals {
            health: MAX_HEALTH,
            food: MAX_FOOD,
            air: MAX_AIR,
            hurt_at: f64::NEG_INFINITY,
            fall_from: None,
        }
    }
}

impl Vitals {
    pub fn hurt(&mut self, amount: f32, now: f64) {
        if amount <= 0.0 {
            return;
        }
        self.health -= amount;
        if amount >= 0.5 {
            self.hurt_at = now;
        }
    }
}

/// Where the player wakes after dying.
#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct Spawn(pub DVec3);

fn material_at(world: &VoxelWorld, p: DVec3) -> mc2_voxel::material::MaterialId {
    world.voxel((p * 16.0).floor().as_ivec3())
}

/// Every fixed tick: falls, breath, burning, hunger, healing, and waking
/// at the spawn after dying.
pub fn live(
    time: Res<Time>,
    voxels: Res<Voxels>,
    spawn: Res<Spawn>,
    mut interaction: ResMut<Interaction>,
    mut players: Query<(&Player, &mut Body, &mut Vitals)>,
) {
    let dt = time.fixed_dt as f32;
    let now = time.elapsed;
    let creative = interaction.inventory.creative;
    let world = &voxels.0;
    for (p, mut b, mut v) in &mut players {
        if creative {
            *v = Vitals {
                hurt_at: v.hurt_at,
                ..Vitals::default()
            };
            continue;
        }
        // Falls: from the highest point of each drop to the landing.
        let walking = p.mode == MoveMode::Walk && !p.seated;
        if !walking {
            v.fall_from = None;
        } else if !p.on_ground {
            v.fall_from = Some(v.fall_from.map_or(b.feet.y, |top| top.max(b.feet.y)));
        } else if let Some(top) = v.fall_from.take() {
            let into_water = material_at(world, b.feet + DVec3::Y * 0.1).get().kind == Kind::Liquid;
            let drop = top - b.feet.y;
            if drop > SAFE_FALL && !into_water {
                v.hurt((drop - 3.0).round() as f32, now);
            }
        }
        // Breath, and what the body stands in.
        let eye = material_at(world, b.feet + DVec3::Y * EYE);
        if eye == ids::WATER {
            v.air -= dt;
            if v.air <= 0.0 {
                v.air = 0.0;
                v.hurt(DROWNING * dt, now);
            }
        } else {
            v.air = (v.air + 3.0 * dt).min(MAX_AIR);
        }
        let body = [0.1, 0.9].map(|h| material_at(world, b.feet + DVec3::Y * h));
        if body.contains(&ids::LAVA) || eye == ids::LAVA {
            v.hurt(LAVA * dt, now);
        } else if body.contains(&ids::FIRE) {
            v.hurt(BURNING * dt, now);
        }
        // Hunger, by time and distance.
        let moved = b.feet - b.prev_feet;
        let metres = moved.x.hypot(moved.z) as f32;
        let sprint = if metres > 5.0 * dt {
            SPRINT_HUNGER
        } else {
            1.0
        };
        v.food -= HUNGER_IDLE * dt + HUNGER_PER_M * metres * sprint;
        if v.food >= WELL_FED && v.health < MAX_HEALTH {
            v.health = (v.health + HEAL * dt).min(MAX_HEALTH);
            v.food -= HEAL_HUNGER * dt;
        }
        v.food = v.food.clamp(0.0, MAX_FOOD);
        if v.food == 0.0 && v.health > 1.0 {
            v.health = (v.health - STARVING * dt).max(1.0);
        }
        // Dead: wake at the spawn.
        if v.health <= 0.0 {
            b.feet = spawn.0;
            b.prev_feet = spawn.0;
            b.velocity = DVec3::ZERO;
            *v = Vitals {
                food: MAX_FOOD * 0.8,
                hurt_at: now,
                ..Vitals::default()
            };
            interaction.say(now, "you died, and woke where you began");
        }
    }
}

/// Every frame: blasts hurt the player by how near they went off and how
/// much stands between; the right button eats what is held, if hungry.
pub fn hurt_and_eat(
    time: Res<Time>,
    input: Res<Input>,
    voxels: Res<Voxels>,
    physics: Res<Physics>,
    mut interaction: ResMut<Interaction>,
    mut players: Query<(&Body, &mut Vitals), With<Player>>,
) {
    let now = time.elapsed;
    if interaction.inventory.creative {
        return;
    }
    for (b, mut v) in &mut players {
        let chest = b.feet + DVec3::Y * 1.1;
        for &(centre, radius) in &physics.blasts_out {
            let reach = f64::from(radius) * BLAST_REACH;
            let d = chest.distance(centre);
            if d < reach {
                let cover = mc2_audio::occlusion(&voxels.0, centre, chest).gain.sqrt();
                let near = (1.0 - d / reach) as f32;
                v.hurt(BLAST * near * near * cover, now);
            }
        }
        if input.captured
            && input.button_pressed(Button::Secondary)
            && let Some(item) = interaction.held()
            && let Some(meal) = item.food()
        {
            if v.food >= MAX_FOOD - 0.5 {
                interaction.say(now, "not hungry");
            } else {
                let slot = interaction.slot;
                interaction.inventory.take_from(slot);
                v.food = (v.food + meal).min(MAX_FOOD);
                interaction.say(now, format!("ate the {}", item.name()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::IVec3;

    fn game_on(world: VoxelWorld, feet: DVec3) -> crate::Game {
        let mut game = crate::Game::new();
        game.world.resource_mut::<Voxels>().0 = world;
        game.world.resource_mut::<Interaction>().inventory.creative = false;
        game.spawn_player(feet, 0.0, 0.0);
        game
    }

    fn vitals(game: &mut crate::Game) -> Vitals {
        let mut q = game.world.query::<&Vitals>();
        *q.iter(&game.world).next().unwrap()
    }

    fn floor() -> VoxelWorld {
        let mut w = VoxelWorld::new();
        w.fill_box(IVec3::ZERO, IVec3::new(511, 31, 511), ids::GRANITE);
        w
    }

    fn run(game: &mut crate::Game, seconds: f32) {
        for _ in 0..(seconds * 60.0) as usize {
            game.update(1.0 / 60.0);
        }
    }

    #[test]
    fn a_long_fall_hurts_and_a_short_one_does_not() {
        let mut game = game_on(floor(), DVec3::new(8.0, 4.0, 8.0));
        run(&mut game, 2.0);
        assert_eq!(vitals(&mut game).health, MAX_HEALTH, "two metres hurt");
        let mut game = game_on(floor(), DVec3::new(8.0, 12.0, 8.0));
        run(&mut game, 3.0);
        let h = vitals(&mut game).health;
        // Ten metres: seven points, give or take healing.
        assert!((12.0..14.5).contains(&h), "health {h}");
    }

    #[test]
    fn under_water_the_breath_runs_out_and_then_it_hurts() {
        let mut w = floor();
        w.fill_box(IVec3::new(0, 32, 0), IVec3::new(511, 95, 511), ids::WATER);
        let mut game = game_on(w, DVec3::new(8.0, 2.0, 8.0));
        run(&mut game, 5.0);
        let v = vitals(&mut game);
        assert!(
            v.air < MAX_AIR * 0.6 && v.health >= MAX_HEALTH - 0.01,
            "{v:?}"
        );
        run(&mut game, 9.0);
        let v = vitals(&mut game);
        assert_eq!(v.air, 0.0);
        assert!(v.health < MAX_HEALTH - 5.0, "{v:?}");
    }

    #[test]
    fn eating_fills_and_dying_wakes_you_at_the_spawn() {
        let mut game = game_on(floor(), DVec3::new(8.0, 2.0, 8.0));
        run(&mut game, 0.5);
        {
            let mut q = game.world.query::<&mut Vitals>();
            let mut v = q.iter_mut(&mut game.world).next().unwrap();
            v.food = 5.0;
        }
        {
            let mut i = game.world.resource_mut::<Interaction>();
            i.inventory.slots[0] = Some(crate::inventory::Stack::new(crate::items::Item::Bread, 2));
            i.slot = 0;
        }
        game.input().captured = true;
        game.input().button_down(Button::Secondary);
        game.update(1.0 / 60.0);
        game.input().button_up(Button::Secondary);
        let v = vitals(&mut game);
        assert!((v.food - 11.0).abs() < 0.1, "{v:?}");
        // Killed well away from the spawn: back at it, whole.
        {
            let mut q = game.world.query::<(&mut Body, &mut Vitals)>();
            let (mut b, mut v) = q.iter_mut(&mut game.world).next().unwrap();
            b.feet = DVec3::new(20.0, 2.0, 20.0);
            b.prev_feet = b.feet;
            v.health = 0.0;
        }
        run(&mut game, 0.1);
        let feet = {
            let mut q = game.world.query::<&Body>();
            q.iter(&game.world).next().unwrap().feet
        };
        assert!(feet.distance(DVec3::new(8.0, 2.0, 8.0)) < 0.5, "{feet}");
        assert_eq!(vitals(&mut game).health, MAX_HEALTH);
    }
}
