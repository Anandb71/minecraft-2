//! Game layer: player, interaction, blocks, on a standalone bevy_ecs world.

pub mod blocks;
pub mod clock;
pub mod collide;
pub mod input;
pub mod interact;
pub mod items;
pub mod physics;
pub mod physics_host;
pub mod player;
pub mod structure;
pub mod view;

use bevy_ecs::prelude::*;
use bevy_ecs::schedule::Schedule;
use mc2_voxel::world::VoxelWorld;
use mc2_worldgen::stream::ChunkStreamer;

/// The loaded voxel world.
#[derive(Resource, Default)]
pub struct Voxels(pub VoxelWorld);

/// Chunk streaming, absent until terrain has loaded.
#[derive(Resource, Default)]
pub struct Streaming(pub Option<ChunkStreamer>);

/// The game state and its schedules.
pub struct Game {
    pub world: World,
    frame: Schedule,
    fixed: Schedule,
    late: Schedule,
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

impl Game {
    pub fn new() -> Self {
        let mut world = World::new();
        world.insert_resource(Voxels::default());
        world.insert_resource(Streaming::default());
        world.insert_resource(input::Input::default());
        world.insert_resource(input::Time::default());
        world.insert_resource(blocks::BlockLayer::default());
        world.insert_resource(interact::Interaction::default());
        world.insert_resource(view::ViewCamera::default());
        world.insert_resource(clock::WorldClock::default());
        world.insert_resource(physics::Physics::default());
        world.insert_resource(structure::Structure::default());

        let mut frame = Schedule::default();
        frame.add_systems((clock::advance_clock, player::look, player::toggles).chain());
        let mut fixed = Schedule::default();
        fixed.add_systems((player::movement, physics::step_physics).chain());
        let mut late = Schedule::default();
        late.add_systems(
            (
                view::update_view,
                interact::interact,
                physics::light_fuses,
                structure::update_structure,
            )
                .chain(),
        );
        Self {
            world,
            frame,
            fixed,
            late,
        }
    }

    /// Spawns the player standing at `feet`.
    pub fn spawn_player(&mut self, feet: glam::DVec3, yaw: f32, pitch: f32) {
        self.world.spawn((
            player::Player {
                yaw,
                pitch,
                ..Default::default()
            },
            player::Body::at(feet),
        ));
    }

    /// Runs one frame: per-frame input systems, due fixed ticks, then camera
    /// and interaction.
    pub fn update(&mut self, dt: f32) {
        mc2_core::scope!("game.update");
        let ticks = self.world.resource_mut::<input::Time>().advance(dt);
        self.frame.run(&mut self.world);
        for _ in 0..ticks {
            self.fixed.run(&mut self.world);
        }
        self.late.run(&mut self.world);
        self.world.resource_mut::<input::Input>().end_frame();
    }

    pub fn input(&mut self) -> Mut<'_, input::Input> {
        self.world.resource_mut::<input::Input>()
    }

    pub fn clock(&mut self) -> Mut<'_, clock::WorldClock> {
        self.world.resource_mut::<clock::WorldClock>()
    }

    pub fn sky(&self) -> clock::Sky {
        self.world.resource::<clock::WorldClock>().sky()
    }

    pub fn view(&self) -> view::ViewCamera {
        *self.world.resource::<view::ViewCamera>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DVec3, IVec3};
    use mc2_voxel::material::ids;

    #[test]
    fn schedules_run_the_player_on_the_fixed_clock() {
        let mut game = Game::new();
        game.world.resource_mut::<Voxels>().0.fill_box(
            IVec3::ZERO,
            IVec3::new(255, 159, 255),
            ids::GRANITE,
        );
        game.spawn_player(DVec3::new(8.0, 12.0, 8.0), 0.0, 0.0);
        for _ in 0..120 {
            game.update(1.0 / 60.0);
        }
        let view = game.view();
        assert!(
            (view.position.y - (10.0 + player::EYE)).abs() < 0.05,
            "eye at {}",
            view.position.y
        );
    }

    #[test]
    fn breaking_a_block_in_front_fills_the_inventory() {
        let mut game = Game::new();
        {
            let mut v = game.world.resource_mut::<Voxels>();
            v.0.fill_box(IVec3::ZERO, IVec3::new(255, 159, 255), ids::GRANITE);
            // A 1 m block of sandstone at (8, 10, 10).
            v.0.fill_box(
                IVec3::new(128, 160, 160),
                IVec3::new(143, 175, 175),
                ids::SANDSTONE,
            );
        }
        game.spawn_player(DVec3::new(8.5, 10.0, 8.5), 0.0, -0.3);
        for _ in 0..30 {
            game.update(1.0 / 60.0);
        }
        {
            let mut input = game.input();
            input.captured = true;
            input.button_down(input::Button::Primary);
        }
        game.update(1.0 / 60.0);
        let state = game.world.resource::<interact::Interaction>();
        assert_eq!(state.inventory.volumes.get(&ids::SANDSTONE), Some(&4096));
        let v = game.world.resource::<Voxels>();
        assert!(v.0.voxel(IVec3::new(136, 168, 168)).is_air());
    }
}
