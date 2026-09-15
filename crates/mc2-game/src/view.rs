//! The render camera derived from the player each frame.

use crate::Voxels;
use crate::input::Time;
use crate::player::{Body, Player, View};
use bevy_ecs::prelude::*;
use glam::DVec3;
use mc2_voxel::march::raycast_filtered;

const THIRD_PERSON_DISTANCE: f64 = 4.0;

#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct ViewCamera {
    pub position: DVec3,
    pub yaw: f32,
    pub pitch: f32,
}

pub fn update_view(
    voxels: Res<Voxels>,
    time: Res<Time>,
    players: Query<(&Player, &Body)>,
    mut view: ResMut<ViewCamera>,
) {
    let Some((p, b)) = players.iter().next() else {
        return;
    };
    let feet = b.prev_feet.lerp(b.feet, time.alpha);
    let eye = feet + DVec3::Y * (p.eye() - p.step_smoothing);
    view.yaw = p.yaw;
    view.pitch = p.pitch;
    view.position = match p.view {
        View::FirstPerson => eye,
        View::ThirdPerson => {
            // Pull the camera in front of whatever is behind the player.
            let back = -p.forward().as_dvec3();
            let reach = raycast_filtered(&voxels.0, eye, back, THIRD_PERSON_DISTANCE, |m| {
                crate::collide::blocks_movement(m)
            })
            .map_or(THIRD_PERSON_DISTANCE, |h| (h.t - 0.2).max(0.3));
            eye + back * reach + DVec3::Y * 0.3
        }
    };
}
