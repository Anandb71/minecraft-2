//! The render camera derived from the player each frame: at the eyes, over
//! the shoulder, or behind the car being driven.

use crate::Voxels;
use crate::input::Time;
use crate::physics::Physics;
use crate::player::{Body, Player, View};
use crate::vehicles::Garage;
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
    garage: Res<Garage>,
    physics: Res<Physics>,
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
    // Driving, seen from behind: follow the car, pulled in front of
    // anything between.
    let car = garage.driven().and_then(|c| physics.host.body(c.body));
    if let (true, View::ThirdPerson, Some(car)) = (p.seated, p.view, car) {
        let (eye, at) = crate::vehicles::chase(car, time.alpha);
        let to = eye - at;
        let far = to.length();
        let dir = to / far;
        let reach = raycast_filtered(&voxels.0, at, dir, far, crate::collide::blocks_movement)
            .map_or(far, |h| (h.t - 0.3).max(1.0));
        let pos = at + dir * reach;
        let look = (at - pos).normalize();
        view.position = pos;
        view.yaw = look.x.atan2(look.z) as f32;
        view.pitch = look.y.asin() as f32;
        return;
    }
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
