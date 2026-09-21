//! Fire: 1 m blocks that burn, spread and char.
//!
//! A burning block has fuel for so many seconds, by what it is made of. Ten
//! times a second its flames are drawn again, in the air beside its
//! flammable voxels (the flicker), a share of those voxels chars from the
//! outside in (wood to charcoal, leaves and grass to nothing, wool and
//! thatch to ash), and the fire may take its neighbours: flammable blocks
//! catch more readily above it and downwind. Water beside a burning block
//! puts it out, rain now and then; its heat melts snow and ice around it.
//! Charring is an edit like any other, so a burning house can fall down.

use crate::Voxels;
use crate::blocks::{BlockKind, BlockLayer};
use crate::input::Time;
use crate::physics::Physics;
use bevy_ecs::prelude::*;
use glam::{IVec3, Vec2};
use mc2_core::FxHashMap;
use mc2_voxel::coords::{BlockPos, VOXELS_PER_BLOCK};
use mc2_voxel::material::{Kind, MaterialId, ids};
use mc2_voxel::world::VoxelWorld;

/// Fire runs at this rate, seconds a tick.
pub const TICK_S: f32 = 0.1;
/// Most blocks burning at once; beyond it nothing new catches.
pub const MAX_BURNING: usize = 400;
/// Flame voxels drawn a tick for one block, at most.
const FLAMES_PER_BLOCK: usize = 160;
/// Chance a second that a fully flammable neighbour catches.
const SPREAD_PER_S: f32 = 0.6;
/// Seconds a block lit with nothing to burn keeps a small flame.
const GROUND_FIRE_S: f32 = 2.5;
/// Burning blocks nearest the player whose flames flicker every tick; the
/// rest keep the flames they last drew.
const FLICKER_BLOCKS: usize = 24;
/// Voxels one block chars a tick at most; what is left chars through when
/// it burns out.
const MAX_CHAR_PER_TICK: usize = 40;
/// Blocks that finish burning in one tick at most (each chars through all
/// its voxels); the rest keep burning a tick longer.
const BURN_OUTS_PER_TICK: usize = 6;
/// Neighbours a burning block tries each tick, of the thirty it has.
const SPREAD_TRIES: usize = 6;

/// Where fire can spread from a block: its 26 neighbours, and straight up
/// two blocks (fire climbs), with how much more readily it goes that way.
const NEIGHBOURS: [(IVec3, f32); 27] = {
    let mut out = [(IVec3::ZERO, 0.0); 27];
    let mut i = 0;
    let mut dz = -1;
    while dz <= 1 {
        let mut dy = -1;
        while dy <= 1 {
            let mut dx = -1;
            while dx <= 1 {
                if dx != 0 || dy != 0 || dz != 0 {
                    let up = match dy {
                        1 => 3.0,
                        0 => 1.0,
                        _ => 0.4,
                    };
                    out[i] = (IVec3::new(dx, dy, dz), up);
                    i += 1;
                }
                dx += 1;
            }
            dy += 1;
        }
        dz += 1;
    }
    out[26] = (IVec3::new(0, 2, 0), 1.5);
    out
};

#[derive(Clone, Debug)]
struct Burn {
    fuel: f32,
    total: f32,
    flames: Vec<IVec3>,
}

#[derive(Resource)]
pub struct Fire {
    burning: FxHashMap<BlockPos, Burn>,
    /// Blocks to set alight, from other systems (flint and steel, blasts,
    /// lightning), and whether a block with nothing to burn still gets a
    /// short flame.
    pub ignite: Vec<(BlockPos, bool)>,
    /// Wind at the ground, metres a second (x, z); weather sets it.
    pub wind: Vec2,
    /// How hard it is raining, 0..1; weather sets it.
    pub rain: f32,
    acc: f32,
    rng: u64,
    pub caught: u64,
    pub burnt_out: u64,
}

impl Default for Fire {
    fn default() -> Self {
        Self {
            burning: FxHashMap::default(),
            ignite: Vec::new(),
            wind: Vec2::ZERO,
            rain: 0.0,
            acc: 0.0,
            rng: 0x2545_f491_4f6c_dd1d,
            caught: 0,
            burnt_out: 0,
        }
    }
}

/// Seconds a block of `m` burns for (0: it does not burn).
pub fn fuel_s(m: MaterialId) -> f32 {
    match m {
        ids::OAK_LOG | ids::PINE_LOG | ids::BIRCH_LOG => 40.0,
        ids::PLANKS | ids::DARK_PLANKS => 25.0,
        ids::CHARCOAL => 30.0,
        ids::COAL_ORE => 45.0,
        ids::WOOL => 10.0,
        ids::THATCH => 12.0,
        ids::CACTUS => 15.0,
        ids::LEAVES | ids::PINE_NEEDLES | ids::BIRCH_LEAVES => 5.0,
        ids::MOSS => 4.0,
        ids::TALL_GRASS
        | ids::WHEAT
        | ids::FLOWER_RED
        | ids::FLOWER_YELLOW
        | ids::FLOWER_BLUE
        | ids::FLOWER_WHITE => 2.0,
        ids::GRASS => 1.5,
        _ => 0.0,
    }
}

/// What a voxel of `m` becomes once burnt (itself if it does not burn).
pub fn burnt(m: MaterialId) -> MaterialId {
    match m {
        ids::OAK_LOG | ids::PINE_LOG | ids::BIRCH_LOG | ids::PLANKS | ids::DARK_PLANKS => {
            ids::CHARCOAL
        }
        ids::CHARCOAL | ids::WOOL | ids::THATCH | ids::CACTUS => ids::ASH,
        ids::GRASS => ids::DIRT,
        ids::COAL_ORE => ids::SHALE,
        m if m.get().kind == Kind::Foliage && m.get().flammability > 0.0 => ids::AIR,
        m => m,
    }
}

fn flammable(m: MaterialId) -> bool {
    m != ids::FIRE && fuel_s(m) > 0.0
}

/// A block's fuel in seconds, from its dominant material, or for placed
/// things from what they are made of.
fn block_fuel(kind: Option<BlockKind>) -> f32 {
    match kind {
        Some(BlockKind::Solid(m)) | Some(BlockKind::Slab(m)) => fuel_s(m),
        Some(BlockKind::CraftingTable) | Some(BlockKind::Window) => 20.0,
        Some(BlockKind::Torch) => 3.0,
        _ => 0.0,
    }
}

/// What is in a block to burn.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Tinder {
    /// How readily it catches, 0..1.
    flammability: f32,
    /// Seconds it burns for.
    fuel: f32,
    tnt: bool,
}

/// What block `b` holds to burn: placed things by what they are, anything
/// else by 27 voxels spread through it, so a trunk thinner than the block
/// or a few leaves still count.
fn tinder(world: &VoxelWorld, layer: &BlockLayer, b: BlockPos) -> Tinder {
    let kind = layer.block_at(world, b);
    match kind {
        Some(BlockKind::Solid(ids::TNT)) => {
            return Tinder {
                flammability: 1.0,
                fuel: 0.0,
                tnt: true,
            };
        }
        Some(BlockKind::CraftingTable | BlockKind::Window | BlockKind::Torch) => {
            return Tinder {
                flammability: block_flammability(kind),
                fuel: block_fuel(kind),
                tnt: false,
            };
        }
        _ => {}
    }
    let o = b.origin();
    let (mut count, mut flam, mut fuel) = (0, 0.0f32, 0.0f32);
    for z in [2, 8, 13] {
        for y in [2, 8, 13] {
            for x in [2, 8, 13] {
                let m = world.voxel(o + IVec3::new(x, y, z));
                if flammable(m) {
                    count += 1;
                    flam = flam.max(m.get().flammability.max(0.02));
                    fuel = fuel.max(fuel_s(m));
                }
            }
        }
    }
    let share = count as f32 / 27.0;
    Tinder {
        flammability: if count > 0 {
            flam * (0.4 + 0.6 * share)
        } else {
            0.0
        },
        fuel: if count > 0 {
            fuel * (0.35 + 0.65 * share)
        } else {
            0.0
        },
        tnt: false,
    }
}

fn block_flammability(kind: Option<BlockKind>) -> f32 {
    match kind {
        Some(BlockKind::Solid(m)) | Some(BlockKind::Slab(m)) if fuel_s(m) > 0.0 => {
            m.get().flammability.max(0.05)
        }
        Some(BlockKind::CraftingTable) | Some(BlockKind::Window) => 0.4,
        _ => 0.0,
    }
}

impl Fire {
    fn roll(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 40) as f32 / (1u64 << 24) as f32
    }

    fn roll_int(&mut self, n: i32) -> i32 {
        ((self.roll() * n as f32) as i32).min(n - 1)
    }

    pub fn burning(&self) -> usize {
        self.burning.len()
    }

    pub fn is_burning(&self, b: BlockPos) -> bool {
        self.burning.contains_key(&b)
    }

    /// Sets block `b` alight with `fuel` seconds of burning, a short flame
    /// on it if it has none but is `forced` (flint and steel on stone).
    /// False if nothing caught.
    fn light(&mut self, b: BlockPos, fuel: f32, forced: bool) -> bool {
        if self.burning.contains_key(&b) || self.burning.len() >= MAX_BURNING {
            return false;
        }
        let fuel = match fuel {
            f if f > 0.0 => f * (0.8 + 0.4 * self.roll()),
            _ if forced => GROUND_FIRE_S,
            _ => return false,
        };
        self.burning.insert(
            b,
            Burn {
                fuel,
                total: fuel,
                flames: Vec::new(),
            },
        );
        self.caught += 1;
        true
    }

    /// Puts block `b` out, clearing its flames.
    fn put_out(&mut self, world: &mut VoxelWorld, b: BlockPos) {
        if let Some(burn) = self.burning.remove(&b) {
            clear_flames(world, &burn.flames);
        }
    }
}

fn clear_flames(world: &mut VoxelWorld, flames: &[IVec3]) {
    for &v in flames {
        if world.voxel(v) == ids::FIRE {
            world.set_voxel(v, ids::AIR);
        }
    }
}

const FACES: [IVec3; 6] = [
    IVec3::X,
    IVec3::NEG_X,
    IVec3::Y,
    IVec3::NEG_Y,
    IVec3::Z,
    IVec3::NEG_Z,
];

/// Whether water touches block `b`: any of its face centres, inside or
/// just outside.
fn wet(world: &VoxelWorld, b: BlockPos) -> bool {
    let c = b.origin() + VOXELS_PER_BLOCK / 2;
    let h = VOXELS_PER_BLOCK / 2;
    FACES
        .iter()
        .any(|&d| world.voxel(c + d * h).get().kind == Kind::Liquid)
        || world.voxel(c).get().kind == Kind::Liquid
}

/// Nothing solid above the block for a few metres: rain reaches it.
fn open_to_sky(world: &VoxelWorld, b: BlockPos) -> bool {
    let top = b.origin() + IVec3::new(8, VOXELS_PER_BLOCK, 8);
    (0..6).all(|k| !world.voxel(top + IVec3::Y * (k * 16)).is_solid())
}

/// One tick of one burning block: flames, charring, heat. Returns the
/// voxel box charred, if any.
fn burn_block(
    world: &mut VoxelWorld,
    fire: &mut Fire,
    b: BlockPos,
    flames: &mut Vec<IVec3>,
    intensity: f32,
    char_voxels: usize,
    flicker: bool,
) -> Option<(IVec3, IVec3)> {
    let o = b.origin();
    let n = VOXELS_PER_BLOCK;
    // Flames: seeds in the air beside flammable voxels, each a tongue up;
    // drawn again each tick near the player, once elsewhere.
    let want = if flicker || flames.is_empty() {
        clear_flames(world, flames);
        flames.clear();
        let all = FLAMES_PER_BLOCK as f32 * intensity.clamp(0.2, 1.0);
        // Far blocks draw their flames once, and fewer of them.
        (if flicker { all } else { all / 3.0 }) as usize
    } else {
        0
    };
    let mut tries = 0;
    while flames.len() < want && tries < want * 6 {
        tries += 1;
        let v = o + IVec3::new(
            fire.roll_int(n + 2) - 1,
            fire.roll_int(n + 2) - 1,
            fire.roll_int(n + 2) - 1,
        );
        if !world.voxel(v).is_air() {
            continue;
        }
        let beside = FACES.iter().any(|&d| flammable(world.voxel(v + d)));
        let on_top = v.y == o.y + n && world.voxel(v - IVec3::Y).is_solid();
        if !beside && !on_top {
            continue;
        }
        // A tongue two voxels across at its root, narrowing to one, leaning
        // a little as it rises.
        let height = 2 + fire.roll_int((3.0 + 13.0 * intensity) as i32);
        let lean = IVec3::new(fire.roll_int(3) - 1, 0, fire.roll_int(3) - 1);
        for k in 0..height {
            let spine = v + IVec3::Y * k + lean * (k / 5);
            let width = if k < height / 2 { 2 } else { 1 };
            let mut blocked = false;
            for dz in 0..width {
                for dx in 0..width {
                    let f = spine + IVec3::new(dx, 0, dz);
                    if !world.voxel(f).is_air() {
                        blocked |= dx == 0 && dz == 0;
                        continue;
                    }
                    world.set_voxel(f, ids::FIRE);
                    flames.push(f);
                }
            }
            if blocked {
                break;
            }
        }
    }
    // Charring: flammable voxels at the surface go first.
    let mut charred: Option<(IVec3, IVec3)> = None;
    let mut done = 0;
    let mut tries = 0;
    while done < char_voxels && tries < char_voxels * 3 {
        tries += 1;
        let v = o + IVec3::new(fire.roll_int(n), fire.roll_int(n), fire.roll_int(n));
        let m = world.voxel(v);
        if !flammable(m) {
            continue;
        }
        let exposed = FACES.iter().any(|&d| {
            let a = world.voxel(v + d);
            a.is_air() || a == ids::FIRE || a == ids::ASH || a == ids::CHARCOAL
        });
        if !exposed {
            continue;
        }
        world.set_voxel(v, burnt(m));
        done += 1;
        charred = Some(match charred {
            Some((lo, hi)) => (lo.min(v), hi.max(v)),
            None => (v, v),
        });
    }
    // Heat: snow and ice within a block of it melt.
    for _ in 0..4 {
        let v = o + IVec3::new(
            fire.roll_int(n * 3) - n,
            fire.roll_int(n * 3) - n,
            fire.roll_int(n * 3) - n,
        );
        match world.voxel(v) {
            m if m == ids::SNOW || m == ids::ICE => {
                world.set_voxel(v, if m == ids::ICE { ids::WATER } else { ids::AIR });
            }
            _ => {}
        }
    }
    charred
}

/// Ten times a second: new fires, flames, charring, spread, water and
/// rain, burning out. Every frame.
pub fn update_fire(
    mut voxels: ResMut<Voxels>,
    layer: Res<BlockLayer>,
    mut fire: ResMut<Fire>,
    mut physics: ResMut<Physics>,
    time: Res<Time>,
    players: Query<&crate::player::Body>,
) {
    let fire = &mut *fire;
    fire.acc += time.dt;
    if fire.acc < TICK_S {
        return;
    }
    fire.acc = (fire.acc - TICK_S).min(TICK_S);
    mc2_core::scope!("fire.tick");
    let world = &mut voxels.0;

    // Blasts set what is flammable around them alight.
    for (centre, radius) in std::mem::take(&mut physics.blasts_out) {
        let r = (radius * 1.4).ceil() as i32;
        let c = centre.floor().as_ivec3();
        for dz in -r..=r {
            for dy in -r..=r {
                for dx in -r..=r {
                    let d = IVec3::new(dx, dy, dz);
                    if d.length_squared() > r * r || fire.roll() > 0.3 {
                        continue;
                    }
                    fire.ignite.push((BlockPos(c + d), false));
                }
            }
        }
    }
    for (b, forced) in std::mem::take(&mut fire.ignite) {
        let t = tinder(world, &layer, b);
        if t.tnt {
            crate::physics::ignite_block(world, &mut physics, b, 1.5);
            continue;
        }
        fire.light(b, t.fuel, forced);
    }

    // Nearest first, so the flicker goes where it is seen.
    let mut blocks: Vec<BlockPos> = fire.burning.keys().copied().collect();
    if let Some(body) = players.iter().next() {
        let at = body.feet.floor().as_ivec3();
        blocks.sort_by_key(|b| (b.0 - at).length_squared());
    }
    let wind = fire.wind;
    let wind_speed = wind.length();
    let mut catch = Vec::new();
    let mut burn_outs = 0;
    for (rank, b) in blocks.into_iter().enumerate() {
        let Some(mut burn) = fire.burning.remove(&b) else {
            continue;
        };
        // Water puts it out; rain does now and then where it reaches.
        let rained_out =
            fire.rain > 0.05 && fire.roll() < fire.rain * 0.04 && open_to_sky(world, b);
        if wet(world, b) || rained_out {
            clear_flames(world, &burn.flames);
            continue;
        }
        burn.fuel -= TICK_S;
        if burn.fuel <= 0.0 && burn_outs < BURN_OUTS_PER_TICK {
            burn_outs += 1;
            clear_flames(world, &burn.flames);
            fire.burnt_out += 1;
            // What is left of it chars through.
            let o = b.origin();
            let hi = o + (VOXELS_PER_BLOCK - 1);
            for z in o.z..=hi.z {
                for y in o.y..=hi.y {
                    for x in o.x..=hi.x {
                        let v = IVec3::new(x, y, z);
                        let m = world.voxel(v);
                        if flammable(m) && m != ids::CHARCOAL {
                            world.set_voxel(v, burnt(m));
                        }
                    }
                }
            }
            physics.world_edited(o, hi);
            continue;
        }
        let intensity = (burn.fuel.max(0.0) / burn.total * 2.0).clamp(0.25, 1.0);
        // A block's flammable voxels char over its burn time.
        let per_tick =
            ((4096.0 / burn.total * TICK_S * 0.8).ceil() as usize).min(MAX_CHAR_PER_TICK);
        let mut flames = std::mem::take(&mut burn.flames);
        let flicker = rank < FLICKER_BLOCKS;
        if let Some((lo, hi)) =
            burn_block(world, fire, b, &mut flames, intensity, per_tick, flicker)
        {
            physics.world_edited(lo, hi);
        }
        burn.flames = flames;
        // Spread: a few neighbours a tick, the odds scaled to all of them.
        for _ in 0..SPREAD_TRIES {
            let (d, up) = NEIGHBOURS[fire.roll_int(NEIGHBOURS.len() as i32) as usize];
            let nb = BlockPos(b.0 + d);
            if fire.burning.contains_key(&nb) {
                continue;
            }
            let t = tinder(world, &layer, nb);
            let f = t.flammability;
            if f <= 0.0 {
                continue;
            }
            let dir = Vec2::new(d.x as f32, d.z as f32).normalize_or_zero();
            let downwind = if wind_speed > 0.1 {
                (1.0 + dir.dot(wind / wind_speed) * wind_speed / 4.0).max(0.15)
            } else {
                1.0
            };
            let share = NEIGHBOURS.len() as f32 / SPREAD_TRIES as f32;
            let p = f * SPREAD_PER_S * TICK_S * up * downwind * intensity * share;
            if fire.roll() < p {
                catch.push((nb, t));
            }
        }
        fire.burning.insert(b, burn);
    }
    for (nb, t) in catch {
        if t.tnt {
            crate::physics::ignite_block(world, &mut physics, nb, 1.5);
        } else {
            fire.light(nb, t.fuel, false);
        }
    }
    // Blocks the fire reached that water or a blast has since cleared.
    let gone: Vec<BlockPos> = fire
        .burning
        .keys()
        .copied()
        .filter(|&b| tinder(world, &layer, b).fuel <= 0.0 && fire.burning[&b].total > GROUND_FIRE_S)
        .collect();
    for b in gone {
        fire.put_out(world, b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Game;
    use glam::DVec3;

    fn forest_floor(game: &mut Game) {
        let mut v = game.world.resource_mut::<Voxels>();
        // Bedrock under 3 m of granite, 40 m each way; a row of five log
        // blocks on it, a gap, then one more.
        v.0.fill_box(IVec3::ZERO, IVec3::new(639, 15, 639), ids::BEDROCK);
        v.0.fill_box(IVec3::new(0, 16, 0), IVec3::new(639, 63, 639), ids::GRANITE);
        for x in 2..7 {
            let o = IVec3::new(x * 16, 64, 64);
            v.0.fill_box(o, o + 15, ids::OAK_LOG);
        }
        let o = IVec3::new(12 * 16, 64, 64);
        v.0.fill_box(o, o + 15, ids::OAK_LOG);
    }

    #[test]
    fn fire_spreads_through_wood_chars_it_and_draws_flames() {
        let mut game = Game::new();
        forest_floor(&mut game);
        game.spawn_player(DVec3::new(2.5, 4.0, 9.5), 0.0, 0.0);
        game.world
            .resource_mut::<Fire>()
            .ignite
            .push((BlockPos(IVec3::new(2, 4, 4)), true));
        let mut flames_seen = 0;
        for _ in 0..(60 * 30) {
            game.update(1.0 / 60.0);
            let v = &game.world.resource::<Voxels>().0;
            flames_seen = flames_seen.max(
                (0..16 * 16)
                    .filter(|i| v.voxel(IVec3::new(2 * 16 + i % 16, 80, 64 + i / 16)) == ids::FIRE)
                    .count(),
            );
        }
        let fire = game.world.resource::<Fire>();
        assert!(fire.caught >= 3, "caught {}", fire.caught);
        assert!(flames_seen > 0, "no flames on top of the first log");
        // Charcoal where the first log was, and the far log untouched.
        let v = &game.world.resource::<Voxels>().0;
        let charcoal = (0..4096)
            .filter(|i| {
                let l = IVec3::new(i & 15, (i >> 8) & 15, (i >> 4) & 15);
                v.voxel(IVec3::new(32, 64, 64) + l) == ids::CHARCOAL
            })
            .count();
        assert!(charcoal > 200, "{charcoal} charcoal voxels");
        assert!(!fire.is_burning(BlockPos(IVec3::new(12, 4, 4))));
        assert_eq!(v.voxel(IVec3::new(12 * 16 + 8, 72, 72)), ids::OAK_LOG);
    }

    #[test]
    fn water_puts_fire_out_and_stone_does_not_burn() {
        let mut game = Game::new();
        forest_floor(&mut game);
        game.spawn_player(DVec3::new(2.5, 4.0, 9.5), 0.0, 0.0);
        {
            let mut fire = game.world.resource_mut::<Fire>();
            fire.ignite.push((BlockPos(IVec3::new(2, 4, 4)), true));
            fire.ignite.push((BlockPos(IVec3::new(9, 3, 9)), true));
        }
        game.update(0.2);
        let (lit, ground) = {
            let fire = game.world.resource::<Fire>();
            (
                fire.is_burning(BlockPos(IVec3::new(2, 4, 4))),
                fire.is_burning(BlockPos(IVec3::new(9, 3, 9))),
            )
        };
        assert!(lit);
        // Stone lit by hand keeps a short flame, then goes out.
        assert!(ground);
        for _ in 0..60 * 4 {
            game.update(1.0 / 60.0);
        }
        assert!(
            !game
                .world
                .resource::<Fire>()
                .is_burning(BlockPos(IVec3::new(9, 3, 9)))
        );
        // Water beside the log.
        {
            let mut v = game.world.resource_mut::<Voxels>();
            let o = IVec3::new(2 * 16, 64, 5 * 16);
            v.0.fill_box(o, o + 15, ids::WATER);
        }
        game.update(0.2);
        game.update(0.2);
        assert!(
            !game
                .world
                .resource::<Fire>()
                .is_burning(BlockPos(IVec3::new(2, 4, 4)))
        );
    }

    #[test]
    fn burnt_forms() {
        assert_eq!(burnt(ids::OAK_LOG), ids::CHARCOAL);
        assert_eq!(burnt(ids::LEAVES), ids::AIR);
        assert_eq!(burnt(ids::THATCH), ids::ASH);
        assert_eq!(burnt(ids::GRASS), ids::DIRT);
        assert_eq!(burnt(ids::GRANITE), ids::GRANITE);
        assert_eq!(fuel_s(ids::GRANITE), 0.0);
    }
}
