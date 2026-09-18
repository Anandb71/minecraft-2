//! Water: free-surface lattice Boltzmann (D3Q19) over sparse tiles of
//! half-metre cells, run only where water moves.

pub mod lattice;
pub mod sim;
mod tile;

pub use sim::{
    CELL_M, FluidStats, FluidWorld, MAX_SPEED, Params, SLEEP_STEPS, STEP_S, STILL_SPEED, Terrain,
};
pub use tile::{Kind, TILE, TILE_CELLS, index_of, local_of, near_slot, need_of, slot_offset};
