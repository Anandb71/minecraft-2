//! Water: free-surface lattice Boltzmann (D3Q19) over sparse tiles of
//! half-metre cells, run only where water moves.

pub mod lattice;
pub mod sim;
mod tile;

pub use sim::{CELL_M, FluidStats, FluidWorld, Params, STEP_S, Terrain};
pub use tile::Kind;
