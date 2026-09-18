//! Water: free-surface lattice Boltzmann (D3Q19) over sparse tiles of
//! half-metre cells, run only where water moves.

pub mod lattice;
mod tile;

pub use tile::Kind;
