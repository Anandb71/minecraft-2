//! Structural integrity over the 1 m block layer.
//!
//! Every solid block is a node carrying its own weight. Load flows from each
//! node toward anchors (bedrock, and the edge of the region being examined)
//! through the faces it shares with its neighbours, preferring to go down.
//! Each connection is checked in compression, bending and shear against
//! the strength of the weaker material, scaled for how the blocks are
//! joined. Overstressed blocks fail and crumble; whatever no longer reaches
//! an anchor falls as rigid bodies.

pub mod node;
pub mod solve;
pub mod split;

pub use node::{NodeInfo, block_info};
pub use solve::{Outcome, Params, RegionNode, solve};
pub use split::cut;
