//! MINECRAFT 2: a voxel sandbox ray marched on the GPU.

mod app;
mod cli;
mod flycam;
mod headless;
mod logger;
mod world;

fn main() {
    logger::init();
    let args = match cli::parse(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(2);
        }
    };
    let result = match args.mode {
        cli::Mode::Window => app::run(&args),
        _ => headless::run(&args),
    };
    if let Err(e) = result {
        eprintln!("fatal: {e}");
        std::process::exit(1);
    }
}
