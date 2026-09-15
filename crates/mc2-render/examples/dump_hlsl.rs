//! Writes naga's HLSL translation of a shader, for debugging FXC errors.

fn main() {
    let name = std::env::args().nth(1).expect("shader name");
    let lib = mc2_render::shaders::library();
    let src = lib.source(&name).expect("shader source");
    let module = naga::front::wgsl::parse_str(&src).expect("parse");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("validate");
    let options = naga::back::hlsl::Options::default();
    let mut out = String::new();
    let pipeline = naga::back::hlsl::PipelineOptions::default();
    let mut writer = naga::back::hlsl::Writer::new(&mut out, &options, &pipeline);
    writer.write(&module, &info, None).expect("hlsl");
    print!("{out}");
}
