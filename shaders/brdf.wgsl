// Microfacet BRDF terms shared by every shading pass.

// GGX normal distribution, Smith-correlated visibility, Schlick Fresnel.
fn ggx_specular(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32, f0: vec3<f32>) -> vec3<f32> {
    let h = normalize(v + l);
    let nl = max(dot(n, l), 0.0);
    let nv = max(dot(n, v), 1e-4);
    let nh = max(dot(n, h), 0.0);
    let vh = max(dot(v, h), 0.0);
    let a = max(roughness * roughness, 2e-3);
    let a2 = a * a;
    let d = a2 / (PI * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
    let gv = nl * sqrt(nv * nv * (1.0 - a2) + a2);
    let gl = nv * sqrt(nl * nl * (1.0 - a2) + a2);
    let vis = 0.5 / max(gv + gl, 1e-5);
    let f = f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
    return d * vis * f * nl;
}

