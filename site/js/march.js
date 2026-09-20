(() => {
  const VS = `#version 300 es
in vec2 a;
void main(){ gl_Position = vec4(a,0.0,1.0); }`;

  const FS = `#version 300 es
precision highp float;
precision highp sampler3D;
uniform sampler3D vol;
uniform vec2 res;
uniform vec3 camPos;
uniform vec3 camFwd;
uniform vec3 camRight;
uniform vec3 camUp;
uniform vec3 sunDir;
uniform float N;
uniform float tod;
uniform float heat;
out vec4 fragColor;

vec3 sky(vec3 rd){
  float h = clamp(rd.y * 0.55 + 0.45, 0.0, 1.0);
  float day = smoothstep(0.14, 0.32, tod) * (1.0 - smoothstep(0.56, 0.70, tod));
  float dusk = smoothstep(0.56, 0.68, tod) * (1.0 - smoothstep(0.82, 0.94, tod));
  vec3 nightZ = vec3(0.025, 0.03, 0.055);
  vec3 nightH = vec3(0.05, 0.06, 0.12);
  vec3 dayZ = vec3(0.42, 0.66, 0.92);
  vec3 dayH = vec3(0.86, 0.91, 0.96);
  vec3 duskZ = vec3(0.12, 0.10, 0.22);
  vec3 duskH = vec3(0.98, 0.48, 0.22);
  vec3 col = mix(nightH, nightZ, h);
  col = mix(col, mix(dayH, dayZ, h), day);
  col = mix(col, mix(duskH, duskZ, h), dusk);
  float sun = pow(max(dot(rd, sunDir), 0.0), 220.0);
  float glow = pow(max(dot(rd, sunDir), 0.0), 8.0);
  col += vec3(1.0, 0.84, 0.55) * sun * (0.4 + 1.2 * dusk + 0.5 * day);
  col += vec3(0.95, 0.42, 0.18) * glow * dusk * 0.35;
  if(day < 0.25){
    float speckle = fract(sin(dot(rd.xy, vec2(12.9898, 78.233))) * 43758.5453);
    if(speckle > 0.9965) col += vec3(0.75, 0.82, 1.0) * (1.0 - day);
  }
  return col;
}

vec3 pal(int id){
  if(id==1) return vec3(0.34, 0.55, 0.22);
  if(id==2) return vec3(0.45, 0.32, 0.18);
  if(id==3) return vec3(0.48, 0.49, 0.52);
  if(id==4) return vec3(0.28, 0.15, 0.07);
  if(id==5) return vec3(0.14, 0.36, 0.16);
  if(id==6) return vec3(0.62, 0.84, 0.90);
  if(id==7) return vec3(0.78, 0.70, 0.48);
  if(id==8) return vec3(0.82, 0.83, 0.85);
  if(id==9) return vec3(0.08, 0.28, 0.46);
  if(id==10) return vec3(0.78, 0.42, 0.22);
  return vec3(1.0, 0.0, 1.0);
}

int voxel(ivec3 p){
  if(p.x<0||p.y<0||p.z<0||p.x>=int(N)||p.y>=int(N)||p.z>=int(N)) return 0;
  return int(texelFetch(vol, p, 0).r * 255.0 + 0.5);
}

bool march(vec3 ro, vec3 rd, float maxT, out float tHit, out ivec3 cell, out vec3 nrm, out int mat, out float glass, out float steps, out vec3 glassN){
  vec3 inv = 1.0 / rd;
  vec3 t0 = (vec3(0.0) - ro) * inv;
  vec3 t1 = (vec3(N) - ro) * inv;
  vec3 tsm = min(t0, t1);
  vec3 tlg = max(t0, t1);
  float tEnter = max(max(max(tsm.x, tsm.y), tsm.z), 0.0);
  float tExit = min(min(tlg.x, tlg.y), tlg.z);
  tHit = 0.0;
  nrm = vec3(0.0, 1.0, 0.0);
  mat = 0;
  glass = 0.0;
  steps = 0.0;
  glassN = vec3(0.0, 1.0, 0.0);
  if(tExit < tEnter) return false;
  tExit = min(tExit, maxT);
  float t = tEnter + 0.0004;
  vec3 pos = ro + rd * t;
  ivec3 ip = ivec3(floor(pos));
  ivec3 stepv = ivec3(sign(rd));
  vec3 tDelta = abs(inv);
  vec3 tMax;
  tMax.x = ((rd.x >= 0.0) ? (float(ip.x) + 1.0 - pos.x) : (pos.x - float(ip.x))) * tDelta.x;
  tMax.y = ((rd.y >= 0.0) ? (float(ip.y) + 1.0 - pos.y) : (pos.y - float(ip.y))) * tDelta.y;
  tMax.z = ((rd.z >= 0.0) ? (float(ip.z) + 1.0 - pos.z) : (pos.z - float(ip.z))) * tDelta.z;
  for(int i = 0; i < 260; i++){
    if(t >= tExit) break;
    steps += 1.0;
    int id = voxel(ip);
      if(id == 6){
      glass += 0.42;
      glassN = nrm;
    } else if(id > 0){
      tHit = t;
      cell = ip;
      mat = id;
      return true;
    }
    if(tMax.x < tMax.y){
      if(tMax.x < tMax.z){
        ip.x += stepv.x;
        t = tEnter + tMax.x;
        tMax.x += tDelta.x;
        nrm = vec3(-float(stepv.x), 0.0, 0.0);
      } else {
        ip.z += stepv.z;
        t = tEnter + tMax.z;
        tMax.z += tDelta.z;
        nrm = vec3(0.0, 0.0, -float(stepv.z));
      }
    } else {
      if(tMax.y < tMax.z){
        ip.y += stepv.y;
        t = tEnter + tMax.y;
        tMax.y += tDelta.y;
        nrm = vec3(0.0, -float(stepv.y), 0.0);
      } else {
        ip.z += stepv.z;
        t = tEnter + tMax.z;
        tMax.z += tDelta.z;
        nrm = vec3(0.0, 0.0, -float(stepv.z));
      }
    }
  }
  return false;
}

void main(){
  vec2 uv = (gl_FragCoord.xy - 0.5 * res) / res.y;
  uv *= 0.72;
  vec3 rd = normalize(camFwd + uv.x * camRight + uv.y * camUp);
  float tHit;
  ivec3 cell;
  vec3 nrm;
  int mat;
  float glass;
  float steps;
  vec3 glassN;
  vec3 skyC = sky(rd);
  bool hit = march(camPos, rd, 460.0, tHit, cell, nrm, mat, glass, steps, glassN);
  if(!hit){
    vec3 col = skyC;
    col *= exp(-glass * vec3(0.55, 0.22, 0.12));
    col += vec3(0.38, 0.72, 0.84) * min(glass, 2.2) * 0.28;
    float fres = pow(1.0 - abs(dot(rd, glassN)), 2.2);
    float spec = pow(max(dot(reflect(rd, glassN), sunDir), 0.0), 36.0);
    col += vec3(0.8, 0.94, 1.0) * fres * min(glass, 1.4) * 0.55;
    col += vec3(1.0, 0.92, 0.78) * spec * min(glass, 1.6) * 0.65;
    if(heat > 0.5) col = mix(col, vec3(1.0, 0.22, 0.05) * (steps / 200.0), 0.58);
    fragColor = vec4(col, 1.0);
    return;
  }
  vec3 p = camPos + rd * tHit;
  vec3 alb = pal(mat);
  if(mat == 9){
    float spark = pow(max(dot(reflect(rd, nrm), sunDir), 0.0), 40.0);
    alb = mix(alb, skyC, 0.38);
    alb += vec3(0.55, 0.75, 0.9) * spark * 0.45;
  }
  float ndl = max(dot(nrm, sunDir), 0.0);
  float tS;
  ivec3 cS;
  vec3 nS;
  int mS;
  float gS;
  float sS;
  vec3 gN;
  bool shade = march(p + nrm * 0.07, sunDir, 96.0, tS, cS, nS, mS, gS, sS, gN);
  if(shade && mS != 6) ndl *= 0.14;
  else ndl *= mix(1.0, 0.72, clamp(gS, 0.0, 1.0));
  float fill = 0.20 + 0.22 * max(nrm.y, 0.0);
  vec3 col = alb * (fill + ndl * 0.98);
  col *= exp(-glass * vec3(0.42, 0.16, 0.08));
  col = mix(col, vec3(0.42, 0.74, 0.86), 1.0 - exp(-glass * 0.85));
  float fres = pow(1.0 - abs(dot(rd, glassN)), 2.2);
  float spec = pow(max(dot(reflect(rd, glassN), sunDir), 0.0), 36.0);
  col += vec3(0.75, 0.9, 1.0) * fres * min(glass, 1.5) * 0.5;
  col += vec3(1.0, 0.9, 0.75) * spec * min(glass, 1.6) * 0.55;
  float fog = 1.0 - exp(-tHit * 0.011);
  col = mix(col, skyC, fog * 0.42);
  if(heat > 0.5) col = mix(col, vec3(1.0, 0.2, 0.04) * (steps / 200.0), 0.52);
  fragColor = vec4(col, 1.0);
}`;

  function compile(gl, type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    const log = gl.getShaderInfoLog(s) || "";
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error(log);
      window.__shaderLog = log;
      return null;
    }
    if (log) window.__shaderWarn = log;
    return s;
  }

  function hash(x, z) {
    const n = Math.sin(x * 127.1 + z * 311.7) * 43758.5453;
    return n - Math.floor(n);
  }

  function buildVolume(N) {
    const data = new Uint8Array(N * N * N);
    const set = (x, y, z, v) => {
      if (x < 0 || y < 0 || z < 0 || x >= N || y >= N || z >= N) return;
      data[x + y * N + z * N * N] = v;
    };
    const sea = 9;
    const height = (x, z) => {
      const nx = x / N;
      const nz = z / N;
      const e =
        0.62 * Math.sin(x * 0.09) * Math.cos(z * 0.08) +
        0.28 * Math.sin((x + z) * 0.041) +
        0.12 * Math.sin(x * 0.21 + z * 0.17);
      let h = 10 + Math.floor((e * 0.5 + 0.5) * 22);
      h = Math.floor(h / 3) * 3;
      if (nz > 0.78) h = sea - 4;
      else if (nz > 0.7) h = Math.min(h, sea - 1);
      if (nx < 0.12) h = Math.max(h - 3, sea - 2);
      return h;
    };

    for (let z = 0; z < N; z++) {
      for (let x = 0; x < N; x++) {
        const h = height(x, z);
        const sand = z > N * 0.68 || h <= sea;
        for (let y = 0; y < h; y++) {
          let id = 3;
          if (y === h - 1) id = sand ? 7 : 1;
          else if (y > h - 4) id = sand ? 7 : 2;
          set(x, y, z, id);
        }
        if (h < sea) {
          for (let y = h; y < sea; y++) set(x, y, z, 9);
        }
      }
    }

    const cx = (N * 0.52) | 0;
    const cz = (N * 0.42) | 0;
    const gh = height(cx, cz);

    for (let dx = -14; dx <= 14; dx++) {
      for (let dz = -14; dz <= 14; dz++) {
        const h = height(cx + dx, cz + dz);
        if (h > gh) {
          for (let y = gh; y < h; y++) set(cx + dx, y, cz + dz, 0);
        } else {
          for (let y = h; y < gh; y++) {
            set(cx + dx, y, cz + dz, y === gh - 1 ? 1 : 2);
          }
        }
        set(cx + dx, gh - 1, cz + dz, 1);
      }
    }

    for (let dx = -3; dx <= 3; dx++) {
      for (let dz = -3; dz <= 3; dz++) {
        set(cx + dx, gh, cz + dz, 3);
        set(cx + dx, gh + 1, cz + dz, 8);
      }
    }

    for (let dx = -4; dx <= 4; dx++) {
      for (let dz = -4; dz <= 4; dz++) {
        for (let dy = 2; dy <= 9; dy++) {
          const wall = Math.abs(dx) >= 3 || Math.abs(dz) >= 3;
          const roof = dy >= 8;
          const floor = dy === 2;
          if (wall || roof || floor) set(cx + dx, gh + dy, cz + dz, 6);
        }
      }
    }

    const frame = (x, y, z) => set(x, y, z, 10);
    for (let dy = 2; dy <= 9; dy++) {
      frame(cx - 4, gh + dy, cz - 4);
      frame(cx + 4, gh + dy, cz - 4);
      frame(cx - 4, gh + dy, cz + 4);
      frame(cx + 4, gh + dy, cz + 4);
    }
    for (let dx = -4; dx <= 4; dx++) {
      frame(cx + dx, gh + 2, cz - 4);
      frame(cx + dx, gh + 2, cz + 4);
      frame(cx + dx, gh + 9, cz - 4);
      frame(cx + dx, gh + 9, cz + 4);
    }
    for (let dz = -4; dz <= 4; dz++) {
      frame(cx - 4, gh + 2, cz + dz);
      frame(cx + 4, gh + 2, cz + dz);
      frame(cx - 4, gh + 9, cz + dz);
      frame(cx + 4, gh + 9, cz + dz);
    }

    const px = cx - 1;
    const pz = cz;
    for (let y = 3; y <= 7; y++) {
      set(px, gh + y, pz, 4);
      set(px, gh + y, pz + 1, 4);
    }
    for (let dx = -2; dx <= 2; dx++) {
      set(px + dx, gh + 7, pz, 4);
      set(px + dx, gh + 7, pz + 1, 4);
    }
    set(px - 2, gh + 6, pz, 4);
    set(px + 2, gh + 6, pz, 4);
    set(px, gh + 8, pz, 4);
    set(px, gh + 8, pz + 1, 4);

    for (let dx = -2; dx <= 1; dx++) {
      for (let dz = -2; dz <= 1; dz++) {
        for (let y = gh - 5; y < gh; y++) set(cx - 18 + dx, y, cz + 6 + dz, 0);
      }
    }

    for (let i = 0; i < 16; i++) {
      const x = 8 + ((hash(i, 3) * (N - 16)) | 0);
      const z = 8 + ((hash(9, i) * (N * 0.62)) | 0);
      if (Math.hypot(x - cx, z - cz) < 18) continue;
      if (z > N * 0.66) continue;
      const th = height(x, z);
      if (th <= sea) continue;
      const ht = 5 + ((hash(x, z) * 5) | 0);
      for (let y = 1; y <= ht; y++) set(x, th + y, z, 4);
      for (let dy = -1; dy <= 3; dy++) {
        for (let ox = -2; ox <= 2; ox++) {
          for (let oz = -2; oz <= 2; oz++) {
            if (Math.abs(ox) + Math.abs(dy) + Math.abs(oz) < 5) set(x + ox, th + ht + dy, z + oz, 5);
          }
        }
      }
    }

    const bx = cx + 16;
    const bz = cz - 10;
    const bh = height(bx, bz);
    for (let i = 0; i < 10; i++) {
      set(bx + (i % 5), bh + 1 + Math.floor(i / 5), bz, i === 0 ? 10 : i);
    }

    let counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (v < counts.length) counts[v]++;
    }

    return { data, cx, cz, gh, counts };
  }

  function cpuProbe(ro, rd, N, data, skipGlass) {
    const inv = [1 / rd[0], 1 / rd[1], 1 / rd[2]];
    const t0 = [-ro[0] * inv[0], -ro[1] * inv[1], -ro[2] * inv[2]];
    const t1 = [(N - ro[0]) * inv[0], (N - ro[1]) * inv[1], (N - ro[2]) * inv[2]];
    const tsm = [Math.min(t0[0], t1[0]), Math.min(t0[1], t1[1]), Math.min(t0[2], t1[2])];
    const tlg = [Math.max(t0[0], t1[0]), Math.max(t0[1], t1[1]), Math.max(t0[2], t1[2])];
    const tEnter = Math.max(tsm[0], tsm[1], tsm[2], 0);
    const tExit = Math.min(tlg[0], tlg[1], tlg[2]);
    if (tExit < tEnter) return 0;
    const pos = [ro[0] + rd[0] * (tEnter + 0.0004), ro[1] + rd[1] * (tEnter + 0.0004), ro[2] + rd[2] * (tEnter + 0.0004)];
    const ip = [Math.floor(pos[0]), Math.floor(pos[1]), Math.floor(pos[2])];
    const step = [Math.sign(rd[0]) || 1, Math.sign(rd[1]) || 1, Math.sign(rd[2]) || 1];
    const tDelta = [Math.abs(inv[0]), Math.abs(inv[1]), Math.abs(inv[2])];
    const tMax = [
      (rd[0] >= 0 ? ip[0] + 1 - pos[0] : pos[0] - ip[0]) * tDelta[0],
      (rd[1] >= 0 ? ip[1] + 1 - pos[1] : pos[1] - ip[1]) * tDelta[1],
      (rd[2] >= 0 ? ip[2] + 1 - pos[2] : pos[2] - ip[2]) * tDelta[2],
    ];
    for (let i = 0; i < 260; i++) {
      const [x, y, z] = ip;
      if (x < 0 || y < 0 || z < 0 || x >= N || y >= N || z >= N) return 0;
      const id = data[x + y * N + z * N * N];
      if (id && (!skipGlass || id !== 6)) return { id, ix: x, iy: y, iz: z };
      if (tMax[0] < tMax[1]) {
        if (tMax[0] < tMax[2]) {
          ip[0] += step[0];
          tMax[0] += tDelta[0];
        } else {
          ip[2] += step[2];
          tMax[2] += tDelta[2];
        }
      } else if (tMax[1] < tMax[2]) {
        ip[1] += step[1];
        tMax[1] += tDelta[1];
      } else {
        ip[2] += step[2];
        tMax[2] += tDelta[2];
      }
    }
    return 0;
  }

  window.startVoxelWorld = function startVoxelWorld(canvas) {
    const gl = canvas.getContext("webgl2", {
      antialias: false,
      alpha: false,
      powerPreference: "high-performance",
    });
    if (!gl) return null;
    const vs = compile(gl, gl.VERTEX_SHADER, VS);
    const fs = compile(gl, gl.FRAGMENT_SHADER, FS);
    if (!vs || !fs) return null;
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.bindAttribLocation(prog, 0, "a");
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error(gl.getProgramInfoLog(prog));
      return null;
    }
    gl.useProgram(prog);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    const N = 80;
    const built = buildVolume(N);
    const data = built.data;
    const tex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, tex);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, N, N, N, 0, gl.RED, gl.UNSIGNED_BYTE, data);

    const u = (name) => gl.getUniformLocation(prog, name);
    gl.uniform1i(u("vol"), 0);
    gl.uniform1f(u("N"), N);

    const target = [built.cx, built.gh + 5.6, built.cz];
    let yaw = -1.28;
    let pitch = -0.2;
    let dist = 30;
    let tod = 0.72;
    let heat = 0;
    let dragging = false;
    let moved = false;
    let lastX = 0;
    let lastY = 0;
    let dirty = true;
    let paused = false;
    const keys = new Set();
    const pointers = new Map();
    let pinch0 = 0;

    const cross = (a, b) => [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
    const norm = (a) => {
      const l = Math.hypot(a[0], a[1], a[2]) || 1;
      return [a[0] / l, a[1] / l, a[2] / l];
    };
    const cam = () => {
      const cp = Math.cos(pitch);
      const fwd = [Math.sin(yaw) * cp, Math.sin(pitch), Math.cos(yaw) * cp];
      const pos = [target[0] - fwd[0] * dist, target[1] - fwd[1] * dist, target[2] - fwd[2] * dist];
      const right = norm(cross(fwd, [0, 1, 0]));
      const up = norm(cross(right, fwd));
      return { pos, fwd, right, up };
    };
    const sunFrom = (t) => {
      const a = (t - 0.25) * Math.PI * 2;
      return norm([Math.cos(a) * 0.88, Math.sin(a) * 0.78 + 0.06, 0.22]);
    };

    const uploadVoxel = (ix, iy, iz, v) => {
      data[ix + iy * N + iz * N * N] = v;
      gl.bindTexture(gl.TEXTURE_3D, tex);
      gl.texSubImage3D(gl.TEXTURE_3D, 0, ix, iy, iz, 1, 1, 1, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array([v]));
    };

    const carveAt = (cx, cy) => {
      const rect = canvas.getBoundingClientRect();
      const x = ((cx - rect.left) / rect.width) * 2 - 1;
      const y = (1 - (cy - rect.top) / rect.height) * 2 - 1;
      const aspect = rect.width / rect.height;
      const fov = 0.72;
      const c = cam();
      const rd = norm([
        c.fwd[0] + x * aspect * fov * c.right[0] + y * fov * c.up[0],
        c.fwd[1] + x * aspect * fov * c.right[1] + y * fov * c.up[1],
        c.fwd[2] + x * aspect * fov * c.right[2] + y * fov * c.up[2],
      ]);
      const hit = cpuProbe(c.pos, rd, N, data, true);
      if (!hit) return;
      const spots = [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
      ];
      for (const [ox, oy, oz] of spots) {
        const ix = hit.ix + ox;
        const iy = hit.iy + oy;
        const iz = hit.iz + oz;
        if (ix < 0 || iy < 1 || iz < 0 || ix >= N || iy >= N || iz >= N) continue;
        const id = data[ix + iy * N + iz * N * N];
        if (id && id !== 6) uploadVoxel(ix, iy, iz, 0);
      }
      dirty = true;
    };

    canvas.addEventListener("pointerdown", (e) => {
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 2) {
        const pts = [...pointers.values()];
        pinch0 = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
        dragging = false;
        return;
      }
      dragging = true;
      moved = false;
      lastX = e.clientX;
      lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
      if (pointers.has(e.pointerId)) pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 2) {
        const pts = [...pointers.values()];
        const d = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
        if (pinch0) {
          dist = Math.max(7, Math.min(N * 1.4, dist * (pinch0 / d)));
          pinch0 = d;
          dirty = true;
        }
        return;
      }
      if (!dragging) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      if (Math.hypot(dx, dy) > 3) moved = true;
      yaw += dx * 0.008;
      pitch = Math.max(-1.2, Math.min(0.22, pitch - dy * 0.008));
      lastX = e.clientX;
      lastY = e.clientY;
      dirty = true;
    });
    const endPtr = (e) => {
      pointers.delete(e.pointerId);
      pinch0 = 0;
      if (dragging && pointers.size === 0) {
        dragging = false;
        if (!moved) carveAt(e.clientX, e.clientY);
      }
      dragging = pointers.size === 1;
    };
    canvas.addEventListener("pointerup", endPtr);
    canvas.addEventListener("pointercancel", endPtr);
    canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        dist = Math.max(7, Math.min(N * 1.4, dist + e.deltaY * 0.032));
        dirty = true;
      },
      { passive: false }
    );

    addEventListener("keydown", (e) => {
      const el = e.target;
      if (el instanceof Element && el.closest("input, textarea, button, a, dialog")) return;
      keys.add(e.key.toLowerCase());
      if (e.key === "=" || e.key === "+") {
        dist = Math.max(7, dist - 1.6);
        dirty = true;
      }
      if (e.key === "-" || e.key === "_") {
        dist = Math.min(N * 1.4, dist + 1.6);
        dirty = true;
      }
    });
    addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));

    const draw = () => {
      const dpr = Math.min(devicePixelRatio || 1, 1.25);
      let w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
      let h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
      const cap = 1400;
      if (w > cap) {
        h = Math.max(1, Math.floor((h * cap) / w));
        w = cap;
      }
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
        dirty = true;
      }
      if (!dirty) return;
      const c = cam();
      const sun = sunFrom(tod);
      gl.viewport(0, 0, w, h);
      gl.uniform2f(u("res"), w, h);
      gl.uniform3f(u("camPos"), c.pos[0], c.pos[1], c.pos[2]);
      gl.uniform3f(u("camFwd"), c.fwd[0], c.fwd[1], c.fwd[2]);
      gl.uniform3f(u("camRight"), c.right[0], c.right[1], c.right[2]);
      gl.uniform3f(u("camUp"), c.up[0], c.up[1], c.up[2]);
      gl.uniform3f(u("sunDir"), sun[0], sun[1], sun[2]);
      gl.uniform1f(u("tod"), tod);
      gl.uniform1f(u("heat"), heat);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      dirty = false;
    };

    const loop = () => {
      if (!paused) {
        const c = cam();
        const speed = 0.32;
        const flat = norm([c.fwd[0], 0, c.fwd[2]]);
        let moving = false;
        if (keys.has("w") || keys.has("arrowup")) {
          target[0] += flat[0] * speed;
          target[2] += flat[2] * speed;
          moving = true;
        }
        if (keys.has("s") || keys.has("arrowdown")) {
          target[0] -= flat[0] * speed;
          target[2] -= flat[2] * speed;
          moving = true;
        }
        if (keys.has("a") || keys.has("arrowleft")) {
          target[0] -= c.right[0] * speed;
          target[2] -= c.right[2] * speed;
          moving = true;
        }
        if (keys.has("d") || keys.has("arrowright")) {
          target[0] += c.right[0] * speed;
          target[2] += c.right[2] * speed;
          moving = true;
        }
        if (keys.has("e")) {
          target[1] += speed;
          moving = true;
        }
        if (keys.has("q")) {
          target[1] -= speed;
          moving = true;
        }
        if (moving) dirty = true;
        draw();
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);

    window.__mc2 = {
      counts: built.counts,
      gh: built.gh,
      cx: built.cx,
      cz: built.cz,
      uniforms: {
        camPos: u("camPos"),
        camFwd: u("camFwd"),
        vol: u("vol"),
        res: u("res"),
        tod: u("tod"),
      },
      err: () => gl.getError(),
      progLink: gl.getProgramParameter(prog, gl.LINK_STATUS),
      sample(x, y, z) {
        return data[x + y * N + z * N * N];
      },
      cam: cam,
      target,
    };

    return {
      setTime(t) {
        tod = t;
        dirty = true;
      },
      setHeat(on) {
        heat = on ? 1 : 0;
        dirty = true;
      },
      zoom(delta) {
        dist = Math.max(7, Math.min(N * 1.4, dist + delta));
        dirty = true;
      },
      scale() {
        return dist * 0.0625;
      },
      look() {
        const c = cam();
        const hit = cpuProbe(c.pos, c.fwd, N, data, false);
        return hit ? hit.id : 0;
      },
      pause() {
        paused = true;
        keys.clear();
      },
      resume() {
        paused = false;
        dirty = true;
      },
    };
  };
})();
