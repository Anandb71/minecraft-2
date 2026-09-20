(() => {
  const canvas = document.getElementById("world");
  const fallback = document.getElementById("fallback");
  const scaleEl = document.getElementById("scale");
  const lede = document.getElementById("lede");
  const tod = document.getElementById("tod");
  const debugBtn = document.getElementById("debug-btn");
  const photos = document.getElementById("photos");
  const clone = document.getElementById("clone");
  const world = typeof startVoxelWorld === "function" ? startVoxelWorld(canvas) : null;

  if (!world) {
    canvas.hidden = true;
    if (fallback) fallback.hidden = false;
  }

  const gazes = {
    0: "Empty cells are free. The march skips them and only stops on a solid.",
    1: "Terraces. Height is quantized so the land reads as steps, not mush.",
    2: "Dirt under the grass. Same grid, different palette index.",
    3: "Stone. The continent is 16 km of this, at 6.25 cm a cell.",
    4: "Wood. The pick is voxels too, same as the hill.",
    5: "Leaves. A tree is just a handful of cells with a 4-bit color.",
    6: "You're looking through the cube. The pick is on the other side.",
    7: "Sand at the waterline. The sea is another index, not a shader trick.",
    8: "Bright stone. Sixteen colors is the whole language.",
    9: "Water is a palette slot. You can still see the bed if the march keeps going.",
    10: "Clay. Same dirt-orange as the clone button, on purpose.",
  };
  const defaultLede =
    "Look through the cube. The pick is behind the glass. This page marches voxels in your GPU, the same idea as the game, just a tiny field.";

  const tick = () => {
    if (!world) return;
    if (scaleEl) scaleEl.textContent = `this box ≈ ${world.scale().toFixed(1)} m across the view`;
    if (lede && !document.querySelector("dialog[open]")) {
      const id = world.look();
      lede.textContent = gazes[id] || defaultLede;
    }
  };
  tick();
  setInterval(tick, 180);

  tod?.addEventListener("input", () => world?.setTime(Number(tod.value) / 100));
  world?.setTime(Number(tod?.value || 72) / 100);

  debugBtn?.addEventListener("click", () => {
    const on = debugBtn.getAttribute("aria-pressed") !== "true";
    debugBtn.setAttribute("aria-pressed", String(on));
    world?.setHeat(on);
  });

  const openDialog = (d) => {
    if (!d) return;
    world?.pause();
    d.showModal();
  };
  document.getElementById("photos-open")?.addEventListener("click", () => openDialog(photos));
  document.getElementById("clone-open")?.addEventListener("click", () => openDialog(clone));
  document.querySelectorAll("[data-close]").forEach((btn) => {
    btn.addEventListener("click", () => btn.closest("dialog")?.close());
  });
  document.querySelectorAll("dialog").forEach((d) => {
    d.addEventListener("close", () => world?.resume());
    d.addEventListener("click", (e) => {
      if (e.target === d) d.close();
    });
  });

  document.getElementById("zoom-in")?.addEventListener("click", () => world?.zoom(-2.2));
  document.getElementById("zoom-out")?.addEventListener("click", () => world?.zoom(2.2));

  const copyBtn = document.getElementById("copy");
  const cmd = document.getElementById("cmd")?.textContent || "";
  copyBtn?.addEventListener("click", async () => {
    copyBtn.disabled = true;
    try {
      await navigator.clipboard.writeText(cmd);
      copyBtn.textContent = "Copied";
    } catch {
      copyBtn.textContent = "Select the text";
    }
    setTimeout(() => {
      copyBtn.disabled = false;
      copyBtn.textContent = "Copy";
    }, 1400);
  });

  addEventListener("keydown", (e) => {
    if (!(e.target instanceof Element)) return;
    if (e.key === "Escape") {
      photos?.open && photos.close();
      clone?.open && clone.close();
      return;
    }
    if (e.target.closest("input, textarea, button, a, dialog")) return;
    if (e.key.toLowerCase() === "p") openDialog(photos);
    if (e.key.toLowerCase() === "c" && !e.ctrlKey && !e.metaKey) openDialog(clone);
    if (e.key.toLowerCase() === "h") debugBtn?.click();
  });
})();
