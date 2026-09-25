// Minecraft 2's page: chapters whose stills change as you scroll, a strip of
// every still with a lightbox, and a copy button. Everything reads without it.
(() => {
  const doc = document.documentElement;
  doc.classList.add("js");
  const still = matchMedia("(prefers-reduced-motion: reduce)");

  // Top bar: solid once past the hero, tucked away while scrolling down.
  const bar = document.getElementById("bar");
  const progress = document.querySelector(".progress");
  const timeline = CSS.supports("animation-timeline: scroll()");
  let lastY = scrollY;
  let ticking = false;
  const chapters = [...document.querySelectorAll(".chapter")];

  function onScroll() {
    ticking = false;
    const y = scrollY;
    bar.classList.toggle("is-solid", y > innerHeight * 0.6);
    bar.classList.toggle("is-hidden", y > innerHeight && y > lastY + 4);
    if (y < lastY - 4 || y <= innerHeight) bar.classList.remove("is-hidden");
    lastY = y;
    if (!timeline) {
      const max = doc.scrollHeight - innerHeight;
      progress.style.setProperty("--p", max > 0 ? (y / max).toFixed(4) : 0);
    }
    for (const c of chapters) pick(c);
  }

  addEventListener(
    "scroll",
    () => {
      if (!ticking) {
        ticking = true;
        requestAnimationFrame(onScroll);
      }
    },
    { passive: true }
  );

  // Chapters: which still is on depends on how far through the chapter you are.
  for (const c of chapters) {
    const shots = [...c.querySelectorAll(".shot")];
    const caption = c.querySelector(".caption");
    const pips = document.createElement("span");
    pips.className = "pips";
    pips.setAttribute("aria-hidden", "true");
    if (shots.length > 1) shots.forEach(() => pips.append(document.createElement("i")));
    const text = document.createElement("span");
    text.className = "caption-text";
    caption.append(pips, text);
    c._shots = shots;
    c._pips = [...pips.children];
    c._text = text;
    c._on = -1;
    show(c, 0);
  }

  function pick(c) {
    const r = c.getBoundingClientRect();
    if (r.bottom < 0 || r.top > innerHeight) return;
    const run = r.height - innerHeight;
    const t = run > 0 ? Math.min(Math.max(-r.top / run, 0), 0.999) : 0;
    show(c, Math.floor(t * c._shots.length));
  }

  function show(c, i) {
    if (i === c._on) return;
    c._on = i;
    c._shots.forEach((s, j) => {
      s.classList.toggle("is-on", j === i);
      // The next still loads before it is needed.
      if (j <= i + 1) s.querySelector("img").loading = "eager";
    });
    c._pips.forEach((p, j) => p.classList.toggle("on", j === i));
    const cap = c._shots[i].dataset.caption || "";
    if (still.matches || !c._text.textContent) {
      c._text.textContent = cap;
    } else {
      c._text.style.opacity = 0;
      setTimeout(() => {
        c._text.textContent = cap;
        c._text.style.opacity = 1;
      }, 250);
    }
  }

  // Words arrive as a chapter comes into view; the nav marks where you are.
  const nav = new Map(
    [...document.querySelectorAll(".chapters-nav a")].map((a) => [a.hash.slice(1), a])
  );
  const seen = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) e.target.classList.add("in-view");
      }
    },
    { rootMargin: "0px 0px -35% 0px" }
  );
  const here = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        const a = nav.get(e.target.id);
        if (!a) continue;
        if (e.isIntersecting) {
          for (const x of nav.values()) x.removeAttribute("aria-current");
          a.setAttribute("aria-current", "true");
        }
      }
    },
    { rootMargin: "-45% 0px -50% 0px" }
  );
  for (const s of document.querySelectorAll("main > section[id]")) {
    if (s.classList.contains("chapter")) seen.observe(s);
    here.observe(s);
  }

  // The manifesto lights up a word at a time as it passes the middle of the screen.
  const manifesto = document.querySelector(".manifesto p");
  if (manifesto) {
    const wrap = (node) => {
      for (const child of [...node.childNodes]) {
        if (child.nodeType === 3) {
          const frag = document.createDocumentFragment();
          for (const part of child.textContent.split(/(\s+)/)) {
            if (!part) continue;
            if (/^\s+$/.test(part)) {
              frag.append(part);
            } else {
              const w = document.createElement("span");
              w.className = "w";
              w.textContent = part;
              frag.append(w);
            }
          }
          child.replaceWith(frag);
        } else {
          wrap(child);
        }
      }
    };
    wrap(manifesto);
    const words = [...manifesto.querySelectorAll(".w")];
    const light = () => {
      const r = manifesto.getBoundingClientRect();
      const t = (innerHeight * 0.78 - r.top) / (r.height + innerHeight * 0.3);
      const n = still.matches ? words.length : Math.round(Math.min(Math.max(t, 0), 1) * words.length);
      words.forEach((w, i) => w.classList.toggle("lit", i < n));
    };
    addEventListener("scroll", () => requestAnimationFrame(light), { passive: true });
    light();
  }

  // Stills: every chapter's pictures in one strip, each opening full size.
  const strip = document.getElementById("strip");
  const all = [...document.querySelectorAll(".chapter .shot")];
  const box = document.getElementById("lightbox");
  const boxImg = box.querySelector("img");
  const boxCap = box.querySelector("figcaption");
  let at = 0;

  all.forEach((fig, i) => {
    const img = fig.querySelector("img");
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("aria-label", `View still: ${fig.dataset.caption}`);
    const t = document.createElement("img");
    t.src = img.getAttribute("src");
    t.alt = "";
    t.loading = "lazy";
    t.decoding = "async";
    const label = document.createElement("span");
    label.className = "label";
    label.textContent = fig.dataset.caption;
    b.append(t, label);
    b.addEventListener("click", () => open(i));
    li.append(b);
    strip.append(li);
  });

  const wrap = document.createElement("div");
  wrap.className = "strip-wrap";
  strip.replaceWith(wrap);
  wrap.append(strip);
  const stripNav = document.createElement("div");
  stripNav.className = "strip-nav";
  const prev = document.createElement("button");
  const next = document.createElement("button");
  prev.type = next.type = "button";
  prev.textContent = "‹";
  next.textContent = "›";
  prev.setAttribute("aria-label", "Earlier stills");
  next.setAttribute("aria-label", "Later stills");
  stripNav.append(prev, next);
  wrap.append(stripNav);
  const page = (dir) => strip.scrollBy({ left: dir * strip.clientWidth * 0.8, behavior: still.matches ? "auto" : "smooth" });
  prev.addEventListener("click", () => page(-1));
  next.addEventListener("click", () => page(1));
  const ends = () => {
    prev.disabled = strip.scrollLeft < 4;
    next.disabled = strip.scrollLeft + strip.clientWidth > strip.scrollWidth - 4;
  };
  strip.addEventListener("scroll", ends, { passive: true });
  addEventListener("resize", ends);
  ends();

  function big(fig) {
    const img = fig.querySelector("img");
    const set = img.getAttribute("srcset") || "";
    const largest = set.split(",").map((s) => s.trim().split(/\s+/)[0]).pop();
    return largest || img.getAttribute("src");
  }

  function open(i) {
    at = (i + all.length) % all.length;
    const fig = all[at];
    boxImg.src = big(fig);
    boxImg.alt = fig.querySelector("img").alt;
    boxCap.textContent = fig.dataset.caption;
    // Restart the zoom each time the picture changes.
    boxImg.style.animation = "none";
    void boxImg.offsetWidth;
    boxImg.style.animation = "";
    if (!box.open) box.showModal();
  }

  box.querySelector(".lb-prev").addEventListener("click", () => open(at - 1));
  box.querySelector(".lb-next").addEventListener("click", () => open(at + 1));
  box.querySelector("[data-close]").addEventListener("click", () => box.close());
  box.addEventListener("click", (e) => {
    if (e.target === box || e.target.tagName === "FIGURE") box.close();
  });
  box.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") open(at - 1);
    if (e.key === "ArrowRight") open(at + 1);
  });
  let touchX = null;
  box.addEventListener("touchstart", (e) => (touchX = e.touches[0].clientX), { passive: true });
  box.addEventListener("touchend", (e) => {
    if (touchX === null) return;
    const dx = e.changedTouches[0].clientX - touchX;
    if (Math.abs(dx) > 50) open(at + (dx < 0 ? 1 : -1));
    touchX = null;
  });

  // Copy the three commands.
  const copy = document.getElementById("copy");
  const cmd = document.getElementById("cmd");
  copy?.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(cmd.innerText.trim());
      copy.textContent = "Copied";
    } catch {
      const sel = getSelection();
      const range = document.createRange();
      range.selectNodeContents(cmd);
      sel.removeAllRanges();
      sel.addRange(range);
      copy.textContent = "Selected";
    }
    setTimeout(() => (copy.textContent = "Copy"), 1800);
  });

  onScroll();
})();
