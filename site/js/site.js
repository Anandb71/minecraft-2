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

  // The trailer: every chapter's stills full screen, one after another,
  // the chapter's title over its first, a caption over each.
  const reel = document.getElementById("reel");
  const reelImgs = [...reel.querySelectorAll(".reel-stage img")];
  const reelCard = reel.querySelector(".reel-card");
  const reelNo = reelCard.querySelector(".pixel");
  const reelKicker = reelCard.querySelector(".reel-kicker");
  const reelTitle = reelCard.querySelector(".reel-title");
  const reelCap = reelCard.querySelector(".reel-cap");
  const reelBar = reel.querySelector(".reel-bar");
  const pauseBtn = reel.querySelector(".reel-pause");
  const scenes = chapters.flatMap((c) => {
    const no = c.querySelector(".chapter-no");
    const num = no.querySelector(".pixel").textContent;
    const kicker = no.textContent.replace(num, "").trim();
    const title = c.querySelector("h2").textContent;
    return c._shots.map((fig, j) => ({
      num,
      kicker,
      title,
      first: j === 0,
      fig,
      caption: fig.dataset.caption,
    }));
  });
  scenes.forEach(() => {
    const li = document.createElement("li");
    li.append(document.createElement("i"));
    reelBar.append(li);
  });
  const bars = [...reelBar.children];
  let scene = -1;
  let front = 0;
  let timer = 0;
  let left = 0;
  let started = 0;
  let paused = false;
  let shown = 0;

  const length = (s) => (s.first ? 5600 : 4400);
  // Full size on big screens, the small still on phones.
  const srcOf = (s) =>
    innerWidth * devicePixelRatio > 1400 ? big(s.fig) : s.fig.querySelector("img").getAttribute("src");

  function play(i) {
    if (i >= scenes.length) {
      reel.close();
      return;
    }
    i = Math.max(i, 0);
    const s = scenes[i];
    const chapterChanged = scene < 0 || scenes[scene].title !== s.title;
    scene = i;
    const dur = length(s);
    reel.style.setProperty("--dur", `${dur}ms`);
    // Crossfade onto the other image once it has loaded.
    const next = reelImgs[1 - front];
    const show = () => {
      next.classList.remove("on");
      void next.offsetWidth;
      next.classList.add("on");
      reelImgs[front].classList.remove("on");
      front = 1 - front;
    };
    const token = ++shown;
    next.onload = () => token === shown && show();
    next.src = srcOf(s);
    if (next.complete) {
      next.onload = null;
      show();
    }
    if (chapterChanged) {
      reelNo.textContent = s.num;
      reelKicker.textContent = s.kicker;
      reelTitle.textContent = s.title;
      reelCard.classList.remove("enter");
      void reelCard.offsetWidth;
      reelCard.classList.add("enter");
    }
    reelCap.textContent = s.caption;
    reelCap.classList.remove("enter");
    void reelCap.offsetWidth;
    reelCap.classList.add("enter");
    bars.forEach((b, j) => {
      b.className = j < i ? "done" : j === i ? "now" : "";
    });
    // Warm the next still.
    if (scenes[i + 1]) new Image().src = srcOf(scenes[i + 1]);
    run(dur);
  }

  function run(ms) {
    clearTimeout(timer);
    left = ms;
    started = performance.now();
    if (!paused) timer = setTimeout(() => play(scene + 1), ms);
  }

  function setPaused(p) {
    paused = p;
    reel.classList.toggle("paused", p);
    pauseBtn.textContent = p ? "Play" : "Pause";
    pauseBtn.setAttribute("aria-label", p ? "Play" : "Pause");
    if (p) {
      clearTimeout(timer);
      left -= performance.now() - started;
    } else {
      started = performance.now();
      timer = setTimeout(() => play(scene + 1), Math.max(left, 0));
    }
  }

  document.querySelectorAll("[data-reel]").forEach((a) =>
    a.addEventListener("click", (e) => {
      e.preventDefault();
      scene = -1;
      setPaused(false);
      reel.showModal();
      play(0);
    })
  );
  pauseBtn.addEventListener("click", () => setPaused(!paused));
  reel.querySelector(".reel-close").addEventListener("click", () => reel.close());
  reel.addEventListener("close", () => {
    clearTimeout(timer);
    reelImgs.forEach((im) => im.classList.remove("on"));
  });
  reel.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") play(scene + 1);
    if (e.key === "ArrowLeft") play(scene - 1);
    if (e.key === " ") {
      e.preventDefault();
      setPaused(!paused);
    }
  });
  reel.addEventListener("click", (e) => {
    if (e.target.closest("button")) return;
    // Tap the right third to skip on, the left third to go back.
    const x = e.clientX / innerWidth;
    if (x > 0.66) play(scene + 1);
    else if (x < 0.33) play(scene - 1);
    else setPaused(!paused);
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
