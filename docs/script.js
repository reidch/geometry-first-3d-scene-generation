(() => {
  const cfg = window.PROJECT_CONFIG || { links: {}, media: {} };

  function setLink(id, url) {
    const el = document.getElementById(id);
    if (!el || !url) return;
    el.href = url;
    el.classList.remove("disabled");
    el.style.opacity = "";
    if (url.startsWith("#")) {
      el.removeAttribute("target");
      el.removeAttribute("rel");
    }
  }

  function setImage(id, src) {
    if (!src) return;
    const el = document.getElementById(id);
    if (el && el.tagName === "IMG") el.src = src;
  }

  function setVideo(videoId, fallbackId, src, poster) {
    if (!src) return;
    const video = document.getElementById(videoId);
    const fallback = fallbackId ? document.getElementById(fallbackId) : null;
    if (!video) return;
    video.innerHTML = "";
    const source = document.createElement("source");
    source.src = src;
    source.type = src.toLowerCase().endsWith(".webm") ? "video/webm" : "video/mp4";
    video.appendChild(source);
    if (poster) video.poster = poster;
    video.hidden = false;
    if (fallback) fallback.hidden = true;
    video.load();
  }

  setLink("paper-link", cfg.links?.paper);
  setLink("code-link", cfg.links?.code);
  setLink("video-link", cfg.links?.video);

  if (cfg.media?.heroVideo) {
    setVideo("hero-video", "hero-image", cfg.media.heroVideo, cfg.media?.heroImage);
    const note = document.querySelector(".hero-media .media-note");
    if (note) note.hidden = true;
  } else if (cfg.media?.heroImage) {
    setImage("hero-image", cfg.media.heroImage);
    const note = document.querySelector(".hero-media .media-note");
    if (note) note.hidden = true;
  }

  setImage("scene-rococo", cfg.media?.sceneRococo);
  setImage("scene-modern", cfg.media?.sceneModern);
  setImage("scene-anime", cfg.media?.sceneAnime);
  setImage("pipeline-image", cfg.media?.pipeline);
  setImage("style-control", cfg.media?.styleControl);
  setImage("ablation-multiview", cfg.media?.ablationMultiview);
  setImage("ablation-repair", cfg.media?.ablationRepair);

  setVideo("video-rococo", "video-rococo-fallback", cfg.media?.videoRococo, cfg.media?.sceneRococo);
  setVideo("video-modern", "video-modern-fallback", cfg.media?.videoModern, cfg.media?.sceneModern);
  setVideo("video-anime", "video-anime-fallback", cfg.media?.videoAnime, cfg.media?.sceneAnime);

  const copy = document.getElementById("copy-bibtex");
  copy?.addEventListener("click", async () => {
    const text = document.getElementById("bibtex")?.innerText || "";
    try {
      await navigator.clipboard.writeText(text);
      const old = copy.textContent;
      copy.textContent = "Copied";
      setTimeout(() => (copy.textContent = old), 1400);
    } catch {
      copy.textContent = "Select & copy";
    }
  });
})();
