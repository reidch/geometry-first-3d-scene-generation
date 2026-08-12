(() => {
  const cfg = window.PROJECT_CONFIG || { links: {}, media: {} };

  function setLink(id, url) {
    const el = document.getElementById(id);
    if (!el) return;
    if (url) {
      el.href = url;
      el.classList.remove("disabled");
      if (url.startsWith("#")) {
        el.removeAttribute("target");
        el.removeAttribute("rel");
      }
    } else {
      el.href = "#";
      el.addEventListener("click", (e) => e.preventDefault());
      el.title = "Add this URL in config.js";
      el.style.opacity = "0.55";
    }
  }

  function setImage(id, src) {
    if (!src) return;
    const el = document.getElementById(id);
    if (el) el.src = src;
  }

  function setVideo(videoId, fallbackId, src) {
    if (!src) return;
    const video = document.getElementById(videoId);
    const fallback = fallbackId ? document.getElementById(fallbackId) : null;
    if (!video) return;
    const source = document.createElement("source");
    source.src = src;
    source.type = src.toLowerCase().endsWith(".webm") ? "video/webm" : "video/mp4";
    video.appendChild(source);
    video.hidden = false;
    if (fallback) fallback.hidden = true;
    video.load();
  }

  setLink("paper-link", cfg.links?.paper);
  setLink("code-link", cfg.links?.code);
  setLink("video-link", cfg.links?.video);

  if (cfg.media?.heroVideo) {
    setVideo("hero-video", "hero-image", cfg.media.heroVideo);
    if (cfg.media?.heroImage) {
      const heroVideo = document.getElementById("hero-video");
      if (heroVideo) heroVideo.poster = cfg.media.heroImage;
    }
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
  setImage("matched-rococo", cfg.media?.matchedRococo);
  setImage("matched-modern", cfg.media?.matchedModern);
  setImage("ablation-multiview", cfg.media?.ablationMultiview);
  setImage("ablation-reprojection", cfg.media?.ablationReprojection);
  setImage("ablation-repair", cfg.media?.ablationRepair);
  setImage("ablation-camera", cfg.media?.ablationCamera);

  setVideo("video-rococo", "video-rococo-fallback", cfg.media?.videoRococo);
  setVideo("video-modern", "video-modern-fallback", cfg.media?.videoModern);
  setVideo("video-anime", "video-anime-fallback", cfg.media?.videoAnime);

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
