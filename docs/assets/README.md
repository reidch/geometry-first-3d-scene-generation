# Media replacement guide

You do **not** need to rename your files to the placeholder names.
Put your images/videos in this `assets/` folder and edit only `../config.js`.

Recommended files:

- `hero.mp4` — 16:9 overview walkthrough, H.264, muted-friendly, ~10–25 s.
- `hero.webp` — optional fallback/teaser frame.
- `rococo.webp`, `modern.webp`, `anime.webp` — representative final 3DGS views.
- `pipeline.webp` — export of the dissertation pipeline figure at high resolution.
- `rococo_walkthrough.mp4`, `modern_walkthrough.mp4`, `anime_walkthrough.mp4` — short persistent novel-view sequences.
- `matched_rococo.webp`, `matched_modern.webp` — identical evaluation camera, same layout, different appearance.
- `ablation_multiview.webp` — full vs independent generation.
- `ablation_reprojection.webp` — target depth / full reprojection / raw neighbour comparison.
- `ablation_repair.webp` — depth / before repair / after repair.
- `ablation_camera.webp` — complete resolver / fixed-step comparison.
- `dissertation.pdf` — optional local PDF if you want the Paper button to point to a file in the repository.

## Practical compression

For website media, prefer:
- images: WebP, roughly 1600–2200 px wide;
- video: MP4/H.264, 1080p, moderate bitrate;
- avoid very large GIFs.

Then edit `docs/config.js` and set the relative paths, e.g.

```js
heroVideo: "assets/hero.mp4",
sceneRococo: "assets/rococo.webp",
```
