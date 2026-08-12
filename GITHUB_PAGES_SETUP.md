# GitHub Pages setup

This package is designed to be copied directly into the root of your code repository.

Expected layout:

```text
YOUR_REPOSITORY/
├── README.md
├── src/
├── scripts/
├── ...
└── docs/
    ├── index.html
    ├── style.css
    ├── config.js
    ├── script.js
    └── assets/
```

## 1. Copy the `docs/` folder into your repository

Keep your source code unchanged. The project website is isolated under `docs/`.

## 2. Add your media

Put your images/videos in `docs/assets/`.
Then edit only:

```text
docs/config.js
```

Fill in:
- paper URL;
- code URL;
- video URL;
- hero media;
- scene images;
- trajectory videos;
- matched-style views;
- ablation figures.

## 3. Preview locally

Opening `docs/index.html` directly works for most of the page.
For the most realistic local test, from the repository root run:

```bash
python -m http.server 8000 -d docs
```

Then open:

```text
http://localhost:8000
```

## 4. Publish with GitHub Pages

On GitHub:

1. Open the repository.
2. Go to **Settings** → **Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Select branch `main` (or your default branch).
5. Select folder `/docs`.
6. Save.

The project site will normally appear at:

```text
https://YOUR_USERNAME.github.io/YOUR_REPOSITORY/
```

## 5. Suggested final media order

1. Hero walkthrough video.
2. Rococo / Modern / Anime representative views.
3. Dissertation Figure 1.1 pipeline image.
4. Three short 3DGS trajectory videos.
5. Matched Rococo–Modern camera comparison.
6. Four ablation figures.

## 6. Before publishing

- Replace all placeholder media.
- Fill the three top buttons in `config.js`.
- Check the BibTeX school wording if you want it to match the final institutional repository entry exactly.
- Confirm whether the dissertation PDF is public before hosting it.
- Test on desktop and mobile.
