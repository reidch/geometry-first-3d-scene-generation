# A Fully Automated Geometry-First Framework for Controllable Indoor 3D Scene Generation

> Replace this line with one strong teaser image from `docs/assets/`.

**Project Page:** `https://YOUR_USERNAME.github.io/YOUR_REPOSITORY/`  
**Paper:** `UPDATE_LINK`  
**Video:** `UPDATE_LINK`

## Overview

This repository contains the implementation of a fully automated geometry-first framework for controllable indoor 3D scene generation. The system treats explicitly supplied scene geometry as persistent spatial authority while pretrained generative components add object detail, architectural appearance and final scene-level visual style.

The pipeline covers declarative scene compilation, scaffold construction, generated asset reintegration, architectural surface synthesis, physically valid camera planning, graph-guided multi-view generation, structural validation and repair, and depth-supervised 3D Gaussian Splatting reconstruction.

## Highlights

- **Declarative control:** scene hierarchy, transforms, scaffolds, generation routes and appearance controls are defined in structured scene data.
- **Geometry-first coordination:** known geometry drives asset alignment, camera feasibility, room-shell coverage, target-space reprojection and depth-supervised reconstruction.
- **Multi-view consistency:** trusted neighbouring views are reprojected into the target camera and robustly fused before final-view generation.
- **Automatic reliability:** bounded retries, fallbacks, trust state, repair and resumable stages avoid per-scene manual correction.

## Project Page

The full visual overview, videos, quantitative results and ablations are available under `docs/` and can be published directly with GitHub Pages.

## Installation

Add your actual environment and dependency instructions here.

## Quick Start

Add the minimal commands required to run one complete example here.

## Scene Description

Add a small JSON example here showing the main controllable fields without pasting a full production scene.

## Citation

```bibtex
@mastersthesis{chen2026geometryfirst,
  title  = {A Fully Automated Geometry-First Framework for Controllable Indoor 3D Scene Generation},
  author = {Chen, Rundong},
  school = {Trinity College Dublin},
  year   = {2026}
}
```
