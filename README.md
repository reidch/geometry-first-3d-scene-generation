# A Fully Automated Geometry-First Framework for Controllable Indoor 3D Scene Generation

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

The project uses two Python 3.10 environments:

- `world_pipeline` for Stage 00–08.
- `worldmesh-nerfstudio` for Stage 09 reconstruction, rendering and evaluation.

Both environments can be created automatically from the project root:

```bash
bash setup_project_envs.sh
```

The experiments were primarily developed and tested on Ubuntu 24.04 with an NVIDIA RTX 5080 (16 GB) and CUDA 12.8. A similar system-level configuration is recommended.

Pretrained models, including FLUX.1, FLUX.2, Pixal3D and the required depth models, are downloaded automatically by the corresponding stages when they are first needed. The first run therefore requires Internet access and sufficient disk space.

## Quick Start

Activate the main pipeline environment:

```bash
conda activate world_pipeline
```

To run the scene stored in `data/scenes/current/`:

```bash
bash run_pipeline.sh
```

Alternatively, a scene JSON can be selected explicitly:

```bash
SCENE_JSON=data/scenes/examples/<scene>.json bash run_pipeline.sh
```

Outputs are written automatically under `outputs/`. Stage 09 uses the `worldmesh-nerfstudio` environment automatically.

## Scene Description

Each scene is defined by a structured JSON file describing its layout, object hierarchy, scaffold geometry, generation modes and appearance prompts.

Scene JSON files can be written manually. Examples under `data/scenes/examples/` can be copied and modified as templates.

For faster authoring, an example JSON can also be provided to a large language model together with a prompt such as:

> "[Describe the desired room type, layout and visual style], and construct my scene following this JSON template."

The generated JSON can then be edited if needed and passed directly to the pipeline using `SCENE_JSON`. Stage 00 validates the scene description before the remaining stages are executed.

## Citation

```bibtex
@mastersthesis{chen2026geometryfirst,
  title  = {A Fully Automated Geometry-First Framework for Controllable Indoor 3D Scene Generation},
  author = {Chen, Rundong},
  school = {Trinity College Dublin},
  year   = {2026}
}
```
