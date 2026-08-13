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

The project uses two Python 3.10 environments: `world_pipeline` for Stage 00–08 and `world_nerf` for Stage 09 reconstruction, rendering and evaluation. See `environment/README.md` for details, or create both environments automatically with:

```bash
bash tools/setup_python_envs.sh
```

The main experiments were developed and tested on Ubuntu 24.04 with an NVIDIA RTX 5080 (16 GB) and CUDA 12.8. A similar system-level configuration is recommended.

Pretrained models, including FLUX.1, FLUX.2 and Pixal3D, are downloaded automatically by the corresponding stages when they are not already available locally. The first run therefore requires Internet access and sufficient disk space.

## Quick Start

After creating the environments, run the pipeline from the project root:

```bash
conda activate world_pipeline
bash run_pipeline.sh <scene-json>
```

Stage 09 uses the `world_nerf` environment automatically.

## Scene Description

Each scene is defined by a structured JSON file describing its layout, object hierarchy, scaffold geometry, generation modes and appearance prompts.

Scene JSON files can be written manually. Example scene descriptions are provided under `data/scenes/examples/` and can be used as templates.

For faster authoring, an example JSON can also be provided to a large language model together with a prompt such as:

> "[Describe the desired room type, layout and visual style], and construct my scene following this JSON template."

The generated JSON can then be edited as needed and used as the input to the pipeline. Before execution, the scene description is validated by Stage 00 against the expected schema and structural constraints.

## Citation

```bibtex
@mastersthesis{chen2026geometryfirst,
  title  = {A Fully Automated Geometry-First Framework for Controllable Indoor 3D Scene Generation},
  author = {Chen, Rundong},
  school = {Trinity College Dublin},
  year   = {2026}
}
```
