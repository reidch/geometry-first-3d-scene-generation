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

The project uses two Python 3.10 Conda environments:

- `world_pipeline` for Stage 00–08.
- `worldmesh-nerfstudio` for Stage 09 reconstruction, rendering and evaluation.

Both environments can be created automatically from the project root:

```bash
bash setup_project_envs.sh
```

The installer reproduces the package versions used for the dissertation experiments, installs the required CUDA extensions, and installs the WorldMesh-compatible Nerfstudio implementation from the recorded source revision.

The experiments were developed and tested on:

- Ubuntu 24.04
- NVIDIA RTX 5080 16 GB
- CUDA 12.8

### CUDA compilation settings

Native CUDA extensions are compiled only for the requested GPU architecture rather than for multiple architectures.

The default settings target the RTX 5080:

```bash
CUDA_ARCH=12.0 BUILD_JOBS=4 bash setup_project_envs.sh
```

This corresponds to compute capability 12.0 / `sm_120`.

For another GPU, specify the corresponding compute capability. For example, an RTX 4090 uses 8.9:

```bash
CUDA_ARCH=8.9 BUILD_JOBS=4 bash setup_project_envs.sh
```

If native compilation is limited by host memory, reduce the number of parallel build jobs:

```bash
BUILD_JOBS=2 bash setup_project_envs.sh
```

The optional `NVCC_THREADS` variable controls the number of host threads used by each NVCC compilation unit. The default is:

```bash
NVCC_THREADS=1
```

For example:

```bash
CUDA_ARCH=12.0 BUILD_JOBS=4 NVCC_THREADS=1 bash setup_project_envs.sh
```

### WSL 2 CUDA driver linking

The installer automatically searches for the CUDA driver library required by native extensions that link against `-lcuda`.

On WSL 2, the NVIDIA driver library is normally exposed through the Windows NVIDIA driver under `/usr/lib/wsl/lib`. If only `libcuda.so.1` is available, the installer creates a temporary build-only `libcuda.so` symlink and adds the appropriate linker path for compilation.

This does not modify the system NVIDIA driver or install a separate Linux display driver.

### Rebuilding the environments

If the environments already exist, the installer reuses them and installs any missing components. This allows an interrupted installation to be resumed by simply running:

```bash
bash setup_project_envs.sh
```

To remove and rebuild both environments from scratch:

```bash
RECREATE_ENVS=1 bash setup_project_envs.sh
```

### Model weights

The installer prepares the runtime environments only. It does not download pretrained model weights.

Required pretrained models, including FLUX.1, FLUX.2, Pixal3D and the required depth models, are downloaded automatically by the corresponding pipeline stages when first needed. The first pipeline run therefore requires Internet access and sufficient disk space.

Stage 09E image-quality evaluation uses its own isolated lightweight runtime, created automatically by the evaluation pipeline when required. It is not a third project Conda environment.

## Quick Start

After installation, activate the main pipeline environment:

```bash
conda activate world_pipeline
```

To run the scene stored in `data/scenes/current/`:

```bash
bash run_pipeline.sh
```

Alternatively, a scene JSON can be selected explicitly without modifying the project configuration:

```bash
SCENE_JSON=data/scenes/examples/<scene>.json bash run_pipeline.sh
```

Outputs are written automatically under:

```text
outputs/
```

When Stage 09 is reached, the pipeline uses the `worldmesh-nerfstudio` environment automatically.

## Scene Description

Each scene is defined by a structured JSON file describing its layout, object hierarchy, scaffold geometry, generation modes and appearance prompts.

Scene JSON files can be written manually. Examples under `data/scenes/examples/` can be copied and modified as templates.

For faster authoring, an example JSON can also be provided to a large language model together with a natural-language description of the desired room. For example:

> "Create a modern Japanese bedroom with a bed, desk, bookshelf, floor lamp and wall decorations, and construct my scene following this JSON template."

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
