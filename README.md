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

## System Environment

The complete system used for the dissertation experiments was developed and tested in the following environment:

- Windows Subsystem for Linux (WSL 2)
- Ubuntu 24.04
- CUDA 12.8
- Python 3.10
- Blender 4.5
- NVIDIA RTX 5080 16 GB

A similar system-level configuration is recommended for reproduction.

## Installation

The project uses two Python 3.10 Conda environments:

- `world_pipeline` for Stage 00–08.
- `worldmesh-nerfstudio` for Stage 09 reconstruction, rendering and evaluation.

Both environments can be created automatically from the project root:

```bash
bash setup_project_envs.sh
```

The installer reproduces the package versions used for the dissertation experiments, installs the required CUDA extensions, and installs the WorldMesh-compatible Nerfstudio implementation from the recorded source revision.

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

When Stage 08 downloads FLUX.2 for the first time, the download process may not display a visible progress bar for part of the download. In this case, the process is not necessarily stalled. If the terminal appears inactive, it is recommended to monitor the host system's network activity to confirm that model files are still being transferred before interrupting the process.

Stage 09E image-quality evaluation uses its own isolated lightweight runtime, created automatically by the evaluation pipeline when required. It is not a third project Conda environment.

## Reproducibility

A clean end-to-end reproduction test was carried out using the public repository instructions.

For this test, the existing `world_pipeline` and `worldmesh-nerfstudio` environments were removed and recreated from scratch using `setup_project_envs.sh`. The Rococo scene specification `grand_rococo_suite.json` was then processed through the complete production pipeline following the same public instructions provided in this repository.

The reproduction test used:

- WSL 2
- Ubuntu 24.04
- CUDA 12.8
- Python 3.10
- Blender 4.5
- NVIDIA RTX 5080 16 GB
- Scene: `grand_rococo_suite.json`

The complete pipeline executed successfully and reproduced the Rococo scene without relying on the previously existing development environments.

This test verifies that the released environment installer, scene specification and production pipeline can be used together from a clean setup on the documented system configuration.

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

### Running stages manually

On some systems, `run_pipeline.sh` may fail to launch one of the stage scripts because the copied or cloned shell script does not have executable permission. If this happens, the recommended fallback is to execute the stages manually in order from the project root.

For example:

```bash
bash run_stage00.sh
bash run_stage01.sh
bash run_stage02.sh
```

Continue with the remaining stages in numerical order.

If a stage is split into scripts such as:

```text
run_stage06a.sh
run_stage06b.sh
```

the letters indicate ordered sub-stages. Run `06a` before `06b`.

These sub-stage splits are generally used to preserve reusable intermediate results. This avoids repeating an expensive earlier computation when only parameters in a later part of the same stage are changed.

If preferred, executable permission can also be restored explicitly:

```bash
chmod +x run_pipeline.sh run_stage*.sh
```

## Scene Description

Each scene is defined by a structured JSON file describing its layout, object hierarchy, scaffold geometry, generation modes and appearance prompts.

Scene JSON files can be written manually. Examples under `data/scenes/examples/` can be copied and modified as templates.

For LLM-assisted scene authoring, the Rococo scene JSON `grand_rococo_suite.json` is recommended as the reference template because it provides a complete example of the scene schema and scaffold organization.

Provide `grand_rococo_suite.json` to the language model together with a natural-language description of the desired scene. For example:

> "I need a warm and cozy library room, with a row of bookshelves along one wall, and a desk and chair placed at the other end of the room. There should also be a chandelier hanging from the ceiling. Please use the provided JSON file as a structural reference and describe the scaffold for this scene using the same JSON schema and organization."

The generated JSON can then be reviewed and edited if needed before being passed directly to the pipeline using `SCENE_JSON`.

Stage 00 validates the scene description before the remaining stages are executed.

## Citation

```bibtex
@mastersthesis{chen2026geometryfirst,
  title  = {A Fully Automated Geometry-First Framework for Controllable Indoor 3D Scene Generation},
  author = {Chen, Rundong},
  school = {Trinity College Dublin},
  year   = {2026}
}
```
