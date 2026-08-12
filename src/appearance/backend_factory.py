from __future__ import annotations


def create_backend(name, config, auth_config=None):
    if name == "flux1_depth_control_inpaint_nf4_16gb":
        from src.appearance.backends.flux1_depth_control_inpaint_backend import (
            Flux1DepthControlInpaintNF416GBBackend,
        )
        return Flux1DepthControlInpaintNF416GBBackend(config, auth_config=auth_config)
    if name == "flux2_klein_4b_multiref_16gb":
        from src.appearance.backends.flux2_klein_multiref_backend import (
            Flux2Klein4BMultiReferenceBackend,
        )
        return Flux2Klein4BMultiReferenceBackend(config, auth_config=auth_config)
    raise ValueError(
        "Unsupported diffusion backend: " + str(name) +
        ". Supported production backends are FLUX.2 [klein] 4B multi-reference and FLUX.1-Depth-dev NF4 compatibility mode."
    )
