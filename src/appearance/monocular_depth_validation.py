from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
from PIL import Image, ImageFilter

try:
    import cv2
except Exception:  # pragma: no cover - validated at runtime when Stage08 depth gate is enabled
    cv2 = None

from src.cameras.reconstruction_view_metrics import decode_metric_depth
from src.cameras.depth_geometry import depth_convention


_EPS = 1e-8


class DepthAnythingV2Validator:
    """Lazy reusable Transformers Depth Anything V2 inference wrapper."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        self.checkpoint = str(
            self.config.get("checkpoint", "depth-anything/Depth-Anything-V2-Small-hf")
        )
        self.device_name = str(self.config.get("device", "cuda"))
        self.precision = str(self.config.get("precision", "float16"))
        self.local_files_only = bool(self.config.get("local_files_only", False))
        self.processor = None
        self.model = None
        self.device = None
        self.torch_dtype = None

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            raise RuntimeError(
                "Depth Anything V2 validation requires transformers with "
                "AutoModelForDepthEstimation support"
            ) from exc
        if self.device_name.startswith("cuda") and not torch.cuda.is_available():
            if bool(self.config.get("allow_cpu_fallback", True)):
                self.device_name = "cpu"
            else:
                raise RuntimeError("CUDA was requested for monocular depth validation but is unavailable")
        self.device = torch.device(self.device_name)
        if self.device.type == "cuda" and self.precision == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif self.device.type == "cuda" and self.precision == "float16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32
        self.processor = AutoImageProcessor.from_pretrained(
            self.checkpoint,
            local_files_only=self.local_files_only,
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            self.checkpoint,
            torch_dtype=self.torch_dtype,
            local_files_only=self.local_files_only,
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image | str | Path) -> np.ndarray:
        self._load()
        import torch
        import torch.nn.functional as F

        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")
        width, height = image.size
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(
                    device_type="cuda",
                    dtype=self.torch_dtype,
                    enabled=self.torch_dtype in {torch.float16, torch.bfloat16},
                ):
                    prediction = self.model(**inputs).predicted_depth
            else:
                prediction = self.model(**inputs).predicted_depth
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        depth = prediction.float().cpu().numpy().astype(np.float32)
        if not np.isfinite(depth).all():
            raise RuntimeError("Depth Anything V2 returned NaN or Inf")
        return depth

    def runtime_metadata(self) -> Dict[str, Any]:
        return {
            "backend": "transformers_depth_anything_v2",
            "checkpoint": self.checkpoint,
            "device": self.device_name,
            "precision": self.precision,
            "local_files_only": self.local_files_only,
            "model_reused_across_views": True,
        }


def robust_normalize(values: np.ndarray, valid: np.ndarray, lower: float, upper: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    result = np.zeros_like(values, dtype=np.float32)
    selected = values[valid]
    if selected.size == 0:
        return result
    lo, hi = np.quantile(selected, [float(lower), float(upper)])
    scale = max(float(hi - lo), _EPS)
    result[valid] = np.clip((values[valid] - lo) / scale, 0.0, 1.0)
    return result


def depth_edge_map(
    depth: np.ndarray,
    valid: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, Dict[str, Any]]:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    lower = float(config.get("normalization_lower_quantile", 0.02))
    upper = float(config.get("normalization_upper_quantile", 0.98))
    normalized = robust_normalize(depth, valid, lower, upper)
    gx = np.zeros_like(normalized)
    gy = np.zeros_like(normalized)
    gx[:, 1:-1] = 0.5 * (normalized[:, 2:] - normalized[:, :-2])
    gy[1:-1, :] = 0.5 * (normalized[2:, :] - normalized[:-2, :])
    gradient = np.sqrt(gx * gx + gy * gy)
    gradient[~valid] = 0.0
    values = gradient[valid]
    quantile = float(config.get("edge_gradient_quantile", 0.88))
    threshold = float(np.quantile(values, quantile)) if values.size else float("inf")
    minimum = float(config.get("minimum_normalized_gradient", 0.015))
    threshold = max(threshold, minimum)
    edge = valid & (gradient >= threshold)
    return edge, {
        "gradient_threshold": threshold,
        "edge_pixel_count": int(edge.sum()),
        "valid_pixel_count": int(valid.sum()),
        "edge_fraction_of_valid": float(edge.sum() / max(int(valid.sum()), 1)),
    }


def _require_cv2():
    if cv2 is None:
        raise RuntimeError(
            "WorldMesh-style Stage08 depth-edge validation requires OpenCV. "
            "Install opencv-python-headless in the diffusion environment."
        )
    return cv2


def normalize_depth_for_canny(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """WorldMesh-style linear depth normalization for Canny edge extraction.

    The direction of a relative monocular depth scale does not affect edge
    locations, so Depth Anything V2 can be used here without metric alignment.
    """
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(depth)
    result = np.zeros(depth.shape, dtype=np.uint8)
    selected = depth[valid]
    if selected.size == 0:
        return result
    d_min = float(selected.min())
    d_max = float(selected.max())
    if d_max - d_min < _EPS:
        return result
    normalized = 1.0 - (depth[valid] - d_min) / (d_max - d_min)
    result[valid] = np.clip(np.rint(255.0 * normalized), 0.0, 255.0).astype(np.uint8)
    return result


def sharpen_depth_for_canny(depth_normalized: np.ndarray, strength: float) -> np.ndarray:
    """Match WorldMesh's bilateral + Laplacian + unsharp depth enhancement."""
    cv = _require_cv2()
    strength = float(strength)
    if strength <= 1.0:
        return np.asarray(depth_normalized, dtype=np.uint8)
    depth_float = np.asarray(depth_normalized, dtype=np.float32)
    smoothed = cv.bilateralFilter(depth_float, d=5, sigmaColor=20, sigmaSpace=5)
    laplacian = cv.Laplacian(smoothed, cv.CV_32F, ksize=3)
    sharpened = depth_float - (strength - 1.0) * laplacian
    blurred = cv.GaussianBlur(depth_float, (0, 0), sigmaX=1.5)
    unsharp = depth_float + (strength - 1.0) * 0.5 * (depth_float - blurred)
    return np.clip((sharpened + unsharp) / 2.0, 0.0, 255.0).astype(np.uint8)


def extract_canny_depth_edges(
    normalized_depth: np.ndarray, low_threshold: float, high_threshold: float
) -> np.ndarray:
    cv = _require_cv2()
    low = int(round(float(low_threshold) * 255.0))
    high = int(round(float(high_threshold) * 255.0))
    return cv.Canny(np.asarray(normalized_depth, dtype=np.uint8), low, high) > 127


def depth_gradient_magnitude(depth: np.ndarray) -> np.ndarray:
    """Scharr magnitude on log depth, normalized exactly as in WorldMesh."""
    cv = _require_cv2()
    safe = np.log(np.clip(np.asarray(depth, dtype=np.float32), 1e-4, 1e4)).astype(np.float32)
    gx = cv.Scharr(safe, cv.CV_32F, 1, 0)
    gy = cv.Scharr(safe, cv.CV_32F, 0, 1)
    magnitude = np.sqrt(gx * gx + gy * gy)
    maximum = float(magnitude.max()) if magnitude.size else 0.0
    if maximum > 0.0:
        magnitude = magnitude / maximum
    return magnitude.astype(np.float32)


def filter_edges_by_gradient(
    edges: np.ndarray, gradient: np.ndarray, percentile: float
) -> tuple[np.ndarray, float | None]:
    edges = np.asarray(edges, dtype=bool)
    if not edges.any() or float(percentile) <= 0.0:
        return edges, None
    values = np.asarray(gradient, dtype=np.float32)[edges]
    threshold = float(np.percentile(values, float(percentile)))
    return edges & (np.asarray(gradient, dtype=np.float32) >= threshold), threshold


def worldmesh_style_depth_edges(
    predicted: np.ndarray,
    mesh: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """Extract asymmetric filtered depth edges using WorldMesh's published code path.

    The only intentional backend substitution is Depth Anything V2 for Depth Pro.
    """
    cv = _require_cv2()
    predicted = np.asarray(predicted, dtype=np.float32)
    mesh = np.asarray(mesh, dtype=np.float32)
    if predicted.shape != mesh.shape:
        predicted = cv.resize(
            predicted, (mesh.shape[1], mesh.shape[0]), interpolation=cv.INTER_LINEAR
        ).astype(np.float32)

    min_valid_depth = float(config.get("minimum_valid_mesh_depth", 0.1))
    valid_mesh = np.isfinite(mesh) & (mesh >= min_valid_depth)
    valid_predicted = np.isfinite(predicted) & (predicted > 0.0)

    predicted_norm = normalize_depth_for_canny(predicted, valid_predicted)
    mesh_norm = normalize_depth_for_canny(mesh, valid_mesh)
    sharpen_predicted = float(config.get("sharpen_predicted", 4.0))
    sharpen_mesh = float(config.get("sharpen_mesh", 3.0))
    predicted_norm = sharpen_depth_for_canny(predicted_norm, sharpen_predicted)
    mesh_norm = sharpen_depth_for_canny(mesh_norm, sharpen_mesh)

    canny_low = float(config.get("canny_low", 0.1))
    canny_high = float(config.get("canny_high", 0.3))
    predicted_edges = extract_canny_depth_edges(predicted_norm, canny_low, canny_high)
    mesh_edges = extract_canny_depth_edges(mesh_norm, canny_low, canny_high)

    percentile = float(config.get("minimum_gradient_percentile", 25.0))
    mesh_gradient = depth_gradient_magnitude(mesh)
    predicted_gradient = depth_gradient_magnitude(predicted)
    mesh_edges, mesh_gradient_threshold = filter_edges_by_gradient(
        mesh_edges, mesh_gradient, percentile
    )
    combined_gradient = np.maximum(mesh_gradient, predicted_gradient)
    predicted_edges, predicted_gradient_threshold = filter_edges_by_gradient(
        predicted_edges, combined_gradient, percentile
    )

    mesh_meta = {
        "edge_pixel_count": int(mesh_edges.sum()),
        "valid_pixel_count": int(valid_mesh.sum()),
        "canny_low": canny_low,
        "canny_high": canny_high,
        "sharpen_strength": sharpen_mesh,
        "minimum_gradient_percentile": percentile,
        "gradient_threshold": mesh_gradient_threshold,
    }
    predicted_meta = {
        "edge_pixel_count": int(predicted_edges.sum()),
        "valid_pixel_count": int(valid_predicted.sum()),
        "canny_low": canny_low,
        "canny_high": canny_high,
        "sharpen_strength": sharpen_predicted,
        "minimum_gradient_percentile": percentile,
        "gradient_threshold": predicted_gradient_threshold,
        "gradient_filter": "max(mesh_gradient,predicted_gradient)",
    }
    return predicted, mesh_edges, predicted_edges, valid_mesh, mesh_meta, predicted_meta


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(int(radius), 0)
    if radius == 0:
        return np.asarray(mask, dtype=bool)
    image = Image.fromarray((np.asarray(mask, dtype=np.uint8) * 255), mode="L")
    return np.asarray(image.filter(ImageFilter.MaxFilter(2 * radius + 1)), dtype=np.uint8) > 0


def bidirectional_edge_consistency(
    mesh_edges: np.ndarray,
    predicted_edges: np.ndarray,
    radius: int,
    minimum_recall: float,
    minimum_precision: float | None = None,
) -> Dict[str, Any]:
    """Measure mesh->predicted edge recall; keep reverse precision diagnostic-only.

    Stage08 acceptance is intentionally one-way: required mesh depth edges must be
    present in the generated image's predicted depth, while extra predicted edges
    do not reject an otherwise valid image.  Reverse precision is still reported
    for diagnostics so historical artifacts remain easy to compare.
    """
    mesh_edges = np.asarray(mesh_edges, dtype=bool)
    predicted_edges = np.asarray(predicted_edges, dtype=bool)
    dilated_predicted = _dilate(predicted_edges, radius)
    dilated_mesh = _dilate(mesh_edges, radius)
    matched_mesh = mesh_edges & dilated_predicted
    matched_predicted = predicted_edges & dilated_mesh
    extra_predicted = predicted_edges & ~dilated_mesh

    mesh_edge_count = int(mesh_edges.sum())
    predicted_edge_count = int(predicted_edges.sum())
    recall = 1.0 if mesh_edge_count == 0 else float(matched_mesh.sum() / mesh_edge_count)
    precision = (
        1.0
        if predicted_edge_count == 0
        else float(matched_predicted.sum() / predicted_edge_count)
    )
    recall_accepted = bool(recall >= float(minimum_recall))
    return {
        "accepted": recall_accepted,
        "depth_edge_recall": recall,
        "depth_edge_recall_accepted": recall_accepted,
        "predicted_depth_edge_precision": precision,
        "predicted_depth_edge_precision_accepted": None,
        "predicted_depth_edge_precision_diagnostic_only": True,
        "extra_predicted_depth_edge_fraction": float(1.0 - precision),
        "matched_mesh_edge_pixel_count": int(matched_mesh.sum()),
        "matched_predicted_edge_pixel_count": int(matched_predicted.sum()),
        "extra_predicted_edge_pixel_count": int(extra_predicted.sum()),
        "predicted_extra_edge": extra_predicted,
    }


def validate_depth_structure(
    generated_rgb: Image.Image | str | Path,
    frame: Mapping[str, Any],
    predictor: DepthAnythingV2Validator,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """WorldMesh-style one-way edge-recall gate using Depth Anything V2.

    Depth Anything is run only on the generated Stage08 candidate RGB. Stage07
    metric mesh depth remains the ground-truth structural reference.
    """
    predicted = predictor.predict(generated_rgb)
    if depth_convention(frame["depth_encoding"]) != "camera_z":
        raise ValueError("Stage08 structural validation requires camera-Z mesh depth")
    mesh = decode_metric_depth(frame["depth"], frame["depth_encoding"])
    (
        predicted,
        mesh_edges,
        predicted_edges,
        valid_mesh,
        mesh_meta,
        predicted_meta,
    ) = worldmesh_style_depth_edges(predicted, mesh, config)

    radius = max(int(config.get("dilation_pixels", 10)), 0)
    minimum_recall = float(config.get("minimum_depth_edge_recall", 0.50))

    # Acceptance is strictly one-way: only required mesh edges missing from the
    # generated image reduce the gate score. Reverse precision remains diagnostic.
    consistency = bidirectional_edge_consistency(
        mesh_edges & valid_mesh,
        predicted_edges,
        radius,
        minimum_recall,
    )
    return {
        **consistency,
        "minimum_depth_edge_recall": minimum_recall,
        "minimum_predicted_depth_edge_precision": None,
        "edge_match_radius_pixels": radius,
        "mesh_edge": mesh_edges & valid_mesh,
        "predicted_edge": predicted_edges,
        "predicted_depth": predicted,
        "mesh_edge_metadata": mesh_meta,
        "predicted_edge_metadata": predicted_meta,
        "comparison": (
            "worldmesh_style_one_way_mesh_to_predicted_depth_edge_recall_"
            "with_depth_anything_v2"
        ),
        "mesh_depth_convention": "camera_z",
        "depth_estimator_backend": "transformers_depth_anything_v2",
        "worldmesh_backend_substitution": "Depth Anything V2 replaces Depth Pro; edge gate is otherwise WorldMesh-style",
    }


def reference_overlap_error(
    generated_rgb: Image.Image | str | Path,
    warped_rgb: np.ndarray,
    valid_mask: np.ndarray,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    generated = np.asarray(Image.open(generated_rgb).convert("RGB"), dtype=np.float32) / 255.0
    warped = np.asarray(warped_rgb, dtype=np.float32) / 255.0
    mask = np.asarray(valid_mask, dtype=bool)
    if warped.shape[:2] != generated.shape[:2]:
        warped = np.asarray(
            Image.fromarray(np.rint(np.clip(warped, 0.0, 1.0) * 255.0).astype(np.uint8)).resize(
                (generated.shape[1], generated.shape[0]), Image.Resampling.LANCZOS
            ),
            dtype=np.float32,
        ) / 255.0
        mask = np.asarray(
            Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(
                (generated.shape[1], generated.shape[0]), Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        ) > 0
    pixel_count = int(mask.sum())
    if pixel_count == 0:
        return {
            "accepted": True,
            "reference_overlap_rgb_l1": 0.0,
            "maximum_reference_overlap_rgb_l1": float(config.get("maximum_reference_overlap_rgb_l1", 0.10)),
            "reference_pixel_count": 0,
            "reference_overlap_available": False,
        }
    error = float(np.mean(np.abs(generated[mask] - warped[mask])))
    maximum = float(config.get("maximum_reference_overlap_rgb_l1", 0.10))
    return {
        "accepted": error <= maximum,
        "reference_overlap_rgb_l1": error,
        "maximum_reference_overlap_rgb_l1": maximum,
        "reference_pixel_count": pixel_count,
        "reference_overlap_available": True,
    }


def image_validity(image: Image.Image | str | Path, config: Mapping[str, Any]) -> Dict[str, Any]:
    array = np.asarray(Image.open(image).convert("RGB"), dtype=np.float32) / 255.0
    finite = bool(np.isfinite(array).all())
    standard_deviation = float(array.std())
    minimum_std = float(config.get("minimum_rgb_standard_deviation", 0.015))
    accepted = finite and standard_deviation >= minimum_std
    return {
        "accepted": accepted,
        "finite": finite,
        "rgb_standard_deviation": standard_deviation,
        "minimum_rgb_standard_deviation": minimum_std,
    }


def save_validation_images(
    directory: str | Path,
    depth_result: Mapping[str, Any],
    overlap_mask: np.ndarray | None,
    overlap_difference: np.ndarray | None,
) -> Dict[str, str]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, str] = {}
    predicted = np.asarray(depth_result["predicted_depth"], dtype=np.float32)
    valid = np.isfinite(predicted)
    normalized = robust_normalize(predicted, valid, 0.02, 0.98)
    predicted_path = directory / "predicted_depth.png"
    Image.fromarray(np.rint(normalized * 65535.0).astype(np.uint16), mode="I;16").save(predicted_path)
    outputs["predicted_depth"] = str(predicted_path)
    for key, source in (
        ("mesh_depth_edges", depth_result["mesh_edge"]),
        ("predicted_depth_edges", depth_result["predicted_edge"]),
        ("predicted_extra_depth_edges", depth_result.get("predicted_extra_edge", np.zeros_like(depth_result["predicted_edge"], dtype=bool))),
    ):
        path = directory / f"{key}.png"
        Image.fromarray((np.asarray(source, dtype=np.uint8) * 255), mode="L").save(path)
        outputs[key] = str(path)
    if overlap_mask is not None:
        path = directory / "reference_valid_mask.png"
        Image.fromarray((np.asarray(overlap_mask, dtype=np.uint8) * 255), mode="L").save(path)
        outputs["reference_valid_mask"] = str(path)
    if overlap_difference is not None:
        path = directory / "reference_overlap_difference.png"
        Image.fromarray(np.rint(np.clip(overlap_difference, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").save(path)
        outputs["reference_overlap_difference"] = str(path)
    return outputs
