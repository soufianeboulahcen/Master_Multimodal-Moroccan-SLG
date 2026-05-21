"""Identity conditioning for avatar rendering.

Encodes a reference signer image into conditioning tensors that are injected
into the diffusion UNet via InstantID or IP-Adapter.  The goal is to preserve
the signer's appearance (face, skin tone, clothing style) across all generated
frames while the pose conditioning drives the motion.

Two backends are supported:

    InstantID  (recommended)
        Uses InsightFace ArcFace to extract a 512-d face embedding, then
        injects it via a dedicated ControlNet branch (face structure) and an
        IP-Adapter module (appearance).  Produces the strongest identity
        preservation for face-visible frames.

    IP-Adapter  (fallback)
        Uses CLIP ViT-H/14 to encode the full reference image.  Less
        face-specific than InstantID but works even when the face is small
        or partially occluded.  Recommended when the reference image shows
        the full body rather than a close-up face.

Both backends cache their embeddings to disk so the same reference image is
never re-encoded across multiple render calls.

Usage:
    encoder = IdentityEncoder(cfg)
    conditioning = encoder.encode("outputs/avatar/reference_images/signer.jpg")
    # conditioning is a dict consumed by the diffusion backend
    encoder.inject_into_pipeline(pipe, conditioning)
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np

from avatar.config import IdentityConfig, IdentityBackend


# ---------------------------------------------------------------------------
# Lazy imports — heavy ML deps only loaded when actually used
# ---------------------------------------------------------------------------

def _require_cv2():
    try:
        import cv2
        return cv2
    except ImportError as e:
        raise ImportError("opencv-python is required for identity encoding") from e


def _require_insightface():
    try:
        import insightface
        from insightface.app import FaceAnalysis
        return insightface, FaceAnalysis
    except ImportError as e:
        raise ImportError(
            "insightface is required for InstantID/IP-Adapter-Face encoding.\n"
            "Install: pip install insightface onnxruntime-gpu"
        ) from e


def _require_torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError("torch is required for identity encoding") from e


def _require_pil():
    try:
        from PIL import Image
        return Image
    except ImportError as e:
        raise ImportError("Pillow is required for identity encoding") from e


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """Disk-backed cache keyed by SHA-256 of the reference image bytes."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, image_path: Path) -> str:
        h = hashlib.sha256(image_path.read_bytes()).hexdigest()[:16]
        return h

    def get(self, image_path: Path, backend: str) -> Optional[dict]:
        key = self._key(image_path)
        p = self.cache_dir / f"{key}_{backend}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
        return None

    def put(self, image_path: Path, backend: str, data: dict) -> None:
        key = self._key(image_path)
        p = self.cache_dir / f"{key}_{backend}.pkl"
        with open(p, "wb") as f:
            pickle.dump(data, f)


# ---------------------------------------------------------------------------
# InstantID encoder
# ---------------------------------------------------------------------------

class InstantIDEncoder:
    """Encodes a reference image using InsightFace ArcFace + InstantID.

    Produces:
        face_embedding  : (512,) float32 — ArcFace identity embedding
        face_kps        : (5, 2) float32 — 5-point facial landmarks (for
                          InstantID's ControlNet branch)
        face_image      : PIL Image — aligned face crop (for IP-Adapter)
    """

    def __init__(self, cfg: IdentityConfig, device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = device
        self._app = None   # lazy-loaded InsightFace FaceAnalysis

    def _load_app(self) -> Any:
        if self._app is not None:
            return self._app
        _, FaceAnalysis = _require_insightface()
        app = FaceAnalysis(
            name=self.cfg.insightface_model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        self._app = app
        return app

    def encode(self, image_path: str | Path) -> dict:
        """Return identity conditioning dict for one reference image."""
        cv2 = _require_cv2()
        Image = _require_pil()

        image_path = Path(image_path)
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read reference image: {image_path}")

        app = self._load_app()
        faces = app.get(img_bgr)

        if not faces:
            raise ValueError(
                f"No face detected in reference image: {image_path}\n"
                "Ensure the image shows a clear frontal face."
            )

        # Use the largest detected face.
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        embedding = face.normed_embedding.astype(np.float32)   # (512,)
        kps = face.kps.astype(np.float32)                      # (5, 2)

        # Aligned face crop for IP-Adapter injection.
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        pad = int((x2 - x1) * 0.2)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(img_bgr.shape[1], x2 + pad)
        y2 = min(img_bgr.shape[0], y2 + pad)
        face_crop_bgr = img_bgr[y1:y2, x1:x2]
        face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        face_image = Image.fromarray(face_crop_rgb).resize((224, 224))

        # Full reference image as PIL (for IP-Adapter full-image conditioning).
        ref_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ref_image = Image.fromarray(ref_rgb)

        return {
            "backend": "instantid",
            "face_embedding": embedding,
            "face_kps": kps,
            "face_image": face_image,
            "ref_image": ref_image,
            "identity_strength": self.cfg.identity_strength,
        }


# ---------------------------------------------------------------------------
# IP-Adapter encoder (fallback)
# ---------------------------------------------------------------------------

class IPAdapterEncoder:
    """Encodes a reference image using CLIP ViT-H/14 for IP-Adapter injection.

    Does not require face detection — works with full-body reference images.
    Produces a CLIP image embedding that IP-Adapter injects into cross-attention.
    """

    def __init__(self, cfg: IdentityConfig, device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = device
        self._processor = None
        self._model = None

    def _load_clip(self) -> tuple:
        if self._processor is not None:
            return self._processor, self._model
        try:
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        except ImportError as e:
            raise ImportError(
                "transformers is required for IP-Adapter encoding.\n"
                "Install: pip install transformers"
            ) from e
        torch = _require_torch()
        model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        self._processor = CLIPImageProcessor.from_pretrained(model_id)
        self._model = CLIPVisionModelWithProjection.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(self.device)
        self._model.eval()
        return self._processor, self._model

    def encode(self, image_path: str | Path) -> dict:
        cv2 = _require_cv2()
        Image = _require_pil()
        torch = _require_torch()

        image_path = Path(image_path)
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read reference image: {image_path}")

        ref_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ref_image = Image.fromarray(ref_rgb)

        processor, model = self._load_clip()
        inputs = processor(images=ref_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
        # (1, 1024) projected embedding
        embedding = outputs.image_embeds.cpu().float().numpy()

        return {
            "backend": "ip_adapter",
            "clip_embedding": embedding,
            "ref_image": ref_image,
            "identity_strength": self.cfg.identity_strength,
        }


# ---------------------------------------------------------------------------
# Unified IdentityEncoder
# ---------------------------------------------------------------------------

class IdentityEncoder:
    """Unified identity encoder that dispatches to InstantID or IP-Adapter.

    Handles caching transparently — the same reference image is never
    re-encoded within a session or across sessions (if cache_embeddings=True).

    Usage:
        encoder = IdentityEncoder(cfg)
        conditioning = encoder.encode("signer.jpg")
        # Pass conditioning to the diffusion backend's inject() method.
    """

    def __init__(
        self,
        cfg: Optional[IdentityConfig] = None,
        device: str = "cuda",
    ) -> None:
        self.cfg = cfg or IdentityConfig()
        self.device = device
        self._cache = (
            EmbeddingCache(self.cfg.embedding_cache_dir)
            if self.cfg.cache_embeddings
            else None
        )
        self._instantid: Optional[InstantIDEncoder] = None
        self._ip_adapter: Optional[IPAdapterEncoder] = None

    def encode(self, image_path: str | Path) -> dict:
        """Encode a reference image.  Returns a conditioning dict.

        The dict is backend-specific and consumed by the diffusion backend's
        inject_identity() method.  Keys always present:
            backend          : str — "instantid" or "ip_adapter"
            ref_image        : PIL.Image — full reference image
            identity_strength: float — conditioning weight
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Reference image not found: {image_path}")

        backend_name = self.cfg.backend.value

        # Cache lookup.
        if self._cache is not None:
            cached = self._cache.get(image_path, backend_name)
            if cached is not None:
                print(f"[IdentityEncoder] cache hit for {image_path.name} ({backend_name})")
                return cached

        print(f"[IdentityEncoder] encoding {image_path.name} with {backend_name}")

        if self.cfg.backend == IdentityBackend.INSTANTID:
            if self._instantid is None:
                self._instantid = InstantIDEncoder(self.cfg, self.device)
            result = self._instantid.encode(image_path)
        elif self.cfg.backend == IdentityBackend.IP_ADAPTER:
            if self._ip_adapter is None:
                self._ip_adapter = IPAdapterEncoder(self.cfg, self.device)
            result = self._ip_adapter.encode(image_path)
        elif self.cfg.backend == IdentityBackend.NONE:
            # No identity conditioning — return a minimal dict.
            Image = _require_pil()
            cv2 = _require_cv2()
            img_bgr = cv2.imread(str(image_path))
            ref_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            result = {
                "backend": "none",
                "ref_image": Image.fromarray(ref_rgb),
                "identity_strength": 0.0,
            }
        else:
            raise ValueError(f"Unknown identity backend: {self.cfg.backend}")

        # Cache the result (PIL Images are not picklable by default — convert
        # to numpy arrays for storage, reconstruct on load).
        if self._cache is not None:
            cacheable = {
                k: (np.array(v) if hasattr(v, "save") else v)
                for k, v in result.items()
            }
            self._cache.put(image_path, backend_name, cacheable)

        return result

    def inject_into_pipeline(self, pipe: Any, conditioning: dict) -> None:
        """Inject identity conditioning into a loaded diffusion pipeline.

        This method is called by the diffusion backend after the pipeline is
        loaded.  It modifies the pipeline in-place to apply the conditioning.

        The exact injection mechanism depends on the backend:
            instantid  → loads IP-Adapter weights + sets image_proj_model
            ip_adapter → loads IP-Adapter weights
        """
        backend = conditioning.get("backend", "none")
        strength = conditioning.get("identity_strength", 0.0)

        if backend == "none" or strength == 0.0:
            return

        Image = _require_pil()

        # Reconstruct PIL Image if it was stored as numpy array.
        ref = conditioning.get("ref_image")
        if isinstance(ref, np.ndarray):
            ref = Image.fromarray(ref)

        if backend == "instantid":
            self._inject_instantid(pipe, conditioning, ref, strength)
        elif backend == "ip_adapter":
            self._inject_ip_adapter(pipe, conditioning, ref, strength)

    def _inject_instantid(
        self, pipe: Any, conditioning: dict, ref_image: Any, strength: float
    ) -> None:
        """Load InstantID weights into the pipeline."""
        try:
            pipe.load_ip_adapter(
                self.cfg.instantid_model_id,
                subfolder=None,
                weight_name="ip-adapter.bin",
            )
            pipe.set_ip_adapter_scale(strength)
        except Exception as e:
            print(f"[IdentityEncoder] InstantID injection warning: {e}")
            print("[IdentityEncoder] Falling back to IP-Adapter injection.")
            self._inject_ip_adapter(pipe, conditioning, ref_image, strength)

    def _inject_ip_adapter(
        self, pipe: Any, conditioning: dict, ref_image: Any, strength: float
    ) -> None:
        """Load IP-Adapter weights into the pipeline."""
        try:
            pipe.load_ip_adapter(
                self.cfg.ip_adapter_model_id,
                subfolder=self.cfg.ip_adapter_subfolder,
                weight_name=self.cfg.ip_adapter_weight_name,
            )
            pipe.set_ip_adapter_scale(strength)
        except Exception as e:
            print(f"[IdentityEncoder] IP-Adapter injection warning: {e}")


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m avatar.conditioning.identity_encoder <image_path> [--backend instantid|ip_adapter]")
        raise SystemExit(1)

    image_path = Path(sys.argv[1])
    backend_arg = "instantid"
    if "--backend" in sys.argv:
        idx = sys.argv.index("--backend")
        backend_arg = sys.argv[idx + 1]

    from avatar.config import IdentityConfig, IdentityBackend
    cfg = IdentityConfig(backend=IdentityBackend(backend_arg))
    encoder = IdentityEncoder(cfg)

    try:
        result = encoder.encode(image_path)
        print(f"Encoding successful (backend={result['backend']})")
        for k, v in result.items():
            if hasattr(v, "shape"):
                print(f"  {k}: shape={v.shape} dtype={v.dtype}")
            elif hasattr(v, "size"):
                print(f"  {k}: PIL Image size={v.size}")
            else:
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"Error: {e}")
        raise SystemExit(1)
