"""Face identity encoder for avatar video generation.

Wraps IP-Adapter (primary) and InstantID (optional) to extract a stable
face embedding from a reference portrait image. The embedding is reused
for every frame of the generated video to maintain consistent identity.

Usage:
    from scripts.avatar.identity_encoder import IdentityEncoder
    enc = IdentityEncoder(method="ip_adapter", device="cuda")
    embedding = enc.encode("assets/avatar_reference.jpg")
    # pass embedding to generate_avatar_video.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


class IdentityEncoder:
    """Encode a reference portrait into a face identity embedding.

    Parameters
    ----------
    method : "ip_adapter" | "instantid"
        IP-Adapter is simpler and works with SDXL out of the box.
        InstantID requires InsightFace and produces higher fidelity.
    device : str
        Torch device string.
    model_dir : Path, optional
        Directory where model weights are cached.
        Defaults to ~/.cache/huggingface/hub (HF default).
    """

    def __init__(
        self,
        method: str = "ip_adapter",
        device: str = "cuda",
        model_dir: Optional[Path] = None,
    ) -> None:
        if method not in ("ip_adapter", "instantid"):
            raise ValueError(f"method must be ip_adapter or instantid, got {method!r}")
        self.method = method
        self.device = device
        self.model_dir = model_dir
        self._processor = None
        self._model = None

    # ------------------------------------------------------------------
    # Lazy initialisation — only load models when encode() is first called
    # ------------------------------------------------------------------

    def _init_ip_adapter(self) -> None:
        """Load CLIP image encoder used by IP-Adapter."""
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        model_id = "openai/clip-vit-large-patch14-336"
        self._processor = CLIPImageProcessor.from_pretrained(model_id)
        self._model = CLIPVisionModelWithProjection.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        ).to(self.device)
        self._model.eval()

    def _init_instantid(self) -> None:
        """Load InsightFace ArcFace encoder used by InstantID."""
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError(
                "InstantID requires insightface: pip install insightface onnxruntime-gpu"
            ) from e
        self._model = FaceAnalysis(
            name="antelopev2",
            root=str(self.model_dir) if self.model_dir else "~/.insightface",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._model.prepare(ctx_id=0, det_size=(640, 640))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, image_path: str | Path) -> dict:
        """Encode a reference portrait image.

        Returns a dict with keys appropriate for the chosen method:
            ip_adapter:  {"image_embeds": Tensor (1, 257, 1024)}
            instantid:   {"face_embeds": np.ndarray (512,),
                          "face_kps": np.ndarray (5, 2)}
        """
        image = Image.open(image_path).convert("RGB")

        if self.method == "ip_adapter":
            return self._encode_ip_adapter(image)
        else:
            return self._encode_instantid(image)

    def _encode_ip_adapter(self, image: Image.Image) -> dict:
        if self._model is None:
            self._init_ip_adapter()

        inputs = self._processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=torch.float16)

        with torch.no_grad():
            outputs = self._model(pixel_values=pixel_values)
            # image_embeds: (1, 257, 1024) — full patch sequence, not just CLS
            image_embeds = outputs.last_hidden_state

        return {
            "image_embeds": image_embeds.cpu(),
            "method": "ip_adapter",
        }

    def _encode_instantid(self, image: Image.Image) -> dict:
        if self._model is None:
            self._init_instantid()

        import cv2
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        faces = self._model.get(img_bgr)

        if not faces:
            raise ValueError(
                "No face detected in reference image. "
                "Use a clear frontal portrait with good lighting."
            )

        # Use the largest detected face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        return {
            "face_embeds": face.normed_embedding,   # (512,) ArcFace embedding
            "face_kps": face.kps,                   # (5, 2) facial keypoints
            "method": "instantid",
        }

    def save(self, embedding: dict, path: str | Path) -> None:
        """Save embedding to disk for reuse across generation runs."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if embedding["method"] == "ip_adapter":
            torch.save(embedding, path)
        else:
            np.savez(path, **{k: v for k, v in embedding.items() if k != "method"})
        print(f"[identity] saved {embedding['method']} embedding → {path}")

    @staticmethod
    def load(path: str | Path) -> dict:
        """Load a previously saved embedding."""
        path = Path(path)
        if path.suffix in (".pt", ".pth"):
            return torch.load(path, map_location="cpu", weights_only=False)
        data = np.load(path, allow_pickle=False)
        return {k: data[k] for k in data.files}


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Encode a reference portrait for avatar generation")
    p.add_argument("image", help="Path to reference portrait image")
    p.add_argument("--method", default="ip_adapter", choices=["ip_adapter", "instantid"])
    p.add_argument("--out", default=None, help="Output path for saved embedding")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    enc = IdentityEncoder(method=args.method, device=args.device)
    emb = enc.encode(args.image)
    print(f"[identity] encoded with {args.method}")
    for k, v in emb.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {v.shape} {getattr(v, 'dtype', '')}")

    if args.out:
        enc.save(emb, args.out)
    else:
        out = Path(args.image).with_suffix(".embedding.pt")
        enc.save(emb, out)
