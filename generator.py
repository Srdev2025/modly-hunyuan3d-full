"""
Modly Extension: Hunyuan3D 2 Full (Mesh + Texture)
---------------------------------------------------
Nodes:
  - Generate Mesh  : image  → mesh  (uses Hunyuan3D-2mini shape pipeline)
  - Texture Mesh   : image + mesh → mesh  (uses Hunyuan3D-Paint texture pipeline)

Models are downloaded automatically from HuggingFace on first run.
Requirements compiled during Modly's Repair step (see setup.py / requirements below).

Compatible with Modly's extension API (generator.py + manifest.json).
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Lazy imports – only loaded when the node actually runs so Modly can still
# display the extension even if optional deps are missing.
# ---------------------------------------------------------------------------

def _import_torch():
    import torch
    return torch


def _import_trimesh():
    import trimesh
    return trimesh


# ---------------------------------------------------------------------------
# Helper: remove background from image (simple alpha / white BG approach)
# ---------------------------------------------------------------------------

def _remove_background(image_path: str) -> "PIL.Image.Image":
    """
    Returns a PIL RGBA image.  If rembg is available use it; otherwise just
    load the image as-is (Hunyuan3D handles white-BG images fine).
    """
    from PIL import Image
    img = Image.open(image_path).convert("RGBA")
    try:
        from rembg import remove
        img = remove(img)
    except Exception:
        # rembg not installed or failed – continue with original
        pass
    return img


# ---------------------------------------------------------------------------
# Node 1: Generate Mesh
# ---------------------------------------------------------------------------

class GenerateMesh:
    """
    Converts a single image into an untextured GLB mesh using
    Hunyuan3D-2mini (0.6B shape model, ~6 GB VRAM).
    """

    # Model identifiers (HuggingFace)
    SHAPE_MODEL_ID = "tencent/Hunyuan3D-2mini"
    SHAPE_SUBFOLDER = "hunyuan3d-dit-v2-mini-turbo"

    def __init__(self):
        self._pipeline = None

    # ------------------------------------------------------------------
    # Modly calls this to know what parameters to show in the UI
    # ------------------------------------------------------------------
    @staticmethod
    def get_parameters() -> list[dict]:
        return [
            {
                "name": "steps",
                "label": "Inference Steps",
                "type": "int",
                "default": 30,
                "min": 10,
                "max": 100,
            },
            {
                "name": "guidance_scale",
                "label": "Guidance Scale",
                "type": "float",
                "default": 5.5,
                "min": 1.0,
                "max": 15.0,
            },
            {
                "name": "seed",
                "label": "Seed (-1 = random)",
                "type": "int",
                "default": -1,
            },
            {
                "name": "octree_resolution",
                "label": "Mesh Resolution",
                "type": "select",
                "options": ["256", "380", "512"],
                "default": "380",
            },
        ]

    # ------------------------------------------------------------------
    # Lazy-load the shape pipeline
    # ------------------------------------------------------------------
    def _get_pipeline(self, device: str):
        if self._pipeline is None:
            torch = _import_torch()
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

            dtype = torch.float16 if device == "cuda" else torch.float32
            self._pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                self.SHAPE_MODEL_ID,
                subfolder=self.SHAPE_SUBFOLDER,
                torch_dtype=dtype,
            )
            self._pipeline = self._pipeline.to(device)
        return self._pipeline

    # ------------------------------------------------------------------
    # Main entry point called by Modly
    # ------------------------------------------------------------------
    def generate(
        self,
        image: str,          # absolute path to input image
        output_dir: str,     # where to write the output mesh
        parameters: dict[str, Any] | None = None,
        device: str = "cuda",
        **kwargs,
    ) -> str:
        """Returns the path to the generated GLB file."""
        params = parameters or {}
        steps          = int(params.get("steps", 30))
        guidance_scale = float(params.get("guidance_scale", 5.5))
        seed           = int(params.get("seed", -1))
        resolution     = int(params.get("octree_resolution", 380))

        torch = _import_torch()

        # Seed
        if seed < 0:
            import random
            seed = random.randint(0, 2**31 - 1)
        generator = torch.Generator(device=device).manual_seed(seed)

        # Pre-process image
        pil_image = _remove_background(image)

        # Run shape pipeline
        pipeline = self._get_pipeline(device)
        try:
            mesh = pipeline(
                image=pil_image,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                octree_resolution=resolution,
            )[0]
        finally:
            # Free VRAM between stages
            if device == "cuda":
                pipeline.to("cpu")
                torch.cuda.empty_cache()

        # Save as GLB
        out_path = str(Path(output_dir) / "mesh.glb")
        mesh.export(out_path)
        return out_path


# ---------------------------------------------------------------------------
# Node 2: Texture Mesh
# ---------------------------------------------------------------------------

class TextureMesh:
    """
    Paints an existing mesh using Hunyuan3D-Paint (1.3B texture model).
    Requires ~16 GB VRAM total when chained after GenerateMesh.

    Inputs:
        image  – original reference image (same one used for shape gen)
        mesh   – path to GLB/OBJ produced by GenerateMesh
    Output:
        path to a new GLB with baked texture maps
    """

    PAINT_MODEL_ID = "tencent/Hunyuan3D-2"
    PAINT_SUBFOLDER = "hunyuan3d-paint-v2-0-turbo"   # turbo = faster, same quality

    def __init__(self):
        self._pipeline = None

    # ------------------------------------------------------------------
    @staticmethod
    def get_parameters() -> list[dict]:
        return [
            {
                "name": "texture_resolution",
                "label": "Texture Resolution",
                "type": "select",
                "options": ["512", "1024", "2048"],
                "default": "1024",
            },
            {
                "name": "max_views",
                "label": "Multi-View Count",
                "type": "int",
                "default": 6,
                "min": 4,
                "max": 12,
            },
        ]

    # ------------------------------------------------------------------
    def _get_pipeline(self, device: str):
        if self._pipeline is None:
            torch = _import_torch()
            from hy3dgen.texgen import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

            config = Hunyuan3DPaintConfig(
                # will be overridden per-call
            )
            self._pipeline = Hunyuan3DPaintPipeline.from_pretrained(
                self.PAINT_MODEL_ID,
                subfolder=self.PAINT_SUBFOLDER,
            )
            self._pipeline = self._pipeline.to(device)
        return self._pipeline

    # ------------------------------------------------------------------
    def generate(
        self,
        image: str,
        mesh: str,
        output_dir: str,
        parameters: dict[str, Any] | None = None,
        device: str = "cuda",
        **kwargs,
    ) -> str:
        params = parameters or {}
        tex_res   = int(params.get("texture_resolution", 1024))
        max_views = int(params.get("max_views", 6))

        torch = _import_torch()
        trimesh = _import_trimesh()

        # Pre-process image
        pil_image = _remove_background(image)

        # Load input mesh
        input_mesh = trimesh.load(mesh)

        from hy3dgen.texgen import Hunyuan3DPaintConfig
        config = Hunyuan3DPaintConfig(
            max_num_view=max_views,
            resolution=tex_res,
        )

        pipeline = self._get_pipeline(device)
        try:
            textured_mesh = pipeline(
                mesh_path=mesh,
                image_path=pil_image,
                config=config,
            )
        finally:
            if device == "cuda":
                pipeline.to("cpu")
                torch.cuda.empty_cache()

        out_path = str(Path(output_dir) / "mesh_textured.glb")
        textured_mesh.export(out_path)
        return out_path


# ---------------------------------------------------------------------------
# Modly extension entry-points
# ---------------------------------------------------------------------------
# Modly discovers nodes by looking for subclasses of these classes or by
# calling the functions below.  We expose both styles for compatibility.

NODES = {
    "Generate Mesh": GenerateMesh,
    "Texture Mesh":  TextureMesh,
}


def get_node(node_name: str):
    """Factory used by Modly's extension loader."""
    cls = NODES.get(node_name)
    if cls is None:
        raise ValueError(f"Unknown node: {node_name!r}. Available: {list(NODES)}")
    return cls()
