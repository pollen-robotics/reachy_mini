from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, List

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from openai import OpenAI

logger = logging.getLogger(__name__)

# Types & state

Direction = Literal["left", "right", "up", "down", "front"]


@dataclass
class Deps:
    """External dependencies the tools need"""

    reachy_mini: Any
    create_head_pose: Any
    camera: cv2.VideoCapture
    # Optional deps
    image_handler: Optional["OpenAIImageHandler"] = None


# Helpers

def _encode_jpeg_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("Failed to encode image as JPEG.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _read_frame(cap: cv2.VideoCapture, attempts: int = 5) -> np.ndarray:
    """Grab a frame with a small retry."""
    trials, frame, ret = 0, None, False
    while trials < attempts and not ret:
        ret, frame = cap.read()
        trials += 1
        if not ret and trials < attempts:
            time.sleep(0.1)  # Small delay between retries
    if not ret or frame is None:
        logger.error("Failed to capture image from camera after %d attempts", attempts)
        raise RuntimeError("Failed to capture image from camera.")
    return frame


# Image QA (OpenAI)


class OpenAIImageHandler:
    """Minimal wrapper for asking questions about an image using OpenAI Responses."""

    def __init__(
        self, client: Optional[OpenAI] = None, model: str = "gpt-4o-mini"
    ) -> None:
        self.client = client or OpenAI()
        self.model = model

    def ask_about_image(self, img: np.ndarray, question: str) -> str:
        url = "data:image/jpeg;base64," + _encode_jpeg_b64(img)
        messages = [
            {"role": "system", "content": question},
            {"role": "user", "content": [{"type": "input_image", "image_url": url}]},
        ]
        resp = self.client.responses.create(model=self.model, input=messages)
        try:
            return resp.output[0].content[0].text  # legacy path
        except Exception:
            return json.dumps(resp.dict())[:500]


# Face recognition (InsightFace, CPU)


class FaceEngine:
    """CPU-only InsightFace embedding + simple cosine search."""

    def __init__(
        self, det_size=(320, 320), providers: Optional[List[str]] = None
    ) -> None:
        self.app = FaceAnalysis(
            name="buffalo_l", providers=providers or ["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=det_size)

    def embed(self, bgr: np.ndarray) -> np.ndarray | None:
        faces = self.app.get(bgr)  # accepts BGR
        if not faces:
            return None
        f = max(faces, key=lambda F: getattr(F, "det_score", 0.0))
        emb = getattr(f, "normed_embedding", None)
        return emb.astype(np.float32) if emb is not None else None


def _iter_images(root: str) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    rootp = Path(root)
    if not rootp.exists():
        return []
    return [p for p in rootp.rglob("*") if p.suffix.lower() in exts]


def _build_face_db(engine: FaceEngine, root: str) -> tuple[np.ndarray, list[str]]:
    """Scan person folders, compute one embedding per image."""
    paths = _iter_images(root)
    embs, labels = [], []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        emb = engine.embed(img)
        if emb is None:
            continue
        embs.append(emb)
        labels.append(p.parent.name)  # person name = parent folder
    if not embs:
        return np.empty((0, 512), dtype=np.float32), []
    return np.vstack(embs), labels


def _load_or_build_face_db(
    engine: FaceEngine, root: str
) -> tuple[np.ndarray, list[str]]:
    """Cache embeddings for faster startup; rebuild if cache missing/bad."""
    cache = Path(root) / ".face_db_cache.npz"
    if cache.exists():
        try:
            data = np.load(cache, allow_pickle=True)
            embs = data["embs"].astype(np.float32)
            labels = list(data["labels"])
            return embs, labels
        except Exception:
            logger.warning("Face DB cache read failed at %s; rebuilding", cache)
    embs, labels = _build_face_db(engine, root)
    try:
        np.savez_compressed(cache, embs=embs, labels=np.array(labels, dtype=object))
    except Exception:
        logger.warning("Face DB cache write failed at %s", cache)
    return embs, labels


def _cosine_sim_matrix(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Cosine similarity for vector v against rows of M (both float32)."""
    v = v.astype(np.float32)
    M = M.astype(np.float32)
    v_norm = np.linalg.norm(v) + 1e-12
    M_norm = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return (M @ v) / (M_norm[:, 0] * v_norm)


# Tool coroutines


async def move_head(deps: Deps, *, direction: Direction) -> Dict[str, Any]:
    """Move your head in a given direction"""
    logger.info("Tool call: move_head direction=%s", direction)

    # Import and update the SAME global variables that main.py reads
    from reachy_mini_talk.main import movement_manager

    if direction == "left":
        target = deps.create_head_pose(0, 0, 0, 0, 0, 40, degrees=True)
    elif direction == "right":
        target = deps.create_head_pose(0, 0, 0, 0, 0, -40, degrees=True)
    elif direction == "up":
        target = deps.create_head_pose(0, 0, 0, 0, -30, 0, degrees=True)
    elif direction == "down":
        target = deps.create_head_pose(0, 0, 0, 0, 30, 0, degrees=True)
    else:  # front
        target = deps.create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)

    movement_manager.moving_start = time.monotonic()
    movement_manager.moving_for = 1.0
    movement_manager.current_head_pose = target

    # Start the movement
    deps.reachy_mini.goto_target(target, duration=1.0)

    return {"status": f"looking {direction}"}


async def head_tracking(deps: Deps, *, start: bool) -> Dict[str, Any]:
    """Toggle head tracking state - UPDATES GLOBAL STATE."""
    from reachy_mini_talk.main import movement_manager

    movement_manager.is_head_tracking = bool(start)
    status = "started" if start else "stopped"
    logger.info("Tool call: head_tracking %s", status)
    return {"status": f"head tracking {status}"}


async def camera(deps: Deps, *, question: str) -> Dict[str, Any]:
    """
    Capture an image and ask a question about it using OpenAI.
    Returns: {"image_description": '...'} or {"error": '...'}.
    """
    q = (question or "").strip()
    if not q:
        logger.error("camera: empty question")
        return {"error": "question must be a non-empty string"}

    logger.info("Tool call: camera question=%s", q[:120])  # log first 120 chars

    try:
        frame = await asyncio.to_thread(_read_frame, deps.camera)
    except Exception as e:
        logger.exception("camera: failed to capture image")
        return {"error": f"camera capture failed: {type(e).__name__}: {e}"}

    if not deps.image_handler:
        logger.error("camera: image handler not configured")
        return {"error": "image handler not configured"}

    # Best-effort sound; don't fail the tool if it errors.
    try:
        deps.reachy_mini.play_sound(f"hmm{np.random.randint(1, 6)}.wav")
    except Exception:
        logger.debug("camera: optional play_sound failed", exc_info=True)

    try:
        desc = await asyncio.to_thread(deps.image_handler.ask_about_image, frame, q)
        logger.debug(
            "camera: image QA result length=%d",
            len(desc) if isinstance(desc, str) else -1,
        )
        return {"image_description": desc}
    except Exception as e:
        logger.exception("camera: vision pipeline error")
        return {"error": f"vision failed: {type(e).__name__}: {e}"}


# Registration helpers (OpenAI Realtime)

TOOL_SPECS = [
    {
        "type": "function",
        "name": "move_head",
        "description": "Move your head in a given direction: left, right, up, down or front.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["left", "right", "up", "down", "front"],
                }
            },
            "required": ["direction"],
        },
    },
    {
        "type": "function",
        "name": "camera",
        "description": "Take a picture using your camera, ask a question about the picture. Get an answer about the picture",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask about the picture",
                }
            },
            "required": ["question"],
        },
    },
    {
        "type": "function",
        "name": "head_tracking",
        "description": "Start or stop head tracking",
        "parameters": {
            "type": "object",
            "properties": {
                "start": {
                    "type": "boolean",
                    "description": "Whether to start or stop head tracking",
                }
            },
            "required": ["start"],
        },
    },
]


def get_tool_registry(deps: Deps):
    """Map tool name -> coroutine that accepts **kwargs (tool args)."""
    return {
        "move_head": lambda **kw: move_head(deps, **kw),
        "camera": lambda **kw: camera(deps, **kw),
        "head_tracking": lambda **kw: head_tracking(deps, **kw),
    }


async def dispatch_tool_call(name: str, args_json: str, deps: Deps) -> Dict[str, Any]:
    """Utility to execute a tool from streamed function_call arguments."""
    try:
        args = json.loads(args_json or "{}")
    except Exception:
        args = {}
    registry = get_tool_registry(deps)
    func = registry.get(name)
    if not func:
        return {"error": f"unknown tool: {name}"}
    try:
        return await func(**args)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.exception("Tool error in %s: %s", name, error_msg)
        return {"error": error_msg}
