from __future__ import annotations

import copy
import json
import logging
import os
import random
import shutil
import subprocess
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import modal
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


APP_NAME = "ltx-motion-studio"
BACKEND_VERSION = "3.0.0"

COMFY_DIR = Path("/opt/ComfyUI")
WORKFLOW_DIR = Path("/opt/workflows")
RESULT_DIR = Path("/results")
COMFY_URL = "http://127.0.0.1:8188"
COMFY_LOG = Path("/tmp/comfyui.log")

COMFYUI_VERSION = "v0.34.0"
WORKFLOW_TEMPLATE_COMMIT = "db9d5859d09c21a2d4101a1c18f64fc2f70e4fa4"
CONVERTER_COMMIT = "bc8538278f82053b3ca10a44d62d02596f8e1a37"

LTX_REPO = "Lightricks/LTX-2.5"
PROMPT_ENHANCER_REPO = "Comfy-Org/gemma-4"

MODEL_FILES = [
    "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "vae/ltx-2.5-video-vae-bf16.safetensors",
    "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
]
PROMPT_ENHANCER_FILE = "text_encoders/gemma4_e2b_it_int8_convrot.safetensors"

MODEL_NAMES = {
    "unet": "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    "clip": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "video_vae": "ltx-2.5-video-vae-bf16.safetensors",
    "audio_vae": "ltx-2.5-audio-vae-bf16.safetensors",
    "upscaler": "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    "enhancer": "gemma4_e2b_it_int8_convrot.safetensors",
}

GEN_DIMS = {"16:9": (1280, 736), "9:16": (736, 1280)}
FINAL_DIMS = {"16:9": (1280, 720), "9:16": (720, 1280)}
ALLOWED_DURATIONS = {5, 10, 20, 30, 60}
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_PROMPT_CHARS = 12000
TRANSFER_LIMIT_BYTES = 95 * 1024 * 1024
TRANSFER_TARGET_BYTES = 90 * 1024 * 1024
FRAME_RATE = 24

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CAMERA_PROMPTS = {
    "auto": "",
    "static": "The camera remains locked off and stable.",
    "dolly_in": "The camera performs a smooth, slow dolly-in toward the subject.",
    "dolly_out": "The camera performs a smooth, slow dolly-out away from the subject.",
    "pan_left": "The camera pans smoothly to the left while preserving the subject.",
    "pan_right": "The camera pans smoothly to the right while preserving the subject.",
    "orbit_left": "The camera makes a controlled cinematic orbit to the left around the subject.",
    "orbit_right": "The camera makes a controlled cinematic orbit to the right around the subject.",
    "crane_up": "The camera rises smoothly in a cinematic crane-up movement.",
    "handheld": "Use subtle, natural handheld camera movement without excessive shake.",
}


def _clip_text(value: Any, limit: int = 6000) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + " ...[truncated]"


def _tail_file(path: Path, max_chars: int = 5000) -> str:
    try:
        if not path.exists():
            return ""
        data = path.read_text(encoding="utf-8", errors="replace")
        return data[-max_chars:]
    except Exception:
        return ""


def _download_models() -> None:
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing. Add the token to the Modal secret named 'huggingface-secret'."
        )

    base = COMFY_DIR / "models"
    base.mkdir(parents=True, exist_ok=True)

    for filename in MODEL_FILES:
        hf_hub_download(
            repo_id=LTX_REPO,
            filename=filename,
            local_dir=str(base),
            token=token,
        )

    hf_hub_download(
        repo_id=PROMPT_ENHANCER_REPO,
        filename=PROMPT_ENHANCER_FILE,
        local_dir=str(base),
        token=token,
    )


hf_secret = modal.Secret.from_name("huggingface-secret")

converter_install = (
    "git clone --filter=blob:none --no-checkout "
    "https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint.git "
    "/opt/ComfyUI/custom_nodes/workflow-to-api-converter && "
    f"git -C /opt/ComfyUI/custom_nodes/workflow-to-api-converter fetch --depth 1 origin {CONVERTER_COMMIT} && "
    f"git -C /opt/ComfyUI/custom_nodes/workflow-to-api-converter checkout --detach {CONVERTER_COMMIT}"
)

gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("git", "ffmpeg", "curl", "libgl1", "libglib2.0-0")
    .run_commands(
        (
            "git clone --depth 1 "
            f"--branch {COMFYUI_VERSION} "
            "https://github.com/Comfy-Org/ComfyUI.git /opt/ComfyUI"
        ),
        "python -m pip install --upgrade pip",
        (
            "python -m pip install --index-url "
            "https://download.pytorch.org/whl/cu128 torch torchvision torchaudio"
        ),
        "python -m pip install -r /opt/ComfyUI/requirements.txt",
        (
            "python -m pip install 'comfy-kitchen>=0.2.26' requests "
            "huggingface_hub hf_xet 'fastapi[standard]' python-multipart Pillow"
        ),
        converter_install,
        "mkdir -p /opt/workflows",
        (
            "curl --retry 5 --retry-all-errors -fsSL "
            "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/"
            f"{WORKFLOW_TEMPLATE_COMMIT}/templates/video_ltx2_5_i2v.json "
            "-o /opt/workflows/video_ltx2_5_i2v.json"
        ),
        (
            "curl --retry 5 --retry-all-errors -fsSL "
            "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/"
            f"{WORKFLOW_TEMPLATE_COMMIT}/templates/video_ltx2_5_t2v.json "
            "-o /opt/workflows/video_ltx2_5_t2v.json"
        ),
        "python -m json.tool /opt/workflows/video_ltx2_5_i2v.json >/dev/null",
        "python -m json.tool /opt/workflows/video_ltx2_5_t2v.json >/dev/null",
    )
    .run_function(_download_models, secrets=[hf_secret], timeout=3600)
)

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]", "python-multipart", "requests", "Pillow")
    .add_local_dir("web", remote_path="/web")
)

app = modal.App(APP_NAME)
result_volume = modal.Volume.from_name("ltx-motion-results", create_if_missing=True)


def _wait_for_comfy(timeout_seconds: int = 240) -> None:
    import requests

    deadline = time.time() + timeout_seconds
    last_error = "No response"

    while time.time() < deadline:
        try:
            response = requests.get(f"{COMFY_URL}/system_stats", timeout=(3, 15))
            if response.ok:
                return
            last_error = f"HTTP {response.status_code}: {_clip_text(response.text, 1000)}"
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2)

    log_tail = _tail_file(COMFY_LOG)
    raise RuntimeError(
        "ComfyUI did not become ready. "
        f"Last error: {last_error}. Log tail: {_clip_text(log_tail)}"
    )


def _find_main_subgraph(
    workflow: dict[str, Any], expected_name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    definitions = workflow.get("definitions", {}).get("subgraphs", [])
    subgraph_def = next(
        (item for item in definitions if item.get("name") == expected_name), None
    )
    if not subgraph_def:
        names = [x.get("name") for x in definitions]
        raise RuntimeError(
            f"Subgraph not found: {expected_name}. Available: {_clip_text(names)}"
        )

    subgraph_id = subgraph_def.get("id")
    outer_node = next(
        (
            node
            for node in workflow.get("nodes", [])
            if node.get("type") == subgraph_id
        ),
        None,
    )
    if not outer_node:
        raise RuntimeError(f"Outer node not found for subgraph: {expected_name}")
    return outer_node, subgraph_def


def _widget_index_map(subgraph_def: dict[str, Any]) -> dict[str, int]:
    scalar_types = {"STRING", "BOOLEAN", "INT", "FLOAT", "COMBO"}
    labels: list[str] = []
    for item in subgraph_def.get("inputs", []):
        if item.get("type") not in scalar_types:
            continue
        name = item.get("label") or item.get("name")
        if name:
            labels.append(name)
    return {name: index for index, name in enumerate(labels)}


def _disconnect_link(workflow: dict[str, Any], link_id: int | None) -> None:
    if link_id is None:
        return

    workflow["links"] = [
        link for link in workflow.get("links", []) if link and link[0] != link_id
    ]

    for node in workflow.get("nodes", []):
        for inp in node.get("inputs", []) or []:
            if inp.get("link") == link_id:
                inp["link"] = None
        for out in node.get("outputs", []) or []:
            links = out.get("links")
            if isinstance(links, list) and link_id in links:
                out["links"] = [x for x in links if x != link_id] or None


def _set_outer_value(
    workflow: dict[str, Any],
    outer_node: dict[str, Any],
    subgraph_def: dict[str, Any],
    label: str,
    value: Any,
) -> None:
    index_map = _widget_index_map(subgraph_def)
    if label not in index_map:
        raise RuntimeError(
            f"Template parameter missing: {label}. Available: {sorted(index_map.keys())}"
        )

    for inp in outer_node.get("inputs", []) or []:
        input_label = inp.get("label") or inp.get("name")
        if input_label == label:
            _disconnect_link(workflow, inp.get("link"))
            inp["link"] = None

    values = outer_node.setdefault("widgets_values", [])
    index = index_map[label]
    if index >= len(values):
        raise RuntimeError(
            f"Template widgets_values mismatch for {label}: "
            f"index={index}, values={len(values)}"
        )
    values[index] = value


def _patch_save_prefix(workflow: dict[str, Any], prefix: str) -> None:
    found = False
    for node in workflow.get("nodes", []):
        if node.get("type") != "SaveVideo":
            continue
        found = True
        values = node.setdefault("widgets_values", [])
        if values:
            values[0] = prefix
        else:
            node["widgets_values"] = [prefix, "auto", "auto"]
    if not found:
        raise RuntimeError("SaveVideo node was not found in the workflow template")


def _prepare_workflow(
    *,
    mode: Literal["image", "text"],
    prompt: str,
    prompt_enhance: bool,
    duration: int,
    aspect_ratio: Literal["16:9", "9:16"],
    seed: int,
    input_filename: str | None,
    output_prefix: str,
) -> dict[str, Any]:
    if mode == "image":
        template_name = "video_ltx2_5_i2v.json"
        expected_name = "Image to Video (LTX-2.5)"
    else:
        template_name = "video_ltx2_5_t2v.json"
        expected_name = "Text to Video (LTX-2.5)"

    workflow_path = WORKFLOW_DIR / template_name
    if not workflow_path.exists():
        raise RuntimeError(f"Workflow template not found: {workflow_path}")

    try:
        workflow = copy.deepcopy(json.loads(workflow_path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise RuntimeError(f"Could not read workflow template: {type(exc).__name__}: {exc}") from None

    outer, definition = _find_main_subgraph(workflow, expected_name)
    width, height = GEN_DIMS[aspect_ratio]

    settings = (
        ("prompt", prompt),
        ("prompt_enhance", bool(prompt_enhance)),
        ("duration", int(duration)),
        ("width", width),
        ("height", height),
        ("noise_seed", int(seed)),
        ("frame_rate", FRAME_RATE),
        ("unet_name", MODEL_NAMES["unet"]),
        ("video_vae", MODEL_NAMES["video_vae"]),
        ("audio_vae", MODEL_NAMES["audio_vae"]),
        ("clip_name", MODEL_NAMES["clip"]),
        ("upscale_model", MODEL_NAMES["upscaler"]),
        ("prompt_enhance_model", MODEL_NAMES["enhancer"]),
    )

    for label, value in settings:
        _set_outer_value(workflow, outer, definition, label, value)

    if mode == "image":
        if not input_filename:
            raise ValueError("Image-to-video requires an input image")
        load_node = next(
            (
                node
                for node in workflow.get("nodes", [])
                if node.get("type") == "LoadImage"
                and "First Frame" in str(node.get("title", ""))
            ),
            None,
        )
        if not load_node:
            raise RuntimeError("Load First Frame node was not found in the I2V template")
        values = load_node.setdefault("widgets_values", [])
        if values:
            values[0] = input_filename
        else:
            values.append(input_filename)

    _patch_save_prefix(workflow, output_prefix)
    return workflow


def _sanitize_api_workflow(api_workflow: Any) -> dict[str, Any]:
    """
    Defensive repair for the UI-workflow -> API converter.

    LTX audio/video generation in this app is always one sample at a time. Some
    converter versions can mis-map a nearby scalar (commonly frame_rate) into
    batch_size inside nested LTX subgraphs. That produces incompatible video/audio
    latent batch dimensions and a torch.cat/pack_latents tensor-size failure.
    Force every LTX batch_size input to 1 after conversion.
    """
    if not isinstance(api_workflow, dict) or not api_workflow:
        raise RuntimeError("Workflow converter returned an empty or invalid API workflow")

    executable_nodes = 0
    patched_batches = 0

    for node_id, node in api_workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type", ""))
        inputs = node.get("inputs")
        if not class_type or not isinstance(inputs, dict):
            continue
        executable_nodes += 1

        class_upper = class_type.upper()
        is_ltx = (
            "LTX" in class_upper
            or class_type == "EmptyLTXVLatentVideo"
            or class_type == "LTXVImgToVideo"
        )
        if is_ltx and "batch_size" in inputs:
            if inputs.get("batch_size") != 1:
                patched_batches += 1
            inputs["batch_size"] = 1

        # Extra protection for the exact latent producers used by LTX workflows,
        # even if a future converter changes whether the input is emitted.
        if class_type in {
            "EmptyLTXVLatentVideo",
            "LTXVEmptyLatentAudio",
            "LTXVImgToVideo",
        }:
            if inputs.get("batch_size") != 1:
                patched_batches += 1
            inputs["batch_size"] = 1

        # A converted constant frame-rate value should never be zero/negative.
        # Preserve valid links and valid numeric values; only repair impossible values.
        if class_type == "LTXVEmptyLatentAudio":
            # This app deliberately fixes generation to 24 fps. Replacing any
            # converted constant/link here avoids the converter accidentally
            # wiring another scalar into the audio frame-rate slot.
            inputs["frame_rate"] = FRAME_RATE

    if executable_nodes == 0:
        raise RuntimeError("Converted workflow contains no executable ComfyUI nodes")

    # Keep this metadata only in Python logs/diagnostics, not in the Comfy prompt.
    _ = patched_batches
    return api_workflow


def _comfy_process_guard(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None:
        return
    exit_code = proc.poll()
    if exit_code is None:
        return
    raise RuntimeError(
        f"ComfyUI process exited unexpectedly with code {exit_code}. "
        f"Log tail: {_clip_text(_tail_file(COMFY_LOG))}"
    )


def _execution_error_summary(message: Any) -> str:
    payload: Any = message
    if isinstance(message, list) and len(message) > 1:
        payload = message[1]

    if isinstance(payload, dict):
        parts: list[str] = []
        for key in ("node_id", "node_type", "exception_type", "exception_message"):
            value = payload.get(key)
            if value not in (None, ""):
                parts.append(f"{key}={_clip_text(value, 1800)}")
        if parts:
            return "; ".join(parts)

    return _clip_text(payload, 5000)


def _convert_and_execute(
    workflow: dict[str, Any],
    *,
    proc: subprocess.Popen[Any] | None,
    timeout_seconds: int = 1200,
) -> tuple[str, dict[str, Any]]:
    import requests

    _comfy_process_guard(proc)

    try:
        converted = requests.post(
            f"{COMFY_URL}/workflow/convert",
            json=workflow,
            timeout=(10, 180),
        )
    except requests.RequestException as exc:
        _comfy_process_guard(proc)
        raise RuntimeError(
            f"ComfyUI workflow conversion request failed: {type(exc).__name__}: {exc}"
        ) from None

    if not converted.ok:
        raise RuntimeError(
            f"ComfyUI workflow conversion failed ({converted.status_code}): "
            f"{_clip_text(converted.text)}"
        )

    try:
        api_workflow = _sanitize_api_workflow(converted.json())
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Could not parse converted workflow: {type(exc).__name__}: {exc}; "
            f"body={_clip_text(converted.text)}"
        ) from None

    try:
        queued = requests.post(
            f"{COMFY_URL}/prompt",
            json={"prompt": api_workflow},
            timeout=(10, 180),
        )
    except requests.RequestException as exc:
        _comfy_process_guard(proc)
        raise RuntimeError(
            f"ComfyUI queue request failed: {type(exc).__name__}: {exc}"
        ) from None

    if not queued.ok:
        raise RuntimeError(
            f"ComfyUI rejected workflow ({queued.status_code}): {_clip_text(queued.text)}"
        )

    try:
        payload = queued.json()
    except Exception as exc:
        raise RuntimeError(
            f"ComfyUI returned invalid queue JSON: {type(exc).__name__}: {exc}; "
            f"body={_clip_text(queued.text)}"
        ) from None

    if payload.get("node_errors"):
        raise RuntimeError(
            "ComfyUI workflow validation failed: "
            f"{_clip_text(json.dumps(payload['node_errors'], default=str), 6000)}"
        )

    prompt_id = payload.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI did not return a prompt_id: {_clip_text(payload)}")

    deadline = time.time() + timeout_seconds
    consecutive_poll_errors = 0
    last_poll_error = ""

    while time.time() < deadline:
        _comfy_process_guard(proc)
        try:
            history_response = requests.get(
                f"{COMFY_URL}/history/{prompt_id}",
                timeout=(5, 60),
            )
        except requests.RequestException as exc:
            consecutive_poll_errors += 1
            last_poll_error = f"{type(exc).__name__}: {exc}"
            if consecutive_poll_errors >= 5:
                _comfy_process_guard(proc)
            time.sleep(3)
            continue

        if not history_response.ok:
            consecutive_poll_errors += 1
            last_poll_error = (
                f"HTTP {history_response.status_code}: "
                f"{_clip_text(history_response.text, 1500)}"
            )
            time.sleep(3)
            continue

        consecutive_poll_errors = 0

        try:
            history = history_response.json()
        except Exception as exc:
            last_poll_error = f"Invalid history JSON: {type(exc).__name__}: {exc}"
            time.sleep(3)
            continue

        item = history.get(prompt_id)
        if not item:
            time.sleep(3)
            continue

        status = item.get("status", {})
        for message in status.get("messages", []) or []:
            if isinstance(message, list) and message and message[0] == "execution_error":
                summary = _execution_error_summary(message)
                raise RuntimeError(f"ComfyUI execution error: {summary}")

        if status.get("completed"):
            return str(prompt_id), item

        time.sleep(3)

    raise TimeoutError(
        f"ComfyUI generation exceeded {timeout_seconds}s. "
        f"Last polling error: {_clip_text(last_poll_error, 1500)}"
    )


def _candidate_history_paths(history_item: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            filename = value.get("filename")
            if isinstance(filename, str) and Path(filename).suffix.lower() in {".mp4", ".webm", ".mov"}:
                subfolder = str(value.get("subfolder") or "").strip("/\\")
                output_type = str(value.get("type") or "output")
                base = COMFY_DIR / ("temp" if output_type == "temp" else "output")
                path = base / subfolder / filename if subfolder else base / filename
                paths.append(path)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(history_item.get("outputs", {}))
    return paths


def _find_generated_video(
    history_item: dict[str, Any], after_ts: float, prefix_token: str
) -> Path:
    candidates = [
        path
        for path in _candidate_history_paths(history_item)
        if path.exists() and path.is_file()
    ]

    prefixed = [path for path in candidates if prefix_token in str(path)]
    if prefixed:
        return max(prefixed, key=lambda p: p.stat().st_mtime)
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    output_dir = COMFY_DIR / "output"
    filesystem_candidates = [
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".mp4", ".webm", ".mov"}
        and path.stat().st_mtime >= after_ts
        and prefix_token in str(path)
    ]
    if filesystem_candidates:
        return max(filesystem_candidates, key=lambda p: p.stat().st_mtime)

    raise RuntimeError(
        "ComfyUI reported completion but the expected output video could not be located"
    )


def _run_ffmpeg(cmd: list[str], context: str) -> None:
    # Keep routine FFmpeg banners and embedded media metadata (which may include
    # the user's prompt) out of Modal logs and API error responses.
    effective_cmd = cmd
    if cmd and Path(cmd[0]).name == "ffmpeg":
        effective_cmd = [cmd[0], "-hide_banner", "-loglevel", "error", *cmd[1:]]
    try:
        subprocess.run(effective_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr or exc.stdout or str(exc)
        raise RuntimeError(f"{context} failed: {_clip_text(detail, 5000)}") from None


def _video_dimensions(path: Path) -> tuple[int, int]:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", "-of", "json", str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        streams = json.loads(result.stdout).get("streams", [])
        if not streams:
            raise ValueError("no video stream")
        return int(streams[0]["width"]), int(streams[0]["height"])
    except Exception as exc:
        raise RuntimeError(
            f"Could not inspect generated video dimensions: {type(exc).__name__}: {exc}"
        ) from None


def _exact_720p(source: Path, target: Path, aspect_ratio: str, audio: bool) -> None:
    width, height = FINAL_DIMS[aspect_ratio]
    source_width, source_height = _video_dimensions(source)
    logger.info(
        "video_normalization source=%s source_dimensions=%sx%s target_dimensions=%sx%s",
        source.name,
        source_width,
        source_height,
        width,
        height,
    )
    cmd = [
        "ffmpeg", "-y", "-i", str(source),
        "-vf",
        (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1"
        ),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    ]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    cmd.append(str(target))
    _run_ffmpeg(cmd, "720p normalization")


def _extract_last_frame(source: Path, target: Path) -> None:
    _run_ffmpeg(
        [
            "ffmpeg", "-y", "-sseof", "-0.12", "-i", str(source),
            "-frames:v", "1", str(target),
        ],
        "Last-frame extraction",
    )


def _concat_segments(segments: list[Path], target: Path, audio: bool) -> None:
    if not segments:
        raise RuntimeError("No generated segments were available for final assembly")
    if len(segments) == 1:
        shutil.copy2(segments[0], target)
        return

    manifest = target.with_suffix(".txt")
    try:
        manifest.write_text(
            "\n".join(f"file '{path.as_posix()}'" for path in segments),
            encoding="utf-8",
        )
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(manifest),
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        ]
        if audio:
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        else:
            cmd += ["-an"]
        cmd.append(str(target))
        _run_ffmpeg(cmd, "Segment concatenation")
    finally:
        manifest.unlink(missing_ok=True)


def _probe(path: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration,size",
                "-of", "json", str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        fmt = payload.get("format", {})
        return {
            "duration": round(float(fmt.get("duration", 0)), 2),
            "size_bytes": int(fmt.get("size", path.stat().st_size)),
        }
    except Exception as exc:
        raise RuntimeError(f"Could not inspect final video: {type(exc).__name__}: {exc}") from None


def _fit_transfer_guard(path: Path, audio: bool) -> dict[str, Any]:
    stats = _probe(path)
    if stats["size_bytes"] <= TRANSFER_LIMIT_BYTES:
        return stats

    duration = max(float(stats["duration"]), 1.0)
    audio_bps = 192_000 if audio else 0
    total_bps = int((TRANSFER_TARGET_BYTES * 8) / duration)
    video_bps = max(600_000, total_bps - audio_bps)
    compressed = path.with_name(path.stem + "_compressed.mp4")

    cmd = [
        "ffmpeg", "-y", "-i", str(path),
        "-c:v", "libx264", "-preset", "medium",
        "-b:v", str(video_bps), "-maxrate", str(video_bps),
        "-bufsize", str(video_bps * 2),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    ]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    cmd.append(str(compressed))

    try:
        _run_ffmpeg(cmd, "Transfer-size compression")
        compressed_stats = _probe(compressed)
        if compressed_stats["size_bytes"] < stats["size_bytes"]:
            compressed.replace(path)
            stats = compressed_stats
    finally:
        compressed.unlink(missing_ok=True)

    if stats["size_bytes"] > TRANSFER_LIMIT_BYTES:
        raise RuntimeError(
            "Final video is still larger than the 95 MB transfer limit after automatic compression"
        )
    return stats


def _prune_local_results(max_age_hours: int = 2) -> None:
    cutoff = time.time() - max_age_hours * 3600
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for path in RESULT_DIR.glob("*.mp4"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            pass


@app.cls(
    image=gpu_image,
    gpu="L40S",
    cpu=4,
    memory=32768,
    timeout=3600,
    startup_timeout=900,
    scaledown_window=75,
    max_containers=1,
    volumes={RESULT_DIR: result_volume},
)
@modal.concurrent(max_inputs=1)
class LTXWorker:
    @modal.enter()
    def start_comfy(self) -> None:
        COMFY_LOG.unlink(missing_ok=True)
        self._comfy_log_handle = COMFY_LOG.open("a", encoding="utf-8", buffering=1)
        self.proc = subprocess.Popen(
            [
                "python", str(COMFY_DIR / "main.py"),
                "--listen", "127.0.0.1", "--port", "8188", "--disable-auto-launch",
            ],
            cwd=str(COMFY_DIR),
            stdout=self._comfy_log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _wait_for_comfy()

    @modal.method()
    def generate(
        self,
        *,
        job_id: str,
        mode: Literal["image", "text"],
        prompt: str,
        image_bytes: bytes | None,
        image_suffix: str,
        duration: int,
        aspect_ratio: Literal["16:9", "9:16"],
        seed: int,
        camera_motion: str,
        prompt_enhance: bool,
        generate_audio: bool,
    ) -> dict[str, Any]:
        try:
            return self._generate_impl(
                job_id=job_id,
                mode=mode,
                prompt=prompt,
                image_bytes=image_bytes,
                image_suffix=image_suffix,
                duration=duration,
                aspect_ratio=aspect_ratio,
                seed=seed,
                camera_motion=camera_motion,
                prompt_enhance=prompt_enhance,
                generate_audio=generate_audio,
            )
        except (RuntimeError, ValueError, TimeoutError) as exc:
            logger.exception(
                "job_failed job_id=%s error_type=%s", job_id, type(exc).__name__
            )
            raise
        except Exception as exc:
            logger.exception(
                "job_failed job_id=%s error_type=%s", job_id, type(exc).__name__
            )
            raise RuntimeError(
                f"Unexpected worker error: {type(exc).__name__}: {_clip_text(exc, 4000)}"
            ) from None

    def _generate_impl(
        self,
        *,
        job_id: str,
        mode: Literal["image", "text"],
        prompt: str,
        image_bytes: bytes | None,
        image_suffix: str,
        duration: int,
        aspect_ratio: Literal["16:9", "9:16"],
        seed: int,
        camera_motion: str,
        prompt_enhance: bool,
        generate_audio: bool,
    ) -> dict[str, Any]:
        _comfy_process_guard(self.proc)

        if duration not in ALLOWED_DURATIONS:
            raise ValueError("Unsupported duration")
        if aspect_ratio not in GEN_DIMS:
            raise ValueError("Unsupported aspect ratio")
        if camera_motion not in CAMERA_PROMPTS:
            raise ValueError("Unsupported camera motion")
        if mode == "image" and not image_bytes:
            raise ValueError("Image-to-video requires an image")
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"Prompt is too long; maximum is {MAX_PROMPT_CHARS} characters")

        try:
            result_volume.reload()
        except Exception:
            pass
        _prune_local_results()

        work = Path("/tmp") / f"ltx-{job_id}"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)

        cleanup_inputs: list[Path] = []
        source_image: Path | None = None
        started = time.time()

        logger.info(
            "job_started job_id=%s mode=%s duration=%s aspect_ratio=%s audio=%s",
            job_id,
            mode,
            duration,
            aspect_ratio,
            generate_audio,
        )

        try:
            base_prompt = prompt.strip()
            camera = CAMERA_PROMPTS[camera_motion]
            if camera:
                base_prompt = f"{base_prompt}\n\nCamera direction: {camera}"

            if image_bytes:
                suffix = image_suffix.lower()
                if suffix not in ALLOWED_IMAGE_SUFFIXES:
                    suffix = ".png"
                source_image = COMFY_DIR / "input" / f"app_{job_id}_source{suffix}"
                source_image.parent.mkdir(parents=True, exist_ok=True)
                source_image.write_bytes(image_bytes)
                cleanup_inputs.append(source_image)

            if duration <= 10:
                segment_lengths = [duration]
            else:
                segment_lengths = [10] * (duration // 10)
                if duration % 10:
                    segment_lengths.append(duration % 10)

            final_segments: list[Path] = []
            current_source = source_image

            for index, segment_duration in enumerate(segment_lengths):
                segment_started = time.time()
                segment_mode: Literal["image", "text"] = (
                    mode if index == 0 else "image"
                )
                logger.info(
                    "segment_started job_id=%s segment=%s/%s mode=%s duration=%s",
                    job_id,
                    index + 1,
                    len(segment_lengths),
                    segment_mode,
                    segment_duration,
                )
                if index > 0:
                    segment_prompt = (
                        "Continue seamlessly from the supplied first frame. Preserve the exact "
                        "subject identity, wardrobe, environment, lighting, lens character, "
                        "camera direction, motion momentum, and visual style from the previous "
                        "shot. Do not restart the scene. Continue the action naturally. "
                        + base_prompt
                    )
                else:
                    segment_prompt = base_prompt

                input_filename = current_source.name if current_source else None
                prefix_token = f"{job_id}_seg{index:02d}"
                output_prefix = f"app/{prefix_token}"
                segment_seed = (seed + index * 1009) % (2**63 - 1)
                if segment_seed <= 0:
                    segment_seed = 1

                workflow = _prepare_workflow(
                    mode=segment_mode,
                    prompt=segment_prompt,
                    prompt_enhance=prompt_enhance,
                    duration=segment_duration,
                    aspect_ratio=aspect_ratio,
                    seed=segment_seed,
                    input_filename=input_filename,
                    output_prefix=output_prefix,
                )

                generation_started = time.time()
                prompt_id, history_item = _convert_and_execute(
                    workflow,
                    proc=self.proc,
                    timeout_seconds=1200,
                )
                raw = _find_generated_video(history_item, generation_started, prefix_token)

                exact = work / f"segment_{index:02d}.mp4"
                _exact_720p(raw, exact, aspect_ratio, generate_audio)
                final_segments.append(exact)
                logger.info(
                    "segment_completed job_id=%s segment=%s/%s prompt_id=%s elapsed_seconds=%.1f",
                    job_id,
                    index + 1,
                    len(segment_lengths),
                    prompt_id,
                    time.time() - segment_started,
                )

                if index < len(segment_lengths) - 1:
                    next_frame = (
                        COMFY_DIR
                        / "input"
                        / f"app_{job_id}_continuation_{index:02d}.png"
                    )
                    _extract_last_frame(exact, next_frame)
                    cleanup_inputs.append(next_frame)
                    current_source = next_frame

            final_path = RESULT_DIR / f"{job_id}.mp4"
            _concat_segments(final_segments, final_path, generate_audio)
            stats = _fit_transfer_guard(final_path, generate_audio)
            result_volume.commit()

            logger.info(
                "job_completed job_id=%s segments=%s elapsed_seconds=%.1f size_bytes=%s",
                job_id,
                len(final_segments),
                time.time() - started,
                stats["size_bytes"],
            )

            return {
                "status": "completed",
                "job_id": job_id,
                "filename": final_path.name,
                "duration": stats["duration"],
                "size_bytes": stats["size_bytes"],
                "elapsed_seconds": round(time.time() - started, 1),
                "segments": len(final_segments),
                "seed": seed,
                "resolution": "1280x720" if aspect_ratio == "16:9" else "720x1280",
                "fps": FRAME_RATE,
                "audio": generate_audio,
                "backend_version": BACKEND_VERSION,
            }
        finally:
            shutil.rmtree(work, ignore_errors=True)
            for path in cleanup_inputs:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass


web = FastAPI(title="LTX Motion Studio", version=BACKEND_VERSION)


@web.middleware("http")
async def no_cache_api(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response


@web.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "app": APP_NAME,
        "backend_version": BACKEND_VERSION,
        "gpu": "L40S",
        "model": "LTX-2.5 distilled INT8",
        "comfyui": COMFYUI_VERSION,
        "workflow_template_commit": WORKFLOW_TEMPLATE_COMMIT[:12],
        "converter_commit": CONVERTER_COMMIT[:12],
    }


@web.get("/api/config")
def config() -> dict[str, Any]:
    return {
        "durations": sorted(ALLOWED_DURATIONS),
        "native_segment_max_seconds": 10,
        "resolution": "720p",
        "fps": FRAME_RATE,
        "aspects": list(GEN_DIMS.keys()),
        "camera_motions": list(CAMERA_PROMPTS.keys()),
        "monthly_free_credit_usd": 30,
        "gross_usage_cap_usd": 32,
        "gpu_hourly_usd": 1.9512,
        "estimated_seconds_per_10s_clip": 360,
        "backend_version": BACKEND_VERSION,
    }


def _validate_image_bytes(image_bytes: bytes, filename: str) -> str:
    from PIL import Image

    if not image_bytes:
        raise HTTPException(400, "Uploaded image is empty")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image must be 20 MB or smaller")

    suffix = Path(filename or "image.png").suffix.lower() or ".png"
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise HTTPException(400, "Image must be PNG, JPG, JPEG, or WEBP")

    try:
        with Image.open(BytesIO(image_bytes)) as img:
            img.verify()
    except Exception:
        raise HTTPException(400, "Uploaded file is not a valid supported image") from None

    return suffix


@web.post("/api/generate")
async def generate(
    mode: Literal["image", "text"] = Form(...),
    prompt: str = Form(...),
    duration: int = Form(10),
    aspect_ratio: Literal["16:9", "9:16"] = Form("16:9"),
    seed: int = Form(-1),
    camera_motion: str = Form("auto"),
    prompt_enhance: bool = Form(True),
    generate_audio: bool = Form(True),
    image: UploadFile | None = File(None),
) -> dict[str, Any]:
    prompt = prompt.strip()
    if duration not in ALLOWED_DURATIONS:
        raise HTTPException(400, "Unsupported duration")
    if aspect_ratio not in GEN_DIMS:
        raise HTTPException(400, "Unsupported aspect ratio")
    if camera_motion not in CAMERA_PROMPTS:
        raise HTTPException(400, "Unsupported camera motion")
    if not prompt:
        raise HTTPException(400, "Prompt is required")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(400, f"Prompt may not exceed {MAX_PROMPT_CHARS} characters")
    if mode == "image" and image is None:
        raise HTTPException(400, "Image-to-video requires an image")

    image_bytes: bytes | None = None
    image_suffix = ".png"
    if image is not None:
        image_bytes = await image.read()
        image_suffix = _validate_image_bytes(image_bytes, image.filename or "image.png")

    resolved_seed = (
        seed if seed >= 0 else random.SystemRandom().randint(1, 2**31 - 1)
    )
    if resolved_seed == 0:
        resolved_seed = 1

    job_id = uuid.uuid4().hex
    call = await LTXWorker().generate.spawn.aio(
        job_id=job_id,
        mode=mode,
        prompt=prompt,
        image_bytes=image_bytes,
        image_suffix=image_suffix,
        duration=duration,
        aspect_ratio=aspect_ratio,
        seed=resolved_seed,
        camera_motion=camera_motion,
        prompt_enhance=prompt_enhance,
        generate_audio=generate_audio,
    )

    segments = 1 if duration <= 10 else (duration + 9) // 10
    return {
        "status": "accepted",
        "call_id": call.object_id,
        "job_id": job_id,
        "seed": resolved_seed,
        "segments": segments,
        "estimated_seconds": segments * 360,
        "backend_version": BACKEND_VERSION,
    }


@web.get("/api/jobs/{call_id}")
def job_status(call_id: str) -> Any:
    fc = modal.FunctionCall.from_id(call_id)
    try:
        return fc.get(timeout=0)
    except TimeoutError:
        from fastapi.responses import JSONResponse

        return JSONResponse({"status": "running"}, status_code=202)
    except Exception as exc:
        raise HTTPException(
            500,
            f"Generation failed: {_clip_text(f'{type(exc).__name__}: {exc}', 5000)}",
        ) from None


@web.post("/api/jobs/{call_id}/cancel")
def cancel_job(call_id: str) -> dict[str, str]:
    fc = modal.FunctionCall.from_id(call_id)
    try:
        fc.cancel(terminate_containers=True)
    except Exception as exc:
        raise HTTPException(500, f"Could not cancel job: {_clip_text(exc, 1500)}") from None
    return {"status": "cancelled"}


@web.get("/api/download/{job_id}")
def download(job_id: str) -> FileResponse:
    if len(job_id) != 32 or not all(c in "0123456789abcdefABCDEF" for c in job_id):
        raise HTTPException(400, "Invalid job id")

    result_volume.reload()
    path = RESULT_DIR / f"{job_id}.mp4"
    if not path.exists():
        raise HTTPException(404, "Result not found or already expired")

    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"ltx-{job_id[:8]}.mp4",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@web.delete("/api/results/{job_id}")
def delete_result(job_id: str) -> dict[str, str]:
    if len(job_id) != 32 or not all(c in "0123456789abcdefABCDEF" for c in job_id):
        raise HTTPException(400, "Invalid job id")

    result_volume.reload()
    path = RESULT_DIR / f"{job_id}.mp4"
    path.unlink(missing_ok=True)
    result_volume.commit()
    return {"status": "deleted"}


web.mount(
    "/",
    StaticFiles(directory="/web", html=True, check_dir=False),
    name="web",
)


@app.function(
    image=web_image,
    volumes={RESULT_DIR: result_volume},
    timeout=300,
)
@modal.asgi_app()
def frontend() -> FastAPI:
    return web
