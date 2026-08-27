from __future__ import annotations

import copy
import json
import os
import random
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import modal

APP_NAME = "ltx-motion-studio"
COMFY_DIR = Path("/opt/ComfyUI")
WORKFLOW_DIR = Path("/opt/workflows")
RESULT_DIR = Path("/results")
COMFY_URL = "http://127.0.0.1:8188"

# Current optimized model pack used by the ComfyUI LTX-2.5 templates.
LTX_REPO = "Lightricks/LTX-2.5"
MODEL_FILES = [
    "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "vae/ltx-2.5-video-vae-bf16.safetensors",
    "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
]
PROMPT_ENHANCER_REPO = "Comfy-Org/gemma-4"
PROMPT_ENHANCER_FILE = "text_encoders/gemma4_e2b_it_bf16.safetensors"

MODEL_NAMES = {
    "unet": "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    "clip": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "video_vae": "ltx-2.5-video-vae-bf16.safetensors",
    "audio_vae": "ltx-2.5-audio-vae-bf16.safetensors",
    "upscaler": "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    "enhancer": "gemma4_e2b_it_bf16.safetensors",
}

# 1280x720 is presented to the user. The open ComfyUI graph works most cleanly
# with multiples of 32, so generation uses 1280x736 / 736x1280 and FFmpeg crops
# the final 16 pixels to exact 720p.
GEN_DIMS = {
    "16:9": (1280, 736),
    "9:16": (736, 1280),
}
FINAL_DIMS = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
}

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


def _download_models() -> None:
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
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

# Pin ComfyUI to a stable LTX-2.5-capable release. comfy-kitchen is explicitly
# upgraded because the INT8 ConvRot checkpoints require newer loader support.
gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("git", "ffmpeg", "curl", "libgl1", "libglib2.0-0")
    .run_commands(
        "git clone --depth 1 --branch v0.33.1 https://github.com/Comfy-Org/ComfyUI.git /opt/ComfyUI",
        "python -m pip install --upgrade pip",
        "python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio",
        "python -m pip install -r /opt/ComfyUI/requirements.txt",
        "python -m pip install 'comfy-kitchen>=0.2.26' requests huggingface_hub hf_xet",
        "git clone --depth 1 https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint.git /opt/ComfyUI/custom_nodes/workflow-to-api-converter",
        "mkdir -p /opt/workflows",
        "curl -fsSL https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_ltx2_5_i2v.json -o /opt/workflows/video_ltx2_5_i2v.json",
        "curl -fsSL https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_ltx2_5_t2v.json -o /opt/workflows/video_ltx2_5_t2v.json",
    )
    .run_function(_download_models, secrets=[hf_secret], timeout=3600)
)

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]", "python-multipart")
    .add_local_dir("web", remote_path="/web")
)

app = modal.App(APP_NAME)
result_volume = modal.Volume.from_name("ltx-motion-results", create_if_missing=True)


def _wait_for_comfy(timeout_seconds: int = 180) -> None:
    import requests

    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{COMFY_URL}/system_stats", timeout=3)
            if r.ok:
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"ComfyUI did not become ready: {last_error}")


def _find_main_subgraph(workflow: dict[str, Any], expected_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    definitions = workflow.get("definitions", {}).get("subgraphs", [])
    subgraph_def = next((d for d in definitions if d.get("name") == expected_name), None)
    if not subgraph_def:
        raise RuntimeError(f"Subgraph not found: {expected_name}")
    subgraph_id = subgraph_def["id"]
    outer_node = next((n for n in workflow.get("nodes", []) if n.get("type") == subgraph_id), None)
    if not outer_node:
        raise RuntimeError(f"Outer node not found for subgraph: {expected_name}")
    return outer_node, subgraph_def


def _widget_index_map(subgraph_def: dict[str, Any]) -> dict[str, int]:
    # The outer node's widgets_values follows the subgraph's scalar/combo inputs.
    scalar_types = {"STRING", "BOOLEAN", "INT", "FLOAT", "COMBO"}
    labels: list[str] = []
    for item in subgraph_def.get("inputs", []):
        if item.get("type") not in scalar_types:
            continue
        labels.append(item.get("label") or item.get("name"))
    return {name: i for i, name in enumerate(labels)}


def _disconnect_link(workflow: dict[str, Any], link_id: int | None) -> None:
    if link_id is None:
        return
    workflow["links"] = [link for link in workflow.get("links", []) if link[0] != link_id]
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
        raise RuntimeError(f"Template parameter missing: {label}")

    # If the parameter is linked from a helper node (e.g. ResolutionSelector),
    # detach it so the promoted widget value becomes authoritative.
    for inp in outer_node.get("inputs", []) or []:
        if (inp.get("label") or inp.get("name")) == label:
            _disconnect_link(workflow, inp.get("link"))
            inp["link"] = None

    values = outer_node.setdefault("widgets_values", [])
    idx = index_map[label]
    if idx >= len(values):
        raise RuntimeError(f"Template widgets_values mismatch for {label}")
    values[idx] = value


def _patch_save_prefix(workflow: dict[str, Any], prefix: str) -> None:
    for node in workflow.get("nodes", []):
        if node.get("type") == "SaveVideo":
            values = node.setdefault("widgets_values", [])
            if values:
                values[0] = prefix
            else:
                node["widgets_values"] = [prefix, "auto", "auto"]


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
    template_name = "video_ltx2_5_i2v.json" if mode == "image" else "video_ltx2_5_t2v.json"
    expected_name = "Image to Video (LTX-2.5)" if mode == "image" else "Text to Video (LTX-2.5)"
    workflow = json.loads((WORKFLOW_DIR / template_name).read_text(encoding="utf-8"))
    workflow = copy.deepcopy(workflow)
    outer, definition = _find_main_subgraph(workflow, expected_name)

    width, height = GEN_DIMS[aspect_ratio]
    for label, value in (
        ("prompt", prompt),
        ("prompt_enhance", bool(prompt_enhance)),
        ("duration", int(duration)),
        ("width", width),
        ("height", height),
        ("seed", int(seed)),
        ("frame_rate", 24),
        ("unet_name", MODEL_NAMES["unet"]),
        ("video_vae", MODEL_NAMES["video_vae"]),
        ("audio_vae", MODEL_NAMES["audio_vae"]),
        ("clip_name", MODEL_NAMES["clip"]),
        ("upscale_model", MODEL_NAMES["upscaler"]),
        ("prompt_enhance_model", MODEL_NAMES["enhancer"]),
    ):
        _set_outer_value(workflow, outer, definition, label, value)

    if mode == "image":
        if not input_filename:
            raise ValueError("Image-to-video requires an input image")
        load_node = next(
            (
                node
                for node in workflow.get("nodes", [])
                if node.get("type") == "LoadImage" and "First Frame" in str(node.get("title", ""))
            ),
            None,
        )
        if not load_node:
            raise RuntimeError("Load First Frame node was not found in the I2V template")
        load_node["widgets_values"][0] = input_filename

    _patch_save_prefix(workflow, output_prefix)
    return workflow


def _convert_and_execute(workflow: dict[str, Any], timeout_seconds: int = 900) -> None:
    import requests

    converted = requests.post(f"{COMFY_URL}/workflow/convert", json=workflow, timeout=60)
    converted.raise_for_status()
    api_workflow = converted.json()

    queued = requests.post(f"{COMFY_URL}/prompt", json={"prompt": api_workflow}, timeout=30)
    queued.raise_for_status()
    payload = queued.json()
    if payload.get("node_errors"):
        raise RuntimeError(f"ComfyUI workflow validation failed: {payload['node_errors']}")
    prompt_id = payload["prompt_id"]

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        history = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=15).json()
        item = history.get(prompt_id)
        if item:
            status = item.get("status", {})
            messages = status.get("messages", [])
            for message in messages:
                if isinstance(message, list) and message and message[0] == "execution_error":
                    raise RuntimeError(f"ComfyUI execution error: {message[1]}")
            if status.get("completed"):
                return
        time.sleep(3)
    raise TimeoutError(f"ComfyUI generation exceeded {timeout_seconds}s")


def _newest_video(after_ts: float, prefix_token: str) -> Path:
    candidates = [
        p
        for p in (COMFY_DIR / "output").rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".mp4", ".webm", ".mov"}
        and p.stat().st_mtime >= after_ts
        and prefix_token in str(p)
    ]
    if not candidates:
        # Fallback for SaveVideo builds that sanitize the prefix path.
        candidates = [
            p
            for p in (COMFY_DIR / "output").rglob("*")
            if p.is_file() and p.suffix.lower() in {".mp4", ".webm", ".mov"} and p.stat().st_mtime >= after_ts
        ]
    if not candidates:
        raise RuntimeError("ComfyUI completed but no output video was found")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _exact_720p(source: Path, target: Path, aspect_ratio: str, audio: bool) -> None:
    width, height = FINAL_DIMS[aspect_ratio]
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vf",
        f"crop={width}:{height}",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    cmd.append(str(target))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _extract_last_frame(source: Path, target: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-sseof",
            "-0.12",
            "-i",
            str(source),
            "-frames:v",
            "1",
            str(target),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _concat_segments(segments: list[Path], target: Path, audio: bool) -> None:
    if len(segments) == 1:
        shutil.copy2(segments[0], target)
        return

    manifest = target.with_suffix(".txt")
    manifest.write_text("\n".join(f"file '{p.as_posix()}'" for p in segments), encoding="utf-8")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(manifest),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    cmd.append(str(target))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    manifest.unlink(missing_ok=True)


def _probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
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


def _prune_local_results(max_age_hours: int = 2) -> None:
    cutoff = time.time() - max_age_hours * 3600
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for p in RESULT_DIR.glob("*.mp4"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
        except OSError:
            pass


@app.cls(
    image=gpu_image,
    gpu="L40S",
    cpu=4,
    memory=32768,
    timeout=3600,
    startup_timeout=600,
    scaledown_window=75,
    max_containers=1,
    volumes={RESULT_DIR: result_volume},
)
@modal.concurrent(max_inputs=1)
class LTXWorker:
    @modal.enter()
    def start_comfy(self) -> None:
        self.proc = subprocess.Popen(
            [
                "python",
                str(COMFY_DIR / "main.py"),
                "--listen",
                "127.0.0.1",
                "--port",
                "8188",
                "--disable-auto-launch",
            ],
            cwd=str(COMFY_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
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
        if duration not in {5, 10, 20, 30, 60}:
            raise ValueError("Duration must be 5, 10, 20, 30, or 60 seconds")
        if aspect_ratio not in GEN_DIMS:
            raise ValueError("Unsupported aspect ratio")
        if mode == "image" and not image_bytes:
            raise ValueError("Image-to-video requires an image")
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        _prune_local_results()
        work = Path("/tmp") / f"ltx-{job_id}"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)

        base_prompt = prompt.strip()
        camera = CAMERA_PROMPTS.get(camera_motion, "")
        if camera:
            base_prompt = f"{base_prompt}\n\nCamera direction: {camera}"

        source_image: Path | None = None
        if image_bytes:
            suffix = image_suffix if image_suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} else ".png"
            source_image = COMFY_DIR / "input" / f"app_{job_id}_source{suffix}"
            source_image.write_bytes(image_bytes)

        segment_lengths: list[int]
        if duration <= 10:
            segment_lengths = [duration]
        else:
            segment_lengths = [10] * (duration // 10)
            if duration % 10:
                segment_lengths.append(duration % 10)

        final_segments: list[Path] = []
        current_source = source_image
        first_mode = mode
        started = time.time()

        for idx, segment_duration in enumerate(segment_lengths):
            segment_mode: Literal["image", "text"] = first_mode if idx == 0 else "image"
            if idx > 0:
                segment_prompt = (
                    "Continue seamlessly from the supplied first frame. Preserve the exact subject identity, "
                    "wardrobe, environment, lighting, lens character, camera direction, motion momentum, and "
                    "visual style from the previous shot. Do not restart the scene. Continue the action naturally. "
                    + base_prompt
                )
            else:
                segment_prompt = base_prompt

            input_filename = current_source.name if current_source else None
            prefix_token = f"{job_id}_seg{idx:02d}"
            output_prefix = f"app/{prefix_token}"
            workflow = _prepare_workflow(
                mode=segment_mode,
                prompt=segment_prompt,
                prompt_enhance=prompt_enhance,
                duration=segment_duration,
                aspect_ratio=aspect_ratio,
                seed=seed + idx * 1009,
                input_filename=input_filename,
                output_prefix=output_prefix,
            )

            generation_started = time.time()
            _convert_and_execute(workflow, timeout_seconds=900)
            raw = _newest_video(generation_started, prefix_token)
            exact = work / f"segment_{idx:02d}.mp4"
            _exact_720p(raw, exact, aspect_ratio, generate_audio)
            final_segments.append(exact)

            if idx < len(segment_lengths) - 1:
                next_frame = COMFY_DIR / "input" / f"app_{job_id}_continuation_{idx:02d}.png"
                _extract_last_frame(exact, next_frame)
                current_source = next_frame

        final_path = RESULT_DIR / f"{job_id}.mp4"
        _concat_segments(final_segments, final_path, generate_audio)
        stats = _probe(final_path)
        if stats["size_bytes"] > 95 * 1024 * 1024:
            raise RuntimeError(
                "Final video exceeded the 95 MB transfer guard. Reduce duration or lower bitrate before downloading."
            )
        result_volume.commit()

        elapsed = round(time.time() - started, 1)
        return {
            "status": "completed",
            "job_id": job_id,
            "filename": final_path.name,
            "duration": stats["duration"],
            "size_bytes": stats["size_bytes"],
            "elapsed_seconds": elapsed,
            "segments": len(final_segments),
            "seed": seed,
            "resolution": "1280x720" if aspect_ratio == "16:9" else "720x1280",
            "fps": 24,
            "audio": generate_audio,
        }


# -------- Web / API layer (CPU only) --------
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

web = FastAPI(title="LTX Motion Studio", version="2.0.0")


@web.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "app": APP_NAME, "gpu": "L40S", "model": "LTX-2.5 distilled INT8"}


@web.get("/api/config")
def config() -> dict[str, Any]:
    return {
        "durations": [5, 10, 20, 30, 60],
        "native_segment_max_seconds": 10,
        "resolution": "720p",
        "fps": 24,
        "aspects": ["16:9", "9:16"],
        "camera_motions": list(CAMERA_PROMPTS.keys()),
        "monthly_free_credit_usd": 30,
        "gross_usage_cap_usd": 32,
        "gpu_hourly_usd": 1.9512,
        "estimated_seconds_per_10s_clip": 360,
    }


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
    if duration not in {5, 10, 20, 30, 60}:
        raise HTTPException(400, "Unsupported duration")
    if mode == "image" and image is None:
        raise HTTPException(400, "Image-to-video requires an image")
    if not prompt.strip():
        raise HTTPException(400, "Prompt is required")

    image_bytes: bytes | None = None
    image_suffix = ".png"
    if image is not None:
        image_bytes = await image.read()
        if len(image_bytes) > 20 * 1024 * 1024:
            raise HTTPException(413, "Image must be 20 MB or smaller")
        image_suffix = Path(image.filename or "image.png").suffix or ".png"

    resolved_seed = seed if seed >= 0 else random.SystemRandom().randint(1, 2**31 - 1)
    job_id = uuid.uuid4().hex
    call = LTXWorker().generate.spawn(
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
    }


@web.get("/api/jobs/{call_id}")
def job_status(call_id: str) -> Any:
    fc = modal.FunctionCall.from_id(call_id)
    try:
        return fc.get(timeout=0)
    except TimeoutError:
        from fastapi.responses import JSONResponse

        return JSONResponse({"status": "running"}, status_code=202)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Generation failed: {exc}") from exc


@web.post("/api/jobs/{call_id}/cancel")
def cancel_job(call_id: str) -> dict[str, str]:
    fc = modal.FunctionCall.from_id(call_id)
    fc.cancel(terminate_containers=True)
    return {"status": "cancelled"}


@web.get("/api/download/{job_id}")
def download(job_id: str) -> FileResponse:
    if not job_id.replace("-", "").isalnum():
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
    result_volume.reload()
    path = RESULT_DIR / f"{job_id}.mp4"
    path.unlink(missing_ok=True)
    result_volume.commit()
    return {"status": "deleted"}


web.mount("/", StaticFiles(directory="/web", html=True), name="web")


@app.function(image=web_image, volumes={RESULT_DIR: result_volume}, timeout=300)
@modal.asgi_app()
def frontend() -> FastAPI:
    return web
