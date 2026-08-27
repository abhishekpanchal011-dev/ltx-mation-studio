# LTX Motion Studio v2 (Android-first PWA)

For the current Android/tablet deployment path, start with **ANDROID_DEPLOY.md**.

# LTX Motion Studio

A generation-first, Kling-style web application built around the **open-source LTX-2.5 distilled ComfyUI workflow**, deployed serverlessly on **Modal L40S 48 GB**.

## What this build does

- Image-to-video and text-to-video
- 720p output (1280×720 or 720×1280), 24 fps
- Native 5s and 10s clips
- 20s / 30s / 60s jobs generated as linked 10-second segments
- Automatic continuation using the previous segment's last frame
- Automatic FFmpeg stitching and exact 720p crop
- Prompt enhancement using Gemma 4
- Camera-motion presets implemented as prompt direction
- Synchronized LTX audio, with silent export option
- Random or fixed seed
- Background Modal jobs with polling, cancel, progress and direct download
- Browser-only history (no database)
- Results retained in a tiny transient Modal Volume and pruned after ~24 hours
- Main model weights baked into the Modal Image: **no paid model Volume required**

## Model pack

The image build downloads these files:

1. `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors`
2. `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors`
3. `ltx-2.5-video-vae-bf16.safetensors`
4. `ltx-2.5-audio-vae-bf16.safetensors`
5. `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`
6. `gemma4_e2b_it_bf16.safetensors` (prompt enhancer)

The first five come from `Lightricks/LTX-2.5`; the prompt enhancer comes from `Comfy-Org/gemma-4`.

## 1. Prepare Hugging Face

1. Accept the LTX-2.5 model terms on Hugging Face.
2. Create a Hugging Face **read** token that can access gated repositories.

## 2. Install Modal locally

```bash
pip install -r requirements-local.txt
modal setup
```

## 3. Add the Hugging Face secret to Modal

```bash
modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
```

Do not put the token in source code.

## 4. Set the cost guard

In **Modal → Usage & Billing**, set the **Workspace budget to $32/month** now. The current budget is measured before credits, so this pairs with the $30 Starter credit to cap gross usage near the user's requested ceiling.

Starting **September 1, 2026**, Modal supports a separate **net Workspace spend limit**. Set that to **$2/month** when it becomes available.

The UI also keeps a local estimated-usage guard, but the Modal billing setting is the real hard stop.

## 5. Deploy

From this directory:

```bash
modal deploy backend/modal_app.py
```

The first deployment is intentionally large because Modal builds the ComfyUI image and downloads the model files. Subsequent deployments reuse cached image layers unless a model-layer dependency changes.

At the end, Modal prints the public URL for the `frontend` web function. Open it in a browser.

## How the generation path works

### 5s / 10s

`UI → FastAPI → Modal job → L40S → ComfyUI LTX-2.5 template → FFmpeg crop → result`

### 20s / 30s / 60s

`segment 1 → extract last frame → image-to-video continuation → repeat → FFmpeg concat`

Every continuation reuses the master prompt plus an explicit identity/environment/lighting/camera continuity instruction. Segment seeds are deterministic offsets from the original seed.

## Why 1280×736 is generated internally

The open ComfyUI LTX graph is happiest with dimensions divisible by 32. The app therefore generates **1280×736** (or 736×1280) and crops the final 16 pixels with FFmpeg to return exact **1280×720** (or 720×1280).

## Expected cost

With the user's measured expectation of roughly **5–6 minutes for a 10-second clip**, L40S GPU compute is roughly ~$0.20 per 10-second segment at current Modal pricing, before small CPU/RAM overhead. At 1–2 10-second clips per day, the $30 monthly Starter credit should have substantial headroom.

Longer clips cost approximately in proportion to the number of 10-second segments because this MVP prioritizes continuity and reliability over trying to force a single very long diffusion pass.

## Important MVP limitation

Cross-segment **visual** continuity is actively managed using the prior segment's last frame. Audio is generated per segment and concatenated. This is good enough for the first build, but truly continuous dialogue/music across long clips should become a dedicated Phase-2 audio track or audio-conditioning workflow rather than relying on independent segment audio.

## Phase 2

- Character/project library
- First-frame + last-frame editor
- Storyboard / shot timeline
- Native multi-shot controls
- Video-to-video / Retake
- Prompt templates and reusable styles
- Proper auth
- Persistent project history
- 4K temporal video enhancement
- Better cross-segment audio continuity
- Native LTX Python pipeline migration once it clearly outperforms the current ComfyUI INT8 path for this hardware
