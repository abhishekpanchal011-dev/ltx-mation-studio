# LTX Motion Studio v2 - Android/PWA deployment

## What you need
- Modal account
- Hugging Face account with access accepted for `Lightricks/LTX-2.5`
- Hugging Face **Read** token

## 1. Create the Modal secret (browser only)
In Modal Dashboard -> Secrets -> Create Secret -> Hugging Face:
- Secret name: `huggingface-secret`
- Key: `HF_TOKEN`
- Value: your Hugging Face read token

Never put the token in source code or share it in chat.

## 2. Deploy once
Run these commands from the project root on any machine with Python. This is only a deployment step; normal use is Android-only afterwards.

```bash
python -m pip install -r requirements-local.txt
modal setup
modal deploy backend/modal_app.py
```

Run `modal deploy` from the folder that contains `backend/` and `web/` so Modal can include the PWA assets in the web image.

At the end Modal prints the public URL for the `frontend` web function.

## 3. Use on Android
1. Open the Modal `frontend` URL in Chrome on your Android phone/tablet.
2. Tap **Install app** when offered, or Chrome menu -> **Add to Home screen / Install app**.
3. Open LTX Motion from the Android home screen.
4. Choose Image to Video or Text to Video, upload from Gallery or Camera, and Generate.
5. You can leave/reopen the app while generation is running. The active job ID is restored from browser storage and the app resumes polling.
6. Preview and download the MP4 to the Android device.

## Budget
The UI keeps a local gross-usage guard at about $32/month. Modal Starter credits are expected to cover the first $30 of eligible usage. Also configure Modal's workspace billing limit in the dashboard; when a net-after-credits limit is available, use $2 paid spend.

## Runtime model pack
The image build downloads only the current optimized LTX 2.5 set:
- distilled INT8 ConvRot transformer
- Gemma 4 12B LTX projection INT8 ConvRot text encoder
- video VAE
- audio VAE
- x2 latent spatial upscaler
- prompt enhancer model

The model weights are baked into the Modal image at deployment/build time. Google Drive is not required.

## Output retention
Generated MP4s are temporary. The backend prunes old results aggressively; download completed outputs to Android. This is not intended to be a cloud video library.
