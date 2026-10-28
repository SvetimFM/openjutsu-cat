# Openjutsu, but the performer is a cat

A cat walks in front of a Kinect. Its **depth performance** — the raw infrared
silhouette, no RGB, no tracking skeleton — is fed as a control video to
Alibaba's **Wan2.2-Fun-5B-Control** inside ComfyUI, which re-renders the
performance as anything the prompt describes: a spirit cat of blue light in a
dark forest, a tiny knight, a creature of living stained glass. The motion is
the cat's. The costume is the model's.

Built for the **Comfy "Open Source Higgsfield"** call (Oct 2026). Everything
runs on open weights, entirely locally, on a single consumer GPU with 8 GB of
VRAM (an RTX 3070 Ti) — no cloud generation, no partner APIs.

<!-- DEMO: link / GIF goes here once the ≤2 min demo video is cut. -->

## Why this works on 8 GB

The control variant of Wan 2.2 ships as a 5B model, and QuantStack's GGUF
quantization (Q5_K_M, ~3.8 GB) plus ComfyUI-GGUF's loader brings the whole
pipeline — control model, UMT5-XXL text encoder (GGUF), Wan 2.2 VAE — inside
an 8 GB card with room to spare. Measured on the 3070 Ti:

| Benchmark | Settings | Result |
|---|---|---|
| Depth-controlled gen (the entry path) | 640×640, 49 frames, 20 steps, cfg 5.0 | **135 s/clip, ~6.4 GB peak VRAM** |
| Same, long take | 640×640, 97 frames, 20 steps, cfg 5.0 | **210 s/clip, no OOM** |
| Base text-to-video (TI2V-5B GGUF) | 832×480, 49 frames, 20 steps, cfg 5.0 | ~20 s/step (~14 min/clip), ~7.5 GB peak |
| LTX-Video 0.9.8 distilled (bf16) + T5 fp16 | 768×512, 49 frames, 20 steps, cfg 3.0 | **75 s/clip end to end** |

The depth input needs **no preprocessor**: `Wan22FunControlToVideo` takes the
depth mp4 directly (loaded with ComfyUI's core `LoadVideo` +
`GetVideoComponents` nodes). Near = bright, far = dark.

You don't need a Kinect to verify the path: two synthetic depth clips are in
`examples/` — run the workflow against one and the generated character follows
the blob's path.

## Pipeline

```
Kinect v2 depth (512×424) ──► grayscale mp4 (upscale to 640×640, 16 fps)
        │                              │
        │                    depth performance of a real cat
        ▼                              ▼
              Wan22FunControlToVideo (control_video input)
        UMT5-XXL (prompt) ──► Wan2.2-Fun-5B-Control GGUF ──► Wan 2.2 VAE ──► mp4
```

## Requirements

- NVIDIA GPU, 8 GB VRAM (developed on RTX 3070 Ti), 32 GB system RAM
- ComfyUI **v0.42.1** (the Windows portable build; developed on v0.42.0) + the
  [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) node pack;
  `pip install gguf sentencepiece protobuf` into ComfyUI's python. Everything
  else is a core ComfyUI node.
- Disk: **9.4 GB** of models for the cat pillar, **25.5 GB** for both pillars
  (table below) — on an **SSD**, and on a filesystem without a 4 GiB file-size
  cap: **FAT32 cannot hold the 9.8 GB T5 encoder or the 6.3 GB LTX
  checkpoint.** If your ComfyUI lives on a FAT32 volume, put the big files on
  an NTFS/exFAT volume and point ComfyUI at them with an `extra_model_paths.yaml`
  (start from `extra_model_paths.example.yaml`). Learned the hard way:
  downloads to the FAT32 volume failed for a day with "not enough space on the
  disk" while 844 GB sat free.
- The scripts here need only Python's standard library (ComfyUI's embedded
  python is fine); the capture tools need `requirements.txt` and ffmpeg.

## Model manifest — verify byte sizes before first run

Exact sizes and sha256 hashes as served by Hugging Face (re-checked
2026-10-27) are in `models.json`. **Check the byte count, not the file's
existence, and not the downloader's "OK" line** — in our own staging, one
9.8 GB encoder arrived truncated at 4.68 GB while the log printed OK.
`tools/verify_models.py` does the check for you (see below).

| File | Needed for | Exact bytes |
|---|---|---|
| **`models/unet/Wan2.2-Fun-5B-Control-Q5_K_M.gguf`** ([download](https://huggingface.co/QuantStack/Wan2.2-Fun-5B-Control-GGUF/resolve/main/Wan2.2-Fun-5B-Control-Q5_K_M.gguf)) | cat pillar (depth) | 3,821,521,760 |
| **`models/text_encoders/umt5-xxl-encoder-Q5_K_M.gguf`** ([download](https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/main/umt5-xxl-encoder-Q5_K_M.gguf)) | cat pillar (depth) | 4,145,878,880 |
| **`models/vae/Wan2.2_VAE.safetensors`** ([download](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/resolve/main/VAE/Wan2.2_VAE.safetensors)) | cat pillar (depth) | 1,409,400,960 |
| **`models/checkpoints/ltxv-2b-0.9.8-distilled.safetensors`** ([download](https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-2b-0.9.8-distilled.safetensors)) | cinema pillar (LTX) | 6,340,744,492 |
| **`models/text_encoders/t5xxl_fp16.safetensors`** ([download](https://huggingface.co/Comfy-Org/Mochi-Preview/resolve/main/split_files/text_encoders/t5xxl_fp16.safetensors)) | cinema pillar (LTX) | 9,787,841,024 |
| `models/unet/Wan2.2-TI2V-5B-Q5_K_M.gguf` ([download](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/resolve/main/Wan2.2-TI2V-5B-Q5_K_M.gguf)) | `run_local.py base` benchmark | 3,810,603,360 |
| `models/checkpoints/ltxv-2b-0.9.8-distilled-fp8.safetensors` ([download](https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-2b-0.9.8-distilled-fp8.safetensors)) | optional (smaller LTX) | 4,461,695,684 |
| `models/diffusion_models/wan2.1_fun_control_1.3B_bf16.safetensors` ([download](https://huggingface.co/Comfy-Org/Wan-2.1/resolve/main/split_files/diffusion_models/wan2.1_fun_control_1.3B_bf16.safetensors)) | fallback control path | 3,129,105,448 |
| `models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors` ([download](https://huggingface.co/Comfy-Org/Wan-2.1/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors)) | fallback control path | 6,735,906,897 |
| `models/vae/wan_2.1_vae.safetensors` ([download](https://huggingface.co/Comfy-Org/Wan-2.1/resolve/main/split_files/vae/wan_2.1_vae.safetensors)) | fallback control path | 253,815,318 |
| `models/clip_vision/clip_vision_h.safetensors` ([download](https://huggingface.co/Comfy-Org/Wan-2.1/resolve/main/split_files/clip_vision/clip_vision_h.safetensors)) | fallback control path | 1,264,219,396 |
| **Cat pillar only** | | **9,376,801,600** |
| **Both pillars** | | **25,505,387,116** |
| **Everything listed** | | **45,160,733,219** |

Naming traps: LTX-Video **0.9.8** is not LTX-2 (the 19–22B line does not fit
8 GB); the Wan 2.2 5B models use `Wan2.2_VAE.safetensors`, not
`wan_2.1_vae.safetensors` — crossing them produces noise.

## Run it locally

1. Install ComfyUI + ComfyUI-GGUF as above and download the bold models into
   the matching `models/` folders (or onto a second drive + `extra_model_paths.yaml`).
2. Check the install — exact sizes, in the order ComfyUI searches, then every
   node and model the workflows use against your running server:

   ```
   python tools/verify_models.py --comfy-root path/to/ComfyUI [--extra D:/comfy-models]
   python tools/validate_workflows.py --skip-base
   ```

3. Render. The control clip is uploaded for you and the result lands in `./out`:

   ```
   python run_local.py depth
   python run_local.py depth --video my-take.mp4 --frames 97 --prompt "A cat painted in black sumi-e ink brushstrokes walks across a blank white paper background, minimal, elegant, high contrast"
   python run_local.py cinema --prompt "The camera cranes upward from street level, revealing a coastal city at dusk"
   ```

   With no flags, `depth` renders `examples/test-depth-640x640-49f.mp4` with
   the exact graph in `workflow-depth-control-api.json`. `--frames` must be
   4n+1 and no longer than the control clip. Prefer the ComfyUI canvas? Drag
   the workflow JSON onto it — it is ComfyUI API format and loads directly.

## Pillar 2 — Cinema camera studio (LTX core)

Higgsfield's second flagship is cinematic camera control. This suite rebuilds
it on **LTX-Video 2B distilled (bf16) + T5-XXL fp16** — the LTX core running
the whole show, no Wan weights involved — using the same 8 GB card:

- `workflow-ltx-cinema-api.json` — the cinema graph (API format).
- `camera_slate.py` — renders six camera-move prompts on that graph
  (`python camera_slate.py` for all, `python camera_slate.py B D E` for some).
  The moves: **dolly-in** (on a cat, golden hour), **crane-up** (coastal city
  reveal), **orbit** (lighthouse), **side tracking** (cyclist), **FPV dive**
  (river valley), **grass push-in** (red barn).
- Measured 2026-10-26 on the same RTX 3070 Ti: **49 frames in ~75 s per move**
  (768×512, 20 steps, cfg 3.0) — camera language in the prompt is followed
  reliably enough that frame 25 → frame 45 shows the move's geometry change
  (the crane rises, the tracker parallaxes, the dive descends).

Camera control plus performance transfer is the suite: one pillar performs,
the other films.

## Any RGB video can perform, too

Kinect depth is the purest input, but any ordinary video can be turned into a
control performance: `capture/depth_estimate.py` runs **Depth-Anything-V2-
Small** (open weights) over an mp4/mov and writes a control clip in the exact
spec above (640×640, 16 fps, near = bright, clip-global normalization so the
depth doesn't flicker). Verified on real camera footage (ducks crossing a
park path, shot on a Canon) — the estimated depth drives the same Wan control
graph, no Kinect required. The performer keeps its species: ducks stay ducks,
the prompt only supplies the costume.

## Files

- `workflow-depth-control-api.json` — the exact workflow behind the benchmark
  numbers above (API format), saving an mp4.
- `workflow-ltx-cinema-api.json` + `camera_slate.py` — the LTX-core cinema
  camera pillar.
- `run_local.py` — renders either workflow from the command line; builds every
  graph from the JSON files above, so what you run is what is published.
- `comfy_client.py` — the small stdlib ComfyUI client the scripts share.
- `models.json` — model manifest: exact bytes, sha256, download URL.
- `tools/verify_models.py`, `tools/validate_workflows.py` — install checks.
- `examples/` — synthetic depth clips for Kinect-free verification.
- `capture/` — Kinect v2 recorder (`capture/KinectBurst.cs`,
  `capture/build_kinectburst.ps1`), `capture/capture_take.ps1`,
  `capture/encode_depth.py` (raw 512×424 uint16 mm frames → 640×640 / 16 fps
  control mp4, resampled by timestamp), `capture/pick_window.py`,
  `capture/process_take.py`.
- `capture/depth_estimate.py` — RGB video → control-spec depth clip via
  Depth-Anything-V2-Small, for footage that wasn't shot on a depth sensor.
- `tests/` — `python -m unittest discover -s tests -t .` (no GPU or server
  needed; a fake ComfyUI stands in for the real one; capture tests need numpy
  + ffmpeg and skip without them).

## Capture your own performer (Kinect v2)

Depth performances were recorded from a Kinect v2 (the same sensor that runs
the cats' interactive pond at home). Cats do not do second takes on command,
so the good crossings (a fetch run, a loaf-to-prowl stretch) are caught in
long takes and the best stretch is found automatically:

```
powershell -File capture\build_kinectburst.ps1          # once; needs Kinect SDK 2.0
powershell -File capture\capture_take.ps1 -TakeDir D:\takes\luna-01
python capture/process_take.py D:\takes\luna-01
```

- `capture/KinectBurst.cs` opens the sensor once and writes `burst-ready.txt`
  only after a real depth frame is in hand. `capture/capture_take.ps1` goes one
  step further: GO (two beeps) only when the first burst frame has landed in a
  brand-new take folder, never while another program owns the sensor, and the
  take's real frame rate and any sensor stalls are reported when it ends.
- **Timing is measured, not assumed.** The recorder's real interval is the
  requested one plus the time to write each frame, and the sensor sometimes
  stalls for seconds. Measured: asking for 62 ms gave 94 ms (10.6 fps) and a
  4.4 s stall in one take; asking for 28 ms gave 63 ms (15.1 fps), 400/400
  frames. `capture/encode_depth.py` therefore resamples every take onto an
  exact 16 fps grid from the logged timestamps — the performance plays back in
  real time — and marks stalls.
- `capture/process_take.py` encodes the take, picks the highest-motion
  stall-free windows with `capture/pick_window.py` (97 frames for the hero
  shot, 49 for the costume slate), renders the costumes through the shipped
  workflow and writes depth | render side-by-sides into a new
  `process-<time>/` folder inside the take. `--no-render` stops after the
  control clips.

## License

MIT (this repo's code and workflows). Model weights remain under their own
licenses (Wan 2.2 / Alibaba, LTX-Video / Lightricks, GGUF requants by
QuantStack and city96, Depth-Anything-V2).
