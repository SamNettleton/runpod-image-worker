# RunPod Serverless Stable Diffusion v1.5 Endpoint

A production-grade, optimized serverless worker implementation for deploying `runwayml/stable-diffusion-v1-5` on RunPod Serverless.

## Architecture & Performance Optimizations

1. **Global Model Initialization:** The model pipeline is loaded once at the module level during container cold start. This ensures weights are cached in VRAM and subsequent requests avoid re-loading overhead.
2. **Cold-Start Acceleration:** Integrated `hf-transfer` with `HF_HUB_ENABLE_HF_TRANSFER=1` enabled in the environment to utilize high-speed Rust-based downloads when pulling model weights.
3. **Memory Efficiency:** Inference execution is wrapped in `torch.inference_mode()` to reduce memory footprint and maximize compute speed on CUDA devices.
4. **Clean Error Handling & Validation:** Incoming payloads are validated instantly at the handler entry point. Missing parameters or invalid inputs return structured JSON error dictionaries rather than crashing the worker loop.

## Repository Structure

- `handler.py`: The core serverless worker script containing initialization logic and the request handler function.
- `Dockerfile`: An optimized container configuration featuring layer caching (dependencies installed before code copy) and CUDA support.
- `requirements.txt`: Pinned Python dependencies required for execution.

## API Payload Specification

The endpoint accepts a JSON payload structured under an `input` key. 

### Minimal Request (Using Defaults)
```json
{
  "input": {
    "prompt": "A majestic cybernetic owl perched on a neon-lit rooftop"
  }
}
```

### Advanced Customization Request
```json
{
  "input": {
    "prompt": "Cinematic wide-angle shot of an alpine research station at twilight",
    "negative_prompt": "blurry, low quality, underexposed",
    "num_inference_steps": 40,
    "guidance_scale": 8.5
  }
}
```

### Local Testing Instructions

To test the handler locally prior to cloud deployment:

1. Ensure your local environment has a CUDA-compatible GPU and PyTorch installed.

2. Install dependencies:
```Bash
pip install -r requirements.txt
```

3. Run the handler script directly (it will default to running a local test loop using local weights or input mockups):
```Bash
python3 handler.py
```
