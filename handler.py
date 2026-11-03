import base64
import io
import torch
from diffusers import StableDiffusionPipeline
import runpod

# Global initialization (loads once on container cold start)
MODEL_ID = "runwayml/stable-diffusion-v1-5"
print(f"[INFO] Initializing model pipeline for {MODEL_ID}...")

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

try:
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype
    )
    pipe = pipe.to(device)
    
    # Enable memory-efficient attention if available to optimize VRAM
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
        
    print("[INFO] Model pipeline successfully loaded into GPU memory.")
except Exception as e:
    print(f"[ERROR] Critical error loading model during initialization: {e}")
    raise e

def handler(job):
    """
    RunPod serverless worker handler.
    Accepts dynamic input parameters for fine-grained inference configuration.
    """
    try:
        job_input = job.get("input", {})
        
        # Validate that a prompt was provided
        prompt = job_input.get("prompt")
        if not prompt:
            print("[ERROR] Job received missing 'prompt' parameter.")
            return {"error": "Missing required 'prompt' parameter in job input."}
            
        negative_prompt = job_input.get("negative_prompt", None)
        num_inference_steps = int(job_input.get("num_inference_steps", 30))
        guidance_scale = float(job_input.get("guidance_scale", 7.5))
        
        print(f"[INFO] Processing prompt: '{prompt}' | Steps: {num_inference_steps} | Guidance Scale: {guidance_scale}")
        
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
        
        # Convert PIL Image directly to base64 string for API response
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        print("[INFO] Image generation completed and encoded successfully.")
        
        return {
            "image": image_base64,
            "prompt": prompt,
            "parameters": {
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            }
        }
        
    except Exception as e:
        print(f"[ERROR] Error processing job execution: {str(e)}")
        return {"error": str(e)}

# Start the RunPod serverless worker loop
runpod.serverless.start({"handler": handler})