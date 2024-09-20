from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

# Load the AnimateDiff model
model_id = "XingangPan/AnimateDiff"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16,use_auth_token='VPpaDaFqLtfiMDDRrppneXfXwarwiCIyry')
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# Function to generate a video
def generate_video(prompt, num_inference_steps=25, guidance_scale=7.5):
  video_frames = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).frames
  return video_frames

# Example 1: Generate a video of a cat playing
cat_video_frames = generate_video("A cute cat playing with a ball of yarn")

# Example 2: Generate a video of a person dancing
dance_video_frames = generate_video("A person dancing in a vibrant club")

# Example 3: Generate a video of a cityscape at night
cityscape_video_frames = generate_video("A beautiful cityscape at night with sparkling lights")
