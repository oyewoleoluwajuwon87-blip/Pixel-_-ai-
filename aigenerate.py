import sys
import torch
from model import SimpleUNet
import imageio
import numpy as np
import os

prompt = sys.argv[1]

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleUNet().to(device)
model.load_state_dict(torch.load("ai/model.pth"))
model.eval()

frames = []

for i in range(16):
    noise = torch.randn(1, 3, 64, 64).to(device)

    with torch.no_grad():
        image = model(noise).cpu().squeeze()

    img = image.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)

    frames.append(img)

os.makedirs("public/videos", exist_ok=True)

video_path = "public/videos/output.mp4"
imageio.mimsave(video_path, frames, fps=4)

print(video_path)