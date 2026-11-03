# Official PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.2-cuda11.8.0-cudnn8-runtime

# Set non-interactive mode to prevent prompt hangups during build
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory inside the container
WORKDIR /app

# Dependency Caching
# Copy ONLY requirements.txt first. 
# Docker will skip reinstalling heavy packages like PyTorch and Diffusers
# if handler.py is changed later.
COPY requirements.txt .

# Install Python dependencies cleanly
RUN pip install --no-cache-dir -r requirements.txt

# Enable high-performance Rust-based Hugging Face downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Copy the application handler code last
COPY handler.py .

# Start the RunPod serverless worker loop
CMD ["python", "handler.py"]