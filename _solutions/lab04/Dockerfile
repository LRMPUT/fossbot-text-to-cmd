# Start from a small official Python image (Debian slim + Python 3.12)
FROM python:3.12-slim

# All subsequent paths are relative to /app inside the image
WORKDIR /app

# Install Python dependencies first, in two layers, so Docker can cache them
# Layer 1: CPU-only PyTorch (much smaller than the default CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Layer 2: the rest of the requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the dataset
COPY src/  src/
COPY data/ data/

# Default command if someone just runs `docker run <image>`
CMD ["python", "-m", "src.text_to_wheels", "--help"]
