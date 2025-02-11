# Use the official Miniconda image as the base
FROM public.ecr.aws/y0o4y9o3/miniconda3:latest

# Set the working directory
WORKDIR /app

# Copy Conda environment file (if you have one)
COPY environment.yml .

# Create the Conda environment (replace `myenv` with your environment name)
RUN conda env create -f environment.yml -n myenv

# Copy your Python script and other files
COPY main.py .
COPY core/ ./core/
COPY utils/ ./utils/

# Make sure Conda's environment is activated when running commands
ENV PATH /opt/conda/envs/myenv/bin:$PATH

# Set environment variables (optional, if you want defaults)
# ENV LOG_LEVEL=INFO
# ENV INPUT_DIR=/app/input/

# Run the script
CMD ["python", "main.py"]