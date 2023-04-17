# Import necessary modules
import subprocess
import sys

# Define the packages and their versions
PACKAGES = [
    "pytorch-cuda==11.6",
    "tensorflow-gpu==2.11.0",
    "pyts==0.12.0",
    "torch==1.13",
    "torchvision",
    "torchaudio",
    "torch-scatter",
    "torch-sparse",
    "torch-cluster",
    "torch-spline-conv",
    "torch-geometric",
    "ts2vg==1.2.1",
    "pytorch_lightning==1.9.1",
    "torchsummary==1.5.1",
    "dvclive==2.0.2"
]

# Define the command to install packages with mamba
MAMBA_COMMAND = ["mamba", "install"] + PACKAGES + ["-c", "pytorch", "-c", "conda-forge", "-c", "nvidia", "-y"]

# Define the command to install packages with pip
PIP_COMMAND = ["pip", "install"] + PACKAGES + ["--extra-index-url", "https://download.pytorch.org/whl/cu116", "-f", "https://data.pyg.org/whl/torch-1.13.1+cu116.html"]

def install_packages():
    # Try to install packages with mamba
    try:
        subprocess.run(MAMBA_COMMAND, check=True)
    except subprocess.CalledProcessError:
        # If mamba installation fails, install packages with pip
        subprocess.run(PIP_COMMAND, check=True)
    
    print("Packages installed successfully.")

def create_virtual_environment():
    # Check if virtual environment already exists
    if getattr(sys, 'real_prefix', None) or hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
        print("Virtual environment already exists.")
        return
    
    # If virtual environment does not exist, create one
    try:
        subprocess.run(["python", "-m", "venv", "env"], check=True)
        print("Virtual environment created successfully.")
    except subprocess.CalledProcessError:
        print("Error creating virtual environment.")
        return

if __name__ == "__main__":
    create_virtual_environment()
    install_packages()