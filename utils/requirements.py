import subprocess

subprocess.run(["mamba", "install", "pytorch-cuda=11.6", "-c", "pytorch", "-c", "conda-forge", "-c", "nvidia", "-y"])
subprocess.run(["mamba", "install","tensorflow-gpu==2.11.0", "-y"])
subprocess.run(["mamba", "install","-c","conda-forge", "pyts==0.12.0", "-y"])
subprocess.run(["pip", "install", "torch==1.13", "torchvision", "torchaudio", "--extra-index-url", "https://download.pytorch.org/whl/cu116"])
subprocess.run(["pip", "install","torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv", "torch-geometric", "-f", "https://data.pyg.org/whl/torch-1.13.1+cu116.html"])
subprocess.run(["pip", "install","ts2vg==1.2.1"])
subprocess.run(["pip", "install","pytorch_lightning==1.9.1"])
subprocess.run(["pip", "install","torchsummary==1.5.1"])
subprocess.run(["pip", "install","dvclive==2.0.2"])