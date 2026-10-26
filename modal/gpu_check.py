import modal

app = modal.App("learning-distributed")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch")
)

@app.function(
    image=image,
    gpu="T4",
    timeout=300,
)
def check_gpu():
    import torch

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))

    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x

    print("Matrix multiplication completed.")
    print("Output shape:", tuple(y.shape))
    print("Output device:", y.device)

@app.local_entrypoint()
def main():
    check_gpu.remote()
