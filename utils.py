import torch

def print_memory_usage():
    max_gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert bytes to GB
    print(f"Maximum GPU memory allocated: {max_gpu_memory:.1f} GB")