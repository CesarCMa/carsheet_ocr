import torch


def check_gpu_availability():
    """
    Check if GPU is available on the system and print relevant information.
    """
    print("Checking GPU availability...")

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        # Get the number of available GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs available: {gpu_count}")

        # Print information about each GPU
        for i in range(gpu_count):
            print(f"\nGPU {i}:")
            print(f"Name: {torch.cuda.get_device_name(i)}")
            print(
                f"Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB"
            )
            print(f"Memory cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
    else:
        print("No GPU available. Using CPU only.")


if __name__ == "__main__":
    check_gpu_availability()
