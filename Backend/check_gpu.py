import torch
import subprocess
import platform
import sys

def check_gpu():
    """
    Check if the GPU is available and return relevant information.
    
    Returns:
        dict: A dictionary containing GPU availability and detailed information.
    """
    result = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": 0,
        "gpu_names": [],
        "cuda_version": None,
        "pytorch_cuda_version": None,
        "system_info": platform.platform()
    }
    
    if result["cuda_available"]:
        result["gpu_count"] = torch.cuda.device_count()
        result["pytorch_cuda_version"] = torch.version.cuda
        
        # Get the name of each GPU
        for i in range(result["gpu_count"]):
            result["gpu_names"].append(torch.cuda.get_device_name(i))
        
        # Try to get the CUDA version
        try:
            if platform.system() == "Windows":
                # Windows system
                nvcc_output = subprocess.check_output("nvcc --version", shell=True).decode("utf-8")
                for line in nvcc_output.split('\n'):
                    if "release" in line:
                        result["cuda_version"] = line.split("release")[-1].strip().split(",")[0]
                        break
            else:
                # Linux/Mac system
                nvcc_output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
                for line in nvcc_output.split('\n'):
                    if "release" in line:
                        result["cuda_version"] = line.split("release")[-1].strip().split(",")[0]
                        break
        except Exception as e:
            result["cuda_version"] = f"Failed to get CUDA version: {str(e)}"
    
    return result

def print_gpu_info():
    """Print GPU information"""
    info = check_gpu()
    
    print("\n===== GPU Information =====")
    print(f"System Information: {info['system_info']}")
    print(f"PyTorch Version: {torch.__version__}")
    
    if info["cuda_available"]:
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {info['cuda_version'] or 'Unknown'}")
        print(f"PyTorch CUDA Version: {info['pytorch_cuda_version']}")
        print(f"Number of GPUs: {info['gpu_count']}")
        
        for i, gpu_name in enumerate(info["gpu_names"]):
            print(f"  GPU {i}: {gpu_name}")
            
        # Display memory information for each GPU
        for i in range(info["gpu_count"]):
            torch.cuda.set_device(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # Convert to GB
            allocated_memory = torch.cuda.memory_allocated() / 1024**3
            cached_memory = torch.cuda.memory_reserved() / 1024**3
            print(f"  GPU {i} Memory: Total {total_memory:.2f} GB, Allocated {allocated_memory:.2f} GB, Cached {cached_memory:.2f} GB")
    else:
        print(f"CUDA Available: No")
        print("No available GPU detected. CPU will be used for computation.")
        print("If you have an NVIDIA GPU, make sure the correct CUDA and cuDNN versions are installed.")
    
    # Check if there is enough GPU memory for training
    if info["cuda_available"] and info["gpu_count"] > 0:
        device_id = 0
        free_memory = (torch.cuda.get_device_properties(device_id).total_memory - 
                      torch.cuda.memory_allocated(device_id) - 
                      torch.cuda.memory_reserved(device_id)) / 1024**3  # Convert to GB
        
        print(f"\nGPU {device_id} Free Memory: {free_memory:.2f} GB")
        
        if free_memory < 2:
            print("Warning: GPU memory may be insufficient to train large models. Consider reducing the batch size or using a smaller model.")
        elif free_memory < 4:
            print("Tip: GPU memory is moderate. You may need to adjust the batch size to optimize training.")
        else:
            print("GPU memory is sufficient for model training.")

if __name__ == "__main__":
    print_gpu_info()