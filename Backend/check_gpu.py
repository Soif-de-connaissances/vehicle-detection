import torch
import subprocess
import platform
import sys

def check_gpu():
    """
    检查GPU是否可用，并返回相关信息
    
    Returns:
        dict: 包含GPU可用性和详细信息的字典
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
        
        # 获取每个GPU的名称
        for i in range(result["gpu_count"]):
            result["gpu_names"].append(torch.cuda.get_device_name(i))
        
        # 尝试获取CUDA版本
        try:
            if platform.system() == "Windows":
                # Windows系统
                nvcc_output = subprocess.check_output("nvcc --version", shell=True).decode("utf-8")
                for line in nvcc_output.split('\n'):
                    if "release" in line:
                        result["cuda_version"] = line.split("release")[-1].strip().split(",")[0]
                        break
            else:
                # Linux/Mac系统
                nvcc_output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
                for line in nvcc_output.split('\n'):
                    if "release" in line:
                        result["cuda_version"] = line.split("release")[-1].strip().split(",")[0]
                        break
        except Exception as e:
            result["cuda_version"] = f"无法获取CUDA版本: {str(e)}"
    
    return result

def print_gpu_info():
    """打印GPU信息"""
    info = check_gpu()
    
    print("\n===== GPU 信息 =====")
    print(f"系统信息: {info['system_info']}")
    print(f"PyTorch版本: {torch.__version__}")
    
    if info["cuda_available"]:
        print(f"CUDA可用: 是")
        print(f"CUDA版本: {info['cuda_version'] or '未知'}")
        print(f"PyTorch CUDA版本: {info['pytorch_cuda_version']}")
        print(f"GPU数量: {info['gpu_count']}")
        
        for i, gpu_name in enumerate(info["gpu_names"]):
            print(f"  GPU {i}: {gpu_name}")
            
        # 显示每个GPU的内存信息
        for i in range(info["gpu_count"]):
            torch.cuda.set_device(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # 转换为GB
            allocated_memory = torch.cuda.memory_allocated() / 1024**3
            cached_memory = torch.cuda.memory_reserved() / 1024**3
            print(f"  GPU {i} 内存: 总计 {total_memory:.2f} GB, 已分配 {allocated_memory:.2f} GB, 已缓存 {cached_memory:.2f} GB")
    else:
        print(f"CUDA可用: 否")
        print("没有检测到可用的GPU，将使用CPU进行计算。")
        print("如果您有NVIDIA GPU，请确保已安装正确的CUDA和cuDNN版本。")
    
    # 检查是否有足够的GPU内存用于训练
    if info["cuda_available"] and info["gpu_count"] > 0:
        device_id = 0
        free_memory = (torch.cuda.get_device_properties(device_id).total_memory - 
                      torch.cuda.memory_allocated(device_id) - 
                      torch.cuda.memory_reserved(device_id)) / 1024**3  # 转换为GB
        
        print(f"\nGPU {device_id} 可用内存: {free_memory:.2f} GB")
        
        if free_memory < 2:
            print("警告: GPU内存可能不足以训练大型模型。考虑减小批量大小或使用更小的模型。")
        elif free_memory < 4:
            print("提示: GPU内存适中，可能需要调整批量大小以优化训练。")
        else:
            print("GPU内存充足，可以进行模型训练。")

if __name__ == "__main__":
    print_gpu_info()