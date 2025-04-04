import os
import kagglehub

def download_dataset():
    """
    下载或使用本地环形交叉口航拍图像数据集
    
    Returns:
        str: 数据集路径
    """
    # 定义相对路径
    dataset_path = os.path.join("data", "roundabout-aerial-images-for-vehicle-detection")
    # 检查数据集是否已经存在
    if os.path.exists(dataset_path):
        print(f"数据集已存在，跳过下载。路径: {dataset_path}")
        return dataset_path
    
    # 如果本地不存在，尝试从Kaggle下载
    try:
        print("尝试从Kaggle下载数据集...")
        # 使用kagglehub下载数据集，按照官方示例代码
        path = kagglehub.dataset_download("javiersanchezsoriano/roundabout-aerial-images-for-vehicle-detection")
        print(f"数据集下载完成，路径: {path}")
        return path
    except Exception as e:
        print(f"下载数据集时出错: {e}")
        print("请确保已安装kagglehub: pip install kagglehub")
    # 如果下载失败，尝试使用已有的本地路径
    local_path = "D:/学习软件/CDS521/vehicle detection/data/kagglehub/datasets/javiersanchezsoriano/roundabout-aerial-images-for-vehicle-detection/versions/2"
    if os.path.exists(local_path):
        print(f"使用已有的本地数据集路径: {local_path}")
        return local_path
    
    print("警告: 无法下载数据集且本地路径不存在。")
    print("请手动下载数据集或确保本地路径正确。")
    return None

if __name__ == "__main__":
    dataset_path = download_dataset()
    print(f"\n数据集路径: {dataset_path}")
