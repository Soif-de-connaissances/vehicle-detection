import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ultralytics import YOLO
from Backend.create_data_yaml import create_data_yaml

def train_detector(data_yaml_path, epochs=100, batch=16, patience=10):
    """
    训练YOLOv8车辆检测器
    
    Args:
        data_yaml_path: 数据配置文件路径
        epochs: 训练轮数
        batch: 批次大小
        patience: 早停耐心值，连续多少个epoch没有改善则停止训练
    """
    print("开始训练车辆检测器...")
    
    # 检查是否已有训练好的模型
    output_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'vehicle_detection', 'weights', 'best.pt')
    
    if os.path.exists(model_path):
        print(f"检测到已有训练好的车辆检测模型: {model_path}")
        user_input = input("是否要重新训练模型？(y/n): ")
        if user_input.lower() != 'y':
            print("跳过训练，使用已有模型")
            return YOLO(model_path)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化YOLOv8模型
    # 指定缓存目录为项目目录下的.cache文件夹
    cache_dir = os.path.join(os.getcwd(), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TORCH_HOME"] = cache_dir
    
    # 使用预训练的YOLOv8n，但指定下载位置
    model = YOLO('yolov8n.pt')
    
    # 训练模型，明确指定项目路径，并添加早停机制
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        project=output_dir,  # 指定项目路径
        name='vehicle_detection',
        exist_ok=True,
        cache=True,  # 使用缓存加速
        patience=patience  # 添加早停机制
    )
    
    print("车辆检测器训练完成！")
    print(f"模型保存在: {os.path.join(output_dir, 'vehicle_detection')}")
    return model

def train_classifier(crops_dir, batch_size=32, epochs=50, patience=10):
    """
    训练车辆分类器
    
    Args:
        crops_dir: 车辆裁剪图像目录
        batch_size: 批次大小
        epochs: 训练轮数
        patience: 早停耐心值，连续多少个epoch验证损失没有改善则停止训练
    """
    print("开始训练车辆分类器...")
    
    # 创建输出目录
    output_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查是否已有训练好的模型
    best_model_path = os.path.join(output_dir, 'best_vehicle_classifier.pth')
    if os.path.exists(best_model_path):
        print(f"检测到已有训练好的车辆分类模型: {best_model_path}")
        user_input = input("是否要重新训练模型？(y/n): ")
        if user_input.lower() != 'y':
            print("跳过训练，使用已有模型")
            # 加载已有模型
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # 需要获取类别数量，从mapping文件读取
            class_mapping_path = os.path.join(output_dir, 'classifier_class_mapping.txt')
            if os.path.exists(class_mapping_path):
                with open(class_mapping_path, 'r') as f:
                    lines = f.readlines()
                num_classes = len(lines)
                model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
                in_features = model.fc.in_features
                model.fc = torch.nn.Linear(in_features, num_classes)
                model.load_state_dict(torch.load(best_model_path, map_location=device))
                model = model.to(device)
                return model
    
    # 指定缓存目录
    cache_dir = os.path.join(os.getcwd(), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TORCH_HOME"] = cache_dir
    
    # 检查是否有足够的类别
    train_dir = os.path.join(crops_dir, 'train')
    class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    if len(class_dirs) <= 1:
        print("警告：类别数量不足，无法训练分类器")
        return None
    
    # 数据增强和预处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    try:
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        
        val_dir = os.path.join(crops_dir, 'val')
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
    except Exception as e:
        print(f"加载数据集时出错：{e}")
        return None
    
    # 获取类别数量
    num_classes = len(train_dataset.classes)
    print(f"检测到 {num_classes} 个车辆类别")
    
    # 保存类别映射
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    class_mapping_path = os.path.join(output_dir, 'classifier_class_mapping.txt')
    with open(class_mapping_path, 'w') as f:
        for class_name, idx in class_to_idx.items():
            f.write(f"{idx}: {class_name}\n")
    
    # 初始化分类器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True, force_reload=False)
    
    # 修改最后一层以匹配类别数量
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    # 训练模型
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, 'best_vehicle_classifier.pth')
    checkpoint_path = os.path.join(output_dir, 'vehicle_classifier_checkpoint.pth')
    
    # 添加早停机制
    early_stopping_counter = 0
    
    # 用于进度显示
    from tqdm import tqdm
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 添加进度条
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            train_acc = train_correct / train_total if train_total > 0 else 0
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{train_acc:.4f}"
            })
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # 添加验证进度条
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                # 更新进度条
                val_acc = val_correct / val_total if val_total > 0 else 0
                val_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{val_acc:.4f}"
                })
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # 更新学习率
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        
        # 保存检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, checkpoint_path)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"保存最佳模型，验证损失: {val_loss:.4f}")
            # 重置早停计数器
            early_stopping_counter = 0
        else:
            # 增加早停计数器
            early_stopping_counter += 1
            print(f"验证损失没有改善，早停计数: {early_stopping_counter}/{patience}")
            
            # 检查是否应该早停
            if early_stopping_counter >= patience:
                print(f"早停触发！连续 {patience} 个epoch验证损失没有改善")
                break
    
    print("车辆分类器训练完成！")
    print(f"最佳模型保存在: {best_model_path}")
    return model

if __name__ == "__main__":
    # 创建数据配置文件
    data_yaml_path = create_data_yaml("data/processed")
    
    # 训练检测器
    detector = train_detector(data_yaml_path, epochs=100, patience=10)
    
    # 训练分类器
    classifier = train_classifier("data/vehicle_crops", epochs=50, patience=10)