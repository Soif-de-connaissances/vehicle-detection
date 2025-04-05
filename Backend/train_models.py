import os
import torch
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ultralytics import YOLO
from Backend.create_data_yaml import create_data_yaml


def train_detector(data_yaml_path, epochs=50, batch=16, patience=3):
    print("开始训练车辆检测器...")

    # 创建输出目录
    os.makedirs("models", exist_ok=True)

    # 初始化 YOLOv8 模型
    model = YOLO('yolov8n.pt')

    # 一次性训练多个 epoch
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        name='vehicle_detection'
    )

    # 获取最终的 mAP50-95
    try:
        final_map = results.box.map
        print(f"最终 mAP50-95: {final_map:.4f}")
    except AttributeError as e:
        print(f"无法提取 mAP50-95：{e}")

    print("车辆检测器训练完成！")
    return model


def get_data_loaders(crops_dir, batch_size):
    """
    加载训练和验证数据集，并返回数据加载器和类别数量
    """
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

    train_dir = os.path.join(crops_dir, 'train')
    val_dir = os.path.join(crops_dir, 'val')

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    num_classes = len(train_dataset.classes)
    print(f"检测到 {num_classes} 个车辆类别")

    # 保存类别映射
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    with open('models/classifier_class_mapping.txt', 'w') as f:
        for class_name, idx in class_to_idx.items():
            f.write(f"{idx}: {class_name}\n")

    return train_loader, val_loader, num_classes


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    训练一个 epoch
    """
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    train_loader_tqdm = tqdm(train_loader, desc="训练中", leave=False)
    for inputs, targets in train_loader_tqdm:
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

    train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_correct / train_total

    return train_loss, train_acc


def validate_one_epoch(model, val_loader, criterion, device):
    """
    验证一个 epoch
    """
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    val_loader_tqdm = tqdm(val_loader, desc="验证中", leave=False)
    with torch.no_grad():
        for inputs, targets in val_loader_tqdm:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_correct / val_total

    return val_loss, val_acc


def train_classifier(crops_dir, batch_size=32, epochs=30, patience=3):
    print("开始训练车辆分类器...")

    # 检查是否有足够的类别
    train_dir = os.path.join(crops_dir, 'train')
    class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]

    if len(class_dirs) <= 1:
        print("警告：类别数量不足，无法训练分类器")
        return None

    # 数据加载和预处理
    train_loader, val_loader, num_classes = get_data_loaders(crops_dir, batch_size)

    # 初始化分类器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )

    # Early Stopping 参数
    best_val_loss = float('inf')
    no_improvement_epochs = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        start_time = time.time()

        # 训练阶段
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # 验证阶段
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        # 更新学习率
        scheduler.step(val_loss)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.2f}s")

        # 检查验证集损失是否改善
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/best_vehicle_classifier.pth')
            no_improvement_epochs = 0
            print("验证集损失改善，保存最佳模型权重。")
        else:
            no_improvement_epochs += 1
            print(f"验证集损失未改善，已连续 {no_improvement_epochs} 个 epoch。")

        # 检查是否需要提前停止
        if no_improvement_epochs >= patience:
            print(f"验证集损失在 {patience} 个 epoch 内未改善，提前停止训练。")
            break

    print("车辆分类器训练完成！")
    return model


if __name__ == "__main__":
    # 创建数据配置文件
    data_yaml_path = create_data_yaml("data/processed")

    # 训练检测器
    detector = train_detector(data_yaml_path, epochs=50)

    # 训练分类器
    classifier = train_classifier("data/vehicle_crops", epochs=30)