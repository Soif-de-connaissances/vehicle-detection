import os
import torch
import time
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ultralytics import YOLO
from Backend.create_data_yaml import create_data_yaml


def train_detector(data_yaml_path, epochs=30, batch=16, patience=3):
    print("Starting vehicle detector training...")

    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/logs", exist_ok=True)
    os.makedirs("models/plots", exist_ok=True)

    # Initialize YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Train for multiple epochs
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        name='vehicle_detection'
    )

    # Get final mAP50-95
    try:
        final_map = results.box.map
        print(f"Final mAP50-95: {final_map:.4f}")
        
        # Save final performance metrics
        with open("models/logs/detector_final_metrics.txt", 'w') as f:
            f.write(f"mAP50-95: {final_map:.4f}\n")
            
    except AttributeError as e:
        print(f"Unable to extract mAP50-95: {e}")

    print("Vehicle detector training completed!")
    return model


def get_data_loaders(crops_dir, batch_size):
    """
    Load training and validation datasets and return data loaders and the number of classes
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
    print(f"Detected {num_classes} vehicle classes")

    # Save class mapping
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    with open('models/classifier_class_mapping.txt', 'w') as f:
        for class_name, idx in class_to_idx.items():
            f.write(f"{idx}: {class_name}\n")

    return train_loader, val_loader, num_classes


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch
    """
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    train_loader_tqdm = tqdm(train_loader, desc="Training", leave=False)
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
    Validate for one epoch
    """
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    val_loader_tqdm = tqdm(val_loader, desc="Validating", leave=False)
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


def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path):
    """
    Plot training and validation loss and accuracy curves
    """
    plt.figure(figsize=(15, 6))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue', marker='o')
    plt.plot(val_losses, label='Validation Loss', color='red', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.grid(True)
    plt.legend()
    
    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy', color='blue', marker='o')
    plt.plot(val_accs, label='Validation Accuracy', color='red', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Curves')
    plt.grid(True)
    plt.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Training curves saved to {save_path}")


def train_classifier(crops_dir, batch_size=32, epochs=25, patience=2):
    print("Starting vehicle classifier training...")

    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/logs", exist_ok=True)
    os.makedirs("models/plots", exist_ok=True)

    # Check if there are enough classes
    train_dir = os.path.join(crops_dir, 'train')
    class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]

    if len(class_dirs) <= 1:
        print("Warning: Not enough classes to train the classifier")
        return None

    # Load and preprocess data
    train_loader, val_loader, num_classes = get_data_loaders(crops_dir, batch_size)

    # Initialize classifier
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )

    # Early stopping parameters
    best_val_loss = float('inf')
    best_val_acc = 0.0
    no_improvement_epochs = 0

    # Lists to record training progress
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Create log file
    log_file_path = "models/logs/classifier_training_log.csv"
    with open(log_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr', 'time'])

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        start_time = time.time()

        # Training phase
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validation phase
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        # Record current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Update learning rate
        scheduler.step(val_loss)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")

        # Record training progress
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Write to log file
        with open(log_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc, current_lr, epoch_time])

        # Check if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/best_vehicle_classifier.pth')
            no_improvement_epochs = 0
            print("Validation loss improved, saving best model weights.")
        else:
            no_improvement_epochs += 1
            print(f"Validation loss did not improve for {no_improvement_epochs} consecutive epochs.")

        # Plot curves every 5 epochs or at the end of training
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1 or no_improvement_epochs >= patience:
            plot_training_curves(
                train_losses, train_accs, val_losses, val_accs,
                f"models/plots/classifier_training_curves_epoch_{epoch+1}.png"
            )

        # Check if early stopping is needed
        if no_improvement_epochs >= patience:
            print(f"Validation loss did not improve for {patience} consecutive epochs, stopping early.")
            break
    
    # Save final training curves
    plot_training_curves(
        train_losses, train_accs, val_losses, val_accs,
        "models/plots/classifier_training_curves_final.png"
    )
    
    # Save final performance metrics
    with open("models/logs/classifier_final_metrics.txt", 'w') as f:
        f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")
        f.write(f"Best Validation Loss: {best_val_loss:.4f}\n")
        f.write(f"Total Training Epochs: {epoch+1}\n")
    
    print(f"Training log saved to {log_file_path}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print("Vehicle classifier training completed!")
    return model


def evaluate_classifier(model_path, data_dir, batch_size=32):
    """
    Evaluate classifier performance on the test set
    """
    print("Evaluating classifier performance...")
    # Create necessary directories
    os.makedirs("models/evaluation", exist_ok=True)
    
    # Load class mapping
    class_mapping = {}
    with open("models/classifier_class_mapping.txt", 'r') as f:
        for line in f:
            parts = line.strip().split(': ')
            if len(parts) == 2:
                idx, class_name = parts
                class_mapping[int(idx)] = class_name
    
    # Data transformation
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load test data
    test_dir = os.path.join(data_dir, 'test')
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} does not exist, using validation set for evaluation")
        test_dir = os.path.join(data_dir, 'val')
    
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(class_mapping)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Evaluation
    criterion = torch.nn.CrossEntropyLoss()
