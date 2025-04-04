import os
import argparse
from Backend.download_data import download_dataset
from Backend.data_preprocessing import DataPreprocessor
from Backend.create_data_yaml import create_data_yaml
from Backend.train_models import train_detector, train_classifier
from Backend.inference import VehicleDetectionSystem

def parse_args():
    parser = argparse.ArgumentParser(description='车辆检测与分类系统')
    parser.add_argument('--mode', type=str, default='all', choices=['download', 'preprocess', 'train', 'infer', 'all', 'check_gpu'], 
                        help='运行模式')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--image', type=str, default=None, help='用于推理的图像路径')
    parser.add_argument('--video', type=str, default=None, help='用于推理的视频路径')
    parser.add_argument('--force_preprocess', action='store_true', help='强制重新预处理数据')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 创建必要的目录
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    dataset_path = None
    
    if args.mode == 'download' or args.mode == 'all':
        print("=== 下载数据集 ===")
        dataset_path = download_dataset()
    
    if args.mode == 'preprocess' or args.mode == 'all':
        print("\n=== 预处理数据 ===")
        if dataset_path is None:
            # 尝试使用默认路径
            dataset_path = os.path.join("data", "roundabout-aerial-images-for-vehicle-detection")
            if not os.path.exists(dataset_path):
                # 尝试使用本地路径
                dataset_path = "C:/Users/14919/Desktop/岭大/CDS521/vehicle detection/data/kagglehub/datasets/javiersanchezsoriano/roundabout-aerial-images-for-vehicle-detection/versions/2"
            
            print(f"使用数据集路径: {dataset_path}")
        
        preprocessor = DataPreprocessor(dataset_path)
        preprocessor.prepare_dataset(force_preprocess=args.force_preprocess)
        preprocessor.extract_vehicle_crops(force_extract=args.force_preprocess)
    
    if args.mode == 'train' or args.mode == 'all':
        print("\n=== 训练模型 ===")
        data_yaml_path = create_data_yaml("data/processed")
        detector = train_detector(data_yaml_path, epochs=args.epochs)
        classifier = train_classifier("data/vehicle_crops", epochs=args.epochs // 2)
    
    if args.mode == 'infer' or args.mode == 'all' or args.image or args.video:
        print("\n=== 模型推理 ===")
        detector_path = "models/vehicle_detection/weights/best.pt"
        classifier_path = "models/best_vehicle_classifier.pth"
        
        if not os.path.exists(detector_path):
            print(f"错误：检测器模型不存在 {detector_path}")
            return
        
        system = VehicleDetectionSystem(
            detector_path=detector_path,
            classifier_path=classifier_path if os.path.exists(classifier_path) else None
        )
        
        if args.image:
            system.process_image(args.image)
        elif args.mode == 'all':
            # 处理测试集中的一张图像
            test_dir = "data/processed/test/images"
            if os.path.exists(test_dir):
                test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                if test_images:
                    test_image_path = os.path.join(test_dir, test_images[0])
                    system.process_image(test_image_path)

        if args.video:
            system.process_video(args.video)

if __name__ == "__main__":
    main()