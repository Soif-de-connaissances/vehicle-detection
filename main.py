import os
import argparse
from Backend.download_data import download_dataset
from Backend.data_preprocessing import DataPreprocessor
from Backend.create_data_yaml import create_data_yaml
from Backend.train_models import train_detector, train_classifier
from Backend.inference import VehicleDetectionSystem


def setup_environment():
    """Set environment variables to ensure all cache and temporary files are saved in the project directory."""
    import os
    # Create necessary directories
    project_dir = os.getcwd()
    cache_dir = os.path.join(project_dir, ".cache")
    temp_dir = os.path.join(project_dir, ".temp")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    # Set environment variables
    os.environ["TORCH_HOME"] = cache_dir
    os.environ["ULTRALYTICS_HOME"] = cache_dir  # YOLO cache
    os.environ["TMPDIR"] = temp_dir
    os.environ["TEMP"] = temp_dir
    os.environ["TMP"] = temp_dir
    print(f"Environment variables have been set. All cache and temporary files will be saved in the project directory.")

# Set environment variables before importing other modules
setup_environment()


def parse_args():
    parser = argparse.ArgumentParser(description='Vehicle Detection and Classification System')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['download', 'preprocess', 'train', 'infer', 'all', 'check_gpu'],
                        help='Operating mode')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--image', type=str, default=None, help='Path to the image for inference')
    parser.add_argument('--video', type=str, default=None, help='Path to the video for inference')
    parser.add_argument('--force_preprocess', action='store_true', help='Force reprocessing of data')
    return parser.parse_args()


def main():
    args = parse_args()
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    dataset_path = None
    if args.mode == 'download' or args.mode == 'all':
        print("=== Downloading the dataset ===")
        dataset_path = download_dataset()
    if args.mode == 'preprocess' or args.mode == 'all':
        print("\n=== Preprocessing the data ===")
        if dataset_path is None:
            # Try using the default path
            dataset_path = os.path.join("data", "roundabout-aerial-images-for-vehicle-detection")
            if not os.path.exists(dataset_path):
                # Try using the local path
                dataset_path = "C:/Users/14919/Desktop/岭大/CDS521/vehicle detection/data/kagglehub/datasets/javiersanchezsoriano/roundabout-aerial-images-for-vehicle-detection/versions/2"
            print(f"Using dataset path: {dataset_path}")
        preprocessor = DataPreprocessor(dataset_path)
        preprocessor.prepare_dataset(force_preprocess=args.force_preprocess)
        preprocessor.extract_vehicle_crops(force_extract=args.force_preprocess)
    if args.mode == 'train' or args.mode == 'all':
        print("\n=== Training the models ===")
        data_yaml_path = create_data_yaml("data/processed")
        detector = train_detector(data_yaml_path, epochs=args.epochs)
        classifier = train_classifier("data/vehicle_crops", epochs=args.epochs // 2)
    if args.mode == 'infer' or args.mode == 'all' or args.image or args.video:
        print("\n=== Model inference ===")
        detector_path = "models/vehicle_detection/weights/best.pt"
        classifier_path = "models/best_vehicle_classifier.pth"
        if not os.path.exists(detector_path):
            print(f"Error: Detector model does not exist at {detector_path}")
            return
        system = VehicleDetectionSystem(
            detector_path=detector_path,
            classifier_path=classifier_path if os.path.exists(classifier_path) else None
        )
        if args.image:
            system.process_image(args.image)
        elif args.mode == 'all':
            # Process an image from the test set
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