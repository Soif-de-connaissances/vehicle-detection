# Vehicle Detection and Classification System

This is a deep learning-based vehicle detection and classification system that uses YOLOv8 for vehicle detection and ResNet50 for vehicle type classification. The system supports both image and video processing, and provides a web interface based on Streamlit.

## Features

- Vehicle Detection: Uses YOLOv8 model to detect vehicles in images/videos
- Vehicle Classification: Uses ResNet50 model to classify detected vehicles
- Supports both image and video processing
- Provides a web interface for user interaction
- Supports batch processing and data preprocessing
- Customizable training parameters and model configurations

## System Requirements

- Python 3.8+
- CUDA-supported GPU (recommended)
- Minimum 8GB RAM
- Sufficient storage space for datasets and models

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vehicle-detection.git
```

## Usage

### 1. Data Preparation

```bash
python main.py --mode download  # Download dataset
python main.py --mode preprocess  # Preprocess data
```

### 2. Model Training

```bash
python main.py --mode train --epochs 50  # Train detector and classifier
```

### 3. Run Demo

```bash
streamlit run vehicle_detection_demo.py
```

### 4. Command Line Inference

```bash
# Process single image
python main.py --mode infer --image path/to/image.jpg

# Process video
python main.py --mode infer --video path/to/video.mp4
```

## Project Structure

```
vehicle-detection/
├── Backend/                 # Backend processing modules
├── data/                    # Dataset directory
├── models/                  # Model files
├── results/                 # Processing results
├── runs/                    # training results
├── main.py                  # Main program
├── vehicle_detection_demo.py # Demo program
```

## Configuration Options

- `--mode`: Operation mode (download/preprocess/train/infer/all)
- `--epochs`: Number of training epochs
- `--image`: Path to image for inference
- `--video`: Path to video for inference
- `--force_preprocess`: Force data preprocessing

## Contributing

Issues and Pull Requests are welcome to help improve this project.

## License

[MIT License](LICENSE)