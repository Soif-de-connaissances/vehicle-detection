import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

class VehicleDetectionSystem:
    def __init__(self, detector_path, classifier_path=None):
        """
        Initialize the vehicle detection and classification system.

        Args:
            detector_path: Path to the detector model.
            classifier_path: Path to the classifier model (optional).
        """
        # Load the detector
        self.detector = YOLO(detector_path)
        # Load the class mapping
        self.detector_classes = self.detector.names
        # Load the classifier (if provided)
        self.classifier = None
        self.classifier_classes = None
        if classifier_path and os.path.exists(classifier_path):
            # Load the classifier class mapping
            self.classifier_classes = self._load_classifier_classes()
            if self.classifier_classes:
                # Load the classifier model
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.classifier = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
                # Modify the last layer to match the number of classes
                in_features = self.classifier.fc.in_features
                self.classifier.fc = torch.nn.Linear(in_features, len(self.classifier_classes))
                # Load the weights
                self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
                self.classifier.to(self.device)
                self.classifier.eval()
                # Classifier preprocessing
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    def _load_classifier_classes(self):
        """Load the classifier class mapping."""
        classes = {}
        if os.path.exists("models/classifier_class_mapping.txt"):
            with open("models/classifier_class_mapping.txt", 'r') as f:
                for line in f:
                    parts = line.strip().split(': ')
                    if len(parts) == 2:
                        idx = int(parts[0])
                        class_name = parts[1]
                        classes[idx] = class_name
            return classes
        return None

    def process_image(self, image_path, conf=0.25, save_result=True, output_path=None):
        """
        Process a single image.

        Args:
            image_path: Path to the image.
            conf: Detection confidence threshold.
            save_result: Whether to save the result.
            output_path: Output path.

        Returns:
            The processed image and detection results.
        """
        # Detect vehicles
        results = self.detector(image_path, conf=conf)
        # Read the original image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to read image {image_path}")
            return None, None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Extract detection boxes
        boxes = results[0].boxes
        # Classification results
        classifications = []
        # Process each detection box
        for i, box in enumerate(boxes):
            # Get the bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            # Get the detected class
            cls_id = int(box.cls[0].item())
            cls_name = self.detector_classes[cls_id]
            confidence = box.conf[0].item()
            # Classification result
            classification = {
                'box': (x1, y1, x2, y2),
                'detector_class_id': cls_id,
                'detector_class_name': cls_name,
                'detector_confidence': confidence,
                'classifier_class_name': None,
                'classifier_confidence': None
            }
            # If there is a classifier, perform classification
            if self.classifier is not None:
                # Extract the vehicle crop
                vehicle_crop = image_rgb[y1:y2, x1:x2]
                if vehicle_crop.size > 0:
                    # Convert to a PIL image
                    vehicle_pil = Image.fromarray(vehicle_crop)
                    # Preprocess
                    vehicle_tensor = self.transform(vehicle_pil).unsqueeze(0).to(self.device)
                    # Classification
                    with torch.no_grad():
                        outputs = self.classifier(vehicle_tensor)
                        _, predicted = outputs.max(1)
                        confidence = torch.softmax(outputs, dim=1)[0, predicted].item()
                    # Get the class name
                    class_id = predicted.item()
                    class_name = self.classifier_classes.get(class_id, f"class_{class_id}")
                    # Update the classification result
                    classification['classifier_class_name'] = class_name
                    classification['classifier_confidence'] = confidence
            # Save the classification result
            classifications.append(classification)
            # Draw the bounding box and class on the image
            color = (0, 255, 0)  # Green bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            # Add the class label
            if classification['classifier_class_name']:
                label = f"{classification['classifier_class_name']}: {classification['classifier_confidence']:.2f}"
            else:
                label = f"{classification['detector_class_name']}: {classification['detector_confidence']:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Save the result
        if save_result:
            if output_path is None:
                os.makedirs('results/processed_images', exist_ok=True)
                output_path = f"results/processed_images/{os.path.basename(image_path)}"
            cv2.imwrite(output_path, image)
            print(f"Processed image saved to: {output_path}")
        return image, classifications

    def process_video(self, video_path, conf=0.25, output_path=None):
        """
        Process a video.

        Args:
            video_path: Path to the video.
            conf: Detection confidence threshold.
            output_path: Output path.
        """
        if output_path is None:
            os.makedirs('results/processed_videos', exist_ok=True)
            output_path = f"results/processed_videos/{os.path.basename(video_path)}"
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video {video_path}")
            return
        # Get the video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Create the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            print(f"Processing frame {frame_count}")
            # Save the current frame as a temporary image
            temp_frame_path = 'temp_frame.jpg'
            cv2.imwrite(temp_frame_path, frame)
            # Process the frame
            try:
                processed_frame, _ = self.process_image(
                    temp_frame_path, conf=conf, save_result=False
                )
                if processed_frame is not None:
                    # Write the processed frame
                    out.write(processed_frame)
                else:
                    # Write the original frame
                    out.write(frame)
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                # Write the original frame
                out.write(frame)
        # Release the resources
        cap.release()
        out.release()
        # Delete the temporary file
        if os.path.exists(temp_frame_path):
            os.remove(temp_frame_path)
        print(f"Processed video saved to: {output_path}")

def main():
    # Check if the model files exist
    detector_path = "models/vehicle_detection/weights/best.pt"
    classifier_path = "models/best_vehicle_classifier.pth"
    if not os.path.exists(detector_path):
        print(f"Error: Detector model does not exist at {detector_path}")
        return
    # Create the system instance
    system = VehicleDetectionSystem(
        detector_path=detector_path,
        classifier_path=classifier_path if os.path.exists(classifier_path) else None
    )
    # Process an image from the test set
    test_dir = "data/processed/test/images"
    if os.path.exists(test_dir):
        test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if test_images:
            test_image_path = os.path.join(test_dir, test_images[0])
            system.process_image(test_image_path)
    # Process the example video (if available)
    video_path = "data/test_video.mp4"
    if os.path.exists(video_path):
        system.process_video(video_path)

if __name__ == "__main__":
    main()