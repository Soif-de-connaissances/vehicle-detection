import os
import base64
import cv2
import torch
import numpy as np
import streamlit as st
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image


class VehicleDetectionSystem:
    def __init__(
            self,
            detector_path="runs/detect/vehicle_detection22/weights/best.pt",
            classifier_path="models/best_vehicle_classifier.pth",
            class_mapping_path="models/classifier_class_mapping.txt",
            conf_threshold=0.25
    ):
        st.text("Initializing the vehicle detection and classification system...")

        # Save paths and parameters
        self.detector_path = detector_path
        self.classifier_path = classifier_path
        self.class_mapping_path = class_mapping_path
        self.conf_threshold = conf_threshold

        # Load the detector
        st.text("Loading the vehicle detector...")
        self.detector = YOLO(detector_path)

        # Load the class mapping
        st.text("Loading the class mapping...")
        self.class_mapping = self.load_class_mapping()

        # Load the classifier
        st.text("Loading the vehicle classifier...")
        num_classes = len(self.class_mapping)
        self.classifier, self.device = self.load_classifier(num_classes)

        # Define the transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        st.text("System initialization completed!")

    def load_classifier(self, num_classes):
        """Load the classifier model"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.text(f"Using device: {device}")

        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)

        # Modify the last layer to match the number of classes
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)

        # Load the trained weights
        model.load_state_dict(torch.load(self.classifier_path, map_location=device))
        model = model.to(device)
        model.eval()

        return model, device

    def load_class_mapping(self):
        """Load the class mapping"""
        class_mapping = {}
        with open(self.class_mapping_path, 'r') as f:
            for line in f:
                parts = line.strip().split(': ')
                if len(parts) == 2:
                    idx, class_name = parts
                    class_mapping[int(idx)] = class_name

        return class_mapping

    def classify_vehicle(self, crop_img):
        """Classify the cropped vehicle image"""
        # Convert the OpenCV image to a PIL image
        if isinstance(crop_img, np.ndarray):
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            crop_img = Image.fromarray(crop_img)

        # Apply the transformation
        input_tensor = self.transform(crop_img).unsqueeze(0).to(self.device)

        # Make a prediction
        with torch.no_grad():
            output = self.classifier(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            _, predicted = torch.max(output, 1)

        class_idx = predicted.item()
        confidence = probabilities[class_idx].item()
        class_name = self.class_mapping.get(class_idx, f"Unknown class ({class_idx})")

        return class_name, confidence

    def process_image(self, image):
        """Process an image, detect and classify vehicles"""
        if image is None:
            st.error("Please upload an image first")
            return None, "No image detected"

        # Convert to OpenCV format
        if isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[2] == 4:
            # If the image is in RGBA format, convert it to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Save the temporary image
        temp_path = "temp_image.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Read the image
        img = cv2.imread(temp_path)
        if img is None:
            st.error("Unable to read the image")
            return None, "Unable to read the image"

        # Run the detection
        results = self.detector(img, conf=self.conf_threshold)

        # Create a copy of the result image
        result_img = img.copy()

        # Store the detection results
        detections = []

        # Process each detection result
        for i, det in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
            conf = float(det.conf[0].cpu().numpy())

            # Crop the vehicle image
            crop_img = img[y1:y2, x1:x2]
            if crop_img.size == 0:
                continue

            # Classify the vehicle
            class_name, class_conf = self.classify_vehicle(crop_img)

            # Draw the detection box
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw the label
            label = f"{class_name} {conf:.2f}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(result_img, (x1, y1 - t_size[1] - 10), (x1 + t_size[0], y1), (0, 255, 0), -1)
            cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            # Save the detection result
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_name": class_name,
                "class_confidence": class_conf
            })

        # Delete the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Count the detected vehicle types
        vehicle_counts = {}
        for det in detections:
            vehicle_type = det["class_name"]
            if vehicle_type in vehicle_counts:
                vehicle_counts[vehicle_type] += 1
            else:
                vehicle_counts[vehicle_type] = 1

        # Generate the statistics
        stats = f"Detected {len(detections)} vehicles:\n"
        for vehicle_type, count in vehicle_counts.items():
            stats += f"- {vehicle_type}: {count} vehicles\n"

        # Convert to RGB for Streamlit display
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        return result_img, stats

    def process_video(self, video_file):
        """Process a video, detect and classify vehicles"""
        if video_file is None:
            st.error("Please upload a video first")
            return None, None

        # Save the temporary video
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(video_file.read())

        output_path = os.path.join(os.getcwd(), "results", "processed_video.mp4")
        print(f"Output path: {output_path}")  # Add a print statement to check the path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Open the video
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            st.error("Unable to open the video")
            return None, None

        # Get the video information
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        st.text(f"Video information: {width}x{height}, {fps} FPS, {total_frames} frames")

        # Create the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            st.error("Unable to create the output video file")
            cap.release()
            return None, None

        # Create the progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        frame_count = 0
        vehicle_counts = {}  # Used to count vehicle types

        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress * 100:.1f}%)")

            # Run the detection
            results = self.detector(frame, conf=self.conf_threshold)

            # Create a copy of the result frame
            result_frame = frame.copy()

            # Process each detection result
            for i, det in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
                conf = float(det.conf[0].cpu().numpy())

                # Crop the vehicle image
                crop_img = frame[y1:y2, x1:x2]
                if crop_img.size == 0:
                    continue

                # Classify the vehicle
                class_name, class_conf = self.classify_vehicle(crop_img)

                # Update the vehicle statistics
                if class_name in vehicle_counts:
                    vehicle_counts[class_name] += 1
                else:
                    vehicle_counts[class_name] = 1

                # Draw the detection box
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw the label
                label = f"{class_name} {conf:.2f}"
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.rectangle(result_frame, (x1, y1 - t_size[1] - 10), (x1 + t_size[0], y1), (0, 255, 0), -1)
                cv2.putText(result_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            # Add the frame counter
            cv2.putText(result_frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Write the output video
            try:
                out.write(result_frame)
            except Exception as e:
                st.error(f"Error writing frame: {e}")

        # Release the resources
        cap.release()
        out.release()

        # Delete the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        status_text.text(f"Processing completed! A total of {frame_count} frames were processed.")
        st.success(f"The processed video has been saved to: {output_path}")

        # Generate the statistics
        total_vehicles = sum(vehicle_counts.values())
        stats = f"A total of {total_vehicles} vehicles were detected in the video:\n"
        for vehicle_type, count in vehicle_counts.items():
            stats += f"- {vehicle_type}: {count} vehicles\n"

        return output_path, stats, width, height, fps, total_frames, frame_count


def main():
    st.set_page_config(
        page_title="Vehicle Detection and Classification System",
        page_icon="🚗",
        layout="wide"
    )

    # Set the global style
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        background-color: #F8F9FA;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4B8BF5 !important;
        color: white !important;
    }
    .stButton > button {
        width: 100%;
        background-color: #4B8BF5;
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 4px;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #1967D2;
    }
    .empty-result {
        width: 100%;
        height: 300px;
        border: 1px dashed #E0E0E0;
        border-radius: 10px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background-color: #F8F9FA;
    }
    .empty-icon {
        font-size: 48px;
        color: #80868B;
        margin-bottom: 16px;
    }
    .empty-text {
        color: #5F6368;
        font-size: 16px;
        text-align: center;
    }
    .stats-card {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #4B8BF5;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Vehicle Detection and Classification System")
    st.markdown("Upload an image or video, and the system will automatically detect and classify vehicles.")

    # Initialize the system (only on the first run)
    @st.cache_resource
    def load_detection_system():
        return VehicleDetectionSystem(
            detector_path="runs/detect/vehicle_detection22/weights/best.pt",
            classifier_path="models/best_vehicle_classifier.pth",
            class_mapping_path="models/classifier_class_mapping.txt",
            conf_threshold=0.25
        )

    # Use try-except to catch possible initialization errors
    try:
        system = load_detection_system()
    except Exception as e:
        st.error(f"System initialization failed: {str(e)}")
        st.error("Please ensure that the model files exist in the correct paths.")
        st.stop()

    # Create tabs
    tab1, tab2 = st.tabs(["Image Detection", "Video Detection"])

    # Image detection tab
    with tab1:
        st.header("Image Detection")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Input Image")

            # Use Streamlit's upload control directly
            uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image_uploader")

            # Display the detection button below the upload control
            if uploaded_image is not None:
                # Display the detection button between the upload control and the image
                if st.button("Start Detection", key="image_detect"):
                    with st.spinner("Processing the image..."):
                        result_img, stats = system.process_image(np.array(Image.open(uploaded_image)))
                        st.session_state.result_img = result_img
                        st.session_state.stats = stats

            # Display the image only when an image is uploaded
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            st.markdown("### Detection Results")
            if 'result_img' in st.session_state and st.session_state.result_img is not None:
                # Create a result container
                result_container = st.container()
                with result_container:
                    # Display the statistics first
                    stats_html = f"""
                    <div class="stats-card">
                        <h4>Statistics</h4>
                        <pre>{st.session_state.stats.replace(chr(10), '<br>')}</pre>
                    </div>
                    """
                    st.markdown(stats_html, unsafe_allow_html=True)

                    # Then display the detection result image
                    st.image(st.session_state.result_img, caption="Detection Results", use_column_width=True)
            else:
                # Create an empty result area
                st.markdown("""
                <div class="empty-result">
                    <div class="empty-icon">🔍</div>
                    <div class="empty-text">Please upload an image and click Start Detection to view the results.</div>
                </div>
                """, unsafe_allow_html=True)

    # Video detection tab
    with tab2:
        st.header("Video Detection")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Upload Video")
            uploaded_video = st.file_uploader("Click or drag a video here", type=["mp4", "avi", "mov"], label_visibility='hidden')
            if uploaded_video is not None:
                st.video(uploaded_video)

        with col2:
            st.markdown("### Processing Results")
            if uploaded_video is not None:
                if st.button("Start Detection", key="video_detect"):
                    with st.spinner("Processing the video..."):
                        output_path, stats, width, height, fps, total_frames, frame_count = system.process_video(uploaded_video)
                        st.session_state.output_path = output_path
                        st.session_state.video_stats = stats
                        if output_path and os.path.exists(output_path):
                            st.success("Video processing completed!")
                            st.text(f"Video information: {width}x{height}, {fps} FPS, {total_frames} frames")
                            st.text(f"Processing completed! A total of {frame_count} frames were processed.")
                            st.text(f"The processed video has been saved to: {output_path}")
                        else:
                            st.error("Video processing failed.")

            if 'output_path' in st.session_state and st.session_state.output_path and os.path.exists(st.session_state.output_path):
                try:
                    with open(st.session_state.output_path, "rb") as f:
                        video_bytes = f.read()
                        video_base64 = base64.b64encode(video_bytes).decode('utf-8')
                    video_html = f"""
                    <video width="100%" controls>
                        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                    """
                    st.markdown(video_html, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error displaying the video: {e}")
            else:
                st.info("Please upload a video and click Start Detection to view the results.")

            # Display the statistics
            if 'video_stats' in st.session_state:
                stats_html = f"""
                <div class="stats-card">
                    <h4>Statistics</h4>
                    <pre>{st.session_state.video_stats.replace(chr(10), '<br>')}</pre>
                </div>
                """
                st.markdown(stats_html, unsafe_allow_html=True)

        # Video processing instructions
        with st.expander("Video Processing Instructions"):
            st.markdown("""
            - Video processing may take a long time. Please be patient.
            - The processed video will display detection boxes and classification results.
            - The processing speed depends on the video length and resolution.
            - The processed video will be saved in the results directory.
            """)


if __name__ == "__main__":
    main()