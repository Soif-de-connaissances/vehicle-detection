import os
import shutil
import random
import csv
import cv2
import numpy as np
from tqdm import tqdm

class DataPreprocessor:
    def __init__(self, dataset_path):
        """
        Initialize the data preprocessor.

        Args:
            dataset_path: Path to the dataset.
        """
        self.dataset_path = dataset_path
        self.processed_data_dir = "data/processed"
        os.makedirs(self.processed_data_dir, exist_ok=True)

        # Create training, validation, and test directories
        self.train_dir = os.path.join(self.processed_data_dir, "train")
        self.val_dir = os.path.join(self.processed_data_dir, "val")
        self.test_dir = os.path.join(self.processed_data_dir, "test")

        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            os.makedirs(dir_path, exist_ok=True)
            os.makedirs(os.path.join(dir_path, "images"), exist_ok=True)
            os.makedirs(os.path.join(dir_path, "labels"), exist_ok=True)

        # Class mapping dictionary
        self.class_mapping = {}

        # Image file cache
        self.image_files_cache = {}

    def prepare_dataset(self, split_ratio=(0.7, 0.15, 0.15), force_preprocess=False):
        """
        Prepare the dataset.

        Args:
            split_ratio: Ratios for training, validation, and test sets, default is (0.7, 0.15, 0.15).
            force_preprocess: Whether to force re-preprocessing even if preprocessed data exists.
        """
        # Check if preprocessed data already exists
        train_images_dir = os.path.join(self.train_dir, "images")
        val_images_dir = os.path.join(self.val_dir, "images")
        test_images_dir = os.path.join(self.test_dir, "images")

        train_labels_dir = os.path.join(self.train_dir, "labels")
        val_labels_dir = os.path.join(self.val_dir, "labels")
        test_labels_dir = os.path.join(self.test_dir, "labels")

        # Check if directories exist and contain files
        dirs_to_check = [
            (train_images_dir, "Training images"),
            (val_images_dir, "Validation images"),
            (test_images_dir, "Test images"),
            (train_labels_dir, "Training labels"),
            (val_labels_dir, "Validation labels"),
            (test_labels_dir, "Test labels")
        ]

        all_dirs_exist_with_files = True
        for dir_path, dir_desc in dirs_to_check:
            if not os.path.exists(dir_path) or len(os.listdir(dir_path)) == 0:
                print(f"{dir_desc} directory does not exist or is empty: {dir_path}")
                all_dirs_exist_with_files = False
                break

        # If all directories exist and contain files, and not forcing reprocessing
        if all_dirs_exist_with_files and not force_preprocess:
            print("Preprocessed data detected, skipping preprocessing step.")
            print("To re-preprocess, set force_preprocess=True")

            # Load class mapping
            self._load_class_mapping()

            # Print dataset statistics
            train_images = os.listdir(train_images_dir)
            val_images = os.listdir(val_images_dir)
            test_images = os.listdir(test_images_dir)

            print(f"Training set: {len(train_images)} images")
            print(f"Validation set: {len(val_images)} images")
            print(f"Test set: {len(test_images)} images")

            return True

        print("Preparing the dataset...")

        # Find the CSV annotation file
        annotations_file = self._find_annotations_file()
        if not annotations_file:
            print("Error: Annotation file not found")
            return False

        # Find the image directory
        images_dir = self._find_images_directory()
        if not images_dir:
            print("Error: Image directory not found")
            return False

        # Pre-scan all image files
        self._scan_image_files(self.dataset_path)
        print(f"Found {len(self.image_files_cache)} image files")

        # Read the annotation file
        annotations = self._read_annotations(annotations_file)

        # Get all unique image names
        image_names = list(set([ann['image_name'] for ann in annotations]))
        random.shuffle(image_names)

        # Calculate split points
        train_end = int(len(image_names) * split_ratio[0])
        val_end = train_end + int(len(image_names) * split_ratio[1])

        # Split the dataset
        train_images = image_names[:train_end]
        val_images = image_names[train_end:val_end]
        test_images = image_names[val_end:]

        print(f"Training set: {len(train_images)} images")
        print(f"Validation set: {len(val_images)} images")
        print(f"Test set: {len(test_images)} images")

        # Process each split
        self._process_split(train_images, annotations, images_dir, self.train_dir)
        self._process_split(val_images, annotations, images_dir, self.val_dir)
        self._process_split(test_images, annotations, images_dir, self.test_dir)

        # Save class mapping
        self._save_class_mapping()

        # Check if the processed directories contain images
        train_images_processed = os.listdir(os.path.join(self.train_dir, "images"))
        val_images_processed = os.listdir(os.path.join(self.val_dir, "images"))
        test_images_processed = os.listdir(os.path.join(self.test_dir, "images"))

        print(f"Processed training images: {len(train_images_processed)}")
        print(f"Processed validation images: {len(val_images_processed)}")
        print(f"Processed test images: {len(test_images_processed)}")

        if len(train_images_processed) == 0 or len(val_images_processed) == 0:
            print("Warning: Processed image directories are empty! Please check the data preprocessing process.")
            return False

        print("Dataset preparation completed!")
        return True

    def _scan_image_files(self, root_dir):
        """Pre-scan all image files and cache their paths."""
        print(f"Scanning image files...")
        count = 0
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')):
                    # Use the file name as the key and the full path as the value
                    self.image_files_cache[file] = os.path.join(root, file)

                    # Also store the file name without the extension
                    base_name = os.path.splitext(file)[0]
                    if base_name not in self.image_files_cache:
                        self.image_files_cache[base_name] = os.path.join(root, file)

                    count += 1
                    if count % 1000 == 0:
                        print(f"{count} image files scanned...")

    def _find_annotations_file(self):
        """Find the CSV annotation file."""
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.csv'):
                    print(f"Annotation file found: {os.path.join(root, file)}")
                    return os.path.join(root, file)
        return None

    def _find_images_directory(self):
        """Find the image directory."""
        # First, check specific paths
        possible_img_dirs = [
            os.path.join(self.dataset_path, "images"),
            os.path.join(self.dataset_path, "img"),
            os.path.join(self.dataset_path, "image"),
            os.path.join(self.dataset_path, "imgs"),
            self.dataset_path  # The dataset root directory itself
        ]

        # Check if there is an "original/imgs" directory
        for root, dirs, _ in os.walk(self.dataset_path):
            for dir_name in dirs:
                if dir_name.lower() in ['imgs', 'images', 'img', 'image']:
                    possible_img_dirs.append(os.path.join(root, dir_name))

        for dir_path in possible_img_dirs:
            if os.path.exists(dir_path):
                # Check if the directory contains image files
                image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'))]
                if image_files:
                    print(f"Image directory found: {dir_path}, containing {len(image_files)} images")
                    return dir_path

        # If no specific image directory is found, return the dataset root directory
        print(f"No dedicated image directory found, using the dataset root directory: {self.dataset_path}")
        return self.dataset_path

    def _read_annotations(self, annotations_file):
        """Read the CSV annotation file."""
        annotations = []

        with open(annotations_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # Skip the header row

            # Determine the column indices
            image_name_idx = 0  # Assume the first column is the image name
            x_min_idx = 1       # Assume the second column is x_min
            y_min_idx = 2       # Assume the third column is y_min
            x_max_idx = 3       # Assume the fourth column is x_max
            y_max_idx = 4       # Assume the fifth column is y_max
            class_name_idx = 5  # Assume the sixth column is the class name

            for row in reader:
                if len(row) <= max(image_name_idx, x_min_idx, y_min_idx, x_max_idx, y_max_idx, class_name_idx):
                    continue  # Skip rows with incorrect format

                # Check if the coordinate values are empty
                try:
                    x_min = float(row[x_min_idx]) if row[x_min_idx].strip() else 0.0
                    y_min = float(row[y_min_idx]) if row[y_min_idx].strip() else 0.0
                    x_max = float(row[x_max_idx]) if row[x_max_idx].strip() else 0.0
                    y_max = float(row[y_max_idx]) if row[y_max_idx].strip() else 0.0

                    # Skip invalid bounding boxes (width or height is 0)
                    if x_min >= x_max or y_min >= y_max:
                        continue

                    annotation = {
                        'image_name': row[image_name_idx],
                        'x_min': x_min,
                        'y_min': y_min,
                        'x_max': x_max,
                        'y_max': y_max,
                        'class_name': row[class_name_idx] if row[class_name_idx].strip() else 'unknown'
                    }

                    # Update the class mapping
                    if annotation['class_name'] not in self.class_mapping:
                        self.class_mapping[annotation['class_name']] = len(self.class_mapping)

                    annotations.append(annotation)
                except ValueError:
                    # If the conversion fails, skip this row
                    continue

        print(f"{len(annotations)} annotation records read")
        print(f"Class mapping: {self.class_mapping}")

        return annotations

    def _process_split(self, image_names, annotations, images_dir, output_dir):
        """Process the dataset split."""
        processed_count = 0
        skipped_count = 0

        for image_name in tqdm(image_names, desc=f"Processing {os.path.basename(output_dir)}"):
            # Get all annotations for this image
            image_annotations = [ann for ann in annotations if ann['image_name'] == image_name]

            if not image_annotations:
                skipped_count += 1
                continue

            # Find the image file
            image_path = self._find_image_file(image_name)
            if not image_path:
                skipped_count += 1
                continue

            # Ensure the use of the original file extension
            _, ext = os.path.splitext(image_path)
            if not ext:
                ext = '.jpg'  # Default extension

            # Copy the image, ensuring the target file name has the correct extension
            base_name = os.path.splitext(os.path.basename(image_name))[0]
            dst_img_path = os.path.join(output_dir, "images", base_name + ext)

            try:
                shutil.copy2(image_path, dst_img_path)
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"{processed_count} images copied to {os.path.basename(output_dir)}")
            except Exception as e:
                print(f"Error copying image ({image_path} -> {dst_img_path}): {e}")
                skipped_count += 1
                continue

            # Read the image to get its dimensions
            img = cv2.imread(dst_img_path)  # Use the copied image path
            if img is None:
                print(f"Warning: Unable to read image {dst_img_path}")
                os.remove(dst_img_path)  # Delete the unreadable image
                skipped_count += 1
                continue

            img_height, img_width = img.shape[:2]

            # Create the YOLO format annotation file
            label_file = os.path.join(output_dir, "labels", base_name + ".txt")

            with open(label_file, 'w') as f:
                for ann in image_annotations:
                    # Get the class ID
                    class_id = self.class_mapping[ann['class_name']]

                    # Convert to YOLO format (class_id, x_center, y_center, width, height)
                    x_min = ann['x_min']
                    y_min = ann['y_min']
                    x_max = ann['x_max']
                    y_max = ann['y_max']

                    # Calculate the center point and width/height
                    x_center = (x_min + x_max) / 2 / img_width
                    y_center = (y_min + y_max) / 2 / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height

                    # Write in YOLO format
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        print(f"Total {processed_count} images copied to {os.path.basename(output_dir)}")
        print(f"{skipped_count} images skipped")

    def _find_image_file(self, image_name):
        """Find the image file."""
        # First, check the cache
        if image_name in self.image_files_cache:
            return self.image_files_cache[image_name]

        # Try to extract the base name from the file name
        base_name = os.path.basename(image_name)
        if base_name in self.image_files_cache:
            return self.image_files_cache[base_name]

        # Try without the extension
        base_name_no_ext = os.path.splitext(base_name)[0]
        if base_name_no_ext in self.image_files_cache:
            return self.image_files_cache[base_name_no_ext]

        # Try to find a file containing the base name
        for file_name, file_path in self.image_files_cache.items():
            if base_name_no_ext in file_name:
                return file_path

        # If still not found, try a more flexible match
        # For example, if image_name is "original/imgs/00001_frame000058_original.jpg"
        # Try to find a file containing "00001_frame000058"
        parts = base_name_no_ext.split('_')
        if len(parts) >= 2:
            search_pattern = '_'.join(parts[:2])  # For example, "00001_frame000058"
            for file_name, file_path in self.image_files_cache.items():
                if search_pattern in file_name:
                    return file_path

        print(f"Warning: Image file {image_name} not found")
        return None

    def _save_class_mapping(self):
        """Save the class mapping."""
        os.makedirs("models", exist_ok=True)

        with open("models/class_mapping.txt", 'w') as f:
            for class_name, class_id in self.class_mapping.items():
                f.write(f"{class_id}: {class_name}\n")

        print(f"Class mapping saved to models/class_mapping.txt")

    def extract_vehicle_crops(self, output_dir="data/vehicle_crops", force_extract=False):
        """
        Extract vehicle crops from images for classification model training.

        Args:
            output_dir: Output directory.
            force_extract: Whether to force re-extraction even if crop data exists.
        """
        # Check if crop data already exists
        train_crops_dir = os.path.join(output_dir, "train")
        val_crops_dir = os.path.join(output_dir, "val")

        if (os.path.exists(train_crops_dir) and os.path.exists(val_crops_dir) and
            len(os.listdir(train_crops_dir)) > 0 and len(os.listdir(val_crops_dir)) > 0 and
            not force_extract):

            # Calculate the number of classes and the number of samples per class
            train_classes = [d for d in os.listdir(train_crops_dir) if os.path.isdir(os.path.join(train_crops_dir, d))]
            val_classes = [d for d in os.listdir(val_crops_dir) if os.path.isdir(os.path.join(val_crops_dir, d))]

            print("Existing vehicle crop data detected, skipping extraction step.")
            print(f"Training set: {len(train_classes)} classes")
            print(f"Validation set: {len(val_classes)} classes")

            for cls in train_classes:
                cls_dir = os.path.join(train_crops_dir, cls)
                print(f"  {cls}: {len(os.listdir(cls_dir))} samples")

            print("To re-extract, set force_extract=True")
            return True

        print("Extracting vehicle crop images for classification...")

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)

        # Process images in the training set
        self._extract_crops_from_split(self.train_dir, os.path.join(output_dir, "train"))

        # Process images in the validation set
        self._extract_crops_from_split(self.val_dir, os.path.join(output_dir, "val"))

        print("Vehicle crop extraction completed!")
        return True

    def _extract_crops_from_split(self, split_dir, output_dir):
        """Extract vehicle crops from the specified split."""
        images_dir = os.path.join(split_dir, "images")
        labels_dir = os.path.join(split_dir, "labels")

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"Warning: Directory {images_dir} or {labels_dir} does not exist")
            return

        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

        for img_file in tqdm(image_files, desc=f"Extracting crops from {os.path.basename(split_dir)}"):
            # Read the image
            img_path = os.path.join(images_dir, img_file)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Unable to read image {img_path}")
                continue

            # Read the corresponding label file
            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_file)

            if not os.path.exists(label_path):
                continue

            # Read the labels
            with open(label_path, 'r') as f:
                lines = f.readlines()

            img_height, img_width = image.shape[:2]

            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                # YOLO format: class x_center y_center width height
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height

                # Calculate the bounding box coordinates
                x1 = int(max(0, x_center - width / 2))
                y1 = int(max(0, y_center - height / 2))
                x2 = int(min(img_width, x_center + width / 2))
                y2 = int(min(img_height, y_center + height / 2))

                # Extract the crop
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Create the class directory
                class_dir = os.path.join(output_dir, f"class_{class_id}")
                os.makedirs(class_dir, exist_ok=True)

                # Save the crop
                crop_filename = f"{os.path.splitext(img_file)[0]}_crop_{i}.jpg"
                cv2.imwrite(os.path.join(class_dir, crop_filename), crop)

    def _load_class_mapping(self):
        """Load the existing class mapping."""
        class_mapping_path = "models/class_mapping.txt"

        if os.path.exists(class_mapping_path):
            with open(class_mapping_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(': ')
                    if len(parts) == 2:
                        class_id = int(parts[0])
                        class_name = parts[1]
                        self.class_mapping[class_name] = class_id

            print(f"Class mapping loaded: {self.class_mapping}")
            return True

        print(f"Warning: Class mapping file {class_mapping_path} not found")
        return False