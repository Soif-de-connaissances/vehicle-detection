import os
import yaml

def create_data_yaml(data_dir, output_path="data/vehicle_data.yaml"):
    """
    Create a data configuration file required for YOLO training.

    Args:
        data_dir: Data directory.
        output_path: Output YAML file path.
    """
    print("Creating data configuration file...")

    # Read class mapping
    class_names = []
    if os.path.exists("models/class_mapping.txt"):
        with open("models/class_mapping.txt", 'r') as f:
            for line in f:
                parts = line.strip().split(': ')
                if len(parts) == 2:
                    class_id = int(parts[0])
                    class_name = parts[1]
                    # Ensure the list length is sufficient
                    while len(class_names) <= class_id:
                        class_names.append(f"class_{len(class_names)}")
                    class_names[class_id] = class_name

    # If there is no class mapping, try to infer from the label files
    if not class_names:
        train_label_dir = os.path.join(data_dir, "train", "labels")
        class_ids = set()

        for label_file in os.listdir(train_label_dir):
            if not label_file.endswith('.txt'):
                continue

            with open(os.path.join(train_label_dir, label_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        class_ids.add(int(parts[0]))

        class_names = [f"vehicle_{i}" for i in sorted(class_ids)]

    # Create YAML configuration
    data_yaml = {
        'path': os.path.abspath(data_dir),
        'train': os.path.join('train', 'images'),
        'val': os.path.join('val', 'images'),
        'test': os.path.join('test', 'images'),
        'nc': len(class_names),
        'names': class_names
    }

    # Save the YAML file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"Data configuration file saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    create_data_yaml("data/processed")