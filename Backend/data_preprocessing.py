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
        初始化数据预处理器
        
        Args:
            dataset_path: 数据集路径
        """
        self.dataset_path = dataset_path
        self.processed_data_dir = "data/processed"
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # 创建训练、验证和测试目录
        self.train_dir = os.path.join(self.processed_data_dir, "train")
        self.val_dir = os.path.join(self.processed_data_dir, "val")
        self.test_dir = os.path.join(self.processed_data_dir, "test")
        
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            os.makedirs(dir_path, exist_ok=True)
            os.makedirs(os.path.join(dir_path, "images"), exist_ok=True)
            os.makedirs(os.path.join(dir_path, "labels"), exist_ok=True)
        
        # 类别映射字典
        self.class_mapping = {}
        
        # 图像文件缓存
        self.image_files_cache = {}
    
    def prepare_dataset(self, split_ratio=(0.7, 0.15, 0.15), force_preprocess=False):
        """
        准备数据集
        
        Args:
            split_ratio: 训练、验证、测试集的比例，默认为(0.7, 0.15, 0.15)
            force_preprocess: 是否强制重新预处理，即使已存在预处理数据
        """
        # 检查是否已经存在预处理数据
        train_images_dir = os.path.join(self.train_dir, "images")
        val_images_dir = os.path.join(self.val_dir, "images")
        test_images_dir = os.path.join(self.test_dir, "images")
        
        train_labels_dir = os.path.join(self.train_dir, "labels")
        val_labels_dir = os.path.join(self.val_dir, "labels")
        test_labels_dir = os.path.join(self.test_dir, "labels")
        
        # 检查目录是否存在并包含文件
        dirs_to_check = [
            (train_images_dir, "训练图像"),
            (val_images_dir, "验证图像"),
            (test_images_dir, "测试图像"),
            (train_labels_dir, "训练标签"),
            (val_labels_dir, "验证标签"),
            (test_labels_dir, "测试标签")
        ]
        
        all_dirs_exist_with_files = True
        for dir_path, dir_desc in dirs_to_check:
            if not os.path.exists(dir_path) or len(os.listdir(dir_path)) == 0:
                print(f"{dir_desc}目录不存在或为空: {dir_path}")
                all_dirs_exist_with_files = False
                break
        
        # 如果所有目录都存在且包含文件，并且不强制重新处理
        if all_dirs_exist_with_files and not force_preprocess:
            print("检测到已存在预处理数据，跳过预处理步骤。")
            print("如需重新预处理，请设置 force_preprocess=True")
            
            # 加载类别映射
            self._load_class_mapping()
            
            # 打印数据集统计信息
            train_images = os.listdir(train_images_dir)
            val_images = os.listdir(val_images_dir)
            test_images = os.listdir(test_images_dir)
            
            print(f"训练集：{len(train_images)}张图像")
            print(f"验证集：{len(val_images)}张图像")
            print(f"测试集：{len(test_images)}张图像")
            
            return True
        
        print("准备数据集...")
        
        # 查找CSV标注文件
        annotations_file = self._find_annotations_file()
        if not annotations_file:
            print("错误：找不到标注文件")
            return False
        
        # 查找图像目录
        images_dir = self._find_images_directory()
        if not images_dir:
            print("错误：找不到图像目录")
            return False
        
        # 预先扫描所有图像文件
        self._scan_image_files(self.dataset_path)
        print(f"找到 {len(self.image_files_cache)} 个图像文件")
        
        # 读取标注文件
        annotations = self._read_annotations(annotations_file)
        
        # 获取所有唯一的图像名称
        image_names = list(set([ann['image_name'] for ann in annotations]))
        random.shuffle(image_names)
        
        # 计算分割点
        train_end = int(len(image_names) * split_ratio[0])
        val_end = train_end + int(len(image_names) * split_ratio[1])
        
        # 分割数据集
        train_images = image_names[:train_end]
        val_images = image_names[train_end:val_end]
        test_images = image_names[val_end:]
        
        print(f"训练集：{len(train_images)}张图像")
        print(f"验证集：{len(val_images)}张图像")
        print(f"测试集：{len(test_images)}张图像")
        
        # 处理每个分割
        self._process_split(train_images, annotations, images_dir, self.train_dir)
        self._process_split(val_images, annotations, images_dir, self.val_dir)
        self._process_split(test_images, annotations, images_dir, self.test_dir)
        
        # 保存类别映射
        self._save_class_mapping()
        
        # 检查处理后的目录是否包含图像
        train_images_processed = os.listdir(os.path.join(self.train_dir, "images"))
        val_images_processed = os.listdir(os.path.join(self.val_dir, "images"))
        test_images_processed = os.listdir(os.path.join(self.test_dir, "images"))
        
        print(f"处理后的训练图像: {len(train_images_processed)}")
        print(f"处理后的验证图像: {len(val_images_processed)}")
        print(f"处理后的测试图像: {len(test_images_processed)}")
        
        if len(train_images_processed) == 0 or len(val_images_processed) == 0:
            print("警告: 处理后的图像目录为空！请检查数据预处理过程。")
            return False
        
        print("数据集准备完成！")
        return True
    
    def _scan_image_files(self, root_dir):
        """预先扫描所有图像文件并缓存路径"""
        print(f"扫描图像文件...")
        count = 0
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')):
                    # 使用文件名作为键，完整路径作为值
                    self.image_files_cache[file] = os.path.join(root, file)
                    
                    # 同时存储不带扩展名的文件名
                    base_name = os.path.splitext(file)[0]
                    if base_name not in self.image_files_cache:
                        self.image_files_cache[base_name] = os.path.join(root, file)
                    
                    count += 1
                    if count % 1000 == 0:
                        print(f"已扫描 {count} 个图像文件...")
    
    def _find_annotations_file(self):
        """查找CSV标注文件"""
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.csv'):
                    print(f"找到标注文件: {os.path.join(root, file)}")
                    return os.path.join(root, file)
        return None
    
    def _find_images_directory(self):
        """查找图像目录"""
        # 首先检查特定路径
        possible_img_dirs = [
            os.path.join(self.dataset_path, "images"),
            os.path.join(self.dataset_path, "img"),
            os.path.join(self.dataset_path, "image"),
            os.path.join(self.dataset_path, "imgs"),
            self.dataset_path  # 数据集根目录本身
        ]
        
        # 检查是否有 "original/imgs" 目录
        for root, dirs, _ in os.walk(self.dataset_path):
            for dir_name in dirs:
                if dir_name.lower() in ['imgs', 'images', 'img', 'image']:
                    possible_img_dirs.append(os.path.join(root, dir_name))
        
        for dir_path in possible_img_dirs:
            if os.path.exists(dir_path):
                # 检查目录中是否包含图像文件
                image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'))]
                if image_files:
                    print(f"找到图像目录: {dir_path}，包含 {len(image_files)} 张图像")
                    return dir_path
        
        # 如果没有找到明确的图像目录，返回数据集根目录
        print(f"未找到专门的图像目录，使用数据集根目录: {self.dataset_path}")
        return self.dataset_path
    
    def _read_annotations(self, annotations_file):
        """读取CSV标注文件"""
        annotations = []
        
        with open(annotations_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # 跳过标题行
            
            # 确定列的索引
            image_name_idx = 0  # 假设第一列是图像名称
            x_min_idx = 1       # 假设第二列是x_min
            y_min_idx = 2       # 假设第三列是y_min
            x_max_idx = 3       # 假设第四列是x_max
            y_max_idx = 4       # 假设第五列是y_max
            class_name_idx = 5  # 假设第六列是类别名称
            
            for row in reader:
                if len(row) <= max(image_name_idx, x_min_idx, y_min_idx, x_max_idx, y_max_idx, class_name_idx):
                    continue  # 跳过格式不正确的行
                
                # 检查坐标值是否为空
                try:
                    x_min = float(row[x_min_idx]) if row[x_min_idx].strip() else 0.0
                    y_min = float(row[y_min_idx]) if row[y_min_idx].strip() else 0.0
                    x_max = float(row[x_max_idx]) if row[x_max_idx].strip() else 0.0
                    y_max = float(row[y_max_idx]) if row[y_max_idx].strip() else 0.0
                    
                    # 跳过无效的边界框（宽度或高度为0）
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
                    
                    # 更新类别映射
                    if annotation['class_name'] not in self.class_mapping:
                        self.class_mapping[annotation['class_name']] = len(self.class_mapping)
                    
                    annotations.append(annotation)
                except ValueError:
                    # 如果转换失败，跳过这一行
                    continue
        
        print(f"读取了 {len(annotations)} 条标注记录")
        print(f"类别映射: {self.class_mapping}")
        
        return annotations
    
    def _process_split(self, image_names, annotations, images_dir, output_dir):
        """处理数据集分割"""
        processed_count = 0
        skipped_count = 0
        
        for image_name in tqdm(image_names, desc=f"处理 {os.path.basename(output_dir)}"):
            # 获取该图像的所有标注
            image_annotations = [ann for ann in annotations if ann['image_name'] == image_name]
            
            if not image_annotations:
                skipped_count += 1
                continue
            
            # 查找图像文件
            image_path = self._find_image_file(image_name)
            if not image_path:
                skipped_count += 1
                continue
            
            # 确保使用原始文件的扩展名
            _, ext = os.path.splitext(image_path)
            if not ext:
                ext = '.jpg'  # 默认扩展名
            
            # 复制图像，确保目标文件名有正确的扩展名
            base_name = os.path.splitext(os.path.basename(image_name))[0]
            dst_img_path = os.path.join(output_dir, "images", base_name + ext)
            
            try:
                shutil.copy2(image_path, dst_img_path)
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"已复制 {processed_count} 张图像到 {os.path.basename(output_dir)}")
            except Exception as e:
                print(f"复制图像时出错 ({image_path} -> {dst_img_path}): {e}")
                skipped_count += 1
                continue
            
            # 读取图像获取尺寸
            img = cv2.imread(dst_img_path)  # 使用复制后的图像路径
            if img is None:
                print(f"警告：无法读取图像 {dst_img_path}")
                os.remove(dst_img_path)  # 删除无法读取的图像
                skipped_count += 1
                continue
            
            img_height, img_width = img.shape[:2]
            
            # 创建YOLO格式的标注文件
            label_file = os.path.join(output_dir, "labels", base_name + ".txt")
            
            with open(label_file, 'w') as f:
                for ann in image_annotations:
                    # 获取类别ID
                    class_id = self.class_mapping[ann['class_name']]
                    
                    # 转换为YOLO格式 (class_id, x_center, y_center, width, height)
                    x_min = ann['x_min']
                    y_min = ann['y_min']
                    x_max = ann['x_max']
                    y_max = ann['y_max']
                    
                    # 计算中心点和宽高
                    x_center = (x_min + x_max) / 2 / img_width
                    y_center = (y_min + y_max) / 2 / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height
                    
                    # 写入YOLO格式
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        
        print(f"总共复制了 {processed_count} 张图像到 {os.path.basename(output_dir)}")
        print(f"跳过了 {skipped_count} 张图像")
    
    def _find_image_file(self, image_name):
        """查找图像文件"""
        # 首先检查缓存
        if image_name in self.image_files_cache:
            return self.image_files_cache[image_name]
        
        # 尝试从文件名中提取基本名称
        base_name = os.path.basename(image_name)
        if base_name in self.image_files_cache:
            return self.image_files_cache[base_name]
        
        # 尝试不带扩展名
        base_name_no_ext = os.path.splitext(base_name)[0]
        if base_name_no_ext in self.image_files_cache:
            return self.image_files_cache[base_name_no_ext]
        
        # 尝试查找包含基本名称的文件
        for file_name, file_path in self.image_files_cache.items():
            if base_name_no_ext in file_name:
                return file_path
        
        # 如果仍然找不到，尝试更灵活的匹配
        # 例如，如果image_name是"original/imgs/00001_frame000058_original.jpg"
        # 尝试查找包含"00001_frame000058"的文件
        parts = base_name_no_ext.split('_')
        if len(parts) >= 2:
            search_pattern = '_'.join(parts[:2])  # 例如 "00001_frame000058"
            for file_name, file_path in self.image_files_cache.items():
                if search_pattern in file_name:
                    return file_path
        
        print(f"警告：找不到图像文件 {image_name}")
        return None
    
    def _save_class_mapping(self):
        """保存类别映射"""
        os.makedirs("models", exist_ok=True)
        
        with open("models/class_mapping.txt", 'w') as f:
            for class_name, class_id in self.class_mapping.items():
                f.write(f"{class_id}: {class_name}\n")
        
        print(f"类别映射已保存到 models/class_mapping.txt")
    
    def extract_vehicle_crops(self, output_dir="data/vehicle_crops", force_extract=False):
        """
        从图像中提取车辆裁剪，用于分类模型训练
        
        Args:
            output_dir: 输出目录
            force_extract: 是否强制重新提取，即使已存在裁剪数据
        """
        # 检查是否已经存在裁剪数据
        train_crops_dir = os.path.join(output_dir, "train")
        val_crops_dir = os.path.join(output_dir, "val")
        
        if (os.path.exists(train_crops_dir) and os.path.exists(val_crops_dir) and 
            len(os.listdir(train_crops_dir)) > 0 and len(os.listdir(val_crops_dir)) > 0 and
            not force_extract):
            
            # 计算类别数量和每个类别的样本数
            train_classes = [d for d in os.listdir(train_crops_dir) if os.path.isdir(os.path.join(train_crops_dir, d))]
            val_classes = [d for d in os.listdir(val_crops_dir) if os.path.isdir(os.path.join(val_crops_dir, d))]
            
            print("检测到已存在车辆裁剪数据，跳过提取步骤。")
            print(f"训练集: {len(train_classes)} 个类别")
            print(f"验证集: {len(val_classes)} 个类别")
            
            for cls in train_classes:
                cls_dir = os.path.join(train_crops_dir, cls)
                print(f"  {cls}: {len(os.listdir(cls_dir))} 个样本")
            
            print("如需重新提取，请设置 force_extract=True")
            return True
        
        print("提取车辆裁剪图像用于分类...")
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
        
        # 处理训练集中的图像
        self._extract_crops_from_split(self.train_dir, os.path.join(output_dir, "train"))
        
        # 处理验证集中的图像
        self._extract_crops_from_split(self.val_dir, os.path.join(output_dir, "val"))
        
        print("车辆裁剪提取完成！")
        return True
    
    def _extract_crops_from_split(self, split_dir, output_dir):
        """从指定分割中提取车辆裁剪"""
        images_dir = os.path.join(split_dir, "images")
        labels_dir = os.path.join(split_dir, "labels")
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"警告：目录不存在 {images_dir} 或 {labels_dir}")
            return
        
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        for img_file in tqdm(image_files, desc=f"从{os.path.basename(split_dir)}提取裁剪"):
            # 读取图像
            img_path = os.path.join(images_dir, img_file)
            image = cv2.imread(img_path)
            if image is None:
                print(f"警告: 无法读取图像 {img_path}")
                continue
            
            # 读取对应的标签文件
            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_file)
            
            if not os.path.exists(label_path):
                continue
            
            # 读取标签
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            img_height, img_width = image.shape[:2]
            
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                # YOLO格式: class x_center y_center width height
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                # 计算边界框坐标
                x1 = int(max(0, x_center - width / 2))
                y1 = int(max(0, y_center - height / 2))
                x2 = int(min(img_width, x_center + width / 2))
                y2 = int(min(img_height, y_center + height / 2))
                
                # 提取裁剪
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                
                # 创建类别目录
                class_dir = os.path.join(output_dir, f"class_{class_id}")
                os.makedirs(class_dir, exist_ok=True)
                
                # 保存裁剪
                crop_filename = f"{os.path.splitext(img_file)[0]}_crop_{i}.jpg"
                cv2.imwrite(os.path.join(class_dir, crop_filename), crop)

    def _load_class_mapping(self):
        """加载已有的类别映射"""
        class_mapping_path = "models/class_mapping.txt"
        
        if os.path.exists(class_mapping_path):
            with open(class_mapping_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(': ')
                    if len(parts) == 2:
                        class_id = int(parts[0])
                        class_name = parts[1]
                        self.class_mapping[class_name] = class_id
            
            print(f"已加载类别映射: {self.class_mapping}")
            return True
        
        print(f"警告：找不到类别映射文件 {class_mapping_path}")
        return False