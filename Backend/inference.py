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
        初始化车辆检测与分类系统
        
        Args:
            detector_path: 检测器模型路径
            classifier_path: 分类器模型路径（可选）
        """
        # 加载检测器
        self.detector = YOLO(detector_path)
        
        # 加载类别映射
        self.detector_classes = self.detector.names
        
        # 加载分类器（如果提供）
        self.classifier = None
        self.classifier_classes = None
        
        if classifier_path and os.path.exists(classifier_path):
            # 加载分类器类别映射
            self.classifier_classes = self._load_classifier_classes()
            
            if self.classifier_classes:
                # 加载分类器模型
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.classifier = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
                
                # 修改最后一层以匹配类别数量
                in_features = self.classifier.fc.in_features
                self.classifier.fc = torch.nn.Linear(in_features, len(self.classifier_classes))
                
                # 加载权重
                self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
                self.classifier.to(self.device)
                self.classifier.eval()
                
                # 分类器预处理
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
    
    def _load_classifier_classes(self):
        """加载分类器类别映射"""
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
        处理单张图像
        
        Args:
            image_path: 图像路径
            conf: 检测置信度阈值
            save_result: 是否保存结果
            output_path: 输出路径
        
        Returns:
            处理后的图像和检测结果
        """
        # 检测车辆
        results = self.detector(image_path, conf=conf)
        
        # 读取原始图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误：无法读取图像 {image_path}")
            return None, None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 提取检测框
        boxes = results[0].boxes
        
        # 分类结果
        classifications = []
        
        # 处理每个检测框
        for i, box in enumerate(boxes):
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # 获取检测类别
            cls_id = int(box.cls[0].item())
            cls_name = self.detector_classes[cls_id]
            confidence = box.conf[0].item()
            
            # 分类结果
            classification = {
                'box': (x1, y1, x2, y2),
                'detector_class_id': cls_id,
                'detector_class_name': cls_name,
                'detector_confidence': confidence,
                'classifier_class_name': None,
                'classifier_confidence': None
            }
            
            # 如果有分类器，进行分类
            if self.classifier is not None:
                # 提取车辆裁剪
                vehicle_crop = image_rgb[y1:y2, x1:x2]
                
                if vehicle_crop.size > 0:
                    # 转换为PIL图像
                    vehicle_pil = Image.fromarray(vehicle_crop)
                    
                    # 预处理
                    vehicle_tensor = self.transform(vehicle_pil).unsqueeze(0).to(self.device)
                    
                    # 分类
                    with torch.no_grad():
                        outputs = self.classifier(vehicle_tensor)
                        _, predicted = outputs.max(1)
                        confidence = torch.softmax(outputs, dim=1)[0, predicted].item()
                    
                    # 获取类别名称
                    class_id = predicted.item()
                    class_name = self.classifier_classes.get(class_id, f"class_{class_id}")
                    
                    # 更新分类结果
                    classification['classifier_class_name'] = class_name
                    classification['classifier_confidence'] = confidence
            
            # 保存分类结果
            classifications.append(classification)
            
            # 在图像上绘制边界框和类别
            color = (0, 255, 0)  # 绿色边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 添加类别标签
            if classification['classifier_class_name']:
                label = f"{classification['classifier_class_name']}: {classification['classifier_confidence']:.2f}"
            else:
                label = f"{classification['detector_class_name']}: {classification['detector_confidence']:.2f}"
            
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 保存结果
        if save_result:
            if output_path is None:
                os.makedirs('results/processed_images', exist_ok=True)
                output_path = f"results/processed_images/{os.path.basename(image_path)}"
            
            cv2.imwrite(output_path, image)
            print(f"处理后的图像已保存到: {output_path}")
        
        return image, classifications
    
    def process_video(self, video_path, conf=0.25, output_path=None):
        """
        处理视频
        
        Args:
            video_path: 视频路径
            conf: 检测置信度阈值
            output_path: 输出路径
        """
        if output_path is None:
            os.makedirs('results/processed_videos', exist_ok=True)
            output_path = f"results/processed_videos/{os.path.basename(video_path)}"
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频 {video_path}")
            return
        
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"处理帧 {frame_count}")
            
            # 保存当前帧为临时图像
            temp_frame_path = 'temp_frame.jpg'
            cv2.imwrite(temp_frame_path, frame)
            
            # 处理帧
            try:
                processed_frame, _ = self.process_image(
                    temp_frame_path, conf=conf, save_result=False
                )
                
                if processed_frame is not None:
                    # 写入处理后的帧
                    out.write(processed_frame)
                else:
                    # 写入原始帧
                    out.write(frame)
            except Exception as e:
                print(f"处理帧 {frame_count} 时出错: {e}")
                # 写入原始帧
                out.write(frame)
        
        # 释放资源
        cap.release()
        out.release()
        
        # 删除临时文件
        if os.path.exists(temp_frame_path):
            os.remove(temp_frame_path)
        
        print(f"处理后的视频已保存到: {output_path}")

def main():
    # 检查模型文件是否存在
    detector_path = "models/vehicle_detection/weights/best.pt"
    classifier_path = "models/best_vehicle_classifier.pth"
    
    if not os.path.exists(detector_path):
        print(f"错误：检测器模型不存在 {detector_path}")
        return
    
    # 创建系统实例
    system = VehicleDetectionSystem(
        detector_path=detector_path,
        classifier_path=classifier_path if os.path.exists(classifier_path) else None
    )
    
    # 处理测试集中的图像
    test_dir = "data/processed/test/images"
    if os.path.exists(test_dir):
        test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if test_images:
            test_image_path = os.path.join(test_dir, test_images[0])
            system.process_image(test_image_path)
    
    # 处理示例视频（如果有）
    video_path = "data/test_video.mp4"
    if os.path.exists(video_path):
        system.process_video(video_path)

if __name__ == "__main__":
    main()