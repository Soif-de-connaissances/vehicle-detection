import os
import cv2
import torch
import numpy as np
import gradio as gr
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import tempfile

class VehicleDetectionSystem:
    def __init__(
        self, 
        detector_path="models/vehicle_detection/weights/best.pt",
        classifier_path="models/best_vehicle_classifier.pth",
        class_mapping_path="models/classifier_class_mapping.txt",
        conf_threshold=0.25
    ):
        print("初始化车辆检测与分类系统...")
        
        # 保存路径和参数
        self.detector_path = detector_path
        self.classifier_path = classifier_path
        self.class_mapping_path = class_mapping_path
        self.conf_threshold = conf_threshold
        
        # 加载检测器
        print("加载车辆检测器...")
        self.detector = YOLO(detector_path)
        
        # 加载类别映射
        print("加载类别映射...")
        self.class_mapping = self.load_class_mapping()
        
        # 加载分类器
        print("加载车辆分类器...")
        num_classes = len(self.class_mapping)
        self.classifier, self.device = self.load_classifier(num_classes)
        
        # 定义变换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("系统初始化完成！")
    
    def load_classifier(self, num_classes):
        """加载分类器模型"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        
        # 修改最后一层以匹配类别数量
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
        
        # 加载训练好的权重
        model.load_state_dict(torch.load(self.classifier_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        return model, device
    
    def load_class_mapping(self):
        """加载类别映射"""
        class_mapping = {}
        with open(self.class_mapping_path, 'r') as f:
            for line in f:
                parts = line.strip().split(': ')
                if len(parts) == 2:
                    idx, class_name = parts
                    class_mapping[int(idx)] = class_name
        
        return class_mapping
    
    def classify_vehicle(self, crop_img):
        """对车辆裁剪图像进行分类"""
        # 将OpenCV图像转换为PIL图像
        if isinstance(crop_img, np.ndarray):
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            crop_img = Image.fromarray(crop_img)
        
        # 应用变换
        input_tensor = self.transform(crop_img).unsqueeze(0).to(self.device)
        
        # 进行预测
        with torch.no_grad():
            output = self.classifier(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            _, predicted = torch.max(output, 1)
        
        class_idx = predicted.item()
        confidence = probabilities[class_idx].item()
        class_name = self.class_mapping.get(class_idx, f"未知类别({class_idx})")
        
        return class_name, confidence
    
    def process_image(self, image):
        """处理图像，检测并分类车辆"""
        # 保存临时图像
        temp_path = "temp_image.jpg"
        cv2.imwrite(temp_path, image)
        
        # 读取图像
        img = cv2.imread(temp_path)
        if img is None:
            raise ValueError(f"无法读取图像")
        
        # 运行检测
        results = self.detector(img, conf=self.conf_threshold)
        
        # 创建结果图像的副本
        result_img = img.copy()
        
        # 存储检测结果
        detections = []
        
        # 处理每个检测结果
        for i, det in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
            conf = float(det.conf[0].cpu().numpy())
            
            # 裁剪车辆图像
            crop_img = img[y1:y2, x1:x2]
            if crop_img.size == 0:
                continue
            
            # 分类车辆
            class_name, class_conf = self.classify_vehicle(crop_img)
            
            # 绘制检测框
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"{class_name} {conf:.2f}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(result_img, (x1, y1 - t_size[1] - 10), (x1 + t_size[0], y1), (0, 255, 0), -1)
            cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # 保存检测结果
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_name": class_name,
                "class_confidence": class_conf
            })
        
        # 删除临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # 统计检测到的车辆类型
        vehicle_counts = {}
        for det in detections:
            vehicle_type = det["class_name"]
            if vehicle_type in vehicle_counts:
                vehicle_counts[vehicle_type] += 1
            else:
                vehicle_counts[vehicle_type] = 1
        
        # 生成统计信息
        stats = f"检测到 {len(detections)} 辆车:\n"
        for vehicle_type, count in vehicle_counts.items():
            stats += f"- {vehicle_type}: {count} 辆\n"
        
        return result_img, stats
    
    def process_video(self, video, progress=None):
        """处理视频，检测并分类车辆"""
        # 保存临时视频
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
        
        output_path = "results/processed_video.mp4"
        
        # 确保输出目录存在
        os.makedirs("results", exist_ok=True)
        
        # 保存上传的视频
        with open(temp_path, "wb") as f:
            f.write(video)
        
        # 打开视频
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频")
        
        # 获取视频信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: {width}x{height}, {fps} FPS, {total_frames} 帧")
        
        # 创建输出视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        # 处理每一帧
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if progress is not None:
                progress((frame_count / total_frames))
            
            print(f"处理帧 {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
            
            # 运行检测
            results = self.detector(frame, conf=self.conf_threshold)
            
            # 创建结果帧的副本
            result_frame = frame.copy()
            
            # 处理每个检测结果
            for i, det in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
                conf = float(det.conf[0].cpu().numpy())
                
                # 裁剪车辆图像
                crop_img = frame[y1:y2, x1:x2]
                if crop_img.size == 0:
                    continue
                
                # 分类车辆
                class_name, class_conf = self.classify_vehicle(crop_img)
                
                # 绘制检测框
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制标签
                label = f"{class_name} {conf:.2f}"
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.rectangle(result_frame, (x1, y1 - t_size[1] - 10), (x1 + t_size[0], y1), (0, 255, 0), -1)
                cv2.putText(result_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # 添加帧计数器
            cv2.putText(result_frame, f"Frame: {frame_count}/{total_frames}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 写入输出视频
            out.write(result_frame)
        
        # 释放资源
        cap.release()
        out.release()
        
        # 删除临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        print(f"处理后的视频已保存到: {output_path}")
        return output_path


def main():
    # 初始化系统
    system = VehicleDetectionSystem(
        detector_path="models/vehicle_detection/weights/best.pt",
        classifier_path="models/best_vehicle_classifier.pth",
        class_mapping_path="models/classifier_class_mapping.txt",
        conf_threshold=0.25
    )
    
    # 创建Gradio界面
    with gr.Blocks(title="车辆检测与分类系统") as demo:
        gr.Markdown("# 车辆检测与分类系统")
        gr.Markdown("上传图像或视频，系统将自动检测和分类车辆")
        
        with gr.Tab("图像检测"):
            with gr.Row():
                image_input = gr.Image(label="上传图像")
                image_output = gr.Image(label="检测结果")
            
            stats_output = gr.Textbox(label="统计信息", lines=5)
            image_button = gr.Button("开始检测")
            image_button.click(system.process_image, inputs=[image_input], outputs=[image_output, stats_output])
        
        with gr.Tab("视频检测"):
            with gr.Row():
                video_input = gr.Video(label="上传视频")
                video_output = gr.Video(label="检测结果")
            
            video_button = gr.Button("开始检测")
            video_button.click(
                system.process_video, 
                inputs=[video_input], 
                outputs=[video_output]
            )
            
            gr.Markdown("""
            ### 视频处理说明
            - 视频处理可能需要较长时间，请耐心等待
            - 处理后的视频将显示检测框和分类结果
            - 处理速度取决于视频长度和分辨率
            """)
    
    # 启动演示界面
    demo.launch()

if __name__ == "__main__":
    main()