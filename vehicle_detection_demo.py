import os
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
        st.text("初始化车辆检测与分类系统...")

        # 保存路径和参数
        self.detector_path = detector_path
        self.classifier_path = classifier_path
        self.class_mapping_path = class_mapping_path
        self.conf_threshold = conf_threshold

        # 加载检测器
        st.text("加载车辆检测器...")
        self.detector = YOLO(detector_path)

        # 加载类别映射
        st.text("加载类别映射...")
        self.class_mapping = self.load_class_mapping()

        # 加载分类器
        st.text("加载车辆分类器...")
        num_classes = len(self.class_mapping)
        self.classifier, self.device = self.load_classifier(num_classes)

        # 定义变换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        st.text("系统初始化完成！")

    def load_classifier(self, num_classes):
        """加载分类器模型"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.text(f"使用设备: {device}")

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
        if image is None:
            st.error("请先上传图像")
            return None, "未检测到图像"

        # 转换为OpenCV格式
        if isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[2] == 4:
            # 如果图像是RGBA格式，转换为RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # 保存临时图像
        temp_path = "temp_image.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # 读取图像
        img = cv2.imread(temp_path)
        if img is None:
            st.error("无法读取图像")
            return None, "无法读取图像"

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

        # 转换为RGB以便Streamlit显示
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        return result_img, stats

    def process_video(self, video_file):
        """处理视频，检测并分类车辆"""
        if video_file is None:
            st.error("请先上传视频")
            return None

        # 保存临时视频
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(video_file.read())

        output_path = "results/processed_video.mp4"

        # 确保输出目录存在
        os.makedirs("results", exist_ok=True)

        # 打开视频
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            st.error("无法打开视频")
            return None

        # 获取视频信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        st.text(f"视频信息: {width}x{height}, {fps} FPS, {total_frames} 帧")

        # 创建输出视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 创建进度条
        progress_bar = st.progress(0)
        status_text = st.empty()

        frame_count = 0

        # 处理每一帧
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"处理帧 {frame_count}/{total_frames} ({progress * 100:.1f}%)")

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

        status_text.text(f"处理完成! 共处理 {frame_count} 帧")
        st.success(f"处理后的视频已保存到: {output_path}")

        return output_path


def main():
    st.set_page_config(
        page_title="车辆检测与分类系统",
        page_icon="🚗",
        layout="wide"
    )

    st.title("车辆检测与分类系统")
    st.markdown("上传图像或视频，系统将自动检测和分类车辆")

    # 初始化系统（仅在第一次运行时）
    @st.cache_resource
    def load_detection_system():
        return VehicleDetectionSystem(
            detector_path="runs/detect/vehicle_detection22/weights/best.pt",
            classifier_path="models/best_vehicle_classifier.pth",
            class_mapping_path="models/classifier_class_mapping.txt",
            conf_threshold=0.25
        )

    # 使用try-except捕获可能的初始化错误
    try:
        system = load_detection_system()
    except Exception as e:
        st.error(f"系统初始化失败: {str(e)}")
        st.error("请确保模型文件存在于正确的路径")
        st.stop()

    # 创建选项卡
    tab1, tab2 = st.tabs(["图像检测", "视频检测"])

    # 图像检测选项卡
    with tab1:
        st.header("图像检测")

        uploaded_image = st.file_uploader("上传图像", type=["jpg", "jpeg", "png"])

        col1, col2 = st.columns(2)

        with col1:
            if uploaded_image is not None:
                # 显示原始图像
                image = Image.open(uploaded_image)
                st.image(image, caption="上传的图像", use_column_width=True)

                # 转换为numpy数组
                image_np = np.array(image)

                # 添加检测按钮
                if st.button("开始检测", key="image_detect"):
                    with st.spinner("正在处理图像..."):
                        result_img, stats = system.process_image(image_np)

                        # 保存结果到session state
                        st.session_state.result_img = result_img
                        st.session_state.stats = stats

        with col2:
            # 显示检测结果
            if 'result_img' in st.session_state and st.session_state.result_img is not None:
                st.image(st.session_state.result_img, caption="检测结果", use_column_width=True)
                st.text_area("统计信息", st.session_state.stats, height=150)

    # 视频检测选项卡
    with tab2:
        st.header("视频检测")

        uploaded_video = st.file_uploader("上传视频", type=["mp4", "avi", "mov"])

        if uploaded_video is not None:
            # 显示上传的视频
            st.video(uploaded_video)

            # 添加检测按钮
            if st.button("开始检测", key="video_detect"):
                with st.spinner("正在处理视频..."):
                    output_path = system.process_video(uploaded_video)

                    if output_path and os.path.exists(output_path):
                        # 显示处理后的视频
                        st.success("视频处理完成!")
                        st.video(output_path)
                    else:
                        st.error("视频处理失败")

        # 视频处理说明
        with st.expander("视频处理说明"):
            st.markdown("""
            - 视频处理可能需要较长时间，请耐心等待
            - 处理后的视频将显示检测框和分类结果
            - 处理速度取决于视频长度和分辨率
            - 处理后的视频将保存在results目录下
            """)


if __name__ == "__main__":
    main()