"""
实时检测文件
调用迁移训练得到的ckPlus_best_model.pth进行实时检测
"""



import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# 导入自定义模型和YOLO
sys.path.append(str(Path(__file__).parent))
from model import MiniXception
from ultralytics import YOLO  # 导入YOLO

# 表情标签 - 同时提供英文标签作为备用
emotion_labels = ['愤怒', '厌恶', '恐惧', '高兴', '悲伤', '惊讶', '中性']
emotion_labels_en = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 颜色映射（BGR格式）
emotion_colors = {
    '愤怒': (0, 0, 255),     # 红色
    '厌恶': (0, 128, 255),   # 橙色
    '恐惧': (0, 255, 255),   # 黄色
    '高兴': (0, 255, 0),     # 绿色
    '悲伤': (255, 0, 0),     # 蓝色
    '惊讶': (255, 0, 255),   # 粉色
    '中性': (255, 255, 255)  # 白色
}

class EmotionDetector:
    def __init__(self, emotion_model_path, yolo_model_path, device=None):
        """初始化表情检测器"""
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        print(f"使用设备: {self.device}")
        
        # 加载表情识别模型
        self.emotion_model = MiniXception(num_classes=7, input_channels=1).to(self.device)
        self.emotion_model.load_state_dict(torch.load(emotion_model_path, map_location=self.device))
        self.emotion_model.eval()
        print(f"已加载表情识别模型: {emotion_model_path}")
        
        # 加载YOLO模型用于人脸检测 - 简化导入方式
        self.face_detector = YOLO(yolo_model_path)  # 简化的YOLO导入
        self.face_detector.conf = 0.5  # 置信度阈值
        print(f"已加载人脸检测模型: {yolo_model_path}")
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor()
        ])
    
    def detect_face(self, frame):
        """使用YOLO模型检测人脸"""
        results = self.face_detector(frame)  # 使用YOLO直接检测
        faces = []
        
        # 处理检测结果
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # 获取坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # 获取置信度
                conf = float(box.conf[0].cpu().numpy())
                
                # 只有在置信度足够高的情况下才处理
                if conf > 0.5:
                    faces.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf
                    })
        
        return faces
    
    def preprocess_face(self, frame, face_box):
        """预处理检测到的人脸，准备进行表情识别"""
        x1, y1, x2, y2 = face_box
        
        # 提取人脸区域
        face_img = frame[y1:y2, x1:x2]
        
        # 如果人脸区域太小则跳过
        if face_img.shape[0] < 5 or face_img.shape[1] < 5:
            return None
            
        # 转换为灰度图，然后转换为PIL图像进行处理
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        pil_img = Image.fromarray(gray)
        
        # 应用图像变换
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        return tensor
    
    def recognize_emotion(self, face_tensor):
        """识别人脸表情"""
        with torch.no_grad():
            outputs = self.emotion_model(face_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            
            # 获取最高概率的表情和对应概率
            emotion_idx = torch.argmax(probabilities).item()
            confidence = probabilities[emotion_idx].item()
            
            return {
                "emotion": emotion_labels[emotion_idx],
                "confidence": confidence
            }
    
    def draw_result(self, frame, face_box, emotion_result):
        """在帧上绘制检测结果"""
        x1, y1, x2, y2 = face_box
        emotion = emotion_result["emotion"]
        confidence = emotion_result["confidence"]
        
        # 为边框选择颜色
        color = emotion_colors.get(emotion, (255, 255, 255))
        
        # 绘制人脸边框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 准备显示的文本 - 使用英文标签替代中文，避免乱码
        emotion_idx = emotion_labels.index(emotion)
        emotion_text = emotion_labels_en[emotion_idx]  # 使用英文表情标签
        
        # 显示英文文本和置信度
        text = f"{emotion_text}: {confidence:.2f}"
        cv2.putText(frame, text, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
        # 如果要在图像上显示中文，可以使用以下函数
        # 将英文识别结果显示在边框上方，中文显示在边框下方
        frame = self.put_chinese_text(frame, emotion, (x1, y2 + 30), color)
        
        return frame
    
    def put_chinese_text(self, img, text, position, color):
        """使用PIL在OpenCV图像上绘制中文文本"""
        # 转换OpenCV图像到PIL格式
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # 加载一个支持中文的字体，如果没有指定的字体，尝试使用系统字体
        try:
            # 首先尝试使用微软雅黑字体
            font_path = 'C:/Windows/Fonts/msyh.ttc'  # Windows系统中文字体路径
            font = ImageFont.truetype(font_path, 24)  # 字体大小为24
        except IOError:
            try:
                # 尝试使用其他常见中文字体
                font_path = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'  # Linux系统字体
                font = ImageFont.truetype(font_path, 24)
            except IOError:
                # 如果都失败了，使用默认字体
                font = ImageFont.load_default()
                print("警告: 无法加载中文字体，将使用默认字体")
        
        # 创建绘图对象
        draw = ImageDraw.Draw(img_pil)
        
        # 绘制中文文本
        # 注意：PIL中颜色顺序是RGB，而OpenCV是BGR
        color_rgb = (color[2], color[1], color[0])  # 转换BGR到RGB
        draw.text(position, text, font=font, fill=color_rgb)
        
        # 转换回OpenCV格式
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img
    
    def process_frame(self, frame):
        """处理单帧图像，检测人脸和表情"""
        # 检测人脸
        faces = self.detect_face(frame)
        
        # 对每个检测到的人脸进行处理
        for face in faces:
            face_box = face["bbox"]
            face_tensor = self.preprocess_face(frame, face_box)
            
            # 如果预处理成功，进行表情识别
            if face_tensor is not None:
                emotion_result = self.recognize_emotion(face_tensor)
                frame = self.draw_result(frame, face_box, emotion_result)
        
        return frame

def run_detection(emotion_model_path='ckplus_best_model.pth', 
                 yolo_model_path='best.pt',
                 camera_id=0):
    """运行实时表情检测"""
    # 初始化检测器
    detector = EmotionDetector(emotion_model_path, yolo_model_path)
    
    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    # 创建窗口
    cv2.namedWindow("表情检测", cv2.WINDOW_NORMAL)
    
    print("开始表情检测，按 'q' 键退出...")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("无法获取图像帧，退出...")
            break
        
        # 处理帧并检测表情
        processed_frame = detector.process_frame(frame)
        
        # 显示处理后的帧
        cv2.imshow("表情检测", processed_frame)
        
        # 检查键盘输入，退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='实时人脸表情检测')
    parser.add_argument('--emotion_model', type=str, default='ckplus_best_model.pth',
                        help='表情识别模型路径')
    parser.add_argument('--yolo_model', type=str, default='best.pt',
                        help='YOLO人脸检测模型路径')
    parser.add_argument('--camera', type=int, default=0,
                        help='摄像头ID (默认: 0)')
    
    args = parser.parse_args()
    
    run_detection(
        emotion_model_path=args.emotion_model,
        yolo_model_path=args.yolo_model,
        camera_id=args.camera
    )
