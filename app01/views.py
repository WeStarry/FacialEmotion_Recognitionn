import os
import cv2
import torch
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from ultralytics import YOLO
from models.model import MiniXception
import time

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
face_detector = YOLO(os.path.join(settings.BASE_DIR, 'models', 'best.pt'))
emotion_model = MiniXception().to(device)
emotion_model.load_state_dict(torch.load(os.path.join(settings.BASE_DIR, 'models', 'ckplus_best_model.pth'), map_location=device))
emotion_model.eval()

# 表情标签映射
EMOTION_LABELS = {
    0: '生气',
    1: '厌恶',
    2: '恐惧',
    3: '开心',
    4: '难过',
    5: '惊讶',
    6: '中性'
}

def preprocess_face(face_img):
    """预处理人脸图像用于表情识别"""
    face_img = cv2.resize(face_img, (48, 48))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = face_img / 255.0
    face_img = torch.from_numpy(face_img).unsqueeze(0).unsqueeze(0).float().to(device)
    return face_img

@api_view(['POST'])
def detect_image(request):
    """处理图片表情识别请求"""
    try:
        image_file = request.FILES['image']
        image = Image.open(image_file)
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 人脸检测
        results = face_detector(image_np)
        faces_data = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_img = image_np[y1:y2, x1:x2]
                
                # 表情识别
                face_tensor = preprocess_face(face_img)
                with torch.no_grad():
                    emotion_pred = emotion_model(face_tensor)
                    probabilities = torch.softmax(emotion_pred, dim=1)[0]
                    emotion_probs = {EMOTION_LABELS[i]: float(prob) for i, prob in enumerate(probabilities)}
                    emotion_idx = torch.argmax(emotion_pred).item()
                    emotion_label = EMOTION_LABELS[emotion_idx]
                
                faces_data.append({
                    'bbox': [x1, y1, x2, y2],
                    'emotion': emotion_label,
                    'probabilities': emotion_probs,
                    'confidence': float(box.conf[0])  # 添加人脸检测的置信度
                })
        
        return Response({
            'status': 'success',
            'faces': faces_data
        })
        
    except KeyError as e:
        print(f'KeyError: {str(e)}')
        return Response({
            'status': 'error',
            'message': f'缺少必要的字段: {str(e)}'
        }, status=400)
    except Exception as e:
        print(f'Error processing image: {str(e)}')
        import traceback
        traceback.print_exc()
        return Response({
            'status': 'error',
            'message': f'处理图片时出错: {str(e)}'
        }, status=500)

@api_view(['POST'])
def detect_video(request):
    """处理视频表情识别请求"""
    try:
        if 'video' not in request.FILES:
            return Response({
                'status': 'error',
                'message': '请上传视频文件'
            }, status=400)
            
        video_file = request.FILES['video']
        
        # 验证文件类型
        if not video_file.content_type.startswith('video/'):
            return Response({
                'status': 'error',
                'message': '请上传有效的视频文件'
            }, status=400)
        
        # 创建media目录
        video_dir = os.path.join(settings.MEDIA_ROOT, 'videos')
        results_dir = os.path.join(settings.MEDIA_ROOT, 'results')
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存上传的视频文件
        filename = f"video_{int(time.time())}.mp4"
        video_path = os.path.join(video_dir, filename)
        with open(video_path, 'wb+') as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)
        
        # 定义输出视频路径
        output_filename = f"result_{filename}"
        output_path = os.path.join(results_dir, output_filename)
        
        # 打开视频进行处理
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return Response({
                'status': 'error',
                'message': '无法打开视频文件'
            }, status=500)
        
        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 创建视频写入器 (尝试使用 H.264 编码)
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*'avc1') 
        output_path_h264 = output_path.replace('.mp4', '_h264.mp4') # 改个名避免混淆
        output_filename_h264 = os.path.basename(output_path_h264)
        out = cv2.VideoWriter(output_path_h264, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"无法打开 VideoWriter，路径: {output_path_h264}, FourCC: avc1")
             # 如果avc1失败，尝试回退到mp4v
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                print(f"无法打开 VideoWriter，路径: {output_path}, FourCC: mp4v")
                # 如果两者都失败，返回错误
                cap.release()
                return Response({
                    'status': 'error',
                    'message': '无法创建视频写入器'
                }, status=500)
            else:
                 output_filename = os.path.basename(output_path)
                 print("回退到 mp4v 编码成功")
        else:
            output_filename = output_filename_h264 # 使用新文件名
            print("使用 avc1 (H.264) 编码")

        # 每隔几帧进行一次检测，以提高性能
        process_frame_interval = 5  # 每5帧处理一次
        
        # 加载中文字体
        font_path = os.path.join(settings.BASE_DIR, 'assets', 'fonts', 'SimHei.ttf') # 假设字体路径
        try:
            font_size = 20 # 调整字体大小
            font = ImageFont.truetype(font_path, font_size)
            print(f"成功加载字体: {font_path}") # 添加成功加载日志
        except IOError:
             print(f"错误：无法加载字体文件 {font_path}。请确保文件存在且路径正确。将使用默认字体。")
             font = None # 使用None作为标记，稍后处理

        # 处理视频帧
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 是否处理当前帧
            process_this_frame = frame_idx % process_frame_interval == 0
            
            # 将OpenCV帧 (BGR) 转换为 PIL Image (RGB) 以便绘制中文
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)

            if process_this_frame:
                # 使用YOLOv8人脸检测
                results = face_detector(frame) # 检测仍在原始CV帧上进行
                
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        if conf > 0.5:  
                            face_img = frame[y1:y2, x1:x2]
                            if face_img.size == 0:
                                continue
                                
                            face_tensor = preprocess_face(face_img)
                            with torch.no_grad():
                                emotion_pred = emotion_model(face_tensor)
                                emotion_idx = torch.argmax(emotion_pred).item()
                                emotion_label = EMOTION_LABELS[emotion_idx]
                            
                            color = (0, 255, 0)  # 绿色
                            # 使用PIL绘制边界框
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            
                            label = f"{emotion_label} ({conf:.2f})"
                            print(f"Debug: Font object: {font}") # 打印字体对象状态
                            
                            if font:
                                print(f"Debug: Drawing label '{label}' for face at bbox {x1},{y1}-{x2},{y2}")
                                # 使用PIL绘制带背景的文本
                                try:
                                    text_bbox = draw.textbbox((x1, y1), label, font=font) # 获取文本边界框
                                    text_width = text_bbox[2] - text_bbox[0]
                                    text_height = text_bbox[3] - text_bbox[1]
                                    
                                    # 调整文本和背景位置，使其紧贴框顶部
                                    text_bg_y0 = max(0, y1 - text_height - 4) # 背景框上边缘，留一点空隙
                                    text_bg_y1 = y1                     # 背景框下边缘与人脸框上边缘对齐
                                    text_y = text_bg_y0 + 2             # 文本 y 坐标
                                    text_x = x1 + 2                     # 文本 x 坐标
                                    text_bg_x0 = x1
                                    text_bg_x1 = x1 + text_width + 4
                                    
                                    print(f"Debug: Text position (x,y): ({text_x}, {text_y}), BG Rect: [{text_bg_x0},{text_bg_y0} - {text_bg_x1},{text_bg_y1}]")

                                    # 绘制背景矩形 (确保坐标在合理范围内)
                                    draw.rectangle(\
                                        [text_bg_x0, text_bg_y0, \
                                         min(text_bg_x1, frame.shape[1]), min(text_bg_y1, frame.shape[0])],\
                                        fill=color\
                                    )
                                    # 绘制文本 (黑色字体)
                                    draw.text((text_x, text_y), label, font=font, fill=(0, 0, 0))
                                    print(f"Debug: Text drawn successfully.")
                                except Exception as e:
                                     print(f"Error drawing text with PIL: {e}")
                            else:
                                # 如果字体加载失败，回退到OpenCV的英文绘制
                                print(f"Debug: Font not loaded, attempting cv2.putText for label '{label}'")
                                try:
                                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                    print(f"Debug: cv2.putText executed.")
                                except Exception as e:
                                     print(f"Error drawing text with cv2.putText: {e}")
            
            # 将处理后的 PIL Image (RGB) 转换回 OpenCV 帧 (BGR)
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

            # 写入处理后的帧
            out.write(frame)
            frame_idx += 1
        
        # 释放资源
        cap.release()
        out.release()
        
        # 构建输出视频的URL
        video_url = f"{settings.MEDIA_URL}results/{output_filename}"
        print(f"生成的视频URL: {video_url}") # 添加日志
        
        return Response({
            'status': 'success',
            'video_url': video_url,
            'message': '视频处理成功'
        })
        
    except Exception as e:
        print(f'处理视频时出错: {str(e)}')
        import traceback
        traceback.print_exc()
        return Response({
            'status': 'error',
            'message': f'处理视频时出错: {str(e)}'
        }, status=500)

@api_view(['POST'])
def detect_emotion(request):
    """处理实时表情检测请求"""
    try:
        image_file = request.FILES['image']
        image = Image.open(image_file)
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 人脸检测
        results = face_detector(image_np)
        faces_data = []
        emotions_data = {}
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_img = image_np[y1:y2, x1:x2]
                
                # 表情识别
                face_tensor = preprocess_face(face_img)
                with torch.no_grad():
                    emotion_pred = emotion_model(face_tensor)
                    probabilities = torch.softmax(emotion_pred, dim=1)[0]
                    
                    # 更新情绪概率
                    emotions_data = {
                        'angry': float(probabilities[0]),
                        'disgust': float(probabilities[1]),
                        'fear': float(probabilities[2]),
                        'happy': float(probabilities[3]),
                        'sad': float(probabilities[4]),
                        'surprise': float(probabilities[5]),
                        'neutral': float(probabilities[6])
                    }
                    
                    faces_data.append({
                        'box': {
                            'x': x1,
                            'y': y1,
                            'width': x2 - x1,
                            'height': y2 - y1
                        }
                    })
        
        return Response({
            'status': 'success',
            'faces': faces_data,
            'emotions': emotions_data
        })
        
    except Exception as e:
        print(f'Error processing frame: {str(e)}')
        return Response({
            'status': 'error',
            'message': str(e)
        }, status=500)

@api_view(['POST'])
def detect_realtime(request):
    """处理实时表情识别请求"""
    try:
        # 从请求中获取图像数据
        image_data = request.data.get('image')
        if not image_data:
            return Response({
                'status': 'error',
                'message': '未接收到图像数据'
            }, status=400)
            
        # 将base64图像数据转换为numpy数组
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # 人脸检测
        results = face_detector(image_np)
        faces_data = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_img = image_np[y1:y2, x1:x2]
                
                # 表情识别
                face_tensor = preprocess_face(face_img)
                with torch.no_grad():
                    emotion_pred = emotion_model(face_tensor)
                    probabilities = torch.softmax(emotion_pred, dim=1)[0]
                    emotion_probs = {EMOTION_LABELS[i]: float(prob) for i, prob in enumerate(probabilities)}
                    emotion_idx = torch.argmax(emotion_pred).item()
                    emotion_label = EMOTION_LABELS[emotion_idx]
                
                faces_data.append({
                    'bbox': [x1, y1, x2, y2],
                    'emotion': emotion_label,
                    'probabilities': emotion_probs,
                    'confidence': float(box.conf[0])
                })
        
        return Response({
            'status': 'success',
            'faces': faces_data
        })
        
    except Exception as e:
        print(f'实时检测出错: {str(e)}')
        import traceback
        traceback.print_exc()
        return Response({
            'status': 'error',
            'message': f'实时检测出错: {str(e)}'
        }, status=500)
