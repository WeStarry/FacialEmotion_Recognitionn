import os
import cv2
import torch
import base64
import numpy as np
from PIL import Image
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from ultralytics import YOLO
from models.model import MiniXception

# 加载模型
face_detector = YOLO(os.path.join(settings.BASE_DIR, 'models', 'best.pt'))
emotion_model = MiniXception()
emotion_model.load_state_dict(torch.load(os.path.join(settings.BASE_DIR, 'models', 'best_model_fold4.pth')))
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
    face_img = torch.from_numpy(face_img).unsqueeze(0).unsqueeze(0).float()
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
        
        # TODO: 实现视频处理逻辑
        return Response({
            'status': 'success',
            'message': '视频处理功能开发中'
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
