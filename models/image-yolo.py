

"""用于识别图片中的人脸并将人脸区域进行裁剪"""


import os
import cv2
from ultralytics import YOLO

def process_images():
    # 加载YOLO模型
    model = YOLO('best.pt')
    
    # 设置输入和输出目录
    input_dir = 'natural'
    output_dir = 'naturalchange'
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取输入目录中的所有图片
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in image_files:
        # 读取图片
        input_path = os.path.join(input_dir, image_file)
        img = cv2.imread(input_path)
        
        # 使用YOLO进行人脸检测
        results = model(img)
        
        # 获取检测结果
        if len(results[0].boxes) > 0:
            # 获取第一个检测到的人脸（因为已知每张图片只有一个人脸）
            box = results[0].boxes[0].xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box[:4])
            
            # 裁剪人脸区域
            face_crop = img[y1:y2, x1:x2]
            
            # 保存裁剪后的人脸图片
            output_path = os.path.join(output_dir, f'face_{image_file}')
            cv2.imwrite(output_path, face_crop)
            print(f'已处理: {image_file}')
        else:
            print(f'警告: 在图片 {image_file} 中没有检测到人脸')

if __name__ == '__main__':
    process_images()