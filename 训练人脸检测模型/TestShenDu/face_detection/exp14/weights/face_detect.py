import cv2
from ultralytics import YOLO
import os


class FaceDetector:
    def __init__(self, model_path='last.pt'):
        # 确保模型文件存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")

        # 创建输出目录
        os.makedirs('image', exist_ok=True)

        # 加载训练好的模型
        self.model = YOLO(model_path)
        self.device = 'cuda' if next(self.model.parameters()).is_cuda else 'cpu'

    def detect_and_save(self, image_path, output_name='output.jpg', conf_threshold=0.5):
        """
        执行人脸检测并保存结果
        :param image_path: 输入图片路径
        :param output_name: 输出文件名
        :param conf_threshold: 置信度阈值
        :return: 检测到的人脸数量
        """
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片 {image_path}")

        # 执行推理
        results = self.model.predict(
            source=img,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )

        # 解析结果
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        # 统计人脸数量
        face_count = len(boxes)
        print(f"检测到人脸数量: {face_count}")

        # 绘制检测框和置信度
        for idx, (box, conf) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = map(int, box)

            # 绘制矩形框
            color = (0, 255, 0)  # BGR格式
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 添加置信度文本
            label = f"Face {idx + 1}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            print(f"人脸 {idx + 1} 置信度: {conf:.4f}")

        # 保存结果
        output_path = os.path.join('image', output_name)
        cv2.imwrite(output_path, img)
        print(f"结果已保存至: {output_path}")

        return face_count


if __name__ == '__main__':
    try:
        detector = FaceDetector(model_path='best.pt')

        # 输入参数设置
        input_image = "ffe5758283a4aaedb7841bcd137ddfd0.jpg"  # 待检测图片路径
        output_image = "detected.jpg"  # 输出图片文件名

        # 执行检测
        count = detector.detect_and_save(
            image_path=input_image,
            output_name=output_image,
            conf_threshold=0.5
        )

        print(f"\n检测完成，共发现 {count} 张人脸")

    except Exception as e:
        print(f"发生错误: {str(e)}")