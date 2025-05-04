from ultralytics import YOLO
import torch
from multiprocessing import freeze_support  # 添加必要模块


def main():
    # 验证CUDA是否可用
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA设备不可用，请检查显卡驱动和PyTorch安装")

    # 初始化模型
    model = YOLO("yolo11n.pt")  # 确保模型文件路径正确

    # 训练参数配置
    model.train(
        data="config.yaml",  # 验证YAML文件路径
        epochs=50,
        imgsz=320,
        batch=8,
        device=0,  # 显式指定第一个GPU
        workers=4,  # 建议设置为CPU物理核心数的50-75%
        project="face_detection",
        name="exp1"
    )


if __name__ == '__main__':
    freeze_support()  # Windows多进程必需
    main()
    print("模型训练完毕")  # 建议在main()内部执行完成后输出