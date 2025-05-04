import os
from PIL import Image

def resize_images(input_dir='naturalchange', output_dir='naturalchange48', size=(48, 48)):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 支持的图片格式
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        # 检查文件扩展名
        if any(filename.lower().endswith(fmt) for fmt in supported_formats):
            # 构建完整的文件路径
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                # 打开图片
                with Image.open(input_path) as img:
                    # 调整图片大小
                    resized_img = img.resize(size, Image.Resampling.LANCZOS)
                    # 保存调整后的图片
                    resized_img.save(output_path)
                print(f"成功处理图片: {filename}")
            except Exception as e:
                print(f"处理图片 {filename} 时出错: {str(e)}")

if __name__ == "__main__":
    # 调用主函数
    resize_images()
    print("所有图片处理完成！")