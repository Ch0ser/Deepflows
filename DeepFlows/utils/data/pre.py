import os
from PIL import Image


def read_png_images_from_folder(folder_path):
    # 创建一个空列表，用于存储图像
    images = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否为PNG图像
        if filename.endswith('.png'):
            # 构建图像的完整路径
            file_path = os.path.join(folder_path, filename)
            # 打开图像并将其添加到列表中
            with Image.open(file_path) as img:
                images.append(img.copy())  # 使用copy()来防止图像被关闭

    return images
