from PIL import Image
import matplotlib.pyplot as plt
import os


def display_images_with_same_size(image_paths, target_size=(800, 600)):
    """
    读取指定的图片并统一展示大小

    :param image_paths: 图片路径列表
    :param target_size: 目标展示大小，默认为(800, 600)
    """
    images = []
    for path in image_paths:
        if not os.path.exists(path):
            print(f"图片路径不存在：{path}")
            continue
        try:
            # 打开图片
            img = Image.open(path)
            # 调整图片大小为目标大小
            img_resized = img.resize(target_size)
            images.append(img_resized)
        except Exception as e:
            print(f"读取或处理图片出错：{path}，错误信息：{e}")

    # 使用matplotlib展示图片
    fig = plt.figure(figsize=(10, 8))  # 设置展示窗口的大小
    for i, img in enumerate(images):
        ax = fig.add_subplot(1, len(images), i + 1)
        ax.imshow(img)
        ax.set_title(f"Image {i + 1}")
        ax.axis('off')  # 关闭坐标轴
    plt.tight_layout()
    plt.show()


# 示例用法
image_paths = ['JPEGImages/fire_dp3_1.jpg', 'JPEGImages/fire_dp3_2.jpg', 'JPEGImages/fire_dp3_3.jpg', 'JPEGImages/fire_dp3_4.jpg',
               'JPEGImages/fire_dp3_5.jpg']  # 替换为你的图片路径
display_images_with_same_size(image_paths)