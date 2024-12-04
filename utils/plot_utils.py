import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def plot_scatter_with_stats(data, save_path):
    # 计算统计值
    mean_value = np.mean(data)
    max_value = np.max(data)
    lower_95 = np.percentile(data, 95)
    upper_95 = np.percentile(data, 99)
    
    # 创建散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(data)), data, color='blue', label='Data Points')
    
    # 在图上标注统计值
    plt.axhline(y=mean_value, color='green', linestyle='--', label=f'Mean: {mean_value:.2f}')
    plt.axhline(y=max_value, color='red', linestyle='--', label=f'Max: {max_value:.2f}')
    plt.axhline(y=lower_95, color='orange', linestyle='--', label=f'95% Percentile: {lower_95:.2f}')
    plt.axhline(y=upper_95, color='purple', linestyle='--', label=f'99% Percentile: {upper_95:.2f}')

    # 添加图例和标题
    plt.title('Scatter Plot with Statistical Values')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.grid()
    
    # 显示图形
    # plt.show()
    plt.savefig(save_path)    


def draw_rotated_rectangle(image_path, vertices, color, save_path):
    """
    在图像上绘制旋转矩形框

    :param image_path: 图片的路径
    :param vertices: 四个顶点坐标的列表，格式为 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    :param color: 绘制矩形框的颜色，格式为 (r, g, b)
    """
    # 打开图像
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        
        # 绘制矩形框
        draw.line(vertices + [vertices[0]], fill=color, width=2)  # 连接最后一个点到第一个点

        # 显示或保存结果图像
        # img.show()  # 显示图片
        img.save(save_path)  # 可以选择保存图片
        #return img

def draw_label(image_path: str, vertices: list, label: str, color: tuple, save_path: str)-> None:
    """在第一个点上渲染label信息
    
    image_path: str, （待绘制）图片路径
    vertices: list,  [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    label: 待渲染label信息
    color: 渲染文字颜色
    save_path: str, （渲染）图片保存路径
    """
    first_point = vertices[0]
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        
        font_path = "Arial.Unicode.ttf"
        size = max(round(sum(img.size) / 2 * 0.035), 12)
        font = ImageFont.truetype(str(font_path), size)
    
        w, h = font.getsize(label)  # text width, height
        outside = first_point[1] >= h  # label fits outside box
        if first_point[0] > img.size[0] - w:  # size is (w, h), check if label extend beyond right side of image
            first_point = (img.size[0] - w, first_point[1])
        draw.rectangle((first_point[0], first_point[1] - h if outside else first_point[1],
                        first_point[0] + w + 1, first_point[1] + 1 if outside else first_point[1] + h + 1),
                fill=color,
            )
            
        draw.text((first_point[0], first_point[1] - h if outside else first_point[1]), label, fill=(0, 0, 0), font=font)
    img.save(save_path)

    
def draw_axes(image_path, center, save_path):
    """
    在图像上以给定点为中心绘制坐标轴

    :param image_path: 图片的路径
    :param center: 中心点坐标 (x, y)
    """
    # 打开图像
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        
        # 中心点坐标
        x, y = center
        
        # 坐标轴长度（15个像素，纵横各一半，总共30个像素）
        length = 15  

        # 绘制X轴（横轴）
        draw.line([(x - length, y), (x + length, y)], fill="black", width=1)

        # 绘制Y轴（纵轴）
        draw.line([(x, y - length), (x, y + length)], fill="black", width=1)

        # 显示结果图像
        # img.show()  # 显示图片
        # img.save('output_image.jpg')  # 可以选择保存图片
        img.save(save_path)
    
    