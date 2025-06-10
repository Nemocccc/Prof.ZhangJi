import pygame 

"""
    本文件存放一些对图像处理的函数 供解决边缘白色像素问题

"""


def remove_white_background(image_path):
    """
        思路1:将图像中非主体像素变为透明 便可以与背景融为一体 只留下主体像素
        处理所得图像会在原图后加_light后缀  现在tvm_v3.py中所用即为处理后的图像
    """
    img = pygame.image.load(image_path).convert_alpha()
    width, height = img.get_size()
        
    # 遍历所有像素，找到白色并将其变为透明
    for x in range(width):
        for y in range(height):
            r, g, b, a = img.get_at((x, y))  # 获取像素的 RGBA 值
            if r == 255 and g == 255 and b == 255:  # 判断是否是白色
                img.set_at((x, y), (255, 255, 255, 0))  # 将白色背景变为透明
        pygame.image.save(img,f'{image_path}_light.png')
        return img

def set_sandy_background(image_path):
    """
        思路2:将图像中非主体像素变为背景色 便可以与背景融为一体 只留下主体像素
        处理所得图像会在原图后加_sandy后缀  但是考虑到图像在不断移动 与背景会有点突兀

        新思路:实时计算图像周围像素颜色 并对非主体部分进行边缘模糊化 可能效果更好 但计算量大 可尝试
    """

    img = pygame.image.load(image_path).convert_alpha()  # 加载图片并保留透明通道
    width, height = img.get_size()
    
    # 沙黄色 RGB 值 (244, 164, 66)
    sandy_color = (244, 164, 66)
    
    # 遍历所有像素，将背景部分替换为沙黄色
    for x in range(width):
        for y in range(height):
            r, g, b, a = img.get_at((x, y))  # 获取像素的 RGBA 值
            if a == 0:  # 如果是透明区域
                img.set_at((x, y), (*sandy_color, 255))  # 设置为沙黄色，不透明
            elif r > 240 and g > 240 and b > 240:  # 可选：假设背景颜色接近白色
                img.set_at((x, y), (*sandy_color, 255))  # 替换为沙黄色
    pygame.image.save(img,f'{image_path}_sandy.png')
    return img
