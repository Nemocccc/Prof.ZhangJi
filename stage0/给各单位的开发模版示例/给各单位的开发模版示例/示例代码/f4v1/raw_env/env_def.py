WIDTH = 600  # 渲染界面宽度
HEIGHT = 600  # 渲染界面高度
DEEPHEIGHT = int(WIDTH * 0.04)  # 渲染界面侧边距

# 颜色
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

vis0 = 1000  # 定义可探测距离
MAXSTEPS = 1000  # 单局最大步数
RENDER = False  # 是否渲染
VIS_STEPS = 3  # 渲染频率
WRITELOG = False  # 是否将日志写入文件
DISCRETE_ACTION = True  # 是否使用离散动作，离散动作数量默认为10，暂不可修改
