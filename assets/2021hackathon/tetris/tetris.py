# demo作者：邓宏灿
import sys
import random
### 以上模块参见 https://docs.python.org/zh-cn/3/library/ 请自行搜索

import pygame
from pygame.constants import *
from pygame.locals import *
### pygame官方教程 https://www.pygame.org/docs/

SCREEN_SIZE = (400, 400)
INFO_WIDTH = 100 # 右侧信息栏的宽度
SIZE = 16 #block size
SPAWN = (SCREEN_SIZE[0]//2-SIZE, 0) # 方块生成地点的坐标
NX = (SCREEN_SIZE[0]-INFO_WIDTH) // SIZE # NX, NY 都是map_arr 的横纵坐标上限
NY = (SCREEN_SIZE[1]) // SIZE
FramePerSec = pygame.time.Clock()
FPS = 30 # 游戏帧率
SPEED = 5 # 速度调节，数字越小速度越快

class Colors:
    """
        存储了颜色RGB表, 以便使用
    """
    RED = (255, 69, 0)
    BLUE = (65, 105, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GOLD = (255, 215, 0)
    PURPLE = (255, 0, 204)
    GREEN = (51, 255, 51)
    YELLOW = (255, 255, 102)

class UtilBlock:
    """
        与方块绘制有关的工具
    """
    def __init__(self, size=16):
        """
            size: 方块大小，单位像素
        """
        self.size = size

    def single_draw(self, pos, screen, color=(255, 255, 255)):
        """
            画出单个方块
            pos: 左上角坐标
        """
        x, y = pos
        size = self.size

        outer_color = (0, 0, 0)
        outer_size  = (size, size)
        outer_rect  = ((x, y), outer_size)

        block_color = color
        block_size  = (self.size - 2, self.size - 2)
        block_rect  = ((x+1, y+1), block_size)

        pygame.draw.rect(screen, block_color, block_rect, width=0)
        pygame.draw.rect(screen, outer_color, outer_rect, width=1)

    def single_erase(self, pos, screen):
        """
            擦掉单个方块，其实就是用黑色块填充
        """
        x, y = pos
        size = self.size

        erase_color = (0, 0, 0)
        erase_size = (size, size)
        erase_rect = ((x, y), erase_size)
        pygame.draw.rect(screen, erase_color, erase_rect, width=0)

"""定义并存储了各种方块的形状"""
types = {
    'O':( # O型方块
        (1, 1),
        (1, 1),
    ),
    'I':( # I型方块
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
    ),
    'T':(
        (0, 0, 0),
        (1, 1, 1),
        (0, 1, 0),
    ),
    'S':(
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
    ),
    'Z':(
        (0, 1, 0),
        (1, 1, 0),
        (1, 0, 0),
    ),
    'L':(
        (0, 1, 0),
        (0, 1, 0),
        (0, 1, 1),
    ),
    'J':(
        (0, 1, 0),
        (0, 1, 0),
        (1, 1, 0),
    ),
}

# 设置各种方块对应的颜色
colors = {
    'I':Colors.RED,
    'J':Colors.BLUE,
    'L':Colors.GOLD,
    'O':Colors.WHITE,
    'Z':Colors.PURPLE,
    'S':Colors.GREEN,
    'T':Colors.YELLOW,
}

def draw_gamearea(screen, drawer, eraser, map_arr, color_arr):
    """
        画出游戏区域，
        生成一个矩形区域
    """
    if not map_arr:
        return 
    block_size = SIZE
    # 根据 map_arr绘制方块
    for i in range(NY):
        for j in range(NX):
            x = j * block_size
            y = i * block_size
            if map_arr[i][j] == 1:
                drawer((x, y), screen)
            elif i>=1 and j>=1:
                eraser((x, y), screen)

def clear_block(block, pos, map_arr):
    """清除掉当前画的方块的位置， 为下一次作画做准备"""
    sx, sy = pos
    for y in range(len(block)):
        for x in range(len(block[0])):
            if block[y][x] == 1:
                map_arr[sy + y][sx + x] = 0


def judge_lateral(block, pos, map_arr):
    # 判断这些方块是否可以横向移动
    """
        返回：双key字典{"left":bool, "right":bool}
    """
    sx, sy = pos
    res = {"left":True, "right":True}
    for y in range(len(block)):
        for x in range(len(block)): #计算左边
            if block[y][x] == 0:
                continue
            elif block[y][x] == 1:
                if (sx + x - 1) == 0 or map_arr[sy + y][sx + x - 1] == 1:
                    res['left'] = False
                break

        for x in range(len(block)-1, -1, -1):
            if block[y][x] == 0:
                continue
            elif block[y][x] == 1:
                if (sx + x + 1) == (NX - 1) or map_arr[sy + y][sx + x + 1] == 1:
                    res['right'] = False
                break 
    return res

def judgefall(block, pos, map_arr):
    """
        检测当前的方块是否可以继续向下移动
        返回一个布尔值
    """
    sx, sy = pos
    #判断是否可以下落
    for x in range(len(block[0])):
        for y in range(len(block)-1 , -1, -1):
            if block[y][x] == 0:
                continue
            elif block[y][x] == 1:
                if map_arr[sy + y + 1][sx + x] == 1:
                    return False
    return True

def clear_lines(map_arr):
    """清除地图上已经填满了的一行"""
    flag = False
    for y in range(len(map_arr) - 1):
        for x in range(len(map_arr[0])):
            if map_arr[y][x] == 0:
                break
        else:
            ty = y
            if ty > 1:
                flag = True
            while ty > 1:
                map_arr[ty] = map_arr[ty - 1]
                ty -= 1
    return flag

def dock(block, pos, map_arr):
    """使无法移动的方块固定在地图里面"""
    sx, sy = pos
    for y in range(len(block)):
        for x in range(len(block[0])):
            if block[y][x] == 1:
                map_arr[sy + y][sx + x] = 1

def rotate(block): # 旋转方块
    temp = list(block)
    for y in range(len(temp)):
        temp[y] = list(temp[y])

    for x, row in enumerate(temp):
        for y in range(x, len(row)):
            tmp = temp[y][x]
            temp[y][x] = row[y]
            temp[x][y] = tmp

    return temp

def get_blocks():
    """
        随机获取方块类型数组
    """
    alltype = "IJLOZST"
    choice = random.choice(alltype)
    return types[choice], colors[choice]

def set_border(arr, color_arr):
    """为arr变量设置边界"""
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if i==0 or i==len(arr)-1 or j==0 or j==len(arr[i]) - 1:
                arr[i][j] = 1
                color_arr[i][j] = Colors.WHITE

def draw_font(text, pos, screen, color=(255, 255, 255)):
    """在指定的位置绘制文字"""
    if pygame.font:
        font = pygame.font.Font(None, 30)
        surf = font.render(text, True, color)
        screen.blit(surf, (pos))
        pygame.display.flip() # 立刻刷新图像
    else:
        pass

def main():
    pygame.init()
    pygame.display.set_caption("Tetris")
    tool = UtilBlock()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    # 初始化模块和游戏屏幕

    map_arr = [[0 for x in range(NX)] for y in range(NY)]
    color_arr = [[(0,0,0) for x in range(NX)] for y in range(NY)]
    set_border(map_arr, color_arr)
    #标识数组, 0代表该区域没有方块
    game_score = 0

    droping = False # 当前是否有方块正在下落
    fps_counter = 0
    droplist = [i for i in range(0, FPS+1, SPEED)]

    fail = False # 标记是否失败
    draw_font("Hello", (200, 200), screen)
    while True:
        flag = True # 是否允许下落， 一般在左右移动的时候不会下落 flag = False
        if fps_counter >= 30:
            fps_counter = 0
        fps_counter += 1
        if not droping: # 方块正在下落
            block, color = get_blocks()
            sx, sy = (NX // 2 - 1, 1)
            droping = True
            for y in range(len(block)):
                for x in range(len(block[y])):
                    if block[y][x] == 1:
                        map_arr[sy + y][sx + x] = block[y][x]

        draw_gamearea(screen, tool.single_draw, tool.single_erase, map_arr, color_arr)
        clear_block(block, (sx, sy), map_arr) # 在图像改变之前，必须擦掉原来占位为1的区块
        if clear_lines(map_arr): # 每消除一行就记10分
            game_score += 10

        for event in pygame.event.get(): # 获取键盘输入
            if event.type in (QUIT, K_ESCAPE):
                sys.exit()
            if event.type==KEYDOWN:
                key = event.key
                able = judge_lateral(block, (sx, sy), map_arr)
                if key == K_j:
                    block = rotate(block)
                elif key == K_a and able['left']:
                    clear_block(block, (sx, sy), map_arr)
                    flag = False
                    sx -= 1 if sx >= 1 else 0
                elif key == K_d and able['right']:
                    clear_block(block, (sx, sy), map_arr)
                    sx += 1 if sx < NX - 1 else 0
                    flag = False
                elif key == K_s:
                    fps_counter = droplist[0] # 加速方块向下移动

        if fail: # 如果游戏失败就停止绘制方块
            pos = (SCREEN_SIZE[0]//2 - 100, SCREEN_SIZE[1]//2)
            draw_font("Game Over!Score:{}".format(game_score), pos, screen, Colors.RED)
            continue

        if not judgefall(block, (sx, sy), map_arr): # 方块落地，停止移动并创建新方块
            droping = False
            dock(block, (sx, sy), map_arr)
            if sy <= 1:
                fail = True

        if droping and flag and fps_counter in droplist : # 每隔10帧方块才会下落一次
            sy += 1
            if sy >= NY:
                droping = False
        
        for y in range(len(block)): # 将下落的方块复制到地图map_arr上
            for x in range(len(block[0])):
                if block[y][x] == 1:
                    map_arr[sy + y][sx + x] = block[y][x]

        pygame.display.update() # 图像刷新
        FramePerSec.tick(FPS) # 在这里设置游戏FPS 
        # print(FramePerSec.get_fps())


if __name__ == '__main__':
    main()