# demo作者：王嘉睿

from time import time
# time库用于处理时间，详见 https://docs.python.org/zh-cn/3/library/time.html
from sys import platform
# sys --- 系统相关的参数和函数，详见 https://docs.python.org/zh-cn/3/library/sys.html
from random import randrange
# random --- 生成伪随机数，详见 https://docs.python.org/zh-cn/3/library/random.html

# 判断运行平台（提交的作品并不需要做多系统适配，能在你们自己的电脑上运行即可）
if platform.startswith('win'):  # 如果是windows系统 
    # 导入相关库
    from msvcrt import getch, kbhit
    '''msvcrt --- 来自 MS VC++ 运行时的有用例程
        详见 https://docs.python.org/zh-cn/3/library/msvcrt.html
            khbit: 如果有某个按键正在等待被读取则返回 True。
            getch: 读取一个按键并将结果字符返回为一个字节串。
    '''
    from os import system 
    '''os --- 多种操作系统接口
        详见 https://docs.python.org/zh-cn/3/library/os.html
    '''

    def readinput(): # 读取键盘输入的函数
        global drc, st
        get = False  # 记录是否已经获取输入
        while time()-st < DELAY: # 增加延迟，控制帧率
            if kbhit() and not get: # 如果有未获得输入
                try:
                    t = getch().decode('ascii') # 获取键盘事件并转换成字符串
                    print('get keyevent:', t)   # 输出
                    drc = KEYDRC[t](drc)        # 做出响应
                    get = True                  # get设为True
                except:
                    pass                        # 不处理运行错误（例如按键不在wasd内）

    def showworld(): # 绘制地图的函数
        system('cls')   # 清空shell
        for i in world: # i为world的每一行
            print('┃',*i,'┃')   # 左右边框+图标
        print('score: {}; time: {}; framerate: {:.2f}'.format(
            len(body)-1, loop, 1/DELAY))    # 输出运行信息

elif platform.startswith('linux') or platform.startswith('darwin'):
    # 如果是linux或MacOS
    import curses # 详见 https://docs.python.org/zh-cn/3/library/curses.html

    stdscr = curses.initscr()
    stdscr.nodelay(True)

    def readinput():
        global drc, st
        c = -1
        while time()-st < DELAY:
            while c == -1 and time()-st < DELAY:
                c = stdscr.getch()
            try:
                t = chr(c)
                stdscr.addstr(HEIGHT+2, 0, 'get keyevent: '+t)
                drc = KEYDRC[t](drc)
                stdscr.refresh()
            except:
                c = -1

    def showworld():
        stdscr.clear()
        try:
            for index, i in enumerate(world):
                stdscr.addstr(index, 0, ' '.join(['┃', *i, '┃']))
            stdscr.addstr(HEIGHT+1, 0, 'Change the direction to up, left, bottom and right by pressing "w" "a" "s" and "d" accordingly.\nscore: {}; time: {}; framerate: {:.2f}'.format(
                len(body)-1, loop, 1/DELAY))
            stdscr.refresh()
        except curses.error:
            curses.endwin()
            raise BaseException(
                'please enlarge your window to fit the game world (width:{},height:{})'.format(WIDTH, HEIGHT))
else:
    print('Unsupported system.')
    quit()

DELAY = 0.1                         # 设置每帧持续的时间（秒）
WIDTH, HEIGHT = (30, 15)            # 地图宽度和高度
FOODCOUNT=3                         # 同时存在的食物数量

PATTERN = {0: ' ', 1: '■', 2: '●'}  # 空白，蛇，食物对应的图案
DIRECTION = {                       # 四个方向的步进函数
    0: lambda p: (p[0], p[1]-1),    # 向上，纵坐标减一
    1: lambda p: (p[0]+1, p[1]),    # 向右，横坐标加一
    2: lambda p: (p[0], p[1]+1),    # 向下，纵坐标加一
    3: lambda p: (p[0]-1, p[1])     # 向左，横坐标减一
}
KEYDRC = {                          # 按键响应函数（不对反方向的指令做出响应）
    'w': lambda c: 0 if c != 2 else 2,  
    'd': lambda c: 1 if c != 3 else 3,
    's': lambda c: 2 if c != 0 else 0,
    'a': lambda c: 3 if c != 1 else 1
}

# 蛇的身体以二元组列表的方式存储，每个二元组为蛇的一节的坐标，列表尾部为蛇的头部，初始值为地图中心
body = [(int(WIDTH/2), int(HEIGHT/2))]  
food = [(randrange(1, WIDTH-1), randrange(1, HEIGHT-1)) for _ in range(FOODCOUNT)] # 随机生成食物
drc = 1 # 初始方向向右
# 0:up 1:right 2:down 3:left

loop = 0        # 循环次数计数
while 1:        # 主循环
    st = time() # 帧开始时间
    loop += 1   
    # move ==========================
    body.append(DIRECTION[drc](body[-1]))   # 将头部向当前方向偏移一格，作为新的头部（添加至列表末尾）
    if body[-1][0] >= WIDTH or body[-1][0] < 0 or body[-1][1] >= HEIGHT or body[-1][1] < 0 or body[-1] in body[:-1]:
        break # 如果检测到头部超越地图边界，退出循环
    if body[-1] not in food: 
        # 如果蛇的头部没有触碰到食物，则删除蛇的尾部（列表的头部），整体表现为蛇运动了一格
        body.pop(0)
    else:
        food.remove(body[-1]) # 头部碰到食物，删除食物
        while 1:    # 生成新的食物
            newfood = (randrange(1, WIDTH-1), randrange(1, HEIGHT-1))
            if newfood not in body+food: # 防止与蛇身、现有食物重合
                food.append(newfood)
                break

    # output ========================
    world = [[PATTERN[0] for i in range(WIDTH)] for j in range(HEIGHT)] # 生成空白地图
    for x, y in body:
        world[y][x] = PATTERN[1]    # 绘制蛇身
    for x, y in food:
        world[y][x] = PATTERN[2]    # 绘制食物
    showworld()                     # 显示

    # input =========================
    readinput()                     # 接收键盘输入

if platform.startswith('linux') or platform.startswith('darwin'):
    curses.endwin() 

print('GAME OVER!\nscore: {}'.format(len(body)-1))  # 结束游戏，并打印得分信息