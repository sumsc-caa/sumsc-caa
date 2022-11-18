#!/usr/bin/python3
#######################################
# Author: furffy@github.com           #
# License: GNU GPLv3                  #
# Written for SUMSC Hackathon 2022    #
#######################################

from typing import Tuple
import numpy as np
import time
import os
import platform
import random

BORDER = "┏━┓┃█┃┗━┛"


if platform.system() in ["Linux", "Darwin"]:
    def cls():
        os.system("clear")
elif platform.system() in ["Windows"]:
    def cls():
        os.system("cls")
else:
    raise Exception("Unsupported Operating System")


def getShape(w=None, h=None) -> Tuple[int, int]:
    if w is None or h is None:
        size = os.get_terminal_size()
        c, l = size.columns, size.lines
        return (w or c//2-2), (h or l-3)
    else:
        return w, h


def showWorld(w: int, h: int, title: str = ""):
    block = BORDER[4] * 2
    title = ' '+title+' ' if title else ''
    upborder = BORDER[0] + BORDER[1] + title + BORDER[1]*(w*2+1-len(title)) + BORDER[2]
    item2block = lambda item: block if item else '  '
    getrow = lambda row: BORDER[3] + ' ' + ''.join(map(item2block, row)) + ' ' + BORDER[5]
    
    def wrapped(mat: np.ndarray, text=""):
        text = ' ' + text + ' ' if text else ''
        cls()
        print(upborder)
        print(*map(getrow, mat), sep="\n")
        print(BORDER[6], BORDER[7], text, BORDER[7] * (w*2+1-len(text)), BORDER[8], sep="")

    return wrapped

def updateWorld(world: np.ndarray):  # edit in place
    h, w = world.shape
    # padding edges with zeros
    padding = np.zeros((h+2, w+2), dtype=np.uint8)
    padding[1:-1, 1:-1] = world
    # convolution with 3x3 kernel
    neighbours = (
        padding[:-2,  :-2] + padding[:-2, 1:-1] + padding[:-2,  2:] +
        padding[1:-1, :-2] + 0                  + padding[1:-1, 2:] +
        padding[2:,   :-2] + padding[2:,  1:-1] + padding[2:,   2:]
    )

    # Rules [source: wikipedia https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life]
    # Any live cell with fewer than two live neighbours dies, as if by underpopulation.
    # Any live cell with two or three live neighbours lives on to the next generation.
    # Any live cell with more than three live neighbours dies, as if by overpopulation.
    # Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
    world[np.bitwise_or(neighbours < 2, neighbours > 3)] = 0
    world[neighbours == 3] = 1


def worldHash(world: np.ndarray):
    # the hash value is same for same states.
    bitmap = np.packbits(world).tobytes()
    h = hash(bitmap)
    return abs(h)*2 + int(h<0)
    

def loadfromFile(file, w: int, h: int, position = 'cc'):
    # compatiable with plaintext format: https://conwaylife.com/wiki/Plaintext
    lst = []
    for line in file.readlines():
        line=line.strip()
        if line.startswith('!') or len(line) == 0:
            continue
        row = [i != '.' for i in line]
        lst.append(row)
    file.close()
    pw, ph = max(0, *map(len, lst)), len(lst)
    assert pw < w-2 and ph < h-2, "Width and height of the input should be greater than that of the world"
    cw = {'l':1, 'c':(w-pw)//2, 'r': w-pw-2}[position[0]] 
    ch = {'t':1, 'c':(h-ph)//2, 'b': h-ph-2}[position[1]]
    world = np.zeros((h, w), dtype=bool)
    for i, row in enumerate(lst):
        world[ch+i, cw: cw+len(row)] = row
    return world


def main(
    shape: Tuple[int, int] = None, 
    iteration: int = 0, 
    autoexit: bool = False, 
    framerate: int = 10, 
    exitcount: int = 20, 
    file: str = "", 
    position: str = "cc",
    waitinit: bool = False,
    *args, **kwargs
):
    w, h = shape or getShape()
    drawWorld = showWorld(w, h, "Conway's Game of Life")
    interval = 1 / framerate
    
    # initialize world
    if file:
        world = loadfromFile(file, w, h, position)
    else:
        # random initialization
        world = np.zeros((h, w), dtype=bool)
        for i in range(w*h//5):
            world[random.randint(0, h-1), random.randint(0, w-1)] = 1
    
    # main loop
    stateHashRecord = {} 
    t = 0 
    while not iteration or t <= iteration:
        start = time.time()
        whash = worldHash(world)
        occurrance = stateHashRecord.get(whash, None) 
        stateHashRecord[whash] = t
        livecells = world.sum().item()
        drawWorld(
            world, 
            text=( 
                f't={t:d} | live cells: {livecells:d}' +
                f' | hash: {whash:016x}' + 
                (f' | last occurrance: {occurrance}(t-{t-occurrance})' if occurrance is not None else '')
            )
        )
        if waitinit and t == 0:
            input("press ENTER to start simulation")
        
        if autoexit:
            if occurrance is not None:
                if exitcount <= 0:
                    print('20 repeated states detected, exiting.')
                    break
                exitcount -= 1
            if livecells <= 0:
                print("All cells are dead, exiting.")
                break

        time.sleep(max(interval - (time.time() - start), 0))
        updateWorld(world)
        t += 1


if __name__ == "__main__":
    import argparse

    # parse cli arguments
    parser = argparse.ArgumentParser(
        prog="gameoflife",
        description="Simple implementation of Conway's Game of Life in python, CLI version. Written by Furffico for SUMSC Hackathon 2022.",
        epilog="This program is distributed under GNU GENERAL PUBLIC LICENSE version 3, for full text please refer to https://www.gnu.org/licenses/gpl-3.0.html"
    )
    parser.add_argument("file", metavar="FILE", action='store', nargs="?", default=None, type=argparse.FileType("r"),
        help="Input initial state [default: random]")
    parser.add_argument("-s", "--shape",  metavar=("W", "H"), nargs=2, type=int, default=None, dest="shape",
        help="Shape of the world. [default: fill the terminal]")
    parser.add_argument("-t", "--iteration", metavar="T", action='store', default=0, dest='iteration', type=int, 
        help="Maximum count of iterations (set to 0 for infinity). [default: 0]")
    parser.add_argument("-e", "--autoexit", action='store_true', default=False, dest='autoexit',
        help="Terminate the simulation when certain conditions are met (e.g. no live cells, repeated states). [default: False]")
    parser.add_argument("-f", "--fps", metavar="F", action='store', default=10, dest='framerate', type=int,
        help="Set the framerate (frames per second). [default: 10]")
    parser.add_argument("-p", "--position", metavar="P", action='store', default="cc", dest='position', type=str,
        choices = ["lt", "ct", "rt", "lc", "cc", 'rc', 'lb', 'cb', 'rb'],
        help="Select where to put the loaded world. The first letter indicates horizontal position (l,c,r); the second letter indicates vertical position (t, c, b). [default: cc, choices: lt, ct, rt, lc, cc, rc, lb, cb, rb]")
    parser.add_argument("-w", "--waitinit", action='store_true', default=False, dest='waitinit',
        help="Pause the program after the initial state is drawn. [default: False]")
    args = parser.parse_args()

    # validate arguments
    assert args.shape is None or (args.shape[0] > 5 and args.shape[1] > 5)
    assert args.iteration >= 0

    # main program
    try:
        main(**dict(args._get_kwargs()))
    except KeyboardInterrupt:
        print("Interrupted.")

    print("Program exited.")


