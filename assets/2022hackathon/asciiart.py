#!/usr/bin/python3

import numpy as np
from PIL import Image


FULL_SCALE = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^'\`. "
MINIMAL_SCALE = "@%#*+=-:. "


def makeASCIIart(image: np.ndarray, minimal = True) -> list[str]:
    assert len(image.shape) == 2
    scale = MINIMAL_SCALE if minimal else FULL_SCALE
    char_idx = (image/256*len(scale)).astype(int) # 将灰度转化为字符串中对应字符的下标
    result = []
    for row in char_idx:
        result.append("".join(map(scale.__getitem__, row))) # 将下标转为对应的字符并连接
    return result

if __name__=="__main__":
    
    im = Image.open("./trophy.png")
    # preprocessing?
    grey = im.convert("L")    # 将图像转换为灰度图 
    # resize?
    mat = np.asarray(grey)    # 将图像转化为numpy矩阵
    if im.mode == "RGBA":
        mat[np.asarray(im.getchannel('A'))==0]=255 # 将透明像素设为白色

    result = makeASCIIart(mat)
    for row in result:
        print(row)

