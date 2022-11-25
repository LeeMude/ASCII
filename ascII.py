import sys,random,argparse
import numpy as np
import cv2 as cv
#70个等级的灰度
gscale1 = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
#10等级的灰度
gscale2 = '@%#*+=-:. '


def main():
    descStr = "This program converts an image into ASCII art."
    parser = argparse.ArgumentParser(description=descStr)
    # 传入一些参数，包括图像文件、ASCII图像的行数等
    parser.add_argument('--file', dest='imgFile', required=True)
    parser.add_argument('--scale', dest='scale', required=False)
    parser.add_argument('--cols', dest='cols', required=False)
    args = parser.parse_args()
    print(args.imgFile)
    # 设置默认的参数
    cols = 150  # 列
    if args.cols:
        cols = args.cols
    scale = 0.43  # 小方块的宽是高的0.4倍
    if args.scale:
        cols = args.scale

    #把传入的图像转化为灰度图像
    img = cv.imread(args.imgFile,0)
    #图片太大，缩放一下
    img = cv.resize(img,None,fx=1,fy=1,interpolation= cv.INTER_CUBIC)


    cv.imshow('img',img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # 图片的宽和高
    H, W = img.shape
    print(W)
    print(H)
    # res = img[1:200,50:300]
    # cv.imshow('res', res)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    #划分小方块，其中宽为w，高为h
    w = int(W/cols)
    h = int(w/scale)
    print(f"实际小方块的宽为{w}高为{h}")
    #计算实际有多少行和多少列
    cols = int(round(W/w,0))
    rows = int(round(H/h,0))
    print(f"实际行数为{rows}实际列数为{cols}")
    #把图片转换成计算结果的大小
    img = cv.resize(img,(cols*w,rows*h))

    #存储每个方块对应的符号对应的亮度等级
    ascii_img = np.zeros(shape=(rows,cols))
    #print(type(ascii_img))
   # print(type(int(round(0.2745 * np.average(img[0:100, 0:100]), 0))))
    #print(int(round(0.2745 * np.average(img[45*17:100*17, 82*7:100*7]), 0)))

    #计算每个方块的平均亮度 c为列，r为行
    for r in range(rows):
        #方块两个对角点的纵坐标
        y1 = h * r
        y2 = h * (r + 1)
        #print((y1,y2))
        for c in range(cols):
           # 方块两个对角点的横坐标
           x1 = w * c
           x2 = w * (c + 1)
           #print((x1, y1),(x2,y2))

           #计算每个方块的平均亮度，并把相应的亮度等级存到scii_img这个矩阵中
           ave = int(round(0.039 * np.average(img[y1:y2, x1:x2])))
           #ave = np.average(img[x1:x2, y1:y2])
           ascii_img[r,c] = ave
    # print(int(round(0.27 * np.average(img[y1:y2,x1:x2]), 0)))
    # ascii_img[r, c]=int(round(0.27 * np.average(img[y1:y2,x1:x2]), 0))

    print(ascii_img)
    print(type(ascii_img[r,c]))
    a =int(ascii_img[r,c])
    print(type(a))
    #打印
    for r in range(rows):
        for c in range(cols):
            print(gscale2[int(ascii_img[r,c])],end='')
        # 换行
        print()



main()

#python ascII.py --file E:/images/3.jpeg

