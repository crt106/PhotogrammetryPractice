# -*- coding: utf-8 -*-
import operator
import PIL.Image as plm
import math
import matplotlib.pyplot as plt
import numpy as np

from CommonUtils import Util
from Constant import const


def getABC(out: list, xyz):
    """
    求得核线中间变量ABC(基于共面条件方法)
    :param out:外方位元素
    :param xyz: 指定点位的像空间坐标
    :return:
    """
    phi = out[3]
    omega = out[4]
    kappa = out[5]
    a1 = math.cos(phi) * math.cos(kappa) - math.sin(phi) * math.sin(omega) * math.sin(kappa)
    a2 = -math.cos(phi) * math.sin(kappa) - math.sin(phi) * math.sin(omega) * math.cos(kappa)
    a3 = -math.sin(phi) * math.cos(omega)
    b1 = math.cos(omega) * math.sin(kappa)
    b2 = math.cos(omega) * math.cos(kappa)
    b3 = -math.sin(omega)
    c1 = math.sin(phi) * math.cos(kappa) + math.cos(phi) * math.sin(omega) * math.sin(kappa)
    c2 = -math.sin(phi) * math.sin(kappa) + math.cos(phi) * math.sin(omega) * math.cos(kappa)
    c3 = math.cos(phi) * math.cos(omega)
    R = np.mat([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]])
    XYZ = R * xyz
    X = XYZ[0, 0]
    Y = XYZ[1, 0]
    Z = XYZ[2, 0]
    A = Y * c1 - Z * b1
    B = Z * b2 - Y * c2
    C = Z * b3 - Y * c3
    return A, B, C


def getL(matchList: list):
    """
    获得参数L1-L8(利用相对定向直接解进行核线排列 《数字摄影测量学基础 P96》)
    :param matchList:匹配的同名像点
    :return:
    """
    x1 = [one[1] for one in matchList]
    y1 = [one[2] for one in matchList]
    x2 = [one[3] for one in matchList]
    y2 = [one[4] for one in matchList]

    B: np.matrix = []
    Q: np.matrix = []
    # 构建A和Q
    for i in range(len(x1)):
        B.append(
            [1, x1[i], y1[i], x2[i], x1[i] * x2[i], x1[i] * y2[i], y1[i] * x2[i], y1[i] * y2[i]])
        Q.append([y1[i] - y2[i]])
    B = np.mat(B)
    Q = np.mat(Q)
    # 构建初值
    L = np.zeros((8, 1))
    while True:
        l = Q - B * L
        v = (B.T * B).I * B.T * l
        L = L + v
        if Util.valueJudge(v, 1E-15):
            break
    return L


def drawSearchPoint(im1: plm.Image, im2: plm.Image, xlist: list, ylist1: list, ylist2: list,
                    needConvert=True):
    """
    在图像上绘制沿着同名核线搜索的同名像点情况
    :param im1: 左图像
    :param im2: 右图像
    :param xlist: x值列表
    :param ylist1: 左图像y值列表
    :param ylist2: 右图像y值列表
    :param needConvert 是否需要从像平面坐标系转换到像素坐标系
    :return:
    """
    # region 沿同名核线绘制RGB分布
    image2, image4 = im1.load(), im2.load()

    # 转换到像素坐标系并且取整，方便提取像素
    if needConvert:
        for i in range(len(xlist)):
            [x, y1] = Util.CoorPhoto2Pixel([xlist[i], ylist1[i]])
            xlist[i] = x
            [x, y2] = Util.CoorPhoto2Pixel([xlist[i], ylist2[i]])
            ylist1[i] = y1
            ylist2[i] = y2

    xlist = [int(l) for l in xlist]
    ylist1 = [int(l) for l in ylist1]
    ylist2 = [int(l) for l in ylist2]

    # 获取RGB属性
    RGBx = []
    RGB1y = []
    RGB2y = []
    for index in range(len(xlist)):
        try:
            x = xlist[index]
            y1 = ylist1[index]
            y2 = ylist2[index]
            RGB1y.append(image2[x, y1])
            RGB2y.append(image4[x, y2])
            RGBx.append(x)
        except BaseException:
            continue

    # 绘制曲线
    plt.subplot(223)
    plt.title("左图像核线上点的RGB值")
    plt.plot(RGBx, [x[0] for x in RGB1y], color='red')
    plt.plot(RGBx, [x[1] for x in RGB1y], color='green')
    plt.plot(RGBx, [x[2] for x in RGB1y], color='blue')
    plt.ylim(0, 255)

    plt.subplot(224)
    plt.title("右图像核线上点的RGB值")
    plt.plot(RGBx, [x[0] for x in RGB2y], color='red')
    plt.plot(RGBx, [x[1] for x in RGB2y], color='green')
    plt.plot(RGBx, [x[2] for x in RGB2y], color='blue')
    plt.ylim(0, 255)
    # endregion


def draw(f, k1, b1, k2, b2, coor4507_1, coor4507_2):
    """
    在图像上绘制同名核线(基于共面条件)
    :param f: 焦距
    :param k1: 左图像核线参数A/B
    :param b1: 左图像核线参数C/B
    :param k2: 右图像核线参数A/B
    :param b2: 右图像核线参数C/B
    :return:
    """
    im2: plm.Image = plm.open("resource/002.jpg")
    im4: plm.Image = plm.open("resource/004.jpg")
    ix, iy = im2.size[0], im2.size[1]
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure("同名核线")

    xmax = (Util.CoorPixel2Photo([const.iwidth, 0]))[0]
    xmin = (Util.CoorPixel2Photo([0, 0]))[0]

    # region 绘制第一张子图 002核线图
    plt.subplot(221)
    plt.imshow(im2)
    line1x = np.linspace(xmin, xmax, 50)
    line1y = k1 * line1x + b1 * f
    # 坐标转换到像素坐标
    for i in range(len(line1x)):
        [x, y] = Util.CoorPhoto2Pixel([line1x[i], line1y[i]])
        line1x[i] = x
        line1y[i] = y

    point1 = Util.CoorPhoto2Pixel(coor4507_1[0:2])
    plt.plot(line1x, line1y)
    plt.scatter(point1[0], point1[1], s=2, color='red')
    # 限制坐标范围
    plt.xlim(0, ix)
    plt.ylim(iy, 0)
    # endregion

    # region 绘制第二张子图 004 核线图
    plt.subplot(222)
    plt.imshow(im4)
    line2x = np.linspace(xmin, xmax, 50)
    line2y = k2 * line2x + b2 * f
    for i in range(len(line2x)):
        [x, y] = Util.CoorPhoto2Pixel([line2x[i], line2y[i]])
        line2x[i] = x
        line2y[i] = y
    plt.plot(line2x, line2y)
    point2 = Util.CoorPhoto2Pixel(coor4507_2[0:2])
    plt.scatter(point2[0], point2[1], s=2, color='red')
    # 限制坐标范围
    plt.xlim(0, ix)
    plt.ylim(iy, 0)
    # endregion

    # 绘制同名核线搜索结果
    line3x = np.linspace(xmin, xmax, 2000)
    line3y1 = k1 * line3x + b1 * f
    line3y2 = k2 * line3x + b2 * f
    drawSearchPoint(im2, im4, line3x, line3y1, line3y2)

    plt.show()


def draw(matchList: list, coor4507_1, coor4507_2):
    """
    绘制同名核线(基于相对定向直接解进行核线排列)
    :return:
    """
    L = getL(matchList)
    L1 = L[0, 0]
    L2 = L[1, 0]
    L3 = L[2, 0]
    L4 = L[3, 0]
    L5 = L[4, 0]
    L6 = L[5, 0]
    L7 = L[6, 0]
    L8 = L[7, 0]

    x1 = coor4507_1[0]
    y1 = coor4507_1[1]

    xmax = (Util.CoorPixel2Photo([const.iwidth, 0]))[0]
    xmin = (Util.CoorPixel2Photo([0, 0]))[0]

    def tryCoor1(x1, y1, tryx):
        """
        利用L系数和任意取tryx求解tryy(左求右)
        :param x1: 已知像点的x坐标
        :param y1: 已知像点的y坐标
        :param tryx: 任意选取的点的x坐标
        :return:任意选取的点求得的y坐标
        """
        return ((1 - L3) * y1 - L1 - L2 * x1 - L4 * tryx - L5 * x1 * tryx - L7 * y1 * tryx) / (
                1 + L6 * x1 + L8 * y1)

    def tryCoor2(x1, y1, tryx):
        """
        利用L系数和任意取tryx求解tryy(右求左)
        :param x1: 已知像点的x坐标
        :param y1: 已知像点的y坐标
        :param tryx: 任意选取的点的x坐标
        :return:任意选取的点求得的y坐标
        """
        return (y1 + L1 + L2 * tryx + L4 * x1 + L5 * x1 * tryx + L6 * tryx * y1) / (
                1 - L3 - L7 * x1 - L8 * y1)

    # 点位try1，try2是右方影像的随机取点(左求右) try3,try4是反求的左侧影像的点(右求左)
    tryy1 = tryCoor1(x1, y1, xmin)
    tryy2 = tryCoor1(x1, y1, xmax)
    tryy3 = tryCoor2(coor4507_2[0], coor4507_2[1], xmin)
    tryy4 = tryCoor2(coor4507_2[0], coor4507_2[1], xmax)

    [tryx1, tryy1] = Util.CoorPhoto2Pixel([xmin, tryy1])
    [tryx2, tryy2] = Util.CoorPhoto2Pixel([xmax, tryy2])
    [tryx3, tryy3] = Util.CoorPhoto2Pixel([xmin, tryy3])
    [tryx4, tryy4] = Util.CoorPhoto2Pixel([xmax, tryy4])

    im2: plm.Image = plm.open("resource/002.jpg")
    im4: plm.Image = plm.open("resource/004.jpg")
    ix, iy = im2.size[0], im2.size[1]
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure("同名核线")

    # region 绘制第一张子图 002核线图
    plt.subplot(221)
    plt.imshow(im2)
    point1 = Util.CoorPhoto2Pixel(coor4507_1[0:2])
    plt.plot([tryx3, tryx4], [tryy3, tryy4])
    plt.scatter(point1[0], point1[1], s=10, color='red', marker="^")
    # 限制坐标范围
    plt.xlim(0, ix)
    plt.ylim(iy, 0)
    # endregion

    # region 绘制第二张子图 004 核线图
    plt.subplot(222)
    plt.imshow(im4)
    point2 = Util.CoorPhoto2Pixel(coor4507_2[0:2])
    plt.plot([tryx1, tryx2], [tryy1, tryy2])
    plt.scatter(point2[0], point2[1], s=10, color='red', marker="^")
    # 限制坐标范围
    plt.xlim(0, ix)
    plt.ylim(iy, 0)

    # endregion

    def getLine(x1, y1, x2, y2, x: list):
        """
        两点式求直线方程
        :param x1:
        :param y1:
        :param x2:
        :param y2:
        :param x:
        :return:
        """
        return (x - x1) * (y2 - y1) / (x2 - x1) + y1

    x = np.arange(0, const.iwidth, 1)
    y1 = getLine(tryx1, tryy1, tryx2, tryy2, x)
    y2 = getLine(tryx3, tryy3, tryx4, tryy4, x)
    drawSearchPoint(im2, im4, x, y2, y1, False)

    plt.show()


# 主方法
# 手动输入外方位元素,毕竟源txt不规整，用正则又多余
# [Xs,Ys,Zs,φ,ω,κ]
f = const.f
out1 = [4079.39113, -145.16986, -298.34664, 0.21148, 0.06098, -0.08535]
out2 = [3373.40082, -141.55657, 92.77483, -0.01205, 0.09997, 0.04101]
# out2 = [3373.40082, -141.55657, 92.77483, -0.01205, 0.29997, 0.14101]
l1 = Util.readPicPoint('resource/002.txt')
l2 = Util.readPicPoint('resource/004.txt')
# 获取4507点的左图像坐标 并且转换到像空间坐标系
coor4507 = [one[1:] for one in l1 if one[0] == 4507][0]
coor4507_2 = [one[1:] for one in l2 if one[0] == 4507][0]
coor4507.append(-1.0 * f)
coor4507_2.append(-1.0 * f)
# 转换为矩阵
mat4507 = np.mat(coor4507).reshape(3, 1)
mat4507_2 = np.mat(coor4507_2).reshape(3, 1)
A1, B1, C1 = getABC(out1, mat4507)
A2, B2, C2 = getABC(out2, mat4507)
# draw(f, A1 / B1, C1 / B1, A2 / B2, C2 / B2, coor4507, coor4507_2)
draw(Util.matchPicPoint(l1, l2), coor4507, coor4507_2)
