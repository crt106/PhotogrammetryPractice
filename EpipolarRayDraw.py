# -*- coding: utf-8 -*-
import re
import PIL.Image as plm
import math
import matplotlib.pyplot as plt
import numpy as np
from Constant import const


def readPicPoint(filepath):
    """
    读取单个相片点位数据
    :return: 返回有效数据列表
             [点号,x,y]
    """

    resultList = []
    pattern = "\d+\s\d+\.\d+\s\d+\.\d+"
    f = open(filepath, mode='r')
    for line in f:
        if re.match(pattern, line):
            # -----添加数据的时候转换到像平面坐标系-----
            num = int(line.split(' ')[0])
            # x = (float(line.split(' ')[1]) - iwidth / 2) * const.pixel_length - const.x0
            # y = -1 * (float(line.split(' ')[2]) - iheight / 2) * const.pixel_length - const.y0
            u = float(line.split(' ')[1])
            v = float(line.split(' ')[2])
            [x, y] = CoorPixel2Photo([u, v])
            # TODO 从图像数据中获取
            resultList.append([num, x, y])
    return resultList


def getABC(out: list, xyz):
    """
    求得核线中间变量ABC
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


def CoorPixel2Photo(uv, iwidth=const.iwidth, iheight=const.iwidth):
    """
    像素坐标转换为像平面坐标系
    :param uv:
    :param iwidth:
    :param iheight:
    :return:
    """
    x = (uv[0] - iwidth / 2) * const.pixel_length - const.x0
    y = (uv[1] - iheight / 2) * const.pixel_length - const.y0
    return [x, y]


def CoorPhoto2Pixel(xy, iwidth=const.iwidth, iheight=const.iwidth):
    """
    像平面坐标系坐标转换为像素坐标
    :param xy:
    :return:
    """
    u = (xy[0] + const.x0) / const.pixel_length + iwidth / 2
    v = (xy[1] + const.y0) / const.pixel_length + iheight / 2
    return [u, v]


def draw(f, k1, b1, k2, b2, coor4507_1, coor4507_2):
    """
    在图像上绘制同名核线 并且绘制沿着该同名核线搜索的同名像点
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
    # imgdata2, imgdata4 = np.asarray(im2), np.asarray(im4)
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure("同名核线")

    xmax = (CoorPixel2Photo([const.iwidth, 0]))[0]
    xmin = (CoorPixel2Photo([0, 0]))[0]

    # region 绘制第一张子图 002核线图
    plt.subplot(221)
    plt.imshow(im2)
    line1x = np.linspace(xmin, xmax, 50)
    line1y = k1 * line1x + b1 * f
    # 坐标转换到像素坐标
    for i in range(len(line1x)):
        [x, y] = CoorPhoto2Pixel([line1x[i], line1y[i]])
        line1x[i] = x
        line1y[i] = y

    point1 = CoorPhoto2Pixel(coor4507_1[0:2])
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
        [x, y] = CoorPhoto2Pixel([line2x[i], line2y[i]])
        line2x[i] = x
        line2y[i] = y
    plt.plot(line2x, line2y)
    point2 = CoorPhoto2Pixel(coor4507_2[0:2])
    plt.scatter(point2[0], point2[1], s=2, color='red')
    # 限制坐标范围
    plt.xlim(0, ix)
    plt.ylim(iy, 0)
    # endregion

    # region 沿同名核线绘制RGB分布
    image2, image4 = im2.load(), im4.load()
    # TODO 修改x序列取值
    line3x = np.linspace(xmin, xmax, 2000)
    line3y1 = k1 * line3x + b1 * f
    line3y2 = k2 * line3x + b2 * f

    for i in range(len(line3x)):
        [x, y1] = CoorPhoto2Pixel([line3x[i], line3y1[i]])
        line3x[i] = x
        [x, y2] = CoorPhoto2Pixel([line3x[i], line3y2[i]])
        line3y1[i] = y1
        line3y2[i] = y2

    line3x = [int(l) for l in line3x]
    line3y1 = [int(l) for l in line3y1]
    line3y2 = [int(l) for l in line3y2]

    # TODO 提取像素
    RGBx = []
    RGB1y = []
    RGB2y = []
    for index in range(len(line3x)):
        try:
            x = line3x[index]
            y1 = line3y1[index]
            y2 = line3y2[index]
            RGB1y.append(image2[x, y1])
            RGB2y.append(image4[x, y2])
            RGBx.append(x)
        except BaseException:
            continue
    # TODO 绘制曲线

    plt.subplot(223)
    plt.plot(RGBx, [x[0] for x in RGB1y], color='red')
    plt.plot(RGBx, [x[1] for x in RGB1y], color='green')
    plt.plot(RGBx, [x[2] for x in RGB1y], color='blue')

    plt.subplot(224)
    plt.plot(RGBx, [x[0] for x in RGB2y], color='red')
    plt.plot(RGBx, [x[1] for x in RGB2y], color='green')
    plt.plot(RGBx, [x[2] for x in RGB2y], color='blue')
    # endregion

    plt.show()


# 手动输入外方位元素,毕竟源txt不规整，用正则又多余
# [Xs,Ys,Zs,φ,ω,κ]
f = const.f
out1 = [4079.39113, -145.16986, -298.34664, 0.21148, 0.06098, -0.08535]
out2 = [3373.40082, -141.55657, 92.77483, -0.01205, 0.09997, 0.04101]
# out2 = [3373.40082, -141.55657, 92.77483, -0.01205, 0.29997, 0.14101]
l1 = readPicPoint('resource/002.txt')
l2 = readPicPoint('resource/004.txt')
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
draw(f, A1 / B1, C1 / B1, A2 / B2, C2 / B2, coor4507, coor4507_2)
# A1, B1, C1 = getABC(out1, mat4507_2)
# A2, B2, C2 = getABC(out2, mat4507_2)
# draw(f, A1 / B1, C1 / B1, A2 / B2, C2 / B2, coor4507, coor4507_2)
