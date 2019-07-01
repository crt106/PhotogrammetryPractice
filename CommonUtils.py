# 存放通用工具
import re
import numpy as np
from Constant import const


class Util:

    @staticmethod
    def readPicPoint(filepath, convert2mm=True):
        """
        读取单个相片点位数据,并且更具情况使用单位像素宽度转换单位为毫米
        :param filepath:文件路径
        :param convert2mm:是否将单位转换为mm
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
                if convert2mm:
                    [x, y] = Util.CoorPixel2Photo([u, v])
                else:
                    [x, y] = [u, v]
                resultList.append([num, x, y])
        return resultList

    @staticmethod
    def matchPicPoint(p1, p2):
        """
        匹配两组相片点位数据中的同名点
        :param p1: 第一张照片数据
        :param p2: 第二章照片数据
        :return: 返回匹配结果
                 [点号,x1,y1,x2,y2]
        """
        resultList = []
        for point1 in p1:
            num1 = point1[0]
            for point2 in p2:
                num2 = point2[0]
                if num1 == num2:
                    resultList.append([num1, point1[1], point1[2], point2[1], point2[2]])
                    break
        return resultList

    @staticmethod
    def CoorPixel2Photo(uv, iwidth=const.iwidth, iheight=const.iheight):
        """
        像素坐标转换为像平面坐标系
        :param uv:单位mm
        :param iwidth:
        :param iheight:
        :return:
        """
        x = (uv[0] - iwidth / 2) * const.pixel_length - const.x0
        y = (uv[1] - iheight / 2) * const.pixel_length - const.y0
        return [x, y]

    @staticmethod
    def CoorPhoto2Pixel(xy, iwidth=const.iwidth, iheight=const.iheight):
        """
        像平面坐标系坐标转换为像素坐标
        :param xy:单位mm
        :return:
        """
        u = (xy[0] + const.x0) / const.pixel_length + iwidth / 2
        v = (xy[1] + const.y0) / const.pixel_length + iheight / 2
        return [u, v]

    @staticmethod
    def valueJudge(mat, value):
        """
        阈值判断方法 判断一个矩阵中的所有元素的绝对值是否都小于等于某个阈值
        :param mat:
        :param value:
        :return:
        """
        a = np.array(mat)
        for x in np.nditer(a):
            ax = abs(x)
            if ax > value:
                return False
        return True
