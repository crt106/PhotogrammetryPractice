import numpy as np
import re


def readPicPoint(filepath, iwidth=5344, iheight=4008):
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
            num = int(line.split(' ')[0])
            x = float(line.split(' ')[1]) - iwidth / 2
            y = float(line.split(' ')[2]) - iheight / 2
            # 添加数据的时候转换到像平面坐标系
            # TODO 从图像数据中获取
            resultList.append([num, x / 1000.0, y / 1000.0])
    return resultList


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


def getAssistCoordinate(phi, omega, kappa, x, y, f):
    """
    构建微小旋转矩阵
    获得左右侧图像的像点像空间辅助坐标[X1,Y1,Z1]或[X2,Y2,Z2]T
    :return:
    """
    R = np.mat([[1, -1 * kappa, -1 * phi],
                [kappa, 1, -1 * omega],
                [phi, omega, 1]])
    right = np.mat([[x],
                    [y],
                    [-1.0 * f]])
    return R * right


def calculteN(f, Bx, matchPoint, elementMat):
    """
    利用一组匹配的点位计算N与N'
    :param matchPoint: matchPicPoint方法得到的匹配点位
    :param elementMat:相对定向元素矩阵
    :return: N1,N2(N,N'),Bx,By,q
    """
    x1 = matchPoint[1]
    y1 = matchPoint[2]
    x2 = matchPoint[3]
    y2 = matchPoint[4]
    miu = elementMat[3, 0]
    v = elementMat[4, 0]
    # 获取左侧相片的辅助坐标
    assistCoor1 = getAssistCoordinate(0, 0, 0, x1, y1, f)
    # 获取右侧相片的辅助坐标
    assistCoor2 = getAssistCoordinate(elementMat[0, 0], elementMat[1, 0], elementMat[2, 0], x2, y2,
                                      f)

    X1 = assistCoor1[0, 0]
    Y1 = assistCoor1[1, 0]
    Z1 = assistCoor1[2, 0]
    X2 = assistCoor2[0, 0]
    Y2 = assistCoor2[1, 0]
    Z2 = assistCoor2[2, 0]

    By = Bx * miu
    Bz = Bx * v
    N1 = (Bx * Z2 - Bz * X2) / (X1 * Z2 - Z1 * X2)
    N2 = (Bx * Z1 - Bz * X1) / (X1 * Z2 - Z1 * X2)
    q = N1 * Y1 - N2 * Y2 - By
    return N1, N2, q


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


l1 = readPicPoint('resource/002.txt')
l2 = readPicPoint('resource/004.txt')
matchList = matchPicPoint(l1, l2)

# matchList = [[1, -0.49532, -0.22204, 0.0488, -0.3294],
#              [2, 1.1468, -0.1952, 1.75924, -0.31598],
#              [3, 0.12322, 0.2806, 0.52948, 0.17324],
#              [4, 0.33184, 0.28548, 0.7381, 0.18178],
#              [5, -0.52216, 0.38186, -0.79422, 0.25986],
#              [6, 0.15738, 0.4026, -0.1952, 0.29036],
#              [7, -1.33834, 0.48312, -1.23098, 0.33916],
#              [8, 1.3603, 0.57706, 1.26148, 0.49166],
#              [9, -1.9459, 0.5734, -1.94956, 0.40504],
#              [10, 1.71776, 0.70516, 1.2871, 0.61976]]

if len(matchList) > 5:

    """
    初始化相对定向元素 矩阵内容为
    [φ,ω,κ,μ,v]T
    """
    elementMat = np.zeros((5, 1))
    # 焦距f
    f = 35

    loopcount = 0
    # 主循环过程
    while True:
        A = np.zeros((1, 5))
        Q = np.zeros((1, 1))
        # 随意估算Bx
        Bx = matchList[0][1] - matchList[0][3]
        for mpoint in matchList:
            x2 = mpoint[3]
            y2 = mpoint[4]
            assistCoor2 = getAssistCoordinate(elementMat[0, 0], elementMat[1, 0], elementMat[2, 0],
                                              x2, y2, f)
            X2 = assistCoor2[0, 0]
            Y2 = assistCoor2[1, 0]
            Z2 = assistCoor2[2, 0]
            N1, N2, q = calculteN(f, Bx, mpoint, elementMat)
            # 构建误差方程式系数
            para1 = -1.0 * X2 * Y2 * N2 / Z2
            para2 = -1.0 * N2 * (Z2 + pow(Y2, 2) / Z2)
            para3 = X2 * N2
            para4 = Bx
            para5 = -1.0 * Y2 / Z2 * Bx
            A1 = np.mat([para1, para2, para3, para4, para5])
            # 合并矩阵 区分第一次情况
            if A.shape[0] == 1 and valueJudge(A, 0):
                A = A1
                Q = np.mat(q)
            else:
                A = np.row_stack((A, A1))
                Q = np.row_stack((Q, np.mat(q)))
        X = (A.T * A).I * A.T * Q
        elementMat = elementMat + X
        loopcount += 1
        print('迭代次数{}'.format(loopcount))
        print("X:", X)
        print("element:", elementMat)

        if valueJudge(X, 0.3E-4):
            # 迭代完成 输出结果
            print("----------------")
            print("迭代完成({:d}) 最终方位元素:\n".format(loopcount))
            print("| φ:{:^.5f}\t|".format(elementMat[0, 0]))
            print("| ω:{:^.5f}\t|".format(elementMat[1, 0]))
            print("| κ:{:^.5f}\t|".format(elementMat[2, 0]))
            print("| μ:{:^.5f}\t|".format(elementMat[3, 0]))
            print("| v:{:^.5f}\t|".format(elementMat[4, 0]))
            print("---------------")
            break
else:
    print('同名点位数据数量不足 结束运算')