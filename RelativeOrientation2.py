# 利用直接设方向余弦的方法进行间接平差
import numpy as np

from CommonUtils import Util
from Constant import const


def getR(elementMat):
    """
    从原始元素矩阵中提取出旋转矩阵R
    :param elementMat:
    :return:
    """
    return elementMat[3:, :].reshape(3, 3)


def getAssistCoordinate(R, x, y, f):
    """
    构建旋转矩阵
    获得左右侧图像的像点像空间辅助坐标[X1,Y1,Z1]或[X2,Y2,Z2]T
    :return:
    """
    # R = np.mat([[1, -1 * kappa, -1 * phi],
    #             [kappa, 1, -1 * omega],
    #             [phi, omega, 1]])

    right = np.mat([[x],
                    [y],
                    [-1.0 * f]])
    return R * right


def getBandL(elementMat, XYZ1, XYZ2, x2, y2):
    """
    求B矩阵
    :return:
    """
    Bx = elementMat[0, 0]
    By = elementMat[1, 0]
    Bz = elementMat[2, 0]
    X1 = XYZ1[0, 0]
    Y1 = XYZ1[1, 0]
    Z1 = XYZ1[2, 0]
    X2 = XYZ2[0, 0]
    Y2 = XYZ2[1, 0]
    Z2 = XYZ2[2, 0]
    b11 = Y1 * Z2 - Y2 * Z1
    b12 = X2 * Z1 - X1 * Z2
    b13 = X1 * Y2 - X2 * Y1
    b14 = (By * Z1 - Bz * Y1) * (x2 - const.x0)
    b15 = (By * Z1 - Bz * Y1) * (y2 - const.y0)
    b16 = -1 * (By * Z1 - Bz * Y1) * const.f
    b17 = (Bz * X1 - Bx * Z1) * (x2 - const.x0)
    b18 = (Bz * X1 - Bx * Z1) * (y2 - const.y0)
    b19 = -1 * (Bz * X1 - Bx * Z1) * const.f
    b1a = (Bx * Y1 - By * X1) * (x2 - const.x0)
    b1b = (Bx * Y1 - By * X1) * (y2 - const.y0)
    b1c = -1 * (Bx * Y1 - By * X1) * const.f
    B = [b11, b12, b13, b14, b15, b16, b17, b18, b19, b1a, b1b, b1c]
    L = Bz * Y1 * X2 + By * X1 * Z2 + Bx * Y2 * Z1 - Bx * Y1 * Z2 - By * X2 * Z1 - Bz * X1 * Y2
    return B, L


def getCandWx(elementMat):
    """
    获取附加条件方程式中的C和Wx
    :param elementMat:
    :return:
    """
    Bx = elementMat[0, 0]
    By = elementMat[1, 0]
    Bz = elementMat[2, 0]
    a1 = elementMat[3, 0]
    a2 = elementMat[4, 0]
    a3 = elementMat[5, 0]
    b1 = elementMat[6, 0]
    b2 = elementMat[7, 0]
    b3 = elementMat[8, 0]
    c1 = elementMat[9, 0]
    c2 = elementMat[10, 0]
    c3 = elementMat[11, 0]

    C = np.mat([
        [2 * Bx, 2 * By, 2 * Bz, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2 * a1, 2 * a2, 2 * a3, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2 * b1, 2 * b2, 2 * b3, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * c1, 2 * c2, 2 * c3],
        [0, 0, 0, a2, a1, 0, b2, b1, 0, c2, c1, 0],
        [0, 0, 0, a3, 0, a1, b3, 0, b1, c3, 0, c1],
        [0, 0, 0, 0, a3, a3, 0, b3, b2, 0, c3, c2]])

    Wx = np.mat([[Bx * Bx + By * By + Bz * Bz - 1],
                 [a1 * a1 + a2 * a2 * a3 * a3 - 1],
                 [b1 * b1 + b2 * b2 * b3 * b3 - 1],
                 [c1 * c1 + c2 * c2 * c3 * c3 - 1],
                 [a1 * a2 + b1 * b2 + c1 * c2],
                 [a1 * a3 + b1 * b3 + c1 * c3],
                 [a2 * a3 + b2 * b3 + c2 * c3]])
    return C, Wx


def givens_rotation(A):
    """Givens变换"""
    (r, c) = np.shape(A)
    Q = np.identity(r)
    R = np.copy(A)
    (rows, cols) = np.tril_indices(r, -1, c)
    for (row, col) in zip(rows, cols):
        if R[row, col] != 0:  # R[row, col]=0则c=1,s=0,R、Q不变
            r_ = np.hypot(R[col, col], R[row, col])  # d
            c = R[col, col] / r_
            s = -R[row, col] / r_
            G = np.identity(r)
            G[[col, row], [col, row]] = c
            G[row, col] = s
            G[col, row] = -s
            R = np.dot(G, R)  # R=G(n-1,n)*...*G(2n)*...*G(23,1n)*...*G(12)*A
            Q = np.dot(Q, G.T)  # Q=G(n-1,n).T*...*G(2n).T*...*G(23,1n).T*...*G(12).T
    return (Q, R)


l1 = Util.readPicPoint('resource/002.txt')
l2 = Util.readPicPoint('resource/004.txt')
matchList = Util.matchPicPoint(l1, l2)

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
    初始化相对待求定向元素 矩阵内容为
    [Bx,By,Bz,a1,a2,a3,b1,b2,b3,c1,c2,c3]T
    """
    elementMat = np.mat([[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]]).T
    # 焦距f
    f = const.f

    loopcount = 0
    # 主循环过程
    while True:
        B = []
        L = []
        C = []
        Wx = []
        # 随意估算Bx
        Bx = matchList[0][1] - matchList[0][3]
        for mpoint in matchList:
            x1 = mpoint[1]
            y1 = mpoint[2]
            x2 = mpoint[3]
            y2 = mpoint[4]
            assistCoor1 = np.mat([[x1], [y1], [-1 * f]])
            assistCoor2 = getAssistCoordinate(getR(elementMat), x2, y2, f)
            B_one, L_one = getBandL(elementMat, assistCoor1, assistCoor2, x2, y2)
            C, Wx = getCandWx(elementMat)
            B.append(B_one)
            L.append(L_one)

        B = np.mat(B)
        Q, R = givens_rotation(B)
        Q, R = np.mat(Q), np.mat(R)
        L = np.mat(L).T
        NBB = R.T * R
        # 求广义逆
        NBB_I = np.linalg.pinv(NBB)
        Wu1 = B.T * L
        NCC = C * NBB_I * C.T
        NCC_I = np.linalg.pinv(NCC)
        X = (NBB_I - NBB_I * C.T * NCC_I * C * NBB_I) * Wu1 + NBB_I * C.T * NCC_I * Wx
        elementMat = elementMat + X
        loopcount += 1
        print('迭代次数{}'.format(loopcount))
        print("X:", X)
        print("element:", elementMat)

        if Util.valueJudge(X, 0.3E-4):
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
