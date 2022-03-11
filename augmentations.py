"""
更新日志
2021.10.21
-为gray_change增加参数，使其可以调节接受的图片是否归一化，默认是归一化前(None)，若是归一化后，则向参数guiyihua输入归一化值。
-增加对3D输入（image_cube）的图像增强函数。
2021.10.24
-增加了弹性变换的函数，可以进行弹性变换增强。
"""
import numpy as np
import cv2
from scipy import ndimage


def gray_change(pic, addminus = True, time = False, guiyihua = None):
    '''
    此函数用于对pic做随机灰度变换。
    :param pic: 输入图片。注意是归一化之前的图片!
    :param addminus: 是否进行加减变换
    :param time: 是否进行乘系数变换。两个均为False即不变换
    :param guiyihua: 归一化除的数值，None为不进行归一化
    :return: 变换后的图片，待归一化
    '''
    rand1 = np.random.rand()
    rand1 = rand1 * 100 - 50  # -50到50之间的随机数
    if guiyihua:
        rand1 = rand1 / guiyihua  # 除以归一化值
    rand2 = np.random.rand()
    rand2 = rand2 * 0.4 + 0.8  # 0.8到1.2之间的随机数
    pic = pic.astype(np.float32)
    if addminus:
        pic += rand1
    if time:
        pic *= rand2
    return pic


def add_noise(pic, num=300):
    '''
    此函数用于向pic添加随机噪声，将某些点的像素值改为0。
    :param pic: 输入图片，resize后、归一化前的图片。
    :param num: 改变的像素数量
    :return: 改变后的图片
    '''
    pic = pic.astype(np.float32)
    rand1 = np.random.randint(0, 255, [1, num])
    rand2 = np.random.randint(0, 255, [1, num])
    pic[rand1, rand2] = 0
    return pic

def gray_change_3D(cube, addminus=True, time=False, guiyihua=None, uniform=False):
    '''
    此函数用于对3D图像进行逐层灰度随机变化
    :param cube: 输入3D图片
    :param uniform: 图像的灰度变化程度是否在全图统一
    :return: 增强后的图片
    '''
    if uniform:
        cube1 = gray_change(cube, addminus=addminus, time=time, guiyihua=guiyihua)
    else:
        pic_list = []
        for pic in cube:
            pic1 = gray_change(pic, addminus=addminus, time=time, guiyihua=guiyihua)
            pic_list.append(pic1)
        cube1 = np.array(pic_list)
    return cube1


def add_noise_3D(cube, num=300):
    '''
    此函数用于向3D图像cube添加随机噪声，将某些点的像素值改为0。
    :param cube: 输入图片，resize后、归一化前的图片。
    :param num: 每个切片改变的像素数量
    :return: 改变后的图片
    '''
    pic_list = []
    for pic in cube:
        pic1 = add_noise(pic, num=num)
        pic_list.append(pic1)
    cube1 = np.array(pic_list)
    return cube1

def elastic_change(pic, sigma=4, alpha=8):
    '''
    此函数用于向图片pic进行弹性形变。
    :param pic: 输入图片
    :param sigma: 高斯滤波器的方差。sigma越小，滤波的平滑效果越低，弹性变换后的图片越随机
    :param alpha: 变形强度因子。alpha越大，变形强度越强
    :return:变形后的图片
    '''
    sh = pic.shape
    xoff = np.random.rand(sh[0], sh[1]) * 2 - 1
    yoff = np.random.rand(sh[0], sh[1]) * 2 - 1
    xoff = cv2.GaussianBlur(xoff, (5, 5), sigmaX=sigma, sigmaY=sigma)
    yoff = cv2.GaussianBlur(yoff, (5, 5), sigmaX=sigma, sigmaY=sigma)
    xoff = xoff * alpha
    yoff = yoff * alpha
    x1 = np.arange(float(sh[0]))
    y1 = np.arange(float(sh[1]))
    [x, y] = np.meshgrid(x1, y1)
    x += xoff
    y += yoff
    x = x.reshape(1, sh[0] * sh[1])
    y = y.reshape(1, sh[0] * sh[1])
    coordinates = np.concatenate([y, x], axis=0)
    im_changed = ndimage.map_coordinates(pic, coordinates, order=1)
    im_changed = im_changed.reshape([sh[0], sh[1]])
    return im_changed



if __name__ == '__main__':
    im = cv2.imread('1.jpg')
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im1 = elastic_transform(im, 4, 5)
    cv2.imwrite('11.jpg', im1)
