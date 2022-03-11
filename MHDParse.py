import SimpleITK as sitk
from medpy.io import load
import numpy as np
import time

def _set_ww_wc(image, ww, wc):
    '''
    此函数将输入的image通过添加窗宽窗位限制的方式映射到0-255的double数据中
    :param image:float32形式的numpy数组
    :param ww:窗宽
    :param wc:窗位
    :return:float32形式的转化后数组
    '''
    hi = wc + ww / 2
    lo = wc - ww / 2
    image = (image - lo) / (hi - lo)
    image[image < 0] = 0
    image[image > 1] = 1
    image = image * 255
    return image


def read_mhd(mhdpath, ww, wc):
    '''
    此函数将输入的mhd路径转化为numpy数组（float32形式），并添加窗宽窗位
    :param mhdpath:mhd文件路径
    :param ww:窗宽
    :param wc:窗位
    :return:第一项为float32形式的数组，第二项为各种信息
    '''
    t1 = time.time()
    a = sitk.ReadImage(mhdpath)
    # b = sitk.GetImageFromArray(a)
    b = load(mhdpath)
    b = np.float32(b[0])
    if ww and wc:
        b = _set_ww_wc(b, ww, wc)
    b = b.T
    dic = {}
    spacing = a.GetSpacing()
    origin = a.GetOrigin()
    dic['spacing'] = spacing
    dic['origin'] = origin

    return b, dic


def get_info(mhdpath):
    a = sitk.ReadImage(mhdpath)
    dic = {}
    spacing = a.GetSpacing()
    origin = a.GetOrigin()
    dic['spacing'] = spacing
    dic['origin'] = origin
    return dic