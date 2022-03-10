import torch
import numpy as np
import os
import cv2
from scipy.ndimage import zoom
from torch.utils.data import Dataset
import MHDParse
import random
from image_processing_funs.resample import resample
import augmentations as A
import time
import pylidc as pl
from skimage import morphology

"""
git test111···
"""
class CustomDataset(Dataset):
    def __init__(self, dataset, mode, to_learn, num_of_test, gray_aug=True, noise_aug=True,
                 outslices_3D=None, resample=True, nodule_prob=0, out_xy=None, saved_files='resampled_npy',
                 winwidth=1500, wincenter=-250, nodule_bias=0, debug=False):
        """
        数据集初始化
        :param dataset: 训练用的数据集，可以选择LIDC或LUNA，用字符串选择
        :param mode: 训练模式，0为训练，1为测试，2为验证
        :param to_learn: 学习目标，肺或结节
        :param num_of_test: 在10折交叉验证中选择测试集. Value must be between 0 and 9.
        :param gray_aug: 是否进行灰度增强，默认为是
        :param noise_aug: 是否进行随机噪声，默认为是
        :param outslices_3D: 3D输出时每个patch输出的切片数
        :param resample: 是否进行z方向重采样
        :param nodule_prob: Probability that a nodule is guaranteed in the patch.
        :param out_xy: xy length of the output patch.
        :param nodule_bias: When a nodule is guaranteed in the patch, the maximum possible shift in each direction.
        """
        self.dataset = dataset
        self.mode = mode
        self.to_learn = to_learn
        self.num_of_test = num_of_test
        self.gray_aug = gray_aug
        self.noise_aug = noise_aug
        self.large_memory = False
        self.outslices_3D = outslices_3D
        self.resample = resample
        self.nodule_prob = nodule_prob
        self.out_xy = out_xy
        self.saved_files = saved_files
        self.nodule_bias = nodule_bias

        self.guiyihua = 300
        self.winwidth = tuple(winwidth)
        self.wincenter = tuple(wincenter)
        self.base_dist_between_slices = 1.2
        self.base_xy_dist = 0.6

        self.volume_path = []
        self.volume_cube = []
        self.info_lst = []
        self.label = []
        self.label_cube = []

        self.LIDC_val_begin = num_of_test * 91
        self.LIDC_val_end = num_of_test * 91 + 91
        self.LIDC_test_end = num_of_test * 91 + 182

        if dataset == 'LUNA':
            lst = os.listdir('./LUNA16')
            lst.sort()
            flag = -1
            fp = os.open('./LUNA16/annotations.csv', os.O_RDONLY)
            ann = os.read(fp, 999999999)
            ann = str(ann, 'utf-8')
            ann = ann.split('\n')
            for i in lst:
                if 'subset' in i and 'dict' not in i:
                    flag += 1
                    if ((flag != num_of_test and flag != self._cycle_add(num_of_test, 1, 10)) and mode == 0) or (flag == num_of_test and mode == 1) or (flag == self._cycle_add(num_of_test, 1, 10) and mode == 2):  # 向列表中加入数据的条件
                        lst2 = os.listdir(os.path.join('LUNA16', i))
                        lst2.sort()
                        for j in lst2:
                            if j.split('.')[-1] == 'mhd':
                                self.volume_path.append(os.path.join('LUNA16', i, j))
                                if self.large_memory:
                                    cube, info = MHDParse.read_mhd(os.path.join('LUNA16', i, j), self.winwidth, self.wincenter)
                                    self.volume_cube.append(cube)
                                    self.info_lst.append(info)
                                # info = MHDParse.get_info(os.path.join('LUNA16', i, j))
                                # self.info_lst.append(info)
                                # print(flag)
                                label_lst = []
                                if to_learn == 'nodule':
                                    for k in ann:
                                        if k.split(',')[0] + '.mhd' == j:
                                            label_lst.append(k)
                                    self.label.append(label_lst)
                                elif to_learn == 'lung':
                                    for k in os.listdir(os.path.join('LUNA16', 'seg-lungs-LUNA16')):
                                        if k == j:
                                            self.label.append(os.path.join('LUNA16', 'seg-lungs-LUNA16', k))
                                            if self.large_memory:
                                                self.label_cube.append(MHDParse.read_mhd(os.path.join('LUNA16', 'seg-lungs-LUNA16', k), self.winwidth, self.wincenter))

        elif dataset == 'LIDC':
            scans = pl.query(pl.Scan)
            scan_lst = [a for a in scans]
            for idx, scan in enumerate(scan_lst):
                if ((idx < self.LIDC_val_begin or idx >= self.LIDC_test_end) and mode == 0) or ((self.LIDC_val_begin <= idx < self.LIDC_val_end) and mode == 2) or ((self.LIDC_val_end <= idx < self.LIDC_test_end) and mode == 1):
                    self.volume_path.append(scan)

        if debug:
            if len(self.volume_path):
                self.volume_path = self.volume_path[:debug]
            if len(self.label):
                self.label = self.label[:debug]

    def __getitem__(self, item):
        if self.dataset == 'LUNA':
            if self.to_learn == 'lung':
                if self.saved_files:
                    label_path = self.label[item]
                    label_cube, junk = MHDParse.read_mhd(label_path, None, None)
                    volume = self.volume_path[item]
                    uid = volume.split('/')[-1].rstrip('.mhd')
                    info = MHDParse.get_info(volume)
                    volume = np.load(os.path.join('LUNA16', 'np_files', uid + '.npz'))
                    volume = volume["img"]
                    volume = self._set_ww_wc(volume, self.winwidth, self.wincenter)
                    volume = volume.T
                else:
                    label_path = self.label[item]
                    label_cube, junk = MHDParse.read_mhd(label_path, None, None)
                    volume = self.volume_path[item]
                    volume, info = MHDParse.read_mhd(volume, self.winwidth, self.wincenter)
                label_cube[label_cube == 3] = 1
                label_cube[label_cube == 4] = 2
                label_cube[label_cube == 5] = 3
                if self.mode == 0 and self.gray_aug:
                    volume = A.gray_change_3D(volume)
                if self.mode == 0 and self.noise_aug:
                    volume = A.add_noise_3D(volume)
                volume = volume / self.guiyihua

                if self.resample:
                    volume, label_cube, *junk = self._xy_resample(volume, label_cube, info['spacing'][0],
                                                                  info['origin'])
                    z_base = len(volume)
                    slicedist = info['spacing'][2]
                    improve_factor = slicedist / self.base_dist_between_slices
                    z_resample = z_base * improve_factor
                    z_resample = int(z_resample)
                    volume = resample(volume, z_resample)
                    label_cube = resample(label_cube, z_resample)
                    label_cube = np.round(label_cube)
                if self.outslices_3D:
                    height = volume.shape[0]
                    rnd = random.randint(0, height - self.outslices_3D - 1)
                    return volume[rnd:rnd + self.outslices_3D, :, :], label_cube[rnd:rnd + self.outslices_3D, :, :]
                return volume, label_cube

            elif self.to_learn == 'nodule':
                volume = self.volume_path[item]
                if self.saved_files == 'npz':
                    uid = volume.split('/')[-1].rstrip('.mhd')
                    info = MHDParse.get_info(volume)
                    volume = np.load(os.path.join('LUNA16', 'np_files', uid + '.npz'))
                    volume = volume["img"]
                    volume = self._set_ww_wc(volume, self.winwidth, self.wincenter)
                    volume = volume.T
                elif self.saved_files == 'npy':
                    uid = volume.split('/')[-1].rstrip('.mhd')
                    info = MHDParse.get_info(volume)
                    print("Reading saved npy files...")
                    volume = np.load(os.path.join('LUNA16', 'imgs', 'images_' + uid + '.npy'))
                    volume = self._set_ww_wc(volume, self.winwidth, self.wincenter)
                    volume = volume.T
                elif self.saved_files == 'resampled_npy':
                    # pixel_spacing = 0.6(xy) / 1.2(z)
                    uid = volume_path.series_instance_uid
                    print("Reading saved resampled npy files...")
                    volume = np.load(os.path.join('LUNA16', 'resampled_npys', 'imgs', uid + '.npy'))
                    label = np.load(os.path.join('LUNA16', 'resampled_npys', 'labels', uid + '.npy'))
                    volume = volume * self.guiyihua
                else:
                    volume, info = MHDParse.read_mhd(volume, self.winwidth, self.wincenter)
                label = self.label[item]
                if self.mode == 0 and self.gray_aug:
                    volume = A.gray_change_3D(volume)
                if self.mode == 0 and self.noise_aug:
                    volume = A.add_noise_3D(volume)
                volume = volume / self.guiyihua
                if self.resample:
                    volume, origin_new, final_dist = self._xy_resample(volume, None, info['spacing'][0], info['origin'])
                    z_base = len(volume)
                    slicedist = info['spacing'][2]
                    improve_factor = slicedist / self.base_dist_between_slices
                    z_resample = z_base * improve_factor
                    z_resample = round(z_resample)
                    volume = resample(volume, z_resample)
                    center_cube = np.zeros(volume.shape)
                    dxy_cube = np.zeros(volume.shape)
                    final_slice_dist = slicedist * z_base / z_resample
                    for l in label:
                        l = l.split(',')
                        coord0 = l[1]
                        coord1 = l[2]
                        coord2 = l[3]
                        diam = l[4]
                        coord0 = float(coord0)
                        coord1 = float(coord1)
                        coord2 = float(coord2)
                        diam = float(diam)
                        x, y, z, d_x, d_y, d_z = self._conv_to_array_coord((coord0, coord1, coord2),
                                                                           (final_dist, final_dist,
                                                                            final_slice_dist),
                                                                           origin_new, diam)
                        if x > 256 or y > 256 or z > volume.shape[0]:
                            with open('Warnings.txt', 'a') as F:
                                F.write("Warning: coordinate out of bounds at example {}.\n".format(item))
                            continue
                        else:
                            center_cube[z, y, x] = 1
                            dxy_cube[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2] = d_x
                else:
                    center_cube = np.zeros(volume.shape)
                    dxy_cube = np.zeros(volume.shape)
                    for (uid, *coord, diam) in label:
                        x, y, z, d_x, d_y, d_z = self._conv_to_array_coord(coord, info['spacing'], info['origin'], diam)
                        center_cube[z, x, y] = 1
                        dxy_cube[z - 1:z + 2, x - 1:x + 2, y - 1:y + 2] = d_x

                if self.outslices_3D:
                    height = volume.shape[0]
                    rnd = random.randint(0, height - self.outslices_3D - 1)
                    return volume[rnd:rnd + self.outslices_3D, :, :], center_cube[rnd:rnd + self.outslices_3D, :,
                                                                      :], dxy_cube[rnd:rnd + self.outslices_3D, :, :]

                return volume, center_cube, dxy_cube

        elif self.dataset == 'LIDC':
            if self.to_learn == 'nodule_seg':
                volume_path = self.volume_path[item]
                volume_lst = []
                if self.saved_files == 'npz':
                    t1 = time.time()
                    uid = volume_path.series_instance_uid
                    volume = np.load(os.path.join('LUNA16', 'np_files', uid + '.npz'))
                    print("Reading saved npz files...")
                    label = volume["mask"]
                    volume = volume["img"]
                    for ww, wc in zip(self.winwidth, self.wincenter):
                        volume_lst.append(self._set_ww_wc(volume, ww, wc))
                    anns = volume_path.annotations
                    t2 = time.time()
                    t = t2 - t1
                elif self.saved_files == 'npy':
                    uid = volume_path.series_instance_uid
                    print("Reading saved npy files...")
                    volume = np.load(os.path.join('LUNA16', 'imgs', 'images_' + uid + '.npy'))
                    label = np.load(os.path.join('LUNA16', 'masks', 'masks_' + uid + '.npy'))
                    for ww, wc in zip(self.winwidth, self.wincenter):
                        volume_lst.append(self._set_ww_wc(volume, ww, wc))
                    anns = volume_path.annotations
                elif self.saved_files == 'resampled_npy':
                    uid = volume_path.series_instance_uid
                    print("Reading saved resampled npy files...")
                    volume = np.load(os.path.join('LUNA16', 'resampled_npys', 'imgs', uid + '.npy'))
                    label = np.load(os.path.join('LUNA16', 'resampled_npys', 'labels', uid + '.npy'))
                    volume = volume * self.guiyihua
                    for (ww, wc) in zip(self.winwidth, self.wincenter):
                        v1 = volume.copy()
                        lo = wc - ww / 2
                        hi = wc + ww / 2
                        lo_rel = (lo + 1000) / 1500 * 255
                        hi_rel = (hi + 1000) / 1500 * 255
                        if not (-0.02 <= lo_rel <= 255.02 and -0.02 <= hi_rel <= 255.02):
                            raise ValueError("Lower bound and upper bound must be between -1000 and 500.")
                        v1 = (v1 - lo_rel) / (hi_rel - lo_rel)
                        v1[v1 < 0] = 0
                        v1[v1 > 1] = 1
                        v1 *= 255
                        volume_lst.append(v1)
                    anns = volume_path.annotations
                else:
                    t1 = time.time()
                    volume = volume_path.to_volume()
                    for ww, wc in zip(self.winwidth, self.wincenter):
                        v1 = volume.copy()
                        lo = wc - ww / 2
                        hi = wc + ww / 2
                        v1 = (v1 - lo) / (hi - lo)
                        v1[v1 < 0] = 0
                        v1[v1 > 1] = 1
                        v1 = v1 * 255
                        volume_lst.append(v1)
                    t3 = time.time()
                    label = np.zeros(volume.shape)
                    anns = volume_path.annotations
                    for ann in anns:
                        bbox = ann.bbox()
                        mask = ann.boolean_mask()
                        label[bbox] = mask
                    t2 = time.time()
                    t = t2 - t3

                for volume in volume_lst:
                    volume = volume.T
                label = label.T

                if self.mode == 0 and self.gray_aug:
                    for volume in volume_lst:
                        volume = A.gray_change_3D(volume, uniform=True)
                if self.mode == 0 and self.noise_aug:
                    for volume in volume_lst:
                        volume = A.add_noise_3D(volume)
                volume = volume / self.guiyihua

                slice_thickness = volume_path.slice_thickness
                pixel_dist = volume_path.pixel_spacing

                label = np.float32(label)

                if self.resample and self.saved_files != 'resampled_npy':
                    for v in volume_lst:
                        v, label, junk, final_dist = self._xy_resample(v, label, pixel_dist, (0, 0, 0),
                                                                            label_upper_bound=1)
                        z_base = len(v)
                        improve_factor = slice_thickness / self.base_dist_between_slices
                        z_resample = z_base * improve_factor
                        z_resample = int(z_resample)
                        v = resample(v, z_resample)
                        label_cube = resample(label, z_resample)
                        label = np.round(label_cube)
                elif self.saved_files == 'resampled_npy':
                    pass
                else:
                    for volume in volume_lst:
                        volume_lst1 = []
                        label_lst = []
                        for (v, l) in zip(volume, label):
                            volume_lst1.append(cv2.resize(v, (256, 256)))
                            label_lst.append(cv2.resize(l, (256, 256)))
                        volume = np.array(volume_lst1)
                        label = np.array(label_lst)
                t5 = time.time()

                if self.saved_files == 'resampled_npy':
                    self.base_dist_between_slices = 1.2
                    self.base_xy_dist = 0.6
                    final_dist = 1.2

                volume = np.array(volume_lst)

                if self.outslices_3D:
                    height = volume.shape[0]
                    rnd_nodule = np.random.rand()
                    if rnd_nodule < self.nodule_prob and len(anns):
                        ann = random.choice(anns)
                        bbox = ann.bbox()
                        slice_z = bbox[2]
                        middle_slice = (slice_z.start + slice_z.stop) // 2
                        if self.resample:
                            middle_slice = round(middle_slice * slice_thickness / self.base_dist_between_slices)
                        middle_slice += random.randint(-self.nodule_bias, self.nodule_bias)
                        if middle_slice < self.outslices_3D // 2:
                            middle_slice = self.outslices_3D // 2
                        if middle_slice >= height + self.outslices_3D // 2 - self.outslices_3D:
                            middle_slice = height + self.outslices_3D // 2 - self.outslices_3D - 1
                        if self.out_xy:
                            slice_x = bbox[0]
                            middle_x = (slice_x.start + slice_x.stop) // 2
                            if self.resample:
                                middle_x = round(((middle_x - 256) * pixel_dist + 128 * final_dist) / final_dist)
                            middle_x += random.randint(-self.nodule_bias, self.nodule_bias)
                            if middle_x < self.out_xy // 2:
                                middle_x = self.out_xy // 2
                            if middle_x >= 256 + self.out_xy // 2 - self.out_xy:
                                middle_x = 256 + self.out_xy // 2 - self.out_xy - 1
                            slice_y = bbox[1]
                            middle_y = (slice_y.start + slice_y.stop) // 2
                            if self.resample:
                                middle_y = round(((middle_y - 256) * pixel_dist + 128 * final_dist) / final_dist)
                            middle_y += random.randint(-self.nodule_bias, self.nodule_bias)
                            if middle_y < self.out_xy // 2:
                                middle_y = self.out_xy // 2
                            if middle_y >= 256 + self.out_xy // 2 - self.out_xy:
                                middle_y = 256 + self.out_xy // 2 - self.out_xy - 1
                            t4 = time.time()
                            return volume[:, middle_slice - self.outslices_3D // 2:middle_slice - self.outslices_3D // 2 + self.outslices_3D, middle_y - self.out_xy // 2:middle_y - self.out_xy // 2 + self.out_xy,  middle_x - self.out_xy // 2:middle_x - self.out_xy // 2 + self.out_xy].squeeze(), label[:, middle_slice - self.outslices_3D // 2:middle_slice - self.outslices_3D // 2 + self.outslices_3D, middle_y - self.out_xy // 2:middle_y - self.out_xy // 2 + self.out_xy,  middle_x - self.out_xy // 2:middle_x - self.out_xy // 2 + self.out_xy].squeeze()
                        return volume[:, middle_slice - self.outslices_3D // 2:middle_slice - self.outslices_3D // 2 + self.outslices_3D, :, :].squeeze(), label[middle_slice - self.outslices_3D // 2:middle_slice - self.outslices_3D // 2 + self.outslices_3D, :, :].squeeze()
                    rnd = random.randint(0, height - self.outslices_3D - 1)
                    if self.out_xy:
                      rnd1 = random.randint(0, 256 - self.out_xy - 1)
                      rnd2 = random.randint(0, 256 - self.out_xy - 1)
                      return volume[:, rnd:rnd + self.outslices_3D, rnd1:rnd1 + self.out_xy, rnd2:rnd2 + self.out_xy].squeeze(), label[:, rnd:rnd + self.outslices_3D, rnd1:rnd1 + self.out_xy, rnd2:rnd2 + self.out_xy].squeeze()
                    return volume[:, rnd:rnd + self.outslices_3D, :, :].squeeze(), label[:, rnd:rnd + self.outslices_3D, :, :].squeeze()

                return volume.squeeze(), label

    def _cycle_add(self, a, b, cycle):
        if a + b > cycle:
            return (a + b) % cycle
        else:
            return a + b

    def __len__(self):
        return len(self.volume_path)

    def _xy_resample(self, image, label, dist, origin, label_upper_bound=3):
        """
        在xy方向进行重采样，统一FOV至23cm，之后将图像在xy方向重采样为256*256
        :param image: 输入图像，(z, 512, 512)
        :param label: 输入标签
        :param dist: xy方向间距
        :param origin: 原点对应实际坐标，为(x, y, z)
        :return: 重采样的image和label以及新的原点对应的坐标，以及最终的点之间距离（消除取整造成的误差）
        """
        factor = dist / self.base_xy_dist
        crop_len = round(512 / factor)
        final_dist = crop_len * dist / 256
        if crop_len <= 512:
            crop_start = int((512 - crop_len) // 2)
            origin_new = list(origin)
            origin_new[0] = origin[0] + crop_start * dist
            origin_new[1] = origin[1] + crop_start * dist
            image_cropped = image[:, crop_start:round(crop_start + crop_len), crop_start:round(crop_start + crop_len)]
        else:
            pad_start = int((crop_len - 512) // 2)
            origin_new = list(origin)
            origin_new[0] = origin[0] - pad_start * dist
            origin_new[1] = origin[1] - pad_start * dist
            image_cropped = np.pad(image, ((0, 0), (pad_start, crop_len-512-pad_start), (pad_start, crop_len-512-pad_start)), 'constant', constant_values=0)
        image_resized = []
        for i in image_cropped:
            i1 = cv2.resize(i, (256, 256))
            image_resized.append(i1)
        image_cropped = np.array(image_resized)
        if type(label) is np.ndarray:
            if crop_len <= 512:
                crop_start = int((512 - crop_len) // 2)
                label_cropped = label[:, crop_start:round(crop_start + crop_len), crop_start:round(crop_start + crop_len)]
            else:
                pad_start = int((crop_len - 512) // 2)
                label_cropped = np.pad(label, ((0, 0), 
                (pad_start, crop_len - 512 - pad_start), (pad_start, crop_len - 512 - pad_start)), 'constant',
                                       constant_values=0)
            label_resized = []
            for i in label_cropped:
                i1 = cv2.resize(i, (256, 256))
                label_resized.append(i1)
            label_cropped = np.array(label_resized)
            label_cropped = np.round(label_cropped)
            label_cropped[label_cropped > label_upper_bound] = label_upper_bound
            label_cropped[label_cropped < 0] = 0
            return image_cropped, label_cropped, origin_new, final_dist
        else:
            return image_cropped, origin_new, final_dist

    def _conv_to_array_coord(self, real_coord, spacing, origin, diam):
        x = round((real_coord[0] - origin[0]) / spacing[0])
        y = round((real_coord[1] - origin[1]) / spacing[1])
        z = round((real_coord[2] - origin[2]) / spacing[2])
        d_x = diam / spacing[0]
        d_y = diam / spacing[1]
        d_z = diam / spacing[2]
        return x, y, z, d_x, d_y, d_z

    def _set_ww_wc(self, image, ww, wc):
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


if __name__ == '__main__':
    random.seed(32)
    a = CustomDataset('LIDC', 0, 'nodule_seg', 2, nodule_prob=1, outslices_3D=64, out_xy=64)
    b = torch.utils.data.DataLoader(dataset=a, shuffle=True, batch_size=1)
    total_pixel = 0
    pixel_nodule = 0
    i1 = 0
    for num, (volume, center_cube, dxy_cube) in enumerate(b):
        volume = np.array(volume[0])
        center_cube = np.array(center_cube[0])
        dxy_cube = np.array(dxy_cube[0])
        center_cube = morphology.binary_dilation(center_cube, np.ones((3, 1, 1)))
        idx = 0
        for v, c, d in zip(volume, center_cube, dxy_cube):
            pos = np.where(c == 1)
            v = v * 300
            v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
            idx += 1
            for i in range(0, len(pos[0])):
                center_pos = (pos[0][i], pos[1][i])
                diameter = d[pos[0][i]][pos[1][i]]
                p1_pos = (int(pos[1][i] - diameter // 2), int(pos[0][i] - diameter // 2))
                p2_pos = (p1_pos[0] + round(diameter), p1_pos[1] + round(diameter))
                v = cv2.rectangle(v, p1_pos, p2_pos, (0, 0, 255), 2)
                print("Nodule at volume {} slice {}.".format(i1, idx))
            cv2.imwrite(os.path.join('test1', str(i1) + '_' + str(idx) + '.png'), v)
            cv2.imwrite(os.path.join('test1', str(i1) + '_' + str(idx) + '_label.png'), c * 255)
            d[d > 0] = 1
            cv2.imwrite(os.path.join('test1', str(i1) + '_' + str(idx) + '_diameter.png'), d * 255)
        i1 += 1
        if i1 >= 8:
            break
        # num1 = 0
        # if num >= 1:
        #     break
        # for (p, l) in zip(volume, label):
        #     num1 += 1
        #     cv2.imwrite(os.path.join('test', str(num1) + '.png'), p * 300)
        #     cv2.imwrite(os.path.join('test', str(num1) + '_label.png'), l * 255)
