from __future__ import print_function
import cv2
from scipy.ndimage import imread
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
import collections
from itertools import repeat
import scipy.io as scio
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter
import random


def save_density_map(density_map, output_dir, fname='results.png', count=None):

    density_map = 255.0 * (density_map - np.min(density_map) + 1e-10) / (1e-10 + np.max(density_map) - np.min(density_map))
    density_map = density_map.squeeze()
    color_map = cv2.applyColorMap(density_map[:, :, np.newaxis].astype(np.uint8).repeat(3, axis=2), cv2.COLORMAP_JET)
    if count is not None:
        h, w = color_map.shape[:2]
        cv2.putText(color_map,str(count), (0,h-3), cv2.FONT_HERSHEY_PLAIN,
                        2.0, (255, 255, 255), 1)
    cv2.imwrite(os.path.join(output_dir, fname), color_map)

def save_density_map_resize(density_map, output_dir, fname='results.png', count=None):
    wd, ht = density_map.shape[1], density_map.shape[0]
    wd_old, ht_old = wd, ht
    max_size = 1280
    downsize = 32
    resize = False

    if (wd > max_size or ht > max_size):
        nwd = int(wd * 1.0 / max(wd, ht) * max_size)
        nht = int(ht * 1.0 / max(wd, ht) * max_size)
        resize = True
        wd = nwd
        ht = nht

    nht = (ht / downsize) * downsize
    nwd = (wd / downsize) * downsize
    if nht != ht or nwd != wd or resize:
        resize = True
        count = density_map.sum()
        density_map = cv2.resize(density_map, (nwd, nht))
        if density_map.sum() != 0:
            density_map = density_map * count / density_map.sum()

    density_map = 255.0 * (density_map - np.min(density_map) + 1e-10) / (1e-10 + np.max(density_map) - np.min(density_map))
    density_map = density_map.squeeze()
    color_map = cv2.applyColorMap(density_map[:, :, np.newaxis].astype(np.uint8).repeat(3, axis=2), cv2.COLORMAP_JET)
    if count is not None:
        h, w = color_map.shape[:2]
        cv2.putText(color_map,'Pred:' + str(int(count)), (0,h-3), cv2.FONT_HERSHEY_PLAIN,
                        2.0, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(output_dir, fname), color_map)

def save_heatmep_pred(heatmap, output_dir, fname='results.png', count=None):

    heatmap = heatmap.astype(np.float32).squeeze()
    heatmap = gaussian_filter(heatmap,sigma=5)
    heatmap = 255.0 * heatmap/heatmap.max()
    color_map = cv2.applyColorMap(heatmap[:, :, np.newaxis].astype(np.uint8).repeat(3, axis=2), cv2.COLORMAP_JET)
    if count is not None:
        h, w = color_map.shape[:2]
        cv2.putText(color_map,str(count), (0,h-3), cv2.FONT_HERSHEY_PLAIN,
                        2.0, (255, 255, 255), 1)
    cv2.imwrite(os.path.join(output_dir, fname), color_map)


def save_image(data, output_dir, fname='results.png'):
    data = data.squeeze()
    if len(data.shape) == 1:
        data = data[:, :, np.newaxis].astype(np.uint8).repeat(3, axis=2)
    else:
        data = data[:,:,::-1].astype(np.uint8)

    cv2.imwrite(os.path.join(output_dir, fname), data)

def save_image_with_point(data, mask, output_dir, fname='results.png', GT=False):
    data = data.squeeze()
    if len(data.shape) == 1:
        data = data[:, :, np.newaxis].astype(np.uint8).repeat(3, axis=2)
    else:
        data = data[:,:,::-1].astype(np.uint8)

    wd, ht = data.shape[1], data.shape[0]
    wd_old, ht_old = wd, ht

    max_size = 1280
    downsize = 32
    resize=False

    if (wd > max_size or ht > max_size):
        nwd = int(wd * 1.0 / max(wd, ht) * max_size)
        nht = int(ht * 1.0 / max(wd, ht) * max_size)
        resize = True
        wd = nwd
        ht = nht

    nht = (ht / downsize) * downsize
    nwd = (wd / downsize) * downsize
    if nht != ht or nwd != wd or resize:
        resize = True
        data = cv2.resize(data, (nwd, nht))

    mask = mask.astype(np.float32).squeeze()
    ids = np.array(np.where(mask == 1))  # y,x
    ori_ids_y = ids[0, :]
    ori_ids_x = ids[1, :]
    ids = np.stack((ori_ids_x, ori_ids_y),axis=1).astype(np.int16)  # x,y

    if resize:
        w_ratio = (float(nwd) / float(wd_old))
        h_ratio = (float(nht) / float(ht_old))
        ids[:, 0] = ids[:, 0] * w_ratio
        ids[:, 1] = ids[:, 1] * h_ratio

    count = ids.shape[0]
    if GT:
        title = 'GT:'
        color = (0, 255, 0)
        data = cv2.putText(cv2.UMat(data), title + str(count), (0, nht - 3), cv2.FONT_HERSHEY_PLAIN,
                           2.0, (255, 255, 255), 2)
    else:
        title = 'Pred:'
        color = (0, 255, 255)

    radius = 4
    for i in range(ids.shape[0]):
        data = cv2.circle(cv2.UMat(data), (ids[i][0], ids[i][1]), radius, color, 1)

    cv2.imwrite(os.path.join(output_dir, fname), data)


def save_density_raw(density_map, output_dir, fname='results.mat'):
    scio.savemat(os.path.join(output_dir, fname), {'data': density_map})


def get_gauss(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


class Gauss2D(object):
    """docstring for DensityMap"""

    def __init__(self):
        super(Gauss2D, self).__init__()
        self.kernel_set = {}

    def get(self, shape=(3, 3), sigma=0.5):
        if '%d_%d' % (int(shape[0]), int(sigma * 10)) not in self.kernel_set.keys():
            m, n = [(ss - 1.0) / 2.0 for ss in shape]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            # import pdb
            # pdb.set_trace()
            t = h[0][int(m)]
            h[h < t] = 0
            sumh = h.sum()
            if sumh != 0:
                h /= sumh
            self.kernel_set['%d_%d' % (int(shape[0]), int(sigma * 10))] = h
            return h
        else:
            return self.kernel_set['%d_%d' % (int(shape[0]), int(sigma * 10))]

class Gauss2D_HM(object):
    """docstring for DensityMap"""

    def __init__(self):
        super(Gauss2D_HM, self).__init__()
        self.kernel_set = {}

    def get(self, shape=(3, 3), sigma=0.5):
        if '%d_%d' % (int(shape[0]), int(sigma * 10)) not in self.kernel_set.keys():
            m, n = [(ss - 1.0) / 2.0 for ss in shape]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            # import pdb
            # pdb.set_trace()
            t = h[0][int(m)]
            h[h < t] = 0
            self.kernel_set['%d_%d' % (int(shape[0]), int(sigma * 10))] = h
            return h
        else:
            return self.kernel_set['%d_%d' % (int(shape[0]), int(sigma * 10))]

def find_kneighbors(locations, K=6, threhold=0):
    nbt = NearestNeighbors(n_neighbors=K, algorithm="ball_tree").fit(locations)
    distances, indices = nbt.kneighbors(locations)
    return indices


def load_annPoints(fname, annReadFunc):
    data = scio.loadmat(fname)
    annPoints = annReadFunc(data)
    return annPoints


def check_xy(x, y, H, W):
    if x > W + 10 or x < -10 or y > H + 10 or y < -10:
        return False, None, None
    else:
        x = x if x < W else W - 1
        x = x if x > 0 else 0
        y = y if y < H else H - 1
        y = y if y > 0 else 0
        return True, int(x), int(y)


def add_filter(den, filter, x, y, f_sz, c=1.0):
    H, W = den.shape
    h_fsz = f_sz // 2
    x1, x2, y1, y2 = x - h_fsz, x + h_fsz + 1, y - h_fsz, y + h_fsz + 1
    fsum, dfx1, dfx2, dfy1, dfy2 = filter.sum(), 0, 0, 0, 0
    if x1 < 0:
        dfx1 = abs(x1)
        x1 = 0
    if x2 >= W:
        dfx2 = x2 - W + 1
        x2 = W
    if y1 < 0:
        dfy1 = abs(y1)
        y1 = 0
    if y2 >= H:
        dfy2 = y2 - H + 1
        y2 = H
    x1h, x2h, y1h, y2h = dfx1, f_sz - dfx2 + 1, dfy1, f_sz - dfy2 + 1
    den[y1:y2, x1:x2] = den[y1:y2, x1:x2] \
        + c * fsum / filter[y1h:y2h, x1h:x2h].sum() * filter[y1h:y2h, x1h:x2h]
    return den


def add_filter_HM(heatmap, filter, x, y, f_sz, k=1):
    radius = f_sz // 2
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = filter[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def get_density_map_fix(H, W, annPoints, get_gauss, sigma, f_sz):
    den = np.zeros((H, W))
    gt_count = 0
    for i, p in enumerate(annPoints):
        x, y = p
        g, x, y = check_xy(x, y, H, W)
        if g is False:
            # print("point {} out of img {}x{} too much\n".format(p, H, W))
            continue
        else:
            gt_count += 1
        f_sz = int(f_sz) // 2 * 2 + 1
        filter = get_gauss((f_sz, f_sz), sigma)
        den = add_filter(den, filter, x, y, f_sz)
    return den, gt_count


def get_annoted_kneighbors(label_file, label_path, annReadFunc, K):
    annPoints = load_annPoints(os.path.join(label_path, label_file), annReadFunc)
    if len(annPoints) > K:
        kneighbors = find_kneighbors(annPoints, K)
    else:
        kneighbors = None
    return kneighbors


def get_density_map_adaptive(H, W, annPoints, kneighbors, K, get_gauss):
    den = np.zeros((H,W))

    limit = min(min(H,W) / 8.0, 100.0)
    use_limit = False

    gt_count = 0
    for i, p in enumerate(annPoints):
        x, y = p
        g, x, y = check_xy(x, y, H, W)
        if g is False:
            # print("point {} out of img {}x{} too much\n".format(p, H, W))
            continue
        else:
            gt_count += 1
        if len(annPoints) > K:
            dis = ((annPoints[kneighbors[i][1:]][:,0] - annPoints[i][0])**2
                    + (annPoints[kneighbors[i][1:]][:,1] - annPoints[i][1])**2)**0.5

            dis = dis.mean()
        else:
            dis = limit

        # sigma = max(0.3 * dis, 0.1)
        # f_sz = int(6.0 * sigma) // 2 * 2 + 1
        f_sz = max(int(0.3 * dis), 2) // 2 * 2 + 1
        sigma = float(f_sz) / 3.0
        filter = get_gauss((f_sz, f_sz), sigma)
        den = add_filter(den, filter, x, y, f_sz)
    return den, gt_count

def get_heat_map(H, W, annPoints, kneighbors, K, get_gauss):
    den = np.zeros((H,W))

    limit = min(min(H,W) / 8.0, 100.0)
    use_limit = False

    gt_count = 0
    for i, p in enumerate(annPoints):
        x, y = p
        # if random.random() < 0.5:
        #     x = x + random.uniform(-0.01*dis, 0.01*dis)
        #     y = y + random.uniform(-0.01*dis, 0.01*dis)
        g, x, y = check_xy(x, y, H, W)
        if g is False:
            # print("point {} out of img {}x{} too much\n".format(p, H, W))
            continue
        else:
            gt_count += 1

        if len(annPoints) > K:
            dis = ((annPoints[kneighbors[i][1:]][:,0] - annPoints[i][0])**2
                    + (annPoints[kneighbors[i][1:]][:,1] - annPoints[i][1])**2)**0.5

            dis = dis.mean()
        else:
            dis = limit

        # sigma = 0.3 * dis
        # f_sz = int(6.0 * sigma) // 2 * 2 + 1
        f_sz = max(int(0.3 * dis),2) // 2 * 2 + 1
        sigma = float(f_sz)/3.0
        filter = get_gauss((f_sz, f_sz), sigma)
        den = add_filter_HM(den, filter, x, y, f_sz)
    return den, gt_count




def get_density_map_3d(H, W, annPoints, K, S, get_gauss):
    D = len(S)
    ov = 0.5
    S = [9, 25, 49, 81]
    S = np.asarray(S)

    den = np.zeros((D, H, W))

    if len(annPoints) > K:
        kneighbors = find_kneighbors(annPoints, K)

    gt_count = 0
    for i, p in enumerate(annPoints):
        x, y = p
        g, x, y = check_xy(x, y, H, W)
        if g is False:
            # print("point {} out of img {}x{} too much\n".format(p, H, W))
            continue
        else:
            gt_count += 1
        if len(annPoints) > K:
            dis = ((annPoints[kneighbors[i][1:]][:, 0] - annPoints[i][0])**2
                   + (annPoints[kneighbors[i][1:]][:, 1] - annPoints[i][1])**2)**0.5
            dis = dis.mean()
        else:
            dis = min(min(H, W) / 8.0, 100.0)
        DN = np.where(S > dis)[0]
        dn = DN[0] if len(DN) > 0 else D - 1
        vn = np.exp(-((np.arange(D) - dn)**2) / (2 * ov))
        vn = vn / sum(vn)
        for i in range(D):
            hh = vn[i]
            f_sz = S[i]
            sigma = 0.3 * f_sz
            f_sz = int(5.0 * sigma) // 2 * 2 + 1
            filter = get_gauss((f_sz, f_sz), sigma)
            den[i, ...] = add_filter(den[i, ...], filter, x, y, f_sz, hh)

    return den, gt_count


def read_image_label_fix(image_file, label_file, image_path, label_path, \
                            get_gauss, sigma, f_sz, channels, downsize, annReadFunc, test=False):
        
    img = Image.open(os.path.join(image_path, image_file)).convert('RGB')
    wd, ht = img.size
    den = None
    resize = False
    annPoints = load_annPoints(os.path.join(label_path, label_file), annReadFunc)

    if not test:
        den, gt_count = get_density_map_fix(ht, wd, annPoints, get_gauss, sigma, f_sz)

    if not test and (wd < 320 or ht < 320):
        nwd = int(wd * 1.0/ min(wd, ht) * 320)
        nht = int(ht * 1.0/ min(wd, ht) * 320)
        resize = True
        img = img.resize((nwd, nht), resample=Image.BICUBIC)
        print("{} X {} -> {} X {}".format(ht, wd, nht, nwd))
        wd = nwd
        ht = nht


    nht = (ht / downsize) * downsize
    nwd = (wd / downsize) * downsize
    if nht != ht or nwd != wd:
        img = img.resize((nwd, nht), resample=Image.BICUBIC)
        resize = True

    if not test:
        if resize:
            count = den.sum()
            den = cv2.resize(den, (nwd, nht))
            if den.sum() != 0:
                den = den * count / den.sum()

    return img, den, len(annPoints)
    

def read_image_label_apdaptive(image_file, label_file, image_path, label_path, \
                                    get_gauss, kneighbors, channels, downsize, K, annReadFunc, get_gauss2=None, test=False):
    img = Image.open(os.path.join(image_path, image_file)).convert('RGB')
    if not test and 'NWPU'.lower() in image_path.lower():
        if random.random() < 0.01:
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.stack((img,) * 3, axis=-1)
            img = Image.fromarray(img)
    wd, ht = img.size
    wd_old, ht_old = wd, ht
    den = None
    resize = False

    annPoints = load_annPoints(os.path.join(label_path, label_file), annReadFunc)

    max_size = 1600 #1600 #2048
    if test or 'test' in image_path:
        max_size = 2048 # 2560
    min_size = 320 #320

    if (wd > max_size or ht > max_size):
        nwd = int(wd * 1.0/ max(wd, ht) * max_size)
        nht = int(ht * 1.0/ max(wd, ht) * max_size)
        resize = True
        wd = nwd
        ht = nht

    if not test and (wd < min_size or ht < min_size):
        nwd = int(wd * 1.0/ min(wd, ht) * min_size)
        nht = int(ht * 1.0/ min(wd, ht) * min_size)
        resize = True
        # img = img.resize((nwd, nht), resample=Image.BICUBIC)
        # print "{} X {} -> {} X {}".format(ht, wd, nht, nwd)
        wd = nwd
        ht = nht
    # if not test:
    #     if random.random() < 0.1:
    #         ratio = random.uniform(0.8, 1.2)
    #         wd = int(wd*ratio)
    #         ht = int(ht*ratio)
    #         resize = True
    #     if (wd < 320 or ht < 320):
    #         nwd = int(wd * 1.0/ min(wd, ht) * 320)
    #         nht = int(ht * 1.0/ min(wd, ht) * 320)
    #         resize = True
    #         wd = nwd
    #         ht = nht
    nht = (ht / downsize) * downsize
    nwd = (wd / downsize) * downsize
    if nht != ht or nwd != wd or resize:
        img = img.resize((nwd, nht), resample=Image.BICUBIC)
        resize = True
    if True: #not test:
        if resize:
            w_ratio = (float(nwd) / float(wd_old))
            h_ratio = (float(nht) / float(ht_old))
            if annPoints.shape[0] != 0:
                annPoints[:, 0] = annPoints[:, 0] * w_ratio
                annPoints[:, 1] = annPoints[:, 1] * h_ratio

    if not test:
        den, gt_count = get_density_map_adaptive(ht, wd, annPoints, kneighbors, K, get_gauss)

    return  img, den, len(annPoints)

def read_image_label_hm(image_file, label_file, image_path, label_path, \
                               get_gauss, kneighbors, channels, downsize, K, annReadFunc, get_gauss2=None, test=False):
    img = Image.open(os.path.join(image_path, image_file)).convert('RGB')
    if not test and 'NWPU'.lower() in image_path.lower():
        if random.random() < 0.01:
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.stack((img,) * 3, axis=-1)
            img = Image.fromarray(img)
    wd, ht = img.size
    wd_old, ht_old = wd, ht
    den = None
    resize = False

    annPoints = load_annPoints(os.path.join(label_path, label_file), annReadFunc) # x, y

    max_size = 1600 #1600 #2048
    if test or 'test' in image_path:
        max_size = 1600 #2048 # 2560
    min_size = 320 #320

    if (wd > max_size or ht > max_size):
        nwd = int(wd * 1.0/ max(wd, ht) * max_size)
        nht = int(ht * 1.0/ max(wd, ht) * max_size)
        resize = True
        wd = nwd
        ht = nht

    if not test and (wd < min_size or ht < min_size):
        nwd = int(wd * 1.0/ min(wd, ht) * min_size)
        nht = int(ht * 1.0/ min(wd, ht) * min_size)
        resize = True
        # img = img.resize((nwd, nht), resample=Image.BICUBIC)
        # print "{} X {} -> {} X {}".format(ht, wd, nht, nwd)
        wd = nwd
        ht = nht
    # if not test:
    #     if random.random() < 0.1:
    #         ratio = random.uniform(0.8, 1.2)
    #         wd = int(wd*ratio)
    #         ht = int(ht*ratio)
    #         resize = True
    #     if (wd < 320 or ht < 320):
    #         nwd = int(wd * 1.0/ min(wd, ht) * 320)
    #         nht = int(ht * 1.0/ min(wd, ht) * 320)
    #         resize = True
    #         wd = nwd
    #         ht = nht
    nht = (ht / downsize) * downsize
    nwd = (wd / downsize) * downsize
    if nht != ht or nwd != wd or resize:
        img = img.resize((nwd, nht), resample=Image.BICUBIC)
        resize = True
    if True: #not test:
        if resize:
            w_ratio = (float(nwd) / float(wd_old))
            h_ratio = (float(nht) / float(ht_old))
            if annPoints.shape[0] != 0:
                annPoints[:, 0] = annPoints[:, 0] * w_ratio
                annPoints[:, 1] = annPoints[:, 1] * h_ratio

    if not test:
        hm, gt_count = get_heat_map(nht, nwd, annPoints, kneighbors, K, get_gauss)
        if get_gauss2 is not None:
            # dm, gt_count = get_density_map_adaptive(nht, nwd, annPoints, kneighbors, K, get_gauss2)
            dm, gt_count = get_density_map_fix(nht, nwd, annPoints, get_gauss2, 9.0/3, 9)
            den = np.stack([hm, dm],axis=0)
        else:
            den = hm.copy()

    return img, den, len(annPoints)

def read_image_label_3d(image_file, label_file, image_path, label_path, get_gauss, K, S, channels, downsize, annReadFunc):
    img = imread(os.path.join(image_path, image_file), 1)
    img = img.astype(np.float32, copy=False)
    ht = img.shape[0]
    wd = img.shape[1]
    annPoints = load_annPoints(os.path.join(label_path, label_file), annReadFunc)
    den, gt_count = get_density_map_3d(ht, wd, annPoints, K, S, get_gauss)
    denstiy_channels = len(S)

    ht_1 = (ht / downsize) * downsize
    wd_1 = (wd / downsize) * downsize
    img = cv2.resize(img, (wd_1, ht_1))
    img = img.reshape((1, 1, img.shape[0], img.shape[1]))
    if channels != 1:
        img = np.repeat(img, channels, axis=1)

    den_resize = []
    for i in range(denstiy_channels):
        den_ = cv2.resize(den[i], (wd_1, ht_1))
        den_ = den_ * ((wd * ht * 1.0) / (wd_1 * ht_1))
        den_resize.append(den_[np.newaxis, ...])
    den = np.vstack(den_resize)
    den = den.reshape((1, denstiy_channels, den.shape[1], den.shape[2]))
    # gt_count = np.sum(den)

    return img, den, gt_count


def read_image(image_file, image_path, channels, downsize, test=False):
    img = Image.open(os.path.join(image_path, image_file)).convert('RGB')
    wd, ht = img.size
    resize = False

    max_size = 2048  # 2048 # 2560
    min_size = 320  # 320

    if (wd > max_size or ht > max_size):
        nwd = int(wd * 1.0 / max(wd, ht) * max_size)
        nht = int(ht * 1.0 / max(wd, ht) * max_size)
        resize = True
        wd = nwd
        ht = nht

    if wd < min_size or ht < min_size:
        nwd = int(wd * 1.0 / min(wd, ht) * min_size)
        nht = int(ht * 1.0 / min(wd, ht) * min_size)
        resize = True
        # img = img.resize((nwd, nht), resample=Image.BICUBIC)
        # print "{} X {} -> {} X {}".format(ht, wd, nht, nwd)
        wd = nwd
        ht = nht

    nht = (ht / downsize) * downsize
    nwd = (wd / downsize) * downsize
    if nht != ht or nwd != wd or resize:
        img = img.resize((nwd, nht), resample=Image.BICUBIC)

    den = np.zeros((nht, nwd))

    return img, den, 0

# def read_image(image_file, image_path, channels, downsize):
#     # print image_file
#     img = imread(os.path.join(image_path, image_file), 1)
#     img = img.astype(np.float32, copy=False)
#     ht = img.shape[0]
#     wd = img.shape[1]
#
#
#     ht_1 = (ht / downsize) * downsize
#     wd_1 = (wd / downsize) * downsize
#     img = cv2.resize(img, (wd_1, ht_1))
#     img = img.reshape((1, 1, img.shape[0], img.shape[1]))
#     if channels != 1:
#         img = np.repeat(img, channels, axis=1)
#     return img


def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2


def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])


def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha


def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3,))
    image += np.dot(eigvec, eigval * alpha)


def color_aug(image):
    data_rng = np.random.RandomState(123)
    eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                       dtype=np.float32)
    eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)

    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)