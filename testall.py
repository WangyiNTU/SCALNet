from __future__ import print_function
import os
import cv2
# import debug
import torch
import numpy as np
import torch.nn.functional as F
from src.crowd_counting import CrowdCounter
from src import network
from src.RawLoader import ImageDataLoader, basic_config
from src import utils
import argparse
from src.sampler import basic_config as sampler_config
from src.sampler import mode_func as sampler_func
import torchvision.transforms as transforms
from src.datasets import datasets, CreateDataLoader
import src.density_gen as dgen
from src.timer import Timer
import itertools
import time
from PIL import Image
from src.utils import AverageMeter

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

# test data and model file path
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--model_name', type=str)
parser.add_argument('--gpus', type=str, help='gpu_id')
parser.add_argument('--dataset', type=str)
parser.add_argument('--prefix', type=str)
parser.add_argument('--det_thr', type=float, default=0.4)
parser.add_argument('--search_thr', dest='is_search_thr', action='store_true')
parser.add_argument('--no_search_thr', dest='is_search_thr', action='store_false')
parser.set_defaults(is_search_thr=False)
parser.add_argument('--search_start', type=int, default=30)

parser.add_argument('--preload', dest='is_preload', action='store_true')
parser.add_argument('--no-preload', dest='is_preload', action='store_false')
parser.set_defaults(is_preload=True)

parser.add_argument('--wait', dest='is_wait', action='store_true')
parser.add_argument('--no-wait', dest='is_wait', action='store_false')
parser.set_defaults(is_wait=True)

parser.add_argument('--save', dest='save_output', action='store_true', help='save image, and input image is resized')
parser.add_argument('--no-save', dest='save_output', action='store_false')
parser.set_defaults(save_output=False)
parser.add_argument('--test_patch', action='store_true')
parser.add_argument('--save_txt', action='store_true')

# crop adap
parser.add_argument('--test_fixed_size', type=int, default=-1)
parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--epoch', type=int)

parser.add_argument('--name', type=str)
parser.add_argument('--split', type=str)
parser.add_argument('--num_workers', type=int, default=8)

def test_patch(data):
    with torch.no_grad():
        crop_imgs, crop_masks = [], []
        b, c, h, w = data.shape
        rh, rw = 320, 320#576, 768
        for i in range(0, h, rh):
            gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
            for j in range(0, w, rw):
                gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                crop_imgs.append(data[:, :, gis:gie, gjs:gje])
                mask = torch.zeros(b, 1, h, w).cuda()
                mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                crop_masks.append(mask)
        crop_imgs, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks))

        # forward may need repeatng
        crop_preds = []
        crop_preds_dm = []
        nz, bz = crop_imgs.size(0), 36
        for i in range(0, nz, bz):
            gs, gt = i, min(nz, i + bz)
            crop_pred, crop_pred_dm = net(crop_imgs[gs:gt])
            crop_pred = crop_pred.sigmoid_()

            # crop_pred_nms = network._nms(crop_pred.detach())

            # crop_pred_nms = crop_pred_nms[None, :, :]
            # crop_pred = F.softmax(crop_pred,dim=1).data.max(1)
            # crop_pred = crop_pred[1].squeeze_(1)

            crop_preds.append(crop_pred)
            crop_preds_dm.append(crop_pred_dm)
        crop_preds = torch.cat(crop_preds, dim=0)
        crop_preds_dm = torch.cat(crop_preds_dm, dim=0)
        size_1 = crop_preds_dm.shape[0]
        sum_crop_preds_dm = torch.sum(crop_preds_dm.reshape(size_1,-1), dim=1).reshape(size_1,1,1)
        crop_preds_dm = F.interpolate(crop_preds_dm, scale_factor=(2,2))
        sum_crop_preds_dm_2 = torch.sum(crop_preds_dm.reshape(size_1,-1), dim=1).reshape(size_1,1,1)
        crop_preds_dm = crop_preds_dm * sum_crop_preds_dm/sum_crop_preds_dm_2

        # crop_preds_dm = F.interpolate(crop_preds_dm, scale_factor=(2,2)) * 0.25
        # splice them to the original size
        idx = 0
        pred_map = torch.zeros(b, 1, h, w).cuda()
        pred_map_dm = torch.zeros(b, 1, h, w).cuda()
        for i in range(0, h, rh):
            gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
            for j in range(0, w, rw):
                gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                pred_map_dm[:, :, gis:gie, gjs:gje] += crop_preds_dm[idx]
                idx += 1

        # for the overlapping area, compute average value
        mask = crop_masks.sum(dim=0).unsqueeze(0)
        pred_map = pred_map / mask
        pred_map_dm = pred_map_dm / mask
        pred_map_nms = network._nms(pred_map.detach())
    return pred_map, pred_map_nms, pred_map_dm


def test_model_origin_search(net, data_loader, save_output=False, save_path=None, test_fixed_size=-1, test_batch_size=1,
                      gpus=None, args=None):
    timer = Timer()
    timer.tic()
    net.eval()
    maes = [0]*(args.search_start+1)
    mses = [0]*(args.search_start+1)
    mae_dm = 0.0
    mse_dm = 0.0
    detail = ''
    if save_output:
        print(save_path)
    for i, blob in enumerate(data_loader.get_loader(test_batch_size, num_workers=args.num_workers)):
        # if (i * len(gpus) + 1) % 100 == 0:
        #     print("testing %d" % (i + 1))
        if save_output:
            index, fname, data, mask, gt_dens, gt_count = blob
        else:
            index, fname, data, mask, gt_count = blob

        if not args.test_patch:
            with torch.no_grad():
                dens, dm = net(data)
                dens = dens.sigmoid_()
                dens_nms = network._nms(dens.detach())
                dens_nms = dens_nms.data.cpu().numpy()
                dm = dm.data.cpu().numpy()
                ### do not support save image ###
                # if save_output:
                #     image = data.squeeze_().mul_(torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)) \
                #         .add_(torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)).data.cpu().numpy()
                #
                #     dgen.save_image(image.transpose((1, 2, 0)) * 255.0, save_path, fname[0].split('.')[0] + "_0_img.jpg")
                #     gt_dens = gt_dens.data.cpu().numpy()
                #     density_map = dens.data.cpu().numpy()
                #     dgen.save_density_map(gt_dens.squeeze(), save_path, fname[0].split('.')[0] + "_1_gt.jpg")
                #     dgen.save_density_map(density_map.squeeze(), save_path, fname[0].split('.')[0] + "_2_et.jpg")
                #     dens_mask = dens_nms >= args.det_thr
                #     dgen.save_heatmep_pred(dens_mask.squeeze(), save_path, fname[0].split('.')[0] + "_3_et.jpg")
                #     _gt_count = gt_dens.sum().item()
                #     del gt_dens
        else: # TODO
            dens, dens_nms, dm = test_patch(data)
            dens_nms = dens_nms.data.cpu().numpy()
            dm = dm.data.cpu().numpy()

        dm[dm < 0] = 0.0
        gt_count = gt_count.item()
        et_count_dm = np.sum(dm.reshape(test_batch_size, -1), axis=-1)[0]
        et_counts = []
        for j in range(args.search_start,61):
            det_thr = j/100.0
            et_counts.append(np.sum(dens_nms.reshape(test_batch_size, -1) >= det_thr, axis=-1)[0])

        del data

        for j in range(len(et_counts)):
            maes[j] += abs(gt_count - et_counts[j])
            mses[j] += ((gt_count - et_counts[j]) * (gt_count - et_counts[j]))
        mae_dm += abs(gt_count - et_count_dm)
        mse_dm += ((gt_count - et_count_dm) * (gt_count - et_count_dm))
        et_counts_dict = {(k+args.search_start)/100.0:v for k, v in enumerate(et_counts)}
        detail += "index: {}; fname: {}; gt: {}; et: {};\n".format(i, fname[0].split('.')[0], gt_count, et_counts_dict)

    maes = [float(mae) / len(data_loader) for mae in maes]
    mses = [np.sqrt(float(mse) / len(data_loader)) for mse in mses]
    mae_dm = mae_dm / len(data_loader)
    mse_dm = np.sqrt(mse_dm / len(data_loader))
    duration = timer.toc(average=False)
    print("testing time: %d" % duration)
    return maes, mses, mae_dm, mse_dm, detail

def test_model_origin(net, data_loader, save_output=False, save_path=None, test_fixed_size=-1, test_batch_size=1,
                      gpus=None, args=None):
    timer = Timer()
    timer.tic()
    net.eval()
    mae = 0.0
    mse = 0.0
    NAE = 0.0
    NAE_count = 0.0
    mae_dm = 0.0
    mse_dm = 0.0
    NAE_dm = 0.0
    NAE_count_dm = 0.0
    detail = ''
    save_txt_num = 5
    if args.save_txt:
        thr = []
        record = []
        save_txt_path = save_path.replace('density_maps','loc_txt')
        if not os.path.exists(save_txt_path):
            os.mkdir(save_txt_path)
        for j in range(save_txt_num):
            thr.append(args.det_thr + float(j-(save_txt_num//2))*0.01)  # if save_txt_num=5, then [-0.02, 0.02]
            record.append(open(save_txt_path+'/DLA_loc_val_thr_{:.02f}.txt'.format(thr[j]), 'w+'))
    time_per_item = Timer()
    if save_output:
        print(save_path)
    for i, blob in enumerate(data_loader.get_loader(test_batch_size, num_workers=args.num_workers)):
        if (i * len(gpus) + 1) % 100 == 0:
            print("testing %d" % (i + 1))
        if save_output:
            index, fname, data, mask, gt_hm_dens, gt_count = blob
        else:
            index, fname, data, mask, gt_count = blob

        if not args.test_patch:
            with torch.no_grad():
                image_validate = False
                if data.shape[-1] == 1600 or data.shape[-2] == 1600 and i>50:
                    image_validate = True
                    time_per_item.tic()
                dens, dm = net(data)
                if image_validate:
                    time_per_item.toc()
                dens = dens.sigmoid_()
                dens_nms = network._nms(dens.detach())
                dens_nms = dens_nms.data.cpu().numpy()
                dm = dm.data.cpu().numpy()
        else: #TODO
            dens, dens_nms, dm = test_patch(data)
            dens_nms = dens_nms.data.cpu().numpy()
            dm = dm.data.cpu().numpy()

        dm[dm < 0] = 0.0
        gt_count = gt_count.item()
        # et_count = dens.sum().item()
        et_count = np.sum(dens_nms.reshape(test_batch_size, -1) >= args.det_thr, axis=-1)[0]
        et_count_dm = np.sum(dm.reshape(test_batch_size, -1), axis=-1)[0]

        if save_output:
            image = data.clone().squeeze_().mul_(torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)) \
                .add_(torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)).data.cpu().numpy()

            gt_dens = gt_hm_dens[:,0:1,:,:].clone().data.cpu().numpy()
            gt_dm = gt_hm_dens[:,1:2,:,:].clone().data.cpu().numpy()
            hm = dens.data.cpu().numpy()
            dgen.save_density_map(gt_dens.squeeze(), save_path, fname[0].split('.')[0] + "_1_gt_hm.jpg", gt_count)
            dgen.save_density_map(gt_dm.squeeze(), save_path, fname[0].split('.')[0] + "_1_gt_dm.jpg", gt_count)
            dgen.save_density_map(hm.squeeze(), save_path, fname[0].split('.')[0] + "_2_et_hm.jpg")
            dgen.save_density_map(dm.squeeze(), save_path, fname[0].split('.')[0] + "_2_et_dm.jpg", et_count_dm)
            dens_mask = dens_nms >= args.det_thr
            # draw prediction in the image
            dgen.save_image_with_point(image.transpose((1, 2, 0)) * 255.0, dens_mask.copy(), save_path, fname[0].split('.')[0] + "_0_img.jpg")
            # draw GT in the image
            dgen.save_image_with_point(image.transpose((1, 2, 0)) * 255.0, gt_dens, save_path,
                                       fname[0].split('.')[0] + "_0_img_GT.jpg", GT=True)
            dgen.save_heatmep_pred(dens_mask.squeeze(), save_path, fname[0].split('.')[0] + "_3_et.jpg", et_count)
            _gt_count = gt_dens.sum().item()
            del gt_dens

        if args.save_txt:
            ori_img = Image.open(os.path.join(data_loader.dataloader.image_path, fname[0]))
            ori_w, ori_h = ori_img.size
            h, w = data.shape[2], data.shape[3]
            ratio_w = float(ori_w) / w
            ratio_h = float(ori_h) / h
            for j in range(save_txt_num):
                dens_nms_tmp = dens_nms.copy()
                dens_nms_tmp[dens_nms_tmp >= thr[j]] = 1
                dens_nms_tmp[dens_nms_tmp < thr[j]] = 0
                ids = np.array(np.where(dens_nms_tmp == 1))  # y,x
                ori_ids_y = ids[2, :] * ratio_h + ratio_h/2
                ori_ids_x = ids[3, :] * ratio_w + ratio_w/2
                ids = np.vstack((ori_ids_x, ori_ids_y)).astype(np.int16)  # x,y
                et_count_tmp = ids.shape[1]

                loc_str = ''
                for i_id in range(ids.shape[1]):
                    loc_str = loc_str + ' ' + str(ids[0][i_id]) + ' ' + str(ids[1][i_id])  # x, y
                if i == len(data_loader) - 1:
                    record[j].write('{filename} {pred:d}{loc_str}'.format(filename=fname[0].split('.')[0], pred=et_count_tmp,
                                                                       loc_str=loc_str))
                else:
                    record[j].write('{filename} {pred:d}{loc_str}\n'.format(filename=fname[0].split('.')[0],pred=et_count_tmp,
                                                                         loc_str=loc_str))

        del data, dens
        detail += "index: {}; fname: {}; gt: {}; et: {}; dif: {};\n".format(i, fname[0].split('.')[0], gt_count, et_count, gt_count-et_count)
        mae += abs(gt_count - et_count)
        mse += ((gt_count - et_count) * (gt_count - et_count))
        if gt_count != 0:
            NAE_count += 1.0
            NAE += abs(gt_count - et_count) / float(gt_count)

        mae_dm += abs(gt_count - et_count_dm)
        mse_dm += ((gt_count - et_count_dm) * (gt_count - et_count_dm))
        if gt_count != 0:
            NAE_count_dm += 1.0
            NAE_dm += abs(gt_count - et_count_dm) / float(gt_count)

    mae = mae / len(data_loader)
    mse = np.sqrt(mse / len(data_loader))
    NAE = NAE/NAE_count

    mae_dm = mae_dm / len(data_loader)
    mse_dm = np.sqrt(mse_dm / len(data_loader))
    NAE_dm = NAE_dm/NAE_count_dm
    duration = timer.toc(average=False)
    if args.save_txt:
        for j in range(save_txt_num):
            record[j].close()
    detail += "Time per item: %f" % time_per_item.average_time
    print("Time per item: %f" % time_per_item.average_time)
    print("testing time: %d" % duration)
    return mae, mse, NAE, mae_dm, mse_dm, NAE_dm, detail


def test_model_patches(net, data_loader, save_output=False, save_path=None, test_fixed_size=-1, test_batch_size=1,
                       gpus=None, args=None):
    timer = Timer()
    timer.tic()
    net.eval()
    mae = 0.0
    mse = 0.0
    detail = ''
    if save_output:
        print(save_path)
    for i, blob in enumerate(data_loader.get_loader(1)):

        if (i + 1) % 10 == 0:
            print("testing %d" % (i + 1))
        if save_output:
            index, fname, data, mask, gt_dens, gt_count = blob
        else:
            index, fname, data, mask, gt_count = blob

        data = data.squeeze_()
        if len(data.shape) == 3:
            'image small than crop size'
            data = data.unsqueeze_(dim=0)
        mask = mask.squeeze_()
        num_patch = len(data)
        batches = zip(
            [i * test_batch_size for i in range(num_patch // test_batch_size + int(num_patch % test_batch_size != 0))],
            [(i + 1) * test_batch_size for i in range(num_patch // test_batch_size)] + [num_patch])
        with torch.no_grad():
            dens_patch = []
            for batch in batches:
                bat = data[slice(*batch)]
                dens = net(bat).cpu()
                dens_patch += [dens]

            if args.test_fixed_size != -1:
                H, W = mask.shape
                _, _, fixed_size = data[0].shape
                assert args.test_fixed_size == fixed_size
                density_map = torch.zeros((H, W))
                for dens_slice, (x, y) in zip(itertools.chain(*dens_patch),
                                              itertools.product(range(W / fixed_size), range(H / fixed_size))):
                    density_map[y * fixed_size:(y + 1) * fixed_size, x * fixed_size:(x + 1) * fixed_size] = dens_slice
                H = mask.sum(dim=0).max().item()
                W = mask.sum(dim=1).max().item()
                density_map = density_map.masked_select(mask).view(H, W)
            else:
                density_map = dens_patch[0]

            gt_count = gt_count.item()
            et_count = density_map.sum().item()

            if save_output:
                image = data.mul_(torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)) \
                    .add_(torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1))

                if args.test_fixed_size != -1:
                    H, W = mask.shape
                    _, _, fixed_size = data[0].shape
                    assert args.test_fixed_size == fixed_size
                    inital_img = torch.zeros((3, H, W))
                    for img_slice, (x, y) in zip(image,
                                                 itertools.product(range(W / fixed_size), range(H / fixed_size))):
                        inital_img[:, y * fixed_size:(y + 1) * fixed_size,
                        x * fixed_size:(x + 1) * fixed_size] = img_slice
                    H = mask.sum(dim=0).max().item()
                    W = mask.sum(dim=1).max().item()
                    inital_img = inital_img.masked_select(mask).view(3, H, W)
                    image = inital_img

                image = image.data.cpu().numpy()
                dgen.save_image(image.transpose((1, 2, 0)) * 255.0, save_path, fname[0].split('.')[0] + "_0_img.png")
                gt_dens = gt_dens.data.cpu().numpy()
                density_map = density_map.data.cpu().numpy()
                dgen.save_density_map(gt_dens.squeeze(), save_path, fname[0].split('.')[0] + "_1_gt.png")
                dgen.save_density_map(density_map.squeeze(), save_path, fname[0].split('.')[0] + "_2_et.png")
                del gt_dens
            del data, dens

        detail += "index: {}; fname: {}; gt: {}; et: {};\n".format(i, fname[0].split('.')[0], gt_count, et_count)
        mae += abs(gt_count - et_count)
        mse += ((gt_count - et_count) * (gt_count - et_count))
    mae = mae / len(data_loader)
    mse = np.sqrt(mse / len(data_loader))
    duration = timer.toc(average=False)
    print("testing time: %d" % duration)
    return mae, mse, detail


if __name__ == '__main__':
    args = parser.parse_args()
    # set gpu ids
    str_ids = args.gpus.split(',')

    args.gpus = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpus.append(id)
    if len(args.gpus) > 0:
        torch.cuda.set_device(args.gpus[0])
    args.loss = None
    args.test_crop_type = 'Adap'
    args.pretrain = None

    data_loader_test = CreateDataLoader(args, phase='test')

    optimizer = lambda x: torch.optim.Adam(filter(lambda p: p.requires_grad, x.parameters()))
    net = CrowdCounter(optimizer=optimizer, opt=args)

    if args.model_path.endswith('.h5'):
        output_path = args.model_path[:-3] + '/output/'
        if not os.path.exists(args.model_path[:-3]):
            os.mkdir(args.model_path[:-3])

        test_once = True
    else:
        output_path = args.model_path + '/output/'
        test_once = False

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if test_once:
        model_files = [args.model_path]
    elif args.epoch is not None:
        model_files = ['%06d.h5' % args.epoch]
        assert args.save_output
    elif not args.is_wait:
        def list_dir(watch_path):
            return itertools.chain(
                *[[filename] if (os.path.isfile(os.path.join(watch_path, filename)) and '.h5' in filename) \
                      else [] \
                  for filename in os.listdir(watch_path)])


        model_files = list(list_dir(args.model_path))
        model_files.sort()
        model_files = model_files[::-1]
        assert not args.save_output

    else:
        model_files = ['%06d.h5' % epoch for epoch in range(0, 301)]
        assert not args.save_output

    if args.split is not None:
        model_files = ['%06d.h5' % epoch for epoch in map(int, args.split[:-1].split(','))]

    print(model_files)
    best_mae = 9e6
    best_mae_dm = 9e6
    best_mae_log = os.path.join(output_path, 'best_mae_log.txt')
    if os.path.isfile(best_mae_log):
        os.remove(best_mae_log)
    for model_file in model_files:

        epoch = model_file.split('.')[0] if not test_once else '0'

        if int(epoch) > 0 and int(epoch) < 0:
            continue

        output_dir = os.path.join(output_path, epoch)
        file_results = os.path.join(output_dir, 'results.txt')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        output_dir = os.path.join(output_dir, 'density_maps')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        trained_model = os.path.join(args.model_path, epoch + '.h5') if not test_once else args.model_path

        while (not os.path.isfile(trained_model)):
            time.sleep(3)

        network.load_net(trained_model, net)

        if args.is_search_thr:
            test_model_fun = test_model_origin_search
        else:
            test_model_fun = test_model_origin

        if args.test_batch_size != 1 or args.test_fixed_size != -1: # TODO
            test_mae, test_mse, detail = test_model_patches(net, data_loader_test, args.save_output, \
                                                            output_dir, test_fixed_size=args.test_fixed_size,
                                                            test_batch_size=args.test_batch_size, \
                                                            gpus=args.gpus, args=args)
        elif args.is_search_thr:
            test_mae, test_mse, test_mae_dm, test_mse_dm, detail = test_model_fun(net, data_loader_test, args.save_output, \
                                                           output_dir, test_fixed_size=args.test_fixed_size,
                                                           test_batch_size=args.test_batch_size, \
                                                           gpus=args.gpus, args=args)
        else:
            test_mae, test_mse, test_nae, test_mae_dm, test_mse_dm, test_nae_dm, detail = \
                                                        test_model_fun(net, data_loader_test, args.save_output, \
                                                           output_dir, test_fixed_size=args.test_fixed_size,
                                                           test_batch_size=args.test_batch_size, \
                                                           gpus=args.gpus, args=args)

        if args.is_search_thr:
            ind_min = test_mae.index(min(test_mae))
            log_text = 'TEST EPOCH: %s, Det_thr: %.2f, MAE: %.2f, MSE: %0.2f\n' % \
                       (epoch, (args.search_start + ind_min) / 100.0, test_mae[ind_min], test_mse[ind_min])
            log_text_dm = 'TEST EPOCH: %s, MAE_dm: %.2f, MSE_dm: %0.2f\n' % (epoch, test_mae_dm, test_mse_dm)
            print(log_text,log_text_dm)
            with open(file_results, 'w') as f:
                f.write(log_text + '\n')
                f.write(log_text_dm  + '\n')
                f.write(detail)

            if min(test_mae) < best_mae:
                best_mae = min(test_mae)
                ind_min = test_mae.index(min(test_mae))
                log_text_best = 'Best TEST EPOCH: %s, Det_thr: %.2f, MAE: %.2f, MSE: %0.2f' % \
                               (epoch, (args.search_start+ind_min)/100.0, test_mae[ind_min], test_mse[ind_min])
                print(log_text_best)
                with open(best_mae_log, 'a+') as f:
                    f.write(log_text_best+'\n')

            if test_mae_dm < best_mae_dm:
                best_mae_dm = test_mae_dm
                log_text_dm_best = 'Best TEST EPOCH: %s, MAE_dm: %.2f, MSE_dm: %0.2f\n' % (epoch, test_mae_dm, test_mse_dm)
                print(log_text_dm_best)
                with open(best_mae_log, 'a+') as f:
                    f.write(log_text_dm_best+'\n')

        else:
            log_text = 'TEST EPOCH: %s, MAE: %.2f, MSE: %0.2f, NAE: %0.3f\n' % (epoch, test_mae, test_mse, test_nae)
            log_text_dm = 'TEST EPOCH: %s, MAE_dm: %.2f, MSE_dm: %0.2f, NAE_dm: %0.3f\n' % (epoch, test_mae_dm, test_mse_dm, test_nae_dm)
            print(log_text,log_text_dm)

            with open(file_results, 'w') as f:
                f.write('Detection threshold: ' + str(args.det_thr) + '\n')
                f.write(log_text)
                f.write(log_text_dm)
                f.write(detail)