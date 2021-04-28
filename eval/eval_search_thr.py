import os
import sys
import numpy as np
from scipy import spatial as ss
import pdb

import cv2
from utils import hungarian,read_pred_and_gt,AverageMeter,AverageCategoryMeter

gt_file = 'eval/val_gt_loc.txt'
pred_folder = 'saved_models/000018/output/0/loc_txt'
print(pred_folder)

flagError = False
id_std = [i for i in range(3110,3610,1)]
id_std[59] = 3098

num_classes = 6

def log_print(opt_file,message):
    print(message)
    opt_file.write(message)
    opt_file.write('\n')

def compute_metrics(dist_matrix,match_matrix,pred_num,gt_num,sigma,level):
    for i_pred_p in range(pred_num):
        pred_dist = dist_matrix[i_pred_p,:]
        match_matrix[i_pred_p,:] = pred_dist<=sigma

    tp, assign = hungarian(match_matrix)
    fn_gt_index = np.array(np.where(assign.sum(0)==0))[0]
    tp_pred_index = np.array(np.where(assign.sum(1)==1))[0]
    tp_gt_index = np.array(np.where(assign.sum(0)==1))[0]
    fp_pred_index = np.array(np.where(assign.sum(1)==0))[0]
    level_list = level[tp_gt_index]

    tp = tp_pred_index.shape[0]
    fp = fp_pred_index.shape[0]
    fn = fn_gt_index.shape[0]

    tp_c = np.zeros([num_classes])
    fn_c = np.zeros([num_classes])

    for i_class in range(num_classes):
        tp_c[i_class] = (level[tp_gt_index]==i_class).sum()
        fn_c[i_class] = (level[fn_gt_index]==i_class).sum()

    return tp,fp,fn,tp_c,fn_c


def main():
    file_name = pred_folder.split('/')[-5] + '_record.txt'
    opt_file = open(file_name, 'w+')

    for pred_file in os.listdir(pred_folder):
        pred_file_path = os.path.join(pred_folder, pred_file)

        cnt_errors = {'mae':AverageMeter(),'mse':AverageMeter(),'nae':AverageMeter(),}
        metrics_s = {'tp':AverageMeter(), 'fp':AverageMeter(), 'fn':AverageMeter(), 'tp_c':AverageCategoryMeter(num_classes), 'fn_c':AverageCategoryMeter(num_classes)}
        metrics_l = {'tp':AverageMeter(), 'fp':AverageMeter(), 'fn':AverageMeter(), 'tp_c':AverageCategoryMeter(num_classes), 'fn_c':AverageCategoryMeter(num_classes)}

        pred_data, gt_data = read_pred_and_gt(pred_file_path,gt_file)
        for i_sample in id_std:
            # print(i_sample)
            # init
            gt_p,pred_p,fn_gt_index,tp_pred_index,fp_pred_index= [],[],[],[],[]
            tp_s,fp_s,fn_s,tp_l,fp_l,fn_l = [0,0,0,0,0,0]
            tp_c_s = np.zeros([num_classes])
            fn_c_s = np.zeros([num_classes])
            tp_c_l = np.zeros([num_classes])
            fn_c_l = np.zeros([num_classes])

            if gt_data[i_sample]['num'] ==0 and pred_data[i_sample]['num'] !=0:
                pred_p =  pred_data[i_sample]['points']
                fp_pred_index = np.array(range(pred_p.shape[0]))
                fp_s = fp_pred_index.shape[0]
                fp_l = fp_pred_index.shape[0]

            if pred_data[i_sample]['num'] ==0 and gt_data[i_sample]['num'] !=0:
                gt_p = gt_data[i_sample]['points']
                level = gt_data[i_sample]['level']
                fn_gt_index = np.array(range(gt_p.shape[0]))
                fn_s = fn_gt_index.shape[0]
                fn_l = fn_gt_index.shape[0]
                for i_class in range(num_classes):
                    fn_c_s[i_class] = (level[fn_gt_index]==i_class).sum()
                    fn_c_l[i_class] = (level[fn_gt_index]==i_class).sum()

            if gt_data[i_sample]['num'] !=0 and pred_data[i_sample]['num'] !=0:
                pred_p =  pred_data[i_sample]['points']
                gt_p = gt_data[i_sample]['points']
                sigma_s = gt_data[i_sample]['sigma'][:,0]
                sigma_l = gt_data[i_sample]['sigma'][:,1]
                level = gt_data[i_sample]['level']

                # dist
                dist_matrix = ss.distance_matrix(pred_p,gt_p,p=2)
                match_matrix = np.zeros(dist_matrix.shape,dtype=bool)

                # sigma_s and sigma_l
                tp_s,fp_s,fn_s,tp_c_s,fn_c_s = compute_metrics(dist_matrix,match_matrix,pred_p.shape[0],gt_p.shape[0],sigma_s,level)
                tp_l,fp_l,fn_l,tp_c_l,fn_c_l = compute_metrics(dist_matrix,match_matrix,pred_p.shape[0],gt_p.shape[0],sigma_l,level)


            metrics_s['tp'].update(tp_s)
            metrics_s['fp'].update(fp_s)
            metrics_s['fn'].update(fn_s)
            metrics_s['tp_c'].update(tp_c_s)
            metrics_s['fn_c'].update(fn_c_s)
            metrics_l['tp'].update(tp_l)
            metrics_l['fp'].update(fp_l)
            metrics_l['fn'].update(fn_l)
            metrics_l['tp_c'].update(tp_c_l)
            metrics_l['fn_c'].update(fn_c_l)

            gt_count,pred_cnt = gt_data[i_sample]['num'],pred_data[i_sample]['num']
            s_mae = abs(gt_count-pred_cnt)
            s_mse = (gt_count-pred_cnt)*(gt_count-pred_cnt)
            cnt_errors['mae'].update(s_mae)
            cnt_errors['mse'].update(s_mse)

            if gt_count !=0:
                s_nae = abs(gt_count-pred_cnt)/float(gt_count)
                cnt_errors['nae'].update(s_nae)

        ap_s = metrics_s['tp'].sum/(metrics_s['tp'].sum+metrics_s['fp'].sum+1e-20)
        ar_s = metrics_s['tp'].sum/(metrics_s['tp'].sum+metrics_s['fn'].sum+1e-20)
        f1m_s = 2*ap_s*ar_s/(ap_s+ar_s)
        ar_c_s = metrics_s['tp_c'].sum/(metrics_s['tp_c'].sum+metrics_s['fn_c'].sum+1e-20)


        ap_l = metrics_l['tp'].sum/(metrics_l['tp'].sum+metrics_l['fp'].sum+1e-20)
        ar_l = metrics_l['tp'].sum/(metrics_l['tp'].sum+metrics_l['fn'].sum+1e-20)
        f1m_l = 2*ap_l*ar_l/(ap_l+ar_l)
        ar_c_l = metrics_l['tp_c'].sum/(metrics_l['tp_c'].sum+metrics_l['fn_c'].sum+1e-20)

        log_print(opt_file,pred_file)
        log_print(opt_file,'-----Localization performance-----')
        log_print(opt_file,'AP_small: '+str(ap_s))
        log_print(opt_file,'AR_small: '+str(ar_s))
        log_print(opt_file,'F1m_small: '+str(f1m_s))
        log_print(opt_file,'AR_small_category: '+str(ar_c_s))
        log_print(opt_file,'    avg: '+str(ar_c_s.mean()))
        log_print(opt_file,'AP_large: '+str(ap_l))
        log_print(opt_file,'AR_large: '+str(ar_l))
        log_print(opt_file,'F1m_large: '+str(f1m_l))
        log_print(opt_file,'AR_large_category: '+str(ar_c_l))
        log_print(opt_file,'    avg: '+str(ar_c_l.mean()))

        mae = cnt_errors['mae'].avg
        mse = np.sqrt(cnt_errors['mse'].avg)
        nae = cnt_errors['nae'].avg

        log_print(opt_file,'-----Counting performance-----')
        log_print(opt_file,'MAE: '+str(mae))
        log_print(opt_file,'MSE: '+str(mse))
        log_print(opt_file,'NAE: '+str(nae))
        log_print(opt_file,'\n')

    opt_file.close()


if __name__ == '__main__':
    main()
