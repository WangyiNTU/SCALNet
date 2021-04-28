import os
import numpy as np

loc_path = 'DLA_loc_testing_wo_FP.txt'

out_cnt_path = loc_path.replace('loc','cnt')
record = open(out_cnt_path, 'w+')

with open(loc_path) as f:
    id_read = []
    for line in f.readlines():
        line = line.strip().split(' ')
        record.writelines('{filename} {pred:0.2f}\n'.format(filename=line[0],pred=float(line[1])))

record.close()