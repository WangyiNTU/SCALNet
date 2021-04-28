import torch
import numpy as np
import torch.nn.functional as F

def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y

def _neg_loss(pred, gt, shrink=16.0):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds / shrink

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def _fp_loss(pred, seg):
    ''' Modified focal loss. Exactly the same as CornerNet.
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    '''
    neg_inds = seg.eq(0).float()
    pred = pred * neg_inds
    pred = torch.clamp(pred, min=1e-4)
    mask = pred.clone().detach()
    mask[mask > 0.1] = 1.
    mask[mask <= 0.1] = 0.

    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * mask

    num_neg = mask.sum(dim=(1,2,3))
    neg_loss = neg_loss.sum(dim=(1,2,3))

    num_neg[num_neg==0] = 1.
    loss = -((neg_loss) / num_neg).mean()
    return loss

class FocalLoss(torch.nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)

class HFocalLoss(torch.nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(HFocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        loss = 0
        for head in out:
            if head == 'hm':
                loss += self.neg_loss(out[head], target[head])
            else:
                loss += 0.1*self.neg_loss(out[head], target[head], shrink=1)
        return loss

class LocLoss(torch.nn.Module):
    def __init__(self):
        super(LocLoss, self).__init__()
        self.crit = FocalLoss()

    def forward(self, hm, density_map, gt):
        hm = _sigmoid(hm)
        hm_gt = gt[:,0:1,:,:]
        loc_loss = self.crit(hm, hm_gt)
        fp_loss = _fp_loss(hm, hm_gt)
        loc_loss += fp_loss

        dm_gt = gt[:,1:2,:,:].clone()
        _,_,h,w = density_map.shape
        _,_,h_gt,w_gt = dm_gt.shape
        size = int(h_gt/h) * int(w_gt/w)
        # size = 4 #in this case of s2
        dm_gt = F.avg_pool2d(dm_gt,kernel_size=int(h_gt/h)) * (size)
        regression_loss = F.mse_loss(density_map, dm_gt) * 1000

        # disable regression_loss
        # regression_loss = torch.zeros_like(loc_loss).cuda()
        # disable loc_loss
        # loc_loss = torch.zeros_like(regression_loss).cuda()

        return loc_loss, regression_loss

class HLocLoss(torch.nn.Module):
    def __init__(self):
        super(HLocLoss, self).__init__()
        self.crit = HFocalLoss()

    def forward(self, h_hm, hm_gt):
        for head in h_hm:
            h_hm[head] = _sigmoid(h_hm[head])
        h_hm_gt = {'hm': hm_gt}
        h_hm_gt['hm0'] = F.max_pool2d(hm_gt.clone(), 2, 2)
        h_hm_gt['hm1'] = F.max_pool2d(hm_gt.clone(), 4, 4)
        h_hm_gt['hm2'] = F.max_pool2d(hm_gt.clone(), 8, 8)
        h_hm_gt['hm3'] = F.max_pool2d(hm_gt.clone(), 16, 16)
        h_hm_gt['hm4'] = F.max_pool2d(hm_gt.clone(), 32, 32)
        loc_loss = self.crit(h_hm, h_hm_gt)
        fp_loss = _fp_loss(h_hm['hm'], hm_gt)
        return loc_loss + fp_loss