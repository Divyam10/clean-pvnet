import torch.nn as nn
from lib.utils import net_utils
import torch
import numpy as np
import BPnP
import os
# import config as cfg


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net

        self.vote_crit = torch.nn.functional.smooth_l1_loss
        self.seg_crit = nn.CrossEntropyLoss()
        # self.Kpt_3D = np.load('../../data/custom/meta.npy',
        #                       allow_pickle=True).item()['kpt_3d's]
        # self.K = np.load('../../data/custom/meta.npy',
        #                  allow_pickle=True).item()['K']
        meta = np.load("/home/ai/pose_est/clean-pvnet-1.10-bpnp/meta.npy", allow_pickle = True).item()
        self.K=meta['K']
        self.K = torch.from_numpy(self.K).float().cuda()
        self.Kpt_3D=meta['kpt_3d']
        self.Kpt_3D = torch.from_numpy(self.Kpt_3D).float().cuda()
        self.bpnp=BPnP.BPnP.apply
        self.ini_pose=torch.zeros(1, 6, device = 'cuda:0')
        self.ini_pose[0, 5]=99

    def forward(self, batch):
        output=self.net(batch['inp'])

        scalar_stats={}
        loss=0

        if 'pose_test' in batch['meta'].keys():
            loss=torch.tensor(0).to(batch['inp'].device)
            return output, loss, {}, {}

        weight=batch['mask'][:, None].float()
        vote_loss=self.vote_crit(
            output['vertex'] * weight, batch['vertex'] * weight, reduction='sum')
        vote_loss = vote_loss / weight.sum() / batch['vertex'].size(1)
        scalar_stats.update({'vote_loss': vote_loss})
        loss += vote_loss

        mask = batch['mask'].long()
        seg_loss = self.seg_crit(output['seg'], mask)
        scalar_stats.update({'seg_loss': seg_loss})
        loss += seg_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        kpt_2d_pd = output['kpt_2d']
        kpt_2d_gt = batch['kpt_2d']

        P_out = self.bpnp(kpt_2d_gt, self.Kpt_3D, self.K)
        pts2d_pro = BPnP.batch_project(P_out, self.Kpt_3D, self.K)

        loss_bpnp = ((pts2d_pro - kpt_2d_gt)**2).mean() + ((pts2d_pro - kpt_2d_pd)**2).mean()
        print("loss_bpnp:  "  , loss_bpnp.item())
        if loss_bpnp.item() < 0.001:
            exit()
        self.ini_pose = P_out.detach()
        return output, loss, scalar_stats, image_stats
