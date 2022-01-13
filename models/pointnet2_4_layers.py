import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_parts=2, num_classes=1, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0

        self.num_classes = num_classes
        self.normal_channel = normal_channel
        # def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        self.sa1 = PointNetSetAbstractionMsg(2048, radius_list=[0.05, 0.1, 0.2], nsample_list=[16, 32, 64],
                                             in_channel=3+additional_channel,
                                             mlp_list=[[32, 32, 64], [32, 32, 64], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(512, [0.2, 0.4], [64, 128],
                                             in_channel=64+64+128,
                                             mlp_list=[[128, 128, 256], [64, 64, 128]])
        self.sa3 = PointNetSetAbstractionMsg(128, [0.4, 0.8], [64, 128],
                                             in_channel=256+128,
                                             mlp_list=[[128, 128, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None,
                                          in_channel=512+3,
                                          mlp=[256, 512, 1024], group_all=True)
        # fp4 gets input from sa4 and sa3
        self.fp4 = PointNetFeaturePropagation(in_channel=1024+512, mlp=[256, 256])
        # fp3 gets input from fp4 and sa2
        self.fp3 = PointNetFeaturePropagation(in_channel=256+256+128, mlp=[256, 128])
        # fp2 gets input from fp3 and sa1
        self.fp2 = PointNetFeaturePropagation(in_channel=128+64+64+128, mlp=[128, 128])
        # fp1 gets input from fp2 + raw input + num_parts
        self.fp1 = PointNetFeaturePropagation(in_channel=128+6+1+additional_channel, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_parts, 1)

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        # Feature Propagation layers
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points) #sa4, sa3
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) #fp4, sa2
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) #fp3, sa1
        cls_label_one_hot = cls_label.view(B,self.num_classes,1).repeat(1,1,N)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self, weights, batch_size, adaptive, device):
        super(get_loss, self).__init__()
        self.weights = weights
        self.batch_size = batch_size
        self.adaptive = adaptive
        self.device = device

    def forward(self, pred, target, trans_feat, num_points, cur_batch_size):
        start = 0
        stop = num_points
        total_loss = 0

        if self.adaptive:
            for i in range(cur_batch_size):
                frac_tree = torch.sum(target[start:stop]) / len(target[start:stop])
                weights = torch.tensor([1, (1-frac_tree)/frac_tree])
                weights = weights.to(self.device)
                weights = weights.float()
                total_loss += F.nll_loss(pred[start:stop], target[start:stop], weight=weights)
                start += num_points
                stop += num_points

            total_loss = total_loss / self.batch_size
        else:
            total_loss = F.nll_loss(pred, target, weight=self.weights)

        return total_loss