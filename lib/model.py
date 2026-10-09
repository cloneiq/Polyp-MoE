import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.pvtv2_afterRFB import Network as Encoder
from lib.TCGM import MemoryBank
from lib.DynamicConv import DynamicConv


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class PCDAlignment(nn.Module):

    def __init__(self, num_feat=64, deformable_groups=8):
        super(PCDAlignment, self).__init__()

        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1,1)
            if i == 3:
                self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1,1)
            else:
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3,1, 1)
                self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1,1)

            self.dcn_pack[level] = DynamicConv(num_feat * 2 , num_feat)

            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l):


        feature_pyramid = []
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([nbr_feat_l[i-1], ref_feat_l[i-1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == 3:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(self.offset_conv2[level](torch.cat(
                    [offset, upsampled_offset], dim=1)))
                offset = self.lrelu(self.offset_conv3[level](offset))

            feat = self.dcn_pack[level](torch.cat([nbr_feat_l[i - 1], offset], dim=1))
            if i < 3:
                feat = self.feat_conv[level](
                    torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)

            if i > 1:  # upsample offset and features
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

            feature_pyramid.append(feat)

        feature_pyramid=feature_pyramid[::-1]
        return feature_pyramid

class ExpertGating(nn.Module):
    """
     Prior Router
    """
    def __init__(self, in_channels, reduction=16):
        super(ExpertGating, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x_loc, x_sz, x_bd):
        avg_feat = self.pool((x_loc + x_sz + x_bd) / 3.0).view(x_loc.size(0), -1)  # [B, C]
        weights = self.fc(avg_feat).view(-1, 3, 1, 1, 1)  # [B,3,1,1,1]
        stacked = torch.stack([x_loc, x_sz, x_bd], dim=1)  # [B,3,C,H,W]
        fused = (weights * stacked).sum(dim=1)  # [B,C,H,W]
        return fused
class SPMoE(nn.Module):
    "Full code to be released soon"

class MemBranch(nn.Module):

    def __init__(self, indim=32, num=3):
        super(MemBranch, self).__init__()
        self.num = num
        self.Key0 = nn.Conv2d(indim, indim//8, kernel_size=(3,3), padding=(1,1), stride=1)
        self.Key1 = nn.Conv2d(indim, indim//8, kernel_size=(3,3), padding=(1,1), stride=1)
        self.Key2 = nn.Conv2d(indim, indim//8, kernel_size=(3,3), padding=(1,1), stride=1)

        self.Value0 = nn.Conv2d(indim, indim//2, kernel_size=(3,3), padding=(1,1), stride=1)
        self.Value1 = nn.Conv2d(indim, indim//2, kernel_size=(3,3), padding=(1,1), stride=1)
        self.Value2 = nn.Conv2d(indim, indim//2, kernel_size=(3,3), padding=(1,1), stride=1)


    def forward(self, x):

        key = [
            self.Key0(x[0]), self.Key1(x[1]), self.Key2(x[2])
        ]
        value = [
            self.Value0(x[0]), self.Value1(x[1]), self.Value2(x[2])
        ]

        return key, value

class VideoModel(nn.Module):

    def __init__(self, args):
        super(VideoModel, self).__init__()

        self.args = args
        self.extra_channels = 0
        print("Select mask mode: concat, num_mask={}".format(self.extra_channels))

        self.backbone = Encoder(imgsize=self.args.trainsize)
        if self.args.pretrained_weights is not None:
            self.load_backbone(self.args.pretrained_weights)

        self.nlnet = SPMoE(in_planes=32, pyramid_type='conv')

        self.pcd_align = PCDAlignment(num_feat=32)

        self.mem_bank = MemoryBank()
        self.first_case_gt = None

        self.membranch = MemBranch(indim=32)

        self.freeze_bn()

        self.val_mem_pre = None
        self.train_mem_pre = None

    def load_backbone(self, pretrained):
        pretrained_dict = torch.load(pretrained)
        model_dict = self.state_dict()
        print("Load pretrained image model parameters from {}".format(pretrained))

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def freeze_bn(self):
        for m in self.backbone.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                m[1].eval()


    def forward(self, x, case_idx=0, mode='train' , gt_mem=None,use_mem=False, add_mem=False, clear = False):

        if mode == 'train':
            out = self.train_longshort(x, case_idx=case_idx, gt_mem=gt_mem, use_mem=use_mem, add_mem=add_mem, clear=clear)
        elif mode == 'val':
            out = self.val_longshort(x, case_idx=case_idx, gt_mem=gt_mem, use_mem=use_mem, add_mem=add_mem, clear=clear)
        else:
            print('no this mode!')
            out = None

        return out

    def train_longshort(self, x, case_idx=0, gt_mem=None, use_mem=False, add_mem=False, clear = False):

        if clear or case_idx == 0:
            self.train_mem_pre = None

        image_mem, image_cur, image_neibor = x[0],x[1],x[2]

        fmap_cur=self.backbone.feat_net(image_cur)
        fmap_neibor=self.backbone.feat_net(image_neibor)
        fmap_mem=self.backbone.feat_net(image_mem)

        keyQ, valueQ = self.membranch(fmap_cur)
        keyM, valueM00= self.membranch(fmap_mem)

        maskM = [] # maskM_shape:{16,1,44/22/11,44/22/11}
        for i in range(len(fmap_mem)):
            mask = F.interpolate(gt_mem, scale_factor=1/(8*2**i), mode='bilinear')
            maskM.append(mask)

        valueM = [valueM00[i].flatten(start_dim=-2)*(F.softmax(maskM[i].flatten(start_dim=-2),dim=-1)+1) for i in range(3)] # valueM_shape:{16,16,44*44/22*22/11*11}


        corr_vol_mem = self.mem_bank.match_memory(keyQ,valueQ,keyM,valueM,maskM)

        # align
        fmap_ref =[x.clone() for x in fmap_cur] # [batchsize,32,44/22/11,44/22/11]
        falign1=self.pcd_align(corr_vol_mem,fmap_ref)  # [batchsize,32,44/22/11,44/22/11]
        falign2=self.pcd_align(fmap_neibor,fmap_ref)
        # fuse
        corr_vol_neibor = self.nlnet(falign1, falign2)  # 3*[b,32,44/22/11,44/22/11]

        out = self.backbone.decoder(corr_vol_neibor)
        # seperate mask of different scale
        mask_mem = out[-1][::-1]
        self.train_mem_pre = mask_mem
        out = out[:-1]
        return out

    def val_longshort(self, x, case_idx=0, gt_mem=None, use_mem=False, add_mem=False, clear = False):

        if clear or case_idx == 0:
            self.val_mem_pre = None

        image1, image2 = x[0],x[1]
        fmap1=self.backbone.feat_net(image1)
        fmap2=self.backbone.feat_net(image2)  # 3*[b,32,44/22/11,44/22/11]

        # val memory read
        keyQ, valueQ = self.membranch(fmap1)

        if use_mem:
            corr_vol_mem = self.mem_bank.match_memory(keyQ,valueQ,maskM=self.val_mem_pre)
            # print("Use mem")

        else:
            corr_vol_mem = self.mem_bank.match_memory(keyQ,valueQ,keyM_outer=keyQ,valueM_outer=valueQ,maskM=self.val_mem_pre)

        # align
        falign1=self.pcd_align(corr_vol_mem,fmap1)
        falign2=self.pcd_align(fmap2,fmap1)
        # fuse
        corr_vol_neibor = self.nlnet(falign1, falign2)  # 3*[b,32,44/22/11,44/22/11]

        ## decode ##
        out = self.backbone.decoder(corr_vol_neibor)
        mask_mem = out[-1][::-1]    # [1,1,44/22/11,44/22/11]
        self.val_mem_pre = mask_mem
        out = out[:-1]

        ## memory load(should after memory read) ##

        if 0 in case_idx:
            self.mem_bank.add_memory(keyQ, valueQ, mask_mem)
        elif add_mem:
            self.mem_bank.add_memory(keyQ, valueQ, mask_mem)

        return out

class ClinicalBoundaryConstrainedModule(nn.Module):
    """
        可学习边界先验
    """
    def __init__(self, in_channels, template_size=(32, 32), num_templates=4, gamma=0.5):
        super(ClinicalBoundaryConstrainedModule, self).__init__()
        self.in_channels = in_channels
        self.num_templates = num_templates
        self.gamma = gamma

        # 可学习的边界模板 (形状为 num_templates x 1 x H x W)
        self.boundary_templates = nn.Parameter(
            torch.randn(num_templates, 1, *template_size), requires_grad=True
        )

        # 可学习的边缘提取器 (depthwise conv)
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 融合后的通道数保持一致
        self.fuse = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        """
        输入: x [B, C, H, W]
        输出: x_enhanced [B, C, H, W] - 注入边界先验后的特征图
        """
        B, C, H, W = x.shape

        # 提取边界特征
        edge_feat = self.edge_conv(x)  # [B, C, H, W]
        edge_feat_reduced = torch.mean(edge_feat, dim=1, keepdim=True)  # [B, 1, H, W]

        # resize boundary_templates 到输入图大小 (H, W)
        templates_resized = F.interpolate(self.boundary_templates, size=(H, W), mode='bilinear', align_corners=False)  # [K, 1, H, W]

        # 计算匹配相似度（余弦相似度）
        sim_scores = []
        for i in range(self.num_templates):
            sim = F.cosine_similarity(edge_feat_reduced, templates_resized[i].unsqueeze(0), dim=1)  # [B, H, W]
            sim_scores.append(sim.unsqueeze(1))  # [B, 1, H, W]

        sim_scores = torch.cat(sim_scores, dim=1)  # [B, K, H, W]
        attention_weights = F.softmax(sim_scores, dim=1)  # [B, K, H, W]

        # 使用 attention 融合 boundary templates
        weighted_templates = (attention_weights.unsqueeze(2) * templates_resized.unsqueeze(0)).sum(dim=1)  # [B, 1, H, W]

        # 扩展通道数，与原始特征融合
        prior_guidance = weighted_templates.repeat(1, C, 1, 1)  # [B, C, H, W]
        prior_guidance = self.fuse(prior_guidance)

        # 最终融合
        out = x + self.gamma * prior_guidance
        return out

class SizePriorInjectionModuleV4(nn.Module):
    def __init__(self, in_channels, image_size=(512, 512), mean_polyp_area=0.02, gamma = 0.5):
        """
        :param in_channels: 输入特征图的通道数
        :param image_size: 图像的大小 (H, W)
        :param mean_polyp_area: 训练集所有图像中平均息肉区域占比
        === 大小先验参数（learned_size_log） ===
            nlnet.SizePriorInjection_l1.learned_size_log: log_value = 0.888875, exp_value = 2.432392 (像素数估计)
            nlnet.SizePriorInjection_l2.learned_size_log: log_value = 2.238295, exp_value = 9.377330 (像素数估计)
            nlnet.SizePriorInjection_l4.learned_size_log: log_value = 3.640097, exp_value = 38.095539 (像素数估计)
        """
        super(SizePriorInjectionModuleV4, self).__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.mean_polyp_area = mean_polyp_area
        self.gamma = gamma

        # 用于生成前景注意力图的注意力模块
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()  # 输出 [B, 1, H, W]，值越大越可能是前景
        )

        # 可学习的目标面积 (对数空间更稳定)
        self.learned_size_log = nn.Parameter(torch.tensor(np.log(self.mean_polyp_area * image_size[0] * image_size[1])))

        # 通道重标定门
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [B, C, H, W] 输入特征图
        return: 特征图 [B, C, H, W]，加入大小先验调节
        """
        B, C, H, W = x.shape

        # 1. 得到前景注意力估计
        attn_map = self.attention(x)  # [B, 1, H, W]

        # 2. 估计前景区域大小（软面积）
        est_area = attn_map.sum(dim=(2, 3))  # [B, 1]
        target_size = self.learned_size_log.exp()  # [1]

        # 3. 计算偏离程度
        size_diff = (est_area - target_size).abs() / (H * W)
        size_diff = size_diff.view(B, 1, 1, 1)  # 用于广播

        # 4. 生成通道注意力门进行特征调节
        gate_weight = self.gate(x)  # [B, C, H, W]

        # 5. 融合大小偏离感知
        out = x * (1 - self.gamma * size_diff) + gate_weight * (self.gamma * size_diff)

        return out

class LocationPriorModuleV2(nn.Module):
    def __init__(self, in_channels, template_size=(32, 32), gamma=0.5):
        """
        可学习位置先验 + 通道门控机制
        :param in_channels: 输入特征图通道数
        :param template_size: 位置先验模板大小
        :param gamma: 注入强度
        """
        super(LocationPriorModuleV2, self).__init__()
        self.in_channels = in_channels
        self.gamma = gamma

        # 可学习的位置先验模板，初始化为偏下方的高斯（你也可以换成全 0 或 uniform）
        init_template = self._init_gaussian(template_size)
        self.location_template = nn.Parameter(init_template)  # shape: [1, 1, H, W]

        # 映射到 C 通道
        self.template_to_feat = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def _init_gaussian(self, size):
        """初始化高斯模板 (H, W)"""
        H, W = size
        y = torch.linspace(0, 1, H).view(H, 1).expand(H, W)
        x = torch.linspace(0, 1, W).view(1, W).expand(H, W)
        center_y = 0.6
        center_x = 0.5
        sigma_y = 0.2
        sigma_x = 0.2
        gauss = torch.exp(-((x - center_x)**2 / (2 * sigma_x**2) + (y - center_y)**2 / (2 * sigma_y**2)))
        return gauss.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]

    def forward(self, x):
        """
        输入:
            x: [B, C, H, W] - 特征图
        输出:
            out: [B, C, H, W] - 加入位置先验后的特征图
        """
        B, C, H, W = x.shape
        device = x.device

        # 插值至当前大小
        location_map = F.interpolate(self.location_template, size=(H, W), mode='bilinear', align_corners=False)  # [1,1,H,W]
        location_map = location_map.expand(B, -1, H, W).to(device)  # [B,1,H,W]

        # 投影为多通道
        prior_feat = self.template_to_feat(location_map)  # [B, C, H, W]

        out = x + self.gamma * prior_feat

        return out




