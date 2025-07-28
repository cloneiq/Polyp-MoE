import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def softmax_w_top(x, top, return_usage=False):  # x:{16,44*44/22*22/11*11}
    top = min(top, x.shape[1]) # min(top,THW)
    values, indices = torch.topk(x, k=top, dim=1)
    x_exp = torch.softmax(values, dim=1)
    x.zero_().scatter_(1, indices, x_exp) # B * THW * HW

    if return_usage:   
        return x, x.sum(dim=2),x_exp
    else:
        return x,x_exp

class PositionEncodingModule(nn.Module):
    def __init__(self, channels, direction,window_size):
        super(PositionEncodingModule, self).__init__()

        self.channels = channels
        self.windows_size = window_size
        self.direction = direction

        if self.direction == 'H':
            self.pos_encoding = nn.Parameter(torch.randn(1, channels, window_size, 1))
        else:
            self.pos_encoding = nn.Parameter(torch.randn(1, channels, 1, window_size))

    def forward(self, feature):

        pos_enc_expanded = self.pos_encoding.expand(1, self.channels, self.windows_size, self.windows_size)

        return feature + pos_enc_expanded
######   New   ######
class QueryGuidedAffinityFilterV5d(nn.Module):

    def __init__(self, in_channels, reduction=16, spatial_size=(32, 32)):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.GELU(),
            nn.Linear(in_channels // reduction, in_channels),
        )

        self.norm = nn.LayerNorm(in_channels)

        # position_score MLP
        self.project = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.GELU(),
            nn.Linear(in_channels // 2, 1),
        )
        self.norm_project = nn.LayerNorm(in_channels)
        self.temperature = 0.1


    def forward(self, value_query, affinity, prev_mask=None):

        B, C, H, W = value_query.shape

        # Step 1: 引导向量
        pooled = self.pool(value_query).view(B, C)          # [B, C]
        channel_weight = self.mlp(pooled)                   # [B, C]
        channel_weight = self.norm(channel_weight)          # [B, C]
        channel_weight = channel_weight.unsqueeze(2)        # [B, C, 1]

        # Step 2: query 指导位置信息
        value_query_flat = value_query.view(B, C, -1)        # [B, C, H*W]
        fusion = value_query_flat * channel_weight          # [B, C, H*W]
        fusion = fusion.permute(0, 2, 1)                    # [B, H*W, C]
        position_score = self.project(fusion).squeeze(-1)   # [B, H*W]
        position_weight = F.softmax(position_score / self.temperature, dim=-1).unsqueeze(1)  # [B, 1, H*W]

        if prev_mask is not None:

            prev_mask_flat = prev_mask.view(B, 1, -1)       # [B, 1, H*W]

            smooth_noise = torch.rand_like(prev_mask_flat) * 1e-6
            prev_mask_flat = prev_mask_flat + smooth_noise

            prev_mask_weight = prev_mask_flat / (prev_mask_flat.sum(dim=-1, keepdim=True) + 1e-6)

            position_weight = position_weight * (prev_mask_weight + 1e-6)

            position_weight = position_weight / (position_weight.sum(dim=-1, keepdim=True) + 1e-6)


        guided_affinity = affinity * position_weight       # [B, H*W, H*W]

        return F.softmax(guided_affinity, dim=-1)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class MemoryBank(nn.Module):

    def __init__(self, test_mem_length=35, num_values=3, top_k=20, count_usage=True):
        super().__init__()
        self.top_k = top_k                                                  

        self.CK = None
        self.CV = None
        self.num_values = num_values
        self.test_mem_length = test_mem_length


        self.mem_ks = [None for i in range(self.num_values)]
        self.mem_vs = [None for i in range(self.num_values)]
        self.T = 0

        self.position_4 = QueryGuidedAffinityFilterV5d(16, spatial_size=(44,44))
        self.position_2 = QueryGuidedAffinityFilterV5d(16, spatial_size=(22,22))
        self.position_1 = QueryGuidedAffinityFilterV5d(16, spatial_size=(11,11))

        self.CountUsage = count_usage
        if self.CountUsage:
            self.usage_count = [None for i in range(self.num_values)]
            self.life_count = [None for i in range(self.num_values)]

    def _global_matching(self, mk, qk):
        B, CK, NE = mk.shape

        a = mk.pow(2).sum(1).unsqueeze(2)   # [b,hw,1]
        b = 2 * (mk.transpose(1, 2) @ qk)   # [b,thw,hw]
        # c = qk.pow(2).sum(1).unsqueeze(1)  # [B, 1, HW]
        #
        # affinity = (-a + b - c) / math.sqrt(CK)
        affinity = (-a+b) / math.sqrt(CK)  # B, NE, HW; [B, [256|512|...], 256]

        return affinity

    def _readout(self, affinity, value_mem, value_query, maskM = None):

        B, C, H, W = value_query.shape
        HW = H * W

        if H == 44:
            if maskM is not None:
                refined_affinity = self.position_4(value_query,affinity, maskM[0])
            else:
                refined_affinity = self.position_4(value_query, affinity)
        elif H == 22:
            if maskM is not None:
                refined_affinity = self.position_2(value_query,affinity, maskM[1])
            else:
                refined_affinity = self.position_2(value_query, affinity)
        else:
            if maskM is not None:
                refined_affinity = self.position_1(value_query,affinity, maskM[2])
            else:
                refined_affinity = self.position_1(value_query, affinity)

        readout = torch.bmm(value_mem, refined_affinity.transpose(1, 2))  # [B, C, HW]
        readout = readout.view(B, C, H, W)

        return torch.cat([readout, value_query], dim=1)

                                                                            
    def match_memory(self, keyQ, valueQ, keyM_outer=None, valueM_outer=None, maskM=None):

        readout_mems = []

        if (keyM_outer is not None) and (valueM_outer is not None):
            if len(keyM_outer[0].shape) == 4:
                keyM = [x.flatten(start_dim=-2) for x in keyM_outer]
            else:
                keyM = keyM_outer
            
            if len(valueM_outer[0].shape) == 4:
                valueM = [x.flatten(start_dim=-2) for x in valueM_outer]
            else:
                valueM = valueM_outer

        else:
            keyM , valueM = self.mem_ks,self.mem_vs

        for i in range(self.num_values):                                        

            key_query, value_query = keyQ[i], valueQ[i]

            if maskM == None:
                key_mem, value_mem = keyM[i], valueM[i] # key_mem:{16,4,44*44/22*22/11*11,44*44/22*22/11*11} value_mem:{16,16,44*44/22*22/11*11,44*44/22*22/11*11}
            else:
                key_mem, value_mem = keyM[i] * (F.softmax(maskM[i].flatten(start_dim=-2), dim=-1) + 1), valueM[i]

            affinity = self._global_matching(key_mem, key_query.flatten(start_dim=-2))

            if self.CountUsage and (None not in self.mem_ks):

                affinity, usage_new, x_exp = softmax_w_top(affinity, top=self.top_k,return_usage=True)  # B, NE, HW
                self.update_usage(usage_new=usage_new,scale=i)
            else:
                affinity,x_exp = softmax_w_top(affinity, top=self.top_k,return_usage=False)  # B, NE, HW


            readout_mems.append(self._readout(affinity, value_mem, value_query, maskM))    # affinity:{B,hw,hw} , value_mem:{B,C/2,hw} , value_query:{B,C/2,H,W}

        return readout_mems # readout_mems:{B,C,H,W}

    def add_memory(self,keyQ_mem,valueQ_mem,mask_mem):

        if any(x is None for x in [keyQ_mem, valueQ_mem, mask_mem]):
            print("Error: add_memory received None input")
            return

        keys_mem = [keyQ_mem[i].flatten(start_dim=-2)*(F.softmax(mask_mem[i].flatten(start_dim=-2),dim=-1)+1)
                    for i in range(self.num_values)]

        values_mem = [valueQ_mem[i].flatten(start_dim=-2)*(F.softmax(mask_mem[i].flatten(start_dim=-2),dim=-1)+1)
                      for i in range(self.num_values)]
        if self.CountUsage:
            new_count = [torch.zeros((keys_mem[i].shape[0], 1, keys_mem[i].shape[2]), device=keys_mem[0].device, dtype=torch.float32) 
                            for i in range(self.num_values)]
            new_life = [torch.zeros((keys_mem[i].shape[0], 1, keys_mem[i].shape[2]), device=keys_mem[0].device, dtype=torch.float32) + 1e-7
                            for i in range(self.num_values)]
        
        # keys: 3*[b,32,h*w]
        if None in self.mem_ks :                                              
            self.mem_ks = keys_mem
            self.mem_vs = values_mem
            self.CK = keys_mem[0].shape[1]
            self.CV = values_mem[0].shape[1]
            self.hwK0 = keys_mem[0].shape[2]
            self.hwV0 = values_mem[0].shape[2]
            if self.CountUsage:
                self.usage_count = new_count
                self.life_count = new_life
        else:                                                               
            self.mem_ks = [torch.cat([self.mem_ks[i], keys_mem[i]], 2) for i in range(self.num_values)]
            self.mem_vs = [torch.cat([self.mem_vs[i], values_mem[i]], 2) for i in range(self.num_values)]
            
            if self.CountUsage:
                self.usage_count = [torch.cat([self.usage_count[i], new_count[i]], -1) for i in range(self.num_values)]
                self.life_count =[torch.cat([self.life_count[i], new_life[i]], -1) for i in range(self.num_values)]
                if self.T >= self.test_mem_length:
                    self.obsolete_features_removing(self.test_mem_length)

        self.T = self.mem_ks[0].shape[2] // self.hwK0

    def clear_memory(self):

        self.mem_ks = [None for i in range(self.num_values)]
        self.mem_vs = [None for i in range(self.num_values)]
        if self.CountUsage:
            self.usage_count = [None for i in range(self.num_values)]
            self.life_count = [None for i in range(self.num_values)]
        self.T = 0
        # print('clear memory!')

    def update_usage(self, usage_new, scale):
        # increase all life count by 1
        # increase use of indexed elements
        if not self.CountUsage:
            return
        
        self.usage_count[scale] += usage_new.view_as(self.usage_count[scale])
        self.life_count[scale] += 1

    def get_usage_scale(self,scale):
        # return normalized usage
        if not self.CountUsage:
            raise RuntimeError('No count usage!')
        else:
            usage = self.usage_count[scale] / self.life_count[scale]
            return usage

    def obsolete_features_removing(self, max_length: int):

        for i in range(self.num_values):

            usage = self.get_usage_scale(scale=i).flatten()  #[B*T*H*W]
            max_size = max_length * (self.hwK0) * (1/4)**i
            # print('remove:{}'.format(str(int(self.size[i]-max_size))))
            values, index = torch.topk(usage, k=int(self.size[i]-max_size), largest=False, sorted=True)
            survived = (usage > values[-1])  

            self.mem_ks[i] = self.mem_ks[i][:, :, survived] if self.mem_ks[i] is not None else None
            self.mem_vs[i] = self.mem_vs[i][:, :, survived] if self.mem_vs[i] is not None else None
            self.usage_count[i] = self.usage_count[i][:, :, survived]
            self.life_count[i] = self.life_count[i][:, :, survived]

    @property
    def size(self):
        if self.mem_ks[0] is None:
            return [0 for i in range(self.num_values)]
        else:
            return [self.mem_ks[i].shape[-1] for i in range(self.num_values)] #T*H*W