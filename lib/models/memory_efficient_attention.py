import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..models.utils import qkv_transform

class MemoryEfficientAxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        super(MemoryEfficientAxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                         padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        # Permute to consistent (N, W, C, H) layout — matches original axialnet.py convention
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)

        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # QKV transformation
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(
            qkv.reshape(N * W, self.groups, self.group_planes * 2, H),
            [self.group_planes // 2, self.group_planes // 2, self.group_planes],
            dim=2
        )

        # Position embeddings
        all_embeddings = torch.index_select(
            self.relative, 1, self.flatten_index
        ).view(self.group_planes * 2, self.kernel_size, self.kernel_size)

        q_embedding, k_embedding, v_embedding = torch.split(
            all_embeddings,
            [self.group_planes // 2, self.group_planes // 2, self.group_planes],
            dim=0
        )

        # Compute full attention in one shot — no chunking needed at s=0.125 with 256x256 input
        # At layer1: H=128, at layer2: H=64 — both fit comfortably in VRAM at this scale
        qk = torch.einsum('bgci,bgcj->bgij', q, k)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)

        similarity = F.softmax(self.bn_similarity(qk + qr + kr), dim=-1)

        # Compute output
        sv  = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)

        stacked_output = torch.cat([sv, sve], dim=-1)
        output = self.bn_output(
            stacked_output.reshape(N * W, self.out_planes * 2, H)
        ).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        # Restore spatial layout
        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))