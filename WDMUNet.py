import torch
from torchvision import models
import torch.nn as nn
from torch.nn import functional as F
from pytorch_wavelets import DWTForward
from torchvision.models import ResNet34_Weights


class Up_Block_sk(nn.Module):
    def __init__(self, in_channels_up, in_channels_skip, out_channels):
        super(Up_Block_sk, self).__init__()
        # ... 其他层定义 ...
        self.upsample = nn.ConvTranspose2d(in_channels_up, out_channels // 2, kernel_size=2, stride=2)
        self.conv_block = nn.Sequential(
            nn.Conv2d(out_channels // 2 + in_channels_skip, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_up, x_skip):
        x_up = self.upsample(x_up)

        x = torch.cat([x_up, x_skip], dim=1)
        x = self.conv_block(x)
        return x
class DAM(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(DAM, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):

        tgt = self.norm1(tgt)
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + tgt2

        tgt = self.norm2(tgt)
        tgt2 = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = tgt + tgt2

        tgt = self.norm3(tgt)
        tgt2 = self.mlp(tgt)
        tgt = tgt + tgt2

        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, mlp_dim, num_queries, num_classes, img_size):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            DAM(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)
        ])
        self.query_embedding = nn.Embedding(num_queries, embed_dim)
        self.linear_class = nn.Linear(embed_dim, num_classes)
        self.img_size = img_size

    def forward(self, memory, mask=None):
        bs, _, h, w = memory.size()
        query_pos = self.query_embedding.weight.unsqueeze(1).repeat(1, bs, 1)  # 初始化查询


        for layer in self.layers:
            query_pos = layer(query_pos, memory.flatten(2).permute(2, 0, 1))  # 与CNN特征的交叉注意力

        logits = self.linear_class(query_pos)

        logits = logits.permute(1, 2, 0).view(bs, -1, h, w)

        return logits


class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__()
        self.query = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True)
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True)
        )
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        # compress x: [B,C,H,W]-->[B,H*W,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)   # 转置 [B,H*W,C]
        proj_key = self.key(x).view(B, -1, W * H)
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)   # [B,H*W,H*W]
        proj_value = self.value(x).view(B, -1, H * W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))   # [B,C,H*W] * [B,H*W,H*W]
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out


class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)
        affinity = torch.matmul(proj_query, proj_key)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out

class WFEM(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, in_channels_3):
        super(WFEM, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)


        self.attention_s = nn.ModuleList([
            SpatialAttentionBlock(in_channels_1),
            SpatialAttentionBlock(in_channels_2),
            SpatialAttentionBlock(in_channels_3)

        ])

        self.attention_c = nn.ModuleList([
            ChannelAttentionBlock(in_channels_1),
            ChannelAttentionBlock(in_channels_2),
            ChannelAttentionBlock(in_channels_3)
        ])

        self.conv_outs = nn.ModuleList([
            nn.Conv2d(3 * in_channels_1, in_channels_1, 1, 1, 0),
            nn.Conv2d(3 * in_channels_2, in_channels_2, 1, 1, 0),
            nn.Conv2d(3 * in_channels_3, in_channels_3, 1, 1, 0)
        ])

    def forward(self, f1, f2, f3):

        features = [f1, f2, f3]
        outputs = []

        for i in range(3):

            yL, yH = self.wt(features[i])
            yH_sum = yH[0].sum(dim=2, keepdim=False)

            yL = self.attention_c[i](yL)

            attn_f = self.attention_s[i](yH_sum)

            out = (self.conv_outs[i](torch.cat([self.upsample(yL), self.upsample(attn_f), features[i]], dim=1)))
            outputs.append(out)

        return outputs[0], outputs[1], outputs[2]


class UnetDsv(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv, self).__init__()
        self.dsv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(size=scale_factor, mode='bilinear'), )

    def forward(self, input):
        return self.dsv(input)


class WDM_UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, img_size=224):
        super().__init__()
        self.n_classes = n_classes
        resnet = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        filters_resnet = [64, 64, 128, 256, 512]
        filters_decoder = [32, 64, 128, 256, 512]

        self.bridge = WFEM(64, 128, 256)

        self.Conv1 = nn.Sequential(
            nn.Conv2d(n_channels, filters_resnet[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(filters_resnet[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters_resnet[0], filters_resnet[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(filters_resnet[0]),
            nn.ReLU(inplace=True)
        )
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.Conv2 = resnet.layer1
        self.Conv3 = resnet.layer2
        self.Conv4 = resnet.layer3
        self.Conv5 = resnet.layer4

        self.Up5 = Up_Block_sk(filters_resnet[4],  filters_resnet[3], filters_decoder[3])
        self.Up4 = Up_Block_sk(filters_decoder[3], filters_resnet[2], filters_decoder[2])
        self.Up3 = Up_Block_sk(filters_decoder[2], filters_resnet[1], filters_decoder[1])

        num_queries = 196
        embed_dim = 512
        num_heads = 8
        num_layers = 2
        mlp_dim = 1024
        self.transformer_decoder = TransformerDecoder(num_layers, embed_dim, num_heads, mlp_dim, num_queries, n_classes, img_size)

        self.pred = nn.Sequential(
            nn.Conv2d(filters_decoder[1], filters_decoder[1]//2, kernel_size=1),
            nn.BatchNorm2d(filters_decoder[1]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters_decoder[1]//2, n_classes, kernel_size=1),
        )
        self.last_activation = nn.Sigmoid()
        self.final_fusion = ScaleAwareChainFusionNoConv()
        # Deep supervision
        filters = [64, 128, 256]
        self.out_size = (224, 224)
        self.dsv4 = UnetDsv(in_size=filters[2], out_size=64, scale_factor=self.out_size)
        self.dsv3 = UnetDsv(in_size=filters[1], out_size=64, scale_factor=self.out_size)
        self.dsv2 = UnetDsv(in_size=filters[0], out_size=64, scale_factor=self.out_size)


        self.c4 = nn.Conv2d(n_classes, 256, kernel_size=1, stride=1, padding=0)
        self.c3 = nn.Conv2d(n_classes, 128, kernel_size=1, stride=1, padding=0)
        self.c2 = nn.Conv2d(n_classes, 64, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 编码器
        e1 = self.Conv1(x)
        e1_maxp = self.Maxpool(e1)  # 64
        e2 = self.Conv2(e1_maxp)  # 64
        e3 = self.Conv3(e2)  # 128
        e4 = self.Conv4(e3)  # 256
        e5 = self.Conv5(e4)  # 512

        c2, c3, c4 = self.bridge(e2, e3, e4)

        transformer_output = self.transformer_decoder(e5)  # [batch_size, n_classes, 14, 14]

        dt4 = self.c4(F.interpolate(transformer_output, size=(28, 28), mode='bilinear',
                                                     align_corners=False))
        dt3 = self.c3(F.interpolate(transformer_output, size=(56, 56), mode='bilinear',
                                                     align_corners=False))
        dt2 = self.c2(F.interpolate(transformer_output, size=(112, 112), mode='bilinear',
                                                     align_corners=False))
        d4 = self.Up5(e5, c4) + dt4  # 256
        d3 = self.Up4(d4, c3) + dt3  # 128
        d2 = self.Up3(d3, c2) + dt2  # 64

        dsv4 = self.dsv4(d4)
        dsv3 = self.dsv3(d3)
        dsv2 = self.dsv2(d2)

        x_fin = self.final_fusion(dsv4, dsv3, dsv2)
        if self.n_classes == 1:
            out = self.last_activation(self.pred(x_fin))
        else:
            out = self.pred(x_fin)
        return out

class PoolingAttentionFusion(nn.Module):
    def __init__(self):
        super(PoolingAttentionFusion, self).__init__()

    def forward(self, feat1, feat2):
        # Spatial average pooling (global)
        pool1 = feat1.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        pool2 = feat2.mean(dim=(2, 3), keepdim=True)

        # Stack and apply softmax over the 2 features for each channel
        pooled = torch.cat([pool1, pool2], dim=1)     # [B, 2C, 1, 1]
        weights = F.softmax(pooled.view(feat1.size(0), 2, -1), dim=1)  # [B, 2, C]
        weights = weights.view(feat1.size(0), 2, feat1.size(1), 1, 1)  # [B, 2, C, 1, 1]

        # Split weights
        w1 = weights[:, 0]
        w2 = weights[:, 1]

        # Apply attention and fuse
        fused = feat1 * w1 + feat2 * w2
        return fused

class ScaleAwareChainFusionNoConv(nn.Module):
    def __init__(self):
        super(ScaleAwareChainFusionNoConv, self).__init__()
        self.fuse1 = PoolingAttentionFusion()
        self.fuse2 = PoolingAttentionFusion()

    def forward(self, dsv4, dsv3, dsv2):
        f43 = self.fuse1(dsv4, dsv3)   # 先融合 F4 和 F3
        f432 = self.fuse2(f43, dsv2)   # 再融合 F43 和 F2
        return f432
