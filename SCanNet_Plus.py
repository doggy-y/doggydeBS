"""
SCanNet-Plus: 基于ScanNet的优化版本
优化点：
1. 添加通道注意力模块(Channel Attention)增强重要特征
2. 添加边缘辅助分支(Edge Auxiliary)改善边界检测
作者：[你的名字]
日期：2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils.misc import initialize_weights
from models.CSWin_Transformer import mit


# ==================== 优化模块1: 通道注意力模块 ====================
class ChannelAttention(nn.Module):
    """
    通道注意力模块
    通过全局平均池化和全连接层学习通道间的依赖关系
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        hidden_dim = max(in_planes // ratio, 16)  # 确保隐藏层维度至少为16

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hidden_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


# ==================== 优化模块2: 边缘辅助分支 ====================
class EdgeAuxiliary(nn.Module):
    """
    边缘辅助检测模块
    使用可学习的Sobel算子提取边缘特征
    """
    def __init__(self, in_channels=128):
        super(EdgeAuxiliary, self).__init__()

        # 可学习的Sobel算子
        self.sobel_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.sobel_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)

        # 初始化Sobel权重
        self._init_sobel_weights()

        # 边缘特征融合
        self.edge_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 边缘预测头
        self.edge_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def _init_sobel_weights(self):
        """初始化Sobel算子权重"""
        # Sobel X算子
        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        # Sobel Y算子
        sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        # 为每个通道设置权重（groups卷积的权重形状是 [in_channels, 1, kH, kW]）
        with torch.no_grad():
            for i in range(self.sobel_x.in_channels):
                self.sobel_x.weight[i, 0, :, :] = sobel_x_kernel
                self.sobel_y.weight[i, 0, :, :] = sobel_y_kernel

            # 冻结Sobel权重（可选，如果不希望网络修改Sobel算子）
            # self.sobel_x.weight.requires_grad = False
            # self.sobel_y.weight.requires_grad = False

    def forward(self, x):
        # 提取边缘特征
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)

        # 融合边缘特征
        edge_feat = torch.cat([edge_x, edge_y], dim=1)
        edge_feat = self.edge_fusion(edge_feat)

        # 边缘预测
        edge_map = self.edge_head(edge_feat)

        # 将边缘特征加回原特征
        out = x + edge_feat

        return out, edge_map


# ==================== 基础模块（来自原始ScanNet） ====================
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class _DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels, scale_ratio=1):
        super(_DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2)
        in_channels = in_channels_high + in_channels_low//scale_ratio
        self.transit = nn.Sequential(
            conv1x1(in_channels_low, in_channels_low//scale_ratio),
            nn.BatchNorm2d(in_channels_low//scale_ratio),
            nn.ReLU(inplace=True) )
        self.decode = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) )

    def forward(self, x, low_feat):
        x = self.up(x)
        low_feat = self.transit(low_feat)
        x = torch.cat((x, low_feat), dim=1)
        x = self.decode(x)
        return x

class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet34(pretrained=pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        if in_channels>3:
          newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])

        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128), nn.ReLU())
        initialize_weights(self.head)

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# ==================== SCanNet-Plus 主模型 ====================
class SCanNetPlus(nn.Module):
    """
    SCanNet-Plus: 优化版本
    相比原始SCanNet的改进：
    1. 在变化检测分支添加通道注意力
    2. 在解码器中添加边缘辅助分支
    """
    def __init__(self, in_channels=3, num_classes=7, input_size=512):
        super(SCanNetPlus, self).__init__()
        feat_size = input_size//4

        # Backbone
        self.FCN = FCN(in_channels, pretrained=True)

        # 变化检测分支
        self.resCD = self._make_layer(ResBlock, 256, 128, 6, stride=1)

        # Transformer跨时相交互
        self.transformer = mit(img_size=feat_size, in_chans=128*3, embed_dim=128*3)

        # 解码器
        self.DecCD = _DecoderBlock(128, 128, 128, scale_ratio=2)
        self.Dec1  = _DecoderBlock(128, 64,  128)
        self.Dec2  = _DecoderBlock(128, 64,  128)

        # ==================== 新增优化模块 ====================
        # 1. 在变化检测特征上添加通道注意力
        self.ca_cd = ChannelAttention(128, ratio=16)

        # 2. 在解码器后添加边缘辅助模块
        self.edge_aux_A = EdgeAuxiliary(128)
        self.edge_aux_B = EdgeAuxiliary(128)
        self.edge_aux_CD = EdgeAuxiliary(128)
        # ====================================================

        # 分类器
        self.classifierA = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierB = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierCD = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        # 初始化权重
        initialize_weights(self.Dec1, self.Dec2, self.classifierA, self.classifierB,
                          self.resCD, self.DecCD, self.classifierCD)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def base_forward(self, x):
        x = self.FCN.layer0(x)
        x = self.FCN.maxpool(x)
        x_low = self.FCN.layer1(x)
        x = self.FCN.layer2(x_low)
        x = self.FCN.layer3(x)
        x = self.FCN.layer4(x)
        x = self.FCN.head(x)
        return x, x_low

    def CD_forward(self, x1, x2):
        b,c,h,w = x1.size()
        x = torch.cat([x1,x2], 1)
        xc = self.resCD(x)
        return x1, x2, xc

    def forward(self, x1, x2):
        x_size = x1.size()

        # 特征提取
        x1, x1_low = self.base_forward(x1)
        x2, x2_low = self.base_forward(x2)

        # 变化检测特征提取
        x1, x2, xc = self.CD_forward(x1, x2)

        # 解码
        x1 = self.Dec1(x1, x1_low)
        x2 = self.Dec2(x2, x2_low)
        xc_low = torch.cat([x1_low, x2_low], 1)
        xc = self.DecCD(xc, xc_low)

        # Transformer跨时相交互
        x = torch.cat([x1, x2, xc], 1)
        x = self.transformer(x)
        x1 = x[:, 0:128, :, :]
        x2 = x[:, 128:256, :, :]
        xc = x[:, 256:, :, :]

        # ==================== 应用优化模块 ====================
        # 1. 在变化检测特征上应用通道注意力
        xc = self.ca_cd(xc)

        # 2. 边缘辅助分支（增强特征并输出边缘预测）
        x1, edge_A = self.edge_aux_A(x1)
        x2, edge_B = self.edge_aux_B(x2)
        xc, edge_CD = self.edge_aux_CD(xc)
        # ====================================================

        # 分类
        out1 = self.classifierA(x1)
        out2 = self.classifierB(x2)
        change = self.classifierCD(xc)

        # 上采样
        out_change = F.interpolate(change, x_size[2:], mode='bilinear', align_corners=False)
        out1 = F.interpolate(out1, x_size[2:], mode='bilinear', align_corners=False)
        out2 = F.interpolate(out2, x_size[2:], mode='bilinear', align_corners=False)

        # 边缘预测也上采样（用于计算边缘损失）
        edge_A = F.interpolate(edge_A, x_size[2:], mode='bilinear', align_corners=False)
        edge_B = F.interpolate(edge_B, x_size[2:], mode='bilinear', align_corners=False)
        edge_CD = F.interpolate(edge_CD, x_size[2:], mode='bilinear', align_corners=False)

        # 返回主预测和边缘辅助预测
        return out_change, out1, out2, edge_A, edge_B, edge_CD


# ==================== 测试代码 ====================
if __name__ == '__main__':
    # 测试模型
    model = SCanNetPlus(in_channels=3, num_classes=7, input_size=512)
    model.eval()

    # 测试前向传播
    x1 = torch.randn(2, 3, 512, 512)
    x2 = torch.randn(2, 3, 512, 512)

    with torch.no_grad():
        out_change, out1, out2, edge_A, edge_B, edge_CD = model(x1, x2)

    print(f"Input shape: {x1.shape}")
    print(f"Change output shape: {out_change.shape}")
    print(f"Semantic A output shape: {out1.shape}")
    print(f"Semantic B output shape: {out2.shape}")
    print(f"Edge A output shape: {edge_A.shape}")
    print(f"Edge B output shape: {edge_B.shape}")
    print(f"Edge CD output shape: {edge_CD.shape}")

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
