# !/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# Copy from https://github.com/Halmstad-University/SalsaNext

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rangeview_conv import RangeViewConv2d


class ResContextBlock(nn.Module):
    def __init__(self, custom_padding, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        self.conv1 = RangeViewConv2d(custom_padding, in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = RangeViewConv2d(custom_padding, out_filters, out_filters, (3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.InstanceNorm2d(out_filters)

        self.conv3 = RangeViewConv2d(custom_padding, out_filters, out_filters, (3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.InstanceNorm2d(out_filters)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        output = shortcut + resA2
        return output

class ResBlock(nn.Module):
    def __init__(self, custom_padding, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = RangeViewConv2d(custom_padding, in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = RangeViewConv2d(custom_padding, in_filters, out_filters, kernel_size=(3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.InstanceNorm2d(out_filters)

        self.conv3 = RangeViewConv2d(custom_padding, out_filters, out_filters, kernel_size=(3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.InstanceNorm2d(out_filters)

        self.conv4 = RangeViewConv2d(custom_padding, out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.InstanceNorm2d(out_filters)

        self.conv5 = RangeViewConv2d(custom_padding, out_filters * 3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.InstanceNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.Identity()

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        resA = self.conv4(resA2)
        resA = self.act4(resA)
        resA3 = self.bn3(resA)

        concat = torch.cat((resA1, resA2, resA3), dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn4(resA)
        resA = shortcut + resA

        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)

            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB, torch.empty_like(resB)

class UpBlock(nn.Module):
    def __init__(self, custom_padding, in_filters, out_filters, dropout_rate, drop_out=True):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = RangeViewConv2d(custom_padding, in_filters // 4 + 2 * out_filters, out_filters, (3, 3), padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.InstanceNorm2d(out_filters)

        self.conv2 = RangeViewConv2d(custom_padding, out_filters, out_filters, (3, 3), dilation=2, padding=2)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.InstanceNorm2d(out_filters)

        self.conv3 = RangeViewConv2d(custom_padding, out_filters, out_filters, (2, 2), dilation=2, padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.InstanceNorm2d(out_filters)

        self.conv4 = RangeViewConv2d(custom_padding, out_filters * 3, out_filters, kernel_size=(1, 1))
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.InstanceNorm2d(out_filters)

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

        self.ps2 = nn.PixelShuffle(2)
        self.id = nn.Identity()

    def forward(self, x, skip):
        upA = self.ps2(x)
        if self.drop_out:
            upA = self.dropout1(upA)

        upB = torch.cat((upA, skip), dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB)
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = self.conv2(upE1)
        upE = self.act2(upE)
        upE2 = self.bn2(upE)

        upE = self.conv3(upE2)
        upE = self.act3(upE)
        upE3 = self.bn3(upE)

        concat = torch.cat((upE1, upE2, upE3), dim=1)
        upE = self.conv4(concat)
        upE = self.act4(upE)
        upE = self.bn4(upE)
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE

class SalsaNext(nn.Module):
    def __init__(self, custom_padding=False, multi_scale=1, deep_supervision=1, in_channels=8, nclasses=20, base_channels=32):
        super(SalsaNext, self).__init__()

        self.multi_scale = multi_scale
        self.deep_supervision = deep_supervision

        self.base_channels = base_channels
        self.dropout_ratio = 0.2
        self.downCntx = ResContextBlock(custom_padding, in_channels, base_channels)
        self.downCntx2 = ResContextBlock(custom_padding, base_channels, base_channels)
        self.downCntx3 = ResContextBlock(custom_padding, base_channels, base_channels)

        self.resBlock1 = ResBlock(custom_padding, base_channels, 2 * base_channels, self.dropout_ratio, pooling=True, drop_out=False)
        self.resBlock2 = ResBlock(custom_padding, 2 * base_channels, 2 * 2 * base_channels, self.dropout_ratio, pooling=True)
        self.resBlock3 = ResBlock(custom_padding, 2 * 2 * base_channels, 2 * 4 * base_channels, self.dropout_ratio, pooling=True)
        self.resBlock4 = ResBlock(custom_padding, 2 * 4 * base_channels, 2 * 4 * base_channels, self.dropout_ratio, pooling=True)

        bottleneck_channels = 2 * 4 * base_channels
        self.resBlock5 = ResBlock(custom_padding, bottleneck_channels, bottleneck_channels, self.dropout_ratio, pooling=False)

        multi_channels = [bottleneck_channels, 4 * base_channels, 4 * base_channels, 2 * base_channels, base_channels]
        self.bottleneck_channels = multi_channels[:self.multi_scale]

        self.upBlock1 = UpBlock(custom_padding, 2 * 4 * base_channels, 4 * base_channels, self.dropout_ratio)
        self.upBlock2 = UpBlock(custom_padding, 4 * base_channels, 4 * base_channels, self.dropout_ratio)
        self.upBlock3 = UpBlock(custom_padding, 4 * base_channels, 2 * base_channels, self.dropout_ratio)
        self.upBlock4 = UpBlock(custom_padding, 2 * base_channels, base_channels, self.dropout_ratio, drop_out=False)

        self.ps2 = nn.PixelShuffle(2)


    def forward(self, x):
        downCntx = self.downCntx(x)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)

        down0c, down0b = self.resBlock1(downCntx)
        down1c, down1b = self.resBlock2(down0c)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down5c, _ = self.resBlock5(down3c)

        # print("down0c, down0b", down0c.shape, down0b.shape)
        # print("down1c, down1b", down1c.shape, down1b.shape)
        # print("down2c, down2b", down2c.shape, down2b.shape)
        # print("down3c, down3b", down3c.shape, down3b.shape)
        # print("down5c", down5c.shape)
        
        up4e = self.upBlock1(down5c,down3b)
        up3e = self.upBlock2(up4e, down2b)
        up2e = self.upBlock3(up3e, down1b)
        up1e = self.upBlock4(up2e, down0b)

        # print("up4e", up4e.shape)
        # print("up3e", up3e.shape)
        # print("up2e", up2e.shape)
        # print("up1e", up1e.shape)

        multi_features = [down5c, up4e, up3e, up2e, up1e]
        bottleneck_features = multi_features[:self.multi_scale]
        multi_features = [down5c, self.ps2(up4e), self.ps2(up3e), up1e]
        features = multi_features[-self.deep_supervision:]
        return bottleneck_features, features