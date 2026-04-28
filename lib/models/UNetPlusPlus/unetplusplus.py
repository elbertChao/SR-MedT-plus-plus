import torch
import torch.nn as nn

class StandardUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StandardUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, deep_supervision=False):
        super(UNetPlusPlus, self).__init__()
        self.deep_supervision = deep_supervision
        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1_1 = StandardUnit(in_channels, nb_filter[0])
        self.conv2_1 = StandardUnit(nb_filter[0], nb_filter[1])
        self.conv3_1 = StandardUnit(nb_filter[1], nb_filter[2])
        self.conv4_1 = StandardUnit(nb_filter[2], nb_filter[3])
        self.conv5_1 = StandardUnit(nb_filter[3], nb_filter[4])

        self.up1_2 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.conv1_2 = StandardUnit(nb_filter[0] * 2, nb_filter[0])

        self.up2_2 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.conv2_2 = StandardUnit(nb_filter[1] * 2, nb_filter[1])

        self.up3_2 = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)
        self.conv3_2 = StandardUnit(nb_filter[2] * 2, nb_filter[2])

        self.up4_2 = nn.ConvTranspose2d(nb_filter[4], nb_filter[3], kernel_size=2, stride=2)
        self.conv4_2 = StandardUnit(nb_filter[3] * 2, nb_filter[3])

        self.up1_3 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.conv1_3 = StandardUnit(nb_filter[0] * 3, nb_filter[0])

        self.up2_3 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.conv2_3 = StandardUnit(nb_filter[1] * 3, nb_filter[1])

        self.up3_3 = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)
        self.conv3_3 = StandardUnit(nb_filter[2] * 3, nb_filter[2])

        self.up1_4 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.conv1_4 = StandardUnit(nb_filter[0] * 4, nb_filter[0])

        self.up2_4 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.conv2_4 = StandardUnit(nb_filter[1] * 4, nb_filter[1])

        self.up1_5 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.conv1_5 = StandardUnit(nb_filter[0] * 5, nb_filter[0])

        # raw logits — no sigmoid applied, compatible with cross-entropy and focal loss
        self.output_1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self.output_2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self.output_3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self.output_4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        x1_1 = self.conv1_1(x)
        x2_1 = self.conv2_1(self.pool(x1_1))
        x3_1 = self.conv3_1(self.pool(x2_1))
        x4_1 = self.conv4_1(self.pool(x3_1))
        x5_1 = self.conv5_1(self.pool(x4_1))

        x1_2 = self.conv1_2(torch.cat([x1_1, self.up1_2(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_1, self.up2_2(x3_1)], dim=1))
        x3_2 = self.conv3_2(torch.cat([x3_1, self.up3_2(x4_1)], dim=1))
        x4_2 = self.conv4_2(torch.cat([x4_1, self.up4_2(x5_1)], dim=1))

        x1_3 = self.conv1_3(torch.cat([x1_1, x1_2, self.up1_3(x2_2)], dim=1))
        x2_3 = self.conv2_3(torch.cat([x2_1, x2_2, self.up2_3(x3_2)], dim=1))
        x3_3 = self.conv3_3(torch.cat([x3_1, x3_2, self.up3_3(x4_2)], dim=1))

        x1_4 = self.conv1_4(torch.cat([x1_1, x1_2, x1_3, self.up1_4(x2_3)], dim=1))
        x2_4 = self.conv2_4(torch.cat([x2_1, x2_2, x2_3, self.up2_4(x3_3)], dim=1))

        x1_5 = self.conv1_5(torch.cat([x1_1, x1_2, x1_3, x1_4, self.up1_5(x2_4)], dim=1))

        out1 = self.output_1(x1_2)
        out2 = self.output_2(x1_3)
        out3 = self.output_3(x1_4)
        out4 = self.output_4(x1_5)

        if self.deep_supervision:
            # Ordered Best to Worst so y_out[0] is the final prediction
            return (out4, out3, out2, out1) 
        else:
            return out4