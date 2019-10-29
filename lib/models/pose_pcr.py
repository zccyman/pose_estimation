import torch
import torch.nn as nn


class CAM(nn.Module):
    def __init__(self, input_channels, input_height, input_width):
        super(CAM, self).__init__()

        middle_channels = input_channels // 4

        self.se_branch = nn.Sequential(
            nn.AdaptiveMaxPool2d((input_height, input_width)),
            nn.Conv2d(input_channels, middle_channels,
                      kernel_size=1, bias=False),
            nn.Conv2d(middle_channels, input_channels,
                      kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # dilated conv branch
        self.dilated_branch_num = 4
        for rate in range(self.dilated_branch_num):
            setattr(self, "dilated_conv_" + str(rate),
                    nn.Conv2d(input_channels, middle_channels,
                              kernel_size=3, stride=1, padding=rate + 1, dilation=rate + 1, bias=False))

        self.hdc_branch = nn.Sequential(
            nn.ConvTranspose2d(input_channels, input_channels,
                               kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.res_branch = nn.Sequential(
            nn.Upsample(scale_factor=1, mode='bilinear', align_corners=True),
            nn.Conv2d(input_channels, input_channels,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        # SE branch
        se_out = self.se_branch(x)

        # HDC branch
        dilated_conv_list = []
        for rate in range(self.dilated_branch_num):
            y = getattr(self, "dilated_conv_" + str(rate))(x)
            dilated_conv_list.append(y)
        hdc_out = torch.cat(dilated_conv_list, dim=1)
        hdc_out = self.hdc_branch(hdc_out)
        hdc_out = hdc_out.mul(se_out)

        # Residual branch
        res_out = self.res_branch(x)

        # output
        out = hdc_out + res_out

        return out


class PCR(nn.Module):
    def __init__(self, input_channels, input_height, input_width, levels_num=4, cascade_num_per_level=4):
        super(PCR, self).__init__()

        self.levels_num = levels_num
        self.cascade_num_per_level = cascade_num_per_level

        # levels
        for i in range(self.levels_num):
            cascade_layers = []
            for j in range(self.cascade_num_per_level):
                cascade_layers.append(
                    CAM(input_channels, input_height, input_width))
            setattr(self, "level_" + str(i), nn.Sequential(*cascade_layers))
            setattr(self, "conv1x1_level_" + str(i),
                    nn.Conv2d(input_channels, input_channels,
                              kernel_size=1, bias=False))

    def forward(self, x):
        outs = []
        for i in range(self.levels_num):
            y = getattr(self, "level_" + str(i))(x)
            y = getattr(self, "conv1x1_level_" + str(i))(x)

            if i == 0:
                outs.append(y)
            else:
                y += outs[i - 1]
                outs.append(y)

        return outs


def test_CAM():
    inputs = torch.randn(2, 64, 512, 512)

    model = CAM(64, 512, 512)

    output = model(inputs)
    print(output.shape)
    print("=> test CAM done ...")


def test_PCR():
    inputs = torch.randn(2, 64, 512, 512)

    model = PCR(64, 512, 512, levels_num=4, cascade_num_per_level=4)

    outputs = model(inputs)
    for output in outputs:
        print(output.shape)
    print("=> test PCR done ...")


if __name__ == "__main__":
    test_CAM()
    test_PCR()
