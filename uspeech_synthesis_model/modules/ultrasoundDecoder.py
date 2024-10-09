import torch
import torch.nn as nn
import torch.nn.functional as F

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(1, 1), stride=(1, 1),
                              bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=(1, 1), stride=(1, 1),
                              bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)


    def forward(self, input, pool_size=(1, 1), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x

class UNetDecoder(nn.Module):
    def __init__(self):
        super(UNetDecoder, self).__init__()

        self.up_conv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=1, stride=1)
        self.up_block1 = ConvBlock(in_channels=2048, out_channels=1024)

        self.up_conv2 = nn.ConvTranspose2d(1024, 512, kernel_size=1, stride=1)
        self.up_block2 = ConvBlock(in_channels=1024, out_channels=512)

        self.up_conv3 = nn.ConvTranspose2d(512, 256, kernel_size=1, stride=1)
        self.up_block3 = ConvBlock(in_channels=512, out_channels=256)

        self.up_conv4 = nn.ConvTranspose2d(256, 128, kernel_size=1, stride=1)
        self.up_block4 = ConvBlock(in_channels=256, out_channels=128)

        self.up_conv5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_block5 = ConvBlock(in_channels=128, out_channels=64)

        self.projector = nn.ConvTranspose2d(64, 1, kernel_size=(2, 1), stride=(2, 1), padding=(0, 25), dilation=(1, 1))

    def forward(self, x, encoder_features):
        x = self.up_conv1(x)
        x = torch.cat([x, encoder_features[4]], dim=1)
        x = self.up_block1(x)

        x = self.up_conv2(x)
        x = torch.cat([x, encoder_features[3]], dim=1)
        x = self.up_block2(x)

        x = self.up_conv3(x)
        x = torch.cat([x, encoder_features[2]], dim=1)
        x = self.up_block3(x)

        x = self.up_conv4(x)
        x = torch.cat([x, encoder_features[1]], dim=1)
        x = self.up_block4(x)

        x = self.up_conv5(x)
        if x.size() != encoder_features[0].size():
            padded_x = self.pad_to_match(x, encoder_features[0])
            x = torch.cat([padded_x, encoder_features[0]], dim=1)
        else:
            x = torch.cat([x, encoder_features[0]], dim=1)
        x = self.up_block5(x)
        output = self.projector(x)
        return output

    def pad_to_match(self, tensor, target):
        _, _, h, w = target.size()
        _, _, h_tensor, w_tensor = tensor.size()
        h_pad = (h - h_tensor) // 2
        w_pad = (w - w_tensor) // 2
        padding = (w_pad, w_pad, h_pad + 1, h_pad)
        return F.pad(tensor, padding, mode='reflect')