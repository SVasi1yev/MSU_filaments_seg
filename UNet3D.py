import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np


class Unet3DModel(nn.Module):
    def __init__(self, in_channels, out_channels, model_depth=3, final_activation="sigmoid"):
        super(Unet3DModel, self).__init__()
        self.encoder = EncoderBlock(in_channels=in_channels, model_depth=model_depth)
        self.decoder = DecoderBlock(out_channels=out_channels, model_depth=model_depth)
        self.final_activation = nn.Sigmoid() \
            if final_activation == "sigmoid" \
            else nn.Softmax()

    def forward(self, x):
        x, downsampling_features = self.encoder(x)
        x = self.decoder(x, downsampling_features)
        x = self.final_activation(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                                stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(num_features=out_channels)

    def forward(self, x):
        x = self.batch_norm(self.conv3d(x))
        x = F.relu(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, model_depth=4, pool_size=2):
        super(EncoderBlock, self).__init__()
        self.root_feat_maps = 16
        self.num_conv_blocks = 2
        self.module_dict = nn.ModuleDict()
        for depth in range(model_depth):
            feat_map_channels = 2 ** (depth + 1) * self.root_feat_maps
            for i in range(self.num_conv_blocks):
                if depth == 0:
                    self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
                else:
                    self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
            if depth == model_depth - 1:
                break
            else:
                self.pooling = nn.MaxPool3d(kernel_size=pool_size, stride=2, padding=0)
                self.module_dict["max_pooling_{}".format(depth)] = self.pooling

    def forward(self, x):
        down_sampling_features = []
        for k, op in self.module_dict.items():
            if k.startswith("conv"):
                x = op(x)
                # print('Encoder: ', x.size())
                if k.endswith("1"):
                    down_sampling_features.append(x)
            elif k.startswith("max_pooling"):
                x = op(x)
                # print('Encoder: ', x.size())

        return x, down_sampling_features


class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=2, padding=1, output_padding=1):
        super(ConvTranspose, self).__init__()
        self.conv3d_transpose = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=k_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )

    def forward(self, x):
        return self.conv3d_transpose(x)


class DecoderBlock(nn.Module):
    def __init__(self, out_channels, model_depth=4):
        super(DecoderBlock, self).__init__()
        self.num_conv_blocks = 2
        self.num_feat_maps = 16
        self.module_dict = nn.ModuleDict()

        for depth in range(model_depth - 2, -1, -1):
            feat_map_channels = 2 ** (depth + 1) * self.num_feat_maps
            self.deconv = ConvTranspose(in_channels=feat_map_channels * 4, out_channels=feat_map_channels * 4)
            self.module_dict["deconv_{}".format(depth)] = self.deconv
            for i in range(self.num_conv_blocks):
                if i == 0:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 6, out_channels=feat_map_channels * 2)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
                else:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=feat_map_channels * 2)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
            if depth == 0:
                self.final_conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=out_channels)
                self.module_dict["final_conv"] = self.final_conv

    def forward(self, x, down_sampling_features):
        for k, op in self.module_dict.items():
            if k.startswith("deconv"):
                # print(x.size())
                x = op(x)
                # print(x.size())
                # print(down_sampling_features[int(k[-1])].size())
                x = torch.cat((down_sampling_features[int(k[-1])], x), dim=1)
            elif k.startswith("conv"):
                x = op(x)
            else:
                x = op(x)
        return x


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        # smooth factor
        self.epsilon = epsilon

    def forward(self, targets, logits):
        batch_size = targets.size(0)
        # log_prob = torch.sigmoid(logits)
        logits = logits.view(batch_size, -1).type(torch.FloatTensor)
        targets = targets.view(batch_size, -1).type(torch.FloatTensor)
        intersection = (logits * targets).sum(-1)
        dice_score = 2. * intersection / ((logits + targets).sum(-1) + self.epsilon)
        # dice_score = 1 - dice_score.sum() / batch_size
        return torch.mean(1. - dice_score)


class UNetDataset(torch.utils.data.Dataset):
    def __init__(self, input_, mask, stride, size):
        self.input_ = input_
        self.mask = mask
        self.stride = stride
        self.size = size
        self.coords = []
        self.data_size = input_.shape
        for k in range(0, self.data_size[0], stride):
            for l in range(0, self.data_size[1], stride):
                for m in range(0, self.data_size[2], stride):
                    if (self.data_size[0] - k * size) >= size \
                         and (self.data_size[1] - l * size) >= size \
                         and (self.data_size[2] - m * size) >= size:
                        self.coords.append((k, l, m))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        c = self.coords[idx]
        return self.input_[None, c[0]: c[0]+self.size, c[1]: c[1]+self.size, c[2]: c[2]+self.size], \
            self.mask[None, c[0]: c[0] + self.size, c[1]: c[1] + self.size, c[2]: c[2] + self.size]


class Trainer(object):
    def __init__(self, net, optimizer, criterion, n_epochs):
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.n_epochs = n_epochs

    def train(self, data_loader):
        for epoch in range(self.n_epochs):
            start_time = time.time()
            train_losses = []
            for input_, mask in data_loader:
                input_ = torch.DoubleTensor(input_)
                # mask = torch.IntTensor(mask)

                self.optimizer.zero_grad()

                logits = self.net(input_)
                mask = mask.type(torch.int8)
                # print(logits)
                # print(mask)
                print(mask.sum())
                loss = self.criterion(logits, mask)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
            end_time = time.time()
            print("Epoch {}, training loss {:.4f}, time {:.2f}".format(epoch, np.mean(train_losses),
                                                                       end_time - start_time))


# if __name__ == "__main__":
#     inputs = torch.randn(1, 1, 96, 96, 96)
#     model = Unet3DModel(in_channels=1, out_channels=1)
#     x = model(inputs)
#     print(model)