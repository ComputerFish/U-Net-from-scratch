import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResUNet4(nn.Module):
    def __init__(self, num_classes=2, backbone="resnet50", pretrained=True):
        super(ResUNet4, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # this means model is trainable on all layers
        for param in self.resnet.parameters():
            param.requires_grad = True

        # per layer parameters, if you want to freeze them or something
        # for param in self.resnet.layer4.parameters():
        #     param.requires_grad = False

        self.decoder4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.decoder_sequential_4 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.decoder3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder_sequential_3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.decoder2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_sequential_2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.decoder1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_sequential_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.decoder0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_sequential_0 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.final_convolution = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        encoder1 = self.resnet.maxpool(
            self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x)))
        )
        encoder2 = self.resnet.layer1(encoder1)
        encoder3 = self.resnet.layer2(encoder2)
        encoder4 = self.resnet.layer3(encoder3)
        encoder5 = self.resnet.layer4(encoder4)

        upconv4 = self.decoder4(encoder5)
        skip4 = torch.cat((upconv4, encoder4), dim=1)
        decoder4 = self.decoder_sequential_4(skip4)

        upconv3 = self.decoder3(decoder4)
        skip3 = torch.cat((upconv3, encoder3), dim=1)
        decoder3 = self.decoder_sequential_3(skip3)

        upconv2 = self.decoder2(decoder3)
        skip2 = torch.cat((upconv2, encoder2), dim=1)
        decoder2 = self.decoder_sequential_2(skip2)

        upconv1 = self.decoder1(decoder2)
        # skip1 = torch.cat((upconv1, encoder1), dim=1)
        decoder1 = self.decoder_sequential_1(upconv1)

        upconv0 = self.decoder0(decoder1)
        decoder0 = self.decoder_sequential_0(upconv0)

        out = self.final_convolution(decoder0)

        return out
    

class ResUNet3(nn.Module):
    def __init__(self, num_classes=2, backbone="resnet50", pretrained=True):
        super(ResUNet3, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # this means model is frozen
        # for param in self.resnet.parameters():
        #     param.requires_grad = False

        # per layer parameters, if you want to freeze them or something
        for param in self.resnet.layer4.parameters():
            param.requires_grad = False

        self.decoder4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.decoder_sequential_4 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.decoder3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder_sequential_3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.decoder2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_sequential_2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.decoder1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_sequential_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.decoder0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_sequential_0 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.final_convolution = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        encoder1 = self.resnet.maxpool(
            self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x)))
        )
        encoder2 = self.resnet.layer1(encoder1)
        encoder3 = self.resnet.layer2(encoder2)
        encoder4 = self.resnet.layer3(encoder3)
        encoder5 = self.resnet.layer4(encoder4)

        upconv4 = self.decoder4(encoder5)
        skip4 = torch.cat((upconv4, encoder4), dim=1)
        decoder4 = self.decoder_sequential_4(skip4)

        upconv3 = self.decoder3(decoder4)
        skip3 = torch.cat((upconv3, encoder3), dim=1)
        decoder3 = self.decoder_sequential_3(skip3)

        upconv2 = self.decoder2(decoder3)
        skip2 = torch.cat((upconv2, encoder2), dim=1)
        decoder2 = self.decoder_sequential_2(skip2)

        upconv1 = self.decoder1(decoder2)
        # skip1 = torch.cat((upconv1, encoder1), dim=1)
        decoder1 = self.decoder_sequential_1(upconv1)

        upconv0 = self.decoder0(decoder1)
        decoder0 = self.decoder_sequential_0(upconv0)

        out = self.final_convolution(decoder0)

        return out
    

class ResUNet2(nn.Module):
    def __init__(self, num_classes=2, backbone="resnet50", pretrained=True):
        super(ResUNet2, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # this means model is frozen
        for param in self.resnet.parameters():
            param.requires_grad = True

        # per layer parameters, if you want to freeze them or something
        # for param in self.resnet.layer4.parameters():
        #     param.requires_grad = False

        self.decoder4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.decoder_sequential_4 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.decoder3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder_sequential_3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.decoder2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_sequential_2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.decoder1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_sequential_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.decoder0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_sequential_0 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.final_convolution = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        encoder1 = self.resnet.maxpool(
            self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x)))
        )
        encoder2 = self.resnet.layer1(encoder1)
        encoder3 = self.resnet.layer2(encoder2)
        encoder4 = self.resnet.layer3(encoder3)
        encoder5 = self.resnet.layer4(encoder4)

        upconv4 = self.decoder4(encoder5)
        skip4 = torch.cat((upconv4, encoder4), dim=1)
        decoder4 = self.decoder_sequential_4(skip4)

        upconv3 = self.decoder3(decoder4)
        skip3 = torch.cat((upconv3, encoder3), dim=1)
        decoder3 = self.decoder_sequential_3(skip3)

        upconv2 = self.decoder2(decoder3)
        skip2 = torch.cat((upconv2, encoder2), dim=1)
        decoder2 = self.decoder_sequential_2(skip2)

        upconv1 = self.decoder1(decoder2)
        # skip1 = torch.cat((upconv1, encoder1), dim=1)
        decoder1 = self.decoder_sequential_1(upconv1)

        upconv0 = self.decoder0(decoder1)
        decoder0 = self.decoder_sequential_0(upconv0)

        out = self.final_convolution(decoder0)

        return out
    

class ResUNet1(nn.Module):
    def __init__(self, num_classes=2, backbone="resnet50", pretrained=True):
        super(ResUNet1, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # this means model is frozen
        for param in self.resnet.parameters():
            param.requires_grad = False

        # per layer parameters, if you want to freeze them or something
        # for param in self.resnet.layer4.parameters():
        #     param.requires_grad = False

        self.decoder4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.decoder_sequential_4 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.decoder3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder_sequential_3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.decoder2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_sequential_2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.decoder1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_sequential_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.decoder0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_sequential_0 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.final_convolution = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        encoder1 = self.resnet.maxpool(
            self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x)))
        )
        encoder2 = self.resnet.layer1(encoder1)
        encoder3 = self.resnet.layer2(encoder2)
        encoder4 = self.resnet.layer3(encoder3)
        encoder5 = self.resnet.layer4(encoder4)

        upconv4 = self.decoder4(encoder5)
        skip4 = torch.cat((upconv4, encoder4), dim=1)
        decoder4 = self.decoder_sequential_4(skip4)

        upconv3 = self.decoder3(decoder4)
        skip3 = torch.cat((upconv3, encoder3), dim=1)
        decoder3 = self.decoder_sequential_3(skip3)

        upconv2 = self.decoder2(decoder3)
        skip2 = torch.cat((upconv2, encoder2), dim=1)
        decoder2 = self.decoder_sequential_2(skip2)

        upconv1 = self.decoder1(decoder2)
        # skip1 = torch.cat((upconv1, encoder1), dim=1)
        decoder1 = self.decoder_sequential_1(upconv1)

        upconv0 = self.decoder0(decoder1)
        decoder0 = self.decoder_sequential_0(upconv0)

        out = self.final_convolution(decoder0)

        return out


class UNet(nn.Module):

    # def __init__(self, num_classes=2, backbone='resnet50', pretrained=True):
    #     super(ResNetUNet, self).__init__()


    # For basic UNet
    def __init__(self, num_classes=2):
        super(UNet, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.decoder4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder_sequential_4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.decoder3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_sequential_3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.decoder2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_sequential_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.decoder1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_sequential_1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.final_convolution = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        encoder1 = self.encoder1(x)
        pool1 = self.max_pool1(encoder1)
        encoder2 = self.encoder2(pool1)
        pool2 = self.max_pool2(encoder2)
        encoder3 = self.encoder3(pool2)
        pool3 = self.max_pool3(encoder3)
        encoder4 = self.encoder4(pool3)
        pool4 = self.max_pool4(encoder4)

        # Bottleneck
        bottleneck = self.bottleneck(pool4)

        # Decoder
        upconv4 = self.decoder4(bottleneck)
        skip4 = torch.cat((upconv4, encoder4), dim=1)
        decoder4 = self.decoder_sequential_4(skip4)

        upconv3 = self.decoder3(decoder4)
        skip3 = torch.cat((upconv3, encoder3), dim=1)
        decoder3 = self.decoder_sequential_3(skip3)

        upconv2 = self.decoder2(decoder3)
        skip2 = torch.cat((upconv2, encoder2), dim=1)
        decoder2 = self.decoder_sequential_2(skip2)

        upconv1 = self.decoder1(decoder2)
        skip1 = torch.cat((upconv1, encoder1), dim=1)
        decoder1 = self.decoder_sequential_1(skip1)

        # Final upsampling step
        out = self.final_convolution(decoder1)

        return out
