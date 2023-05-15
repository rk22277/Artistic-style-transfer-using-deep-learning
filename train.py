import os
import random
import re
import time
from collections import namedtuple

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch import optim
from torch.utils import model_zoo
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image
from torch.nn import functional


os.makedirs("style", exist_ok=True)
os.makedirs("weights", exist_ok=True)


TRAIN_FLDR = "/Users/rk/Documents/DeepLearning/project/stylizeGAN/style-transfer-GAN/train_imgs"
SAMPLE_FLDR = "/Users/rk/Documents/DeepLearning/project/stylizeGAN/style-transfer-GAN/result_imgs"
STYLE_IMG = "/Users/rk/Documents/DeepLearning/project/stylizeGAN/style-transfer-GAN/static/icons/tree.jpg"
STYLE_MODEL = "/Users/rk/Documents/DeepLearning/project/stylizeGAN/style-transfer-GAN/weights/tree2.pth"


os.makedirs(SAMPLE_FLDR, exist_ok=True)



apply_augmentation = A.Compose([
    A.VerticalFlip(0.5),
    A.HorizontalFlip(0.5),
    A.RandomRotate90(0.5)],
    p=1,
    additional_targets={"image2" : "image"})



class ImageDataset(Dataset):
    
    def __init__(self, path, transform_):
        
        super(ImageDataset, self).__init__()
        self.files = [f'{path}/{file}' for file in os.listdir(path)]
        self.transform = transform_
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        
        file = self.files[i]
        img = Image.open(file).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor


image_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

save_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

style_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



# !wget http://images.cocodataset.org/zips/train2014.zip && unzip train2014.zip && mv train2014 $TRAIN_FLDR
image_data = ImageDataset(TRAIN_FLDR, transform_=image_transform)


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
            
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_extractor = Vgg19(requires_grad=False).to(device)


np.random.seed(42)
torch.manual_seed(42)


def to_numpy(tensor):
    return tensor.cpu().detach().permute(1, 2, 0).numpy()


def to_tensor(array):
    return torch.from_numpy(array).permute(2, 0, 1).to(device).unsqueeze(0)


def save_test(path, test_image, model, i):
    
    img = Image.open(f"{path}/{test_image}").convert("RGB")
    tensor = save_transform(img).unsqueeze(0).to(device)
    tensor = model(tensor)
    tensor *= torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    tensor += torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    
    save_image(tensor, f'{SAMPLE_FLDR}/test_{i}.jpg')
    return tensor


def gram_matrix(y):

    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)

    return gram


def train(model, style_image, dataset, epochs, batch_size, content_weight, style_weight, augmentation_weight, optimizer, feature_extractor):
    
    train_loader = DataLoader(dataset, batch_size=batch_size)

    style = style_transform(style_image)
    style = style.repeat(batch_size, 1, 1, 1).to(device)

    features_style = feature_extractor(style)
    gram_style = [gram_matrix(layer) for layer in features_style]
    
    last_total = 1e6
    step_to_break = 0

    for e in range(epochs):
        
        model.train()
        agg_content_loss = 0.
        agg_augmentation_loss = 0.
        agg_style_loss = 0.
        count = 0
        
        for batch_id, x in enumerate(train_loader):
            
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            if x.shape[2] % 2 == 1:
                x = x[:, :, :-1, :]
            if x.shape[3] % 2 == 1:
                x = x[:, :, :, :-1]
            y = model(x)
               
            features_y = feature_extractor(y)
            features_x = feature_extractor(x)

            content_loss = content_weight * mse_loss(features_y.relu3_3, features_x.relu3_3)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = gram_matrix(ft_y)
                style_l = style_weight*mse_loss(gm_y, gm_s[:n_batch, :, :])
                style_loss += style_l
        
            total_loss = content_loss + style_loss
                
            total_loss.backward()
            optimizer.step()
            
            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % 100 == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.2f}\tstyle: {:.2f}\ttotal: {:.2f}".format(
                    time.ctime(), e + 1, count, len(dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)
                save_test(TRAIN_FLDR, random.choice(os.listdir(TRAIN_FLDR)), model, e * len(dataset) + batch_id)
                
            if abs(last_total - total_loss) < 1e-5:
                step_to_break += 1
                if step_to_break == 5:
                    break
            else:
                step_to_break = 0
                last_total = total_loss


style_image = Image.open(STYLE_IMG).convert("RGB")


torch.cuda.empty_cache()
model = TransformerNet()
model.to(device);
mse_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), 1e-4, betas=(0.5, 0.999))
train(
    model, 
    style_image, 
    image_data, 
    2, 
    1, 
    content_weight=1e5, 
    style_weight=2e10, 
    augmentation_weight=0, 
    feature_extractor=feature_extractor,
    optimizer=optimizer
)


torch.save(model.state_dict(), STYLE_MODEL)
