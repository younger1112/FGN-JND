from __future__ import print_function
import argparse
import os

import torch
import torchvision.transforms as transforms

from utils import is_image_file, load_img, save_img
import torch
from torchvision import transforms
from PIL import Image
#from ImageFeatNet import ImageFeatNet  # 导入你的模型定义
#import config  # 确保导入了包含配置的模块
# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', default='CPL-Set')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--nepochs', type=int, default=300, help='saved model of which epochs')
parser.add_argument('--cuda', default='true', help='use cuda?')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:2" if opt.cuda else "cpu")
model_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, opt.nepochs)


net_g = torch.load(model_path).to(device)

if opt.direction == "a2b":
    image_dir = "dataset/{}/test/a/".format(opt.dataset)
else:
    image_dir = "dataset/{}/test/b/".format(opt.dataset)

image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)




for image_name in image_filenames:
    img = load_img(image_dir + image_name)
    img = transform(img)
    input = img.unsqueeze(0).to(device)
    
    out = net_g(input)
    # 打印输出列表的长度
    print("Number of outputs in the list: {}".format(len(out)))
    # output is the final reconstructed image i.e. last in the array of outputs of n iterations
    output = out[-2]
    out_img = output.detach().squeeze(0).cpu()
    print(out_img.shape)


    if not os.path.exists(os.path.join("result", opt.dataset)):
        os.makedirs(os.path.join("result", opt.dataset))
    save_img(out_img, "result/{}/{}".format(opt.dataset, image_name))
