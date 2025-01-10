from __future__ import print_function
import argparse
import os
from math import log10
from utils import is_image_file, load_img, save_img
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.transforms.functional import to_tensor
from IQA_pytorch import MS_SSIM,NLPD,GMSD,SSIM,LPIPSvgg
from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from data import get_training_set, get_test_set
import matplotlib.pyplot as plt
# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', default='CPL-Set')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=32, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=32, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=150, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=150, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for 0.0002 adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default="true", help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective 10 ')
parser.add_argument('--num_steps', type=int, default=4)
parser.add_argument('--act_type', type=str, default='prelu',
                    help='type of activation function')
parser.add_argument('--num_cfbs', type=int, default=3,
                    help='number of CFBs in CF_Net')
parser.add_argument('--num_groups', type=int, default=4,
                    help='number of projection groups in SRB and CFB')
parser.add_argument('--upscale_factor', type=int, default=2,
                    help='number of CFBs in CF_Net')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
root_path = "dataset/"
train_set = get_training_set(root_path + opt.dataset, opt.direction)
#test_set = get_test_set(root_path + opt.dataset, opt.direction)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
#testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)

device = torch.device("cuda:1" if opt.cuda else "cpu")

print('===> Building models')
net_g = define_G(3)
net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'pixel', gpu_id=device)

criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)
#LPIPSvgg_metric = LPIPSvgg(channels=3).to(device)
NLPD_metric = NLPD(channels=3).to(device)
GMSD_metric = GMSD(channels=3).to(device)
# setup optimizer
optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt)


generator_losses = []
discriminator_losses = []
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # train
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        real_a, real_b = batch[0].to(device), batch[1].to(device)
        #print("real_a:",real_a.shape)
        fake_b = net_g(real_a)
        #print("fake_b:",fake_b.shape)
        # 打印输出列表的长度
        print("Number of fake_b in the list: {}".format(len(fake_b)))
        
        ######################
        # (1) Update D network
        ######################

        optimizer_d.zero_grad()
        loss_d_fake=0
        loss_d_real=0
        loss_d_l1=0
        for fake_out in fake_b:
            #JND_out=torch.abs(fake_out - real_a)
            
            # train with fake
            fake_ab = torch.cat((real_a, fake_out), 1)
            pred_fake = net_d.forward(fake_ab.detach())
            loss_d_fake += criterionGAN(pred_fake, False)

            # train with real
            real_ab = torch.cat((real_a, real_b), 1)
            pred_real = net_d.forward(real_ab)
            loss_d_real += criterionGAN(pred_real, True)
            #loss_d_l1 += criterionL1(fake_out, real_b) * opt.lamb
        # averaged over n iterations
        loss_d_fake /= len(fake_b)
        loss_d_real /= len(fake_b)
        #loss_d_l1 /= len(fake_b)

        # averaged over batches
        loss_d_fake = torch.mean(loss_d_fake)
        loss_d_real = torch.mean(loss_d_real)
        #loss_d_l1 = torch.mean(loss_d_l1)
        # Combined D loss
        loss_d = (loss_d_fake *0.5+ loss_d_real*0.5)
        discriminator_losses.append(loss_d.item())
        # backpropagate and step
        loss_d.backward()
       
        optimizer_d.step()

        ######################
        # (2) Update G network
        ######################

        optimizer_g.zero_grad()
        loss_g_gan=0
        NLPD_score=0
        GMSD_score=0
        loss_g_l1=0
        for fake_out in fake_b:
            # First, G(A) should fake the discriminator
            fake_ab = torch.cat((real_a, fake_out), 1)
            pred_fake = net_d.forward(fake_ab)
            loss_g_gan += criterionGAN(pred_fake, True)
            # Calculate MS-SSIM
            #ssim_score = ssim_metric(real_a, fake_b, as_loss=True)
            NLPD_score += NLPD_metric(real_b, fake_out, as_loss=True)
            GMSD_score += GMSD_metric(real_b, fake_out, as_loss=True)
            loss_g_l1 += criterionL1(fake_out, real_b) * opt.lamb
      
         # averaged over n iterations
        loss_g_gan /= len(fake_b)
        NLPD_score /= len(fake_b)
        GMSD_score /= len(fake_b)
        loss_g_l1 /= len(fake_b)

        # averaged over batches
        loss_g_gan = torch.mean(loss_g_gan)
        NLPD_score = torch.mean(NLPD_score)
        GMSD_score = torch.mean(GMSD_score)
        loss_g_l1 = torch.mean(loss_g_l1)

        # Combined D loss
        IQA_loss=NLPD_score+GMSD_score

        # Second, G(A) = B
        loss_g = loss_g_gan*0.01 + loss_g_l1*0.5+IQA_loss
        generator_losses.append(loss_g.item())
        loss_g.backward()

        optimizer_g.step()

        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item()))

    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)

    # test
   
    #checkpoint
    if epoch % 50 == 0:
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
            os.mkdir(os.path.join("checkpoint", opt.dataset))
        net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch)
        net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch)
        torch.save(net_g, net_g_model_out_path)
        torch.save(net_d, net_d_model_out_path)
        print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))

       
# 绘制生成器和判别器的损失曲线
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(generator_losses, label="Generator Loss")
plt.plot(discriminator_losses, label="Discriminator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# 保存损失曲线图像到文件
losses_plot_path = "generator_and_discriminator_losses.png"
plt.savefig(losses_plot_path)
plt.close()  # 关闭图形以释放内存
