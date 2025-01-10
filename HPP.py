import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class FHDR(nn.Module):
    def __init__(self, level):
        super(FHDR, self).__init__()
        
        self.level = level

        self.reflect_pad = nn.ReflectionPad2d(1)
        self.feb1 = nn.Conv2d(3, 64, kernel_size=3, padding=0)
        self.feb2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.feedback_block = FeedbackBlock()

        self.hrb1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.hrb2 = nn.Conv2d(64, 3, kernel_size=3, padding=0)

        self.tanh = nn.Tanh()

    def forward(self, input):

        outs = []
        fb_out=[]
        feb1 = F.relu(self.feb1(self.reflect_pad(input)))
        feb2 = F.relu(self.feb2(feb1))

        for i in range(self.level):
            fb_out = self.feedback_block(feb2)
            fb_out_i=fb_out[i]
            FDF_i = fb_out_i + feb1
            
            hrb1_i = F.relu(self.hrb1(FDF_i))
            out_i = self.hrb2(self.reflect_pad(hrb1_i))
            out_i = self.tanh(out_i)
            outs.append(out_i)

        return outs


class FeedbackBlock(nn.Module):
    def __init__(self):
        super(FeedbackBlock, self).__init__()

        self.compress_in = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.DRDB1 = gycblock(64,64)
        self.DRDB2 = gycblock(64,64)
        self.DRDB3 = gycblock(64,64)
        self.last_hidden = None

        self.GFF_3x3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.should_reset = True

    def forward(self, x):
        if self.should_reset:
            #self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden = torch.zeros(x.size(), device='cuda:2')
            self.last_hidden.copy_(x)
            self.should_reset = False
        self.last_hidden = torch.zeros(x.size(), device='cuda:2')
        self.last_hidden.copy_(x)
        output=[]
        out1 = torch.cat((x, self.last_hidden), dim=1)
        out2 = self.compress_in(out1)

        

        out3 = self.DRDB1(out2,1)
        
        out4 = self.DRDB2(out3,2)
       
        out5 = self.DRDB3(out4,3)
        output1 = F.relu(self.GFF_3x3(out3))
        output.append(output1)
        output2 = F.relu(self.GFF_3x3(out4))
        output.append(output2)
        output3 = F.relu(self.GFF_3x3(out5))
        output.append(output3)
        self.last_hidden = output3
        self.last_hidden = Variable(self.last_hidden.data)
        
        return output


class DilatedResidualDenseBlock(nn.Module):
    def __init__(self, nDenselayer=4, growthRate=32):
        super(DilatedResidualDenseBlock, self).__init__()

        nChannels_ = 64
        modules = []

        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.should_reset = True

        self.compress = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.conv_1x1 = nn.Conv2d(nChannels_, 64, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.should_reset:
            #self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden = torch.zeros(x.size(), device='cuda:2')
            self.last_hidden.copy_(x)
            self.should_reset = False
        self.last_hidden = torch.zeros(x.size(), device='cuda:2')
        self.last_hidden.copy_(x)
        cat = torch.cat((x, self.last_hidden), dim=1)

        out = self.compress(cat)
        out = self.dense_layers(out)
        out = self.conv_1x1(out)

        self.last_hidden = out
        self.last_hidden = Variable(out.data)

        return out


class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(
            nChannels,
            growthRate,
            kernel_size=kernel_size,
            padding=(kernel_size - 1),
            bias=False,
            dilation=2,
        )

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out
class gycblock(nn.Module):
    def __init__(self,channels_in,channels_out):
        super(gycblock, self).__init__()
        self.recp7 = nn.Sequential(
            BasicConv(channels_in, channels_in, kernel_size=7, dilation=1, padding=3, groups=channels_in, bias=False),
            BasicConv(channels_in, channels_in, kernel_size=1, dilation=1, bias=False),

            BasicConv(channels_in, channels_in, kernel_size=3, dilation=7, padding=7, groups=channels_in, bias=False,
                      relu=False),
            BasicConv(channels_in, channels_in, kernel_size=1, dilation=1, bias=False)
        )

        self.recp5 = nn.Sequential(
            BasicConv(channels_in, channels_in, kernel_size=5, dilation=1, padding=2, groups=channels_in, bias=False),
            BasicConv(channels_in, channels_in, kernel_size=1, dilation=1, bias=False),

            BasicConv(channels_in, channels_in, kernel_size=3, dilation=5, padding=5, groups=channels_in, bias=False,
                      relu=False),
            BasicConv(channels_in, channels_in, kernel_size=1, dilation=1, bias=False)
        )

        self.recp3 = nn.Sequential(
            BasicConv(channels_in, channels_in, kernel_size=3, dilation=1, padding=1, groups=channels_in, bias=False),
            BasicConv(channels_in, channels_in, kernel_size=1, dilation=1, bias=False),

            BasicConv(channels_in, channels_in, kernel_size=3, dilation=3, padding=3, groups=channels_in, bias=False,
                      relu=False),
            BasicConv(channels_in, channels_in, kernel_size=1, dilation=1, bias=False)
        )

        self.recp1 = nn.Sequential(
            BasicConv(channels_in, channels_out, kernel_size=1, dilation=1, bias=False,relu=True)
        )

    def forward(self, x,level):
        if level==1:
            out = self.recp7(x)
            
        elif level==2:
            out = self.recp5(x)
            
        else: 
            out = self.recp3(x)
            

        return out
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.PReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)

        return x