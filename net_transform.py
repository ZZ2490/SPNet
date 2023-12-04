import math
import torch
from torch import nn
import torch.nn.functional as F
import vgg
from torch.nn import Conv2d, Parameter, Softmax

from Encoder import Transformer


def convblock(in_,out_,ks,st,pad):
    return nn.Sequential(
        nn.Conv2d(in_,out_,ks,st,pad),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )
def convblock2(in_ch, out_ch, rate):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, dilation=rate, padding=rate),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )
class FAModule(nn.Module):
    def __init__(self,in_ch,in_cn):
        super(FAModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv_branch1 = nn.Sequential(nn.Conv2d(in_cn, int(in_cn / 4), 1), self.relu)
        self.conv_branch2 = nn.Sequential(nn.Conv2d(in_cn, int(in_cn / 2), 1), self.relu,
                                          nn.Conv2d(int(in_cn / 2), int(in_cn / 4), 3, 1, 1), self.relu)
        self.conv_branch3 = nn.Sequential(nn.Conv2d(in_cn, int(in_cn / 4), 1), self.relu,
                                          nn.Conv2d(int(in_cn / 4), int(in_cn / 4), 5, 1, 2), self.relu)
        self.conv_branch4 = nn.Sequential(nn.MaxPool2d(3, 1, 1), nn.Conv2d(in_cn, int(in_cn / 4), 1), self.relu)
        self.conv1 = convblock(2*in_ch,512,3,1,1)
        self.casa = CA(2*in_ch)

    def forward(self, x_1, x_2):
        x = self.conv1(self.casa(torch.cat((x_1,x_2),1)))

        x_branch1 = self.conv_branch1(x)
        x_branch2 = self.conv_branch2(x)
        x_branch3 = self.conv_branch3(x)
        x_branch4 = self.conv_branch4(x)

        x = torch.cat((x_branch1, x_branch2, x_branch3, x_branch4), dim=1)
        return x

class GFM(nn.Module):
    def __init__(self, in_c):
        super(GFM, self).__init__()
        self.conv_n1 = convblock(in_c, in_c//2, 1, 1, 0)
        self.conv_n2 = convblock(in_c, in_c//2, 1, 1, 0)
        self.conv_n3 = convblock(in_c, in_c//2, 1, 1, 0)
        self.conv_rt = convblock(in_c//2, in_c, 1, 1, 0)
        self.softmax = nn.Softmax(dim=-1)
        self.gam = Parameter(torch.zeros(1))

    def forward(self, rgb, t):
        b, c, h, w = rgb.size()
        rgb1 = self.conv_n1(rgb)
        rgb2 = rgb1.view(b, -1, w * h)
        t1 = self.conv_n2(t).view(b, -1, w * h)
        t2 = self.conv_n3(t).view(b, -1, w * h).permute(0, 2, 1)
        r_1 = rgb2.permute(0, 2, 1)
        rt_1 = torch.bmm(r_1, t1)
        softmax_rt = self.softmax(rt_1)
        rt_2 = torch.bmm(softmax_rt, t2)
        b1, c1, h1, w1 = rgb1.size()
        rt_2 = rt_2.view(b1, c1, h1, w1)
        rt_2 = self.conv_rt(rt_2)
        rgbt = self.gam*rt_2 + rgb

        return rgbt

class CGFM(nn.Module):
    def __init__(self, in_1, in_2):
        super(CGFM, self).__init__()
        self.gfm = GFM(in_2)
        self.conv_globalinfo = convblock(512, 128, 3, 1, 1)

        self.casa = CA(2*in_1)
        self.conv_l = convblock(2*in_1,128,3,1,1)
        self.conv_m = convblock(in_2, 128, 3, 1, 1)
        self.conv_out = convblock(128, in_2, 3, 1, 1)

    def forward(self, cu_p, cu_s, cu_pf, cu_pb, global_info):#主、次、前、后、全局
        cur_size = cu_p.size()[2:]
        cl_fus1 = self.casa(torch.cat((cu_pf, cu_pb), 1))
        cl_fus = self.conv_l(F.interpolate(cl_fus1, cur_size, mode='bilinear', align_corners=True))

        cm_fus = self.conv_m(self.gfm(cu_p, cu_s))

        global_info = self.conv_globalinfo(F.interpolate(global_info,cur_size,mode='bilinear',align_corners=True))

        fus = cl_fus + cm_fus + global_info

        return self.conv_out(fus)

class CGFM3(nn.Module):
    def __init__(self, in_1, in_2):
        super(CGFM3, self).__init__()
        self.conv_globalinfo = convblock(512, 128, 3, 1, 1)
        self.rt_fus = nn.Sequential(
            nn.Conv2d(in_2, in_2, 3, 1, 1),
            nn.Sigmoid()
        )
        self.casa1 = CA(in_2)
        self.casa = CA(2 * in_1)
        self.conv_l = convblock(2 * in_1, 128, 3, 1, 1)
        self.conv_m = convblock(2*in_2, 128, 3, 1, 1)
        self.conv_out = convblock(128, in_2, 3, 1, 1)

    def forward(self, cu_p, cu_s, cu_pf, cu_pb, global_info):  # 主、次、前、后、全局
        cur_size = cu_p.size()[2:]
        cl_fus1 = self.casa(torch.cat((cu_pf, cu_pb), 1))
        cl_fus = self.conv_l(F.interpolate(cl_fus1, cur_size, mode='bilinear', align_corners=True))

        cu_p = self.casa1(cu_p)
        cu_s = self.casa1(cu_s)
        cross_cat = torch.cat((torch.add(cu_p, torch.mul(cu_p, self.rt_fus(cu_s))), cu_s), 1)
        cm_fus = self.conv_m(cross_cat)

        global_info = self.conv_globalinfo(F.interpolate(global_info, cur_size, mode='bilinear', align_corners=True))

        fus = cl_fus + cm_fus + global_info

        return self.conv_out(fus)

class GFAPF(nn.Module):
    def __init__(self):
        super(GFAPF, self).__init__()
        self.casa = CA(1024)
        self.de_chan = convblock(1024, 256, 3, 1, 1)
        self.b0 = nn.Sequential(
            nn.AdaptiveMaxPool2d(13),
            nn.Conv2d(256, 128, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True)
        )
        self.b1 = nn.Sequential(
            nn.AdaptiveMaxPool2d(9),
            nn.Conv2d(256,128,1,1,0, bias=False),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.AdaptiveMaxPool2d(5),
            nn.Conv2d(256, 128, 1, 1, 0,bias=False),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(256, 128, 1, 1, 0,bias=False),
            nn.ReLU(inplace=True)
        )
        self.fus = convblock(768, 512, 1, 1, 0)
        self.gfm = GFAPM(128)

    def forward(self, rgb, t):
        x_size = rgb.size()[2:]
        x = self.casa(torch.cat((rgb, t), 1))
        x = self.de_chan(x)
        b0 = F.interpolate(self.b0(x), x_size,mode='bilinear',align_corners=True)
        b1 = F.interpolate(self.b1(x), x_size,mode='bilinear',align_corners=True)
        b2 = F.interpolate(self.b2(x), x_size,mode='bilinear',align_corners=True)
        out1 = self.gfm(b0, b1, b2)
        out2 = self.gfm(b1, b2, b0)
        out3 = self.gfm(b2, b0, b1)
        out4 = F.interpolate(self.b3(x), x_size, mode='bilinear',align_corners=True)
        out = self.fus(torch.cat((out1, out2, out3, out4, x), 1))
        return out

class GFAPM(nn.Module):
    def __init__(self, in_c):
        super(GFAPM, self).__init__()
        self.conv_n1 = convblock(in_c, in_c//4, 1, 1, 0)
        self.conv_n2 = convblock(in_c, in_c//4, 1, 1, 0)
        self.conv_n3 = convblock(in_c, in_c, 1, 1, 0)
        self.softmax = nn.Softmax(dim=-1)
        self.gam = Parameter(torch.zeros(1))

    def forward(self, x1, x2, x3):
        b, c, h, w = x1.size()
        rgb1 = self.conv_n1(x1).view(b, -1, w * h).permute(0, 2, 1)
        t1 = self.conv_n2(x2).view(b, -1, w * h)
        rt_1 = torch.bmm(rgb1, t1)
        attention1 = self.softmax(rt_1)
        rt_2 = self.conv_n3(x3).view(b, -1, w * h)
        out = torch.bmm(rt_2, attention1.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        out = self.gam * out + x1

        return out

class CA(nn.Module):
    def __init__(self,in_ch):
        super(CA, self).__init__()
        self.avg_weight = nn.AdaptiveAvgPool2d(1)
        self.max_weight = nn.AdaptiveMaxPool2d(1)
        self.fus = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(in_ch // 2, in_ch, 1, 1, 0),
        )
        self.c_mask = nn.Sigmoid()
    def forward(self, x):
        avg_map_c = self.avg_weight(x)
        max_map_c = self.max_weight(x)
        c_mask = self.c_mask(torch.add(self.fus(avg_map_c), self.fus(max_map_c)))
        return torch.mul(x, c_mask)

class FinalScore(nn.Module):
    def __init__(self):
        super(FinalScore, self).__init__()
        self.score = nn.Conv2d(512, 1, 1, 1, 0)
    def forward(self,f1,xsize):

        score = F.interpolate(self.score(f1), xsize, mode='bilinear', align_corners=True)
        return score

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.global_info = GFAPF()
        self.score_global = nn.Conv2d(512, 1, 1, 1, 0)

        self.fa1 = FAModule(512,512)
        self.fa2 = FAModule(256,512)
        self.fa3 = FAModule(128,512)

        self.gfb4_1 = CGFM(512,512)
        self.gfb3_1 = CGFM(512,256)
        self.gfb2_1 = CGFM3(256,128)

        self.gfb4_2 = CGFM(512, 512) #1/8
        self.gfb3_2 = CGFM(512, 256)#1/4
        self.gfb2_2 = CGFM3(256, 128)#1/2

        self.score_1 = nn.Conv2d(128, 1, 1, 1, 0)
        self.score_2 = nn.Conv2d(128, 1, 1, 1, 0)

        self.refine = FinalScore()


    def forward(self,rgb,t):
        xsize = rgb[0].size()[2:]
        global_info = self.global_info(rgb[4], t[4]) # 512 1/16
        score_global = self.score_global(global_info)

        d1 = self.gfb4_1(rgb[3], t[3], t[4], rgb[4], global_info)
        d2 = self.gfb4_2(t[3], rgb[3], rgb[4], t[4], global_info)
        global_info = self.fa1(d1, d2)

        d3 = self.gfb3_1(rgb[2], t[2], d1, d2, global_info)
        d4 = self.gfb3_2(t[2], rgb[2], d2, d1, global_info)
        global_info = self.fa2(d3, d4)

        d5 = self.gfb2_1(rgb[1], t[1], d3, d4, global_info)
        d6 = self.gfb2_2(t[1], rgb[1], d4, d3, global_info) #1/2 128
        global_info = self.fa3(d5, d6)

        score1 = self.score_1(F.interpolate(d5, xsize, mode='bilinear',align_corners=True))
        score2 = self.score_2(F.interpolate(d6, xsize, mode='bilinear', align_corners=True))
        score = self.refine(global_info, xsize)
        score_global = F.interpolate(score_global, xsize, mode='bilinear', align_corners=True)
        return score, score1, score2, score_global

def Encoder():
    model = Transformer('pvt_v2_b3', pretrained=True)
    return model

class Mnet(nn.Module):
    def __init__(self):
        super(Mnet,self).__init__()
        model = Encoder()
        self.rgb_net = model.encoder
        self.t_net = model.encoder
        self.decoder=Decoder()
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,rgb,t):
        rgb_f= self.rgb_net(rgb)
        t_f= self.t_net(t)
        score,score1,score2,score_g =self.decoder(rgb_f,t_f)

        return score,score1,score2,score_g,self.sigmoid(score),self.sigmoid(score1),self.sigmoid(score2),self.sigmoid(score_g)
    #

    # def load_pretrained_model(self):
    #     st=torch.load("vgg16.pth")
    #     st2={}
    #     for key in st.keys():
    #         st2['base.'+key]=st[key]
    #     self.rgb_net.load_state_dict(st2)
    #     self.t_net.load_state_dict(st2)
    #     print('loading pretrained model success!')