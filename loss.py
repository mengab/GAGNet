import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import BiFlowNet
from option import opt
import time
import GFNet.utils as utils
import numpy as np
import torchvision

####################
# Perceptual Network
####################
# Assume input range is [0, 1]
class VGGFeatureExtractor(nn.Module):
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cuda')):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output

class Basic_Loss(object):
    def __init__(self,opt):
        self.opt =opt
        if self.opt.w_VGG>0:
            print('===>use VGG Loss')
            self.VGGLayers = [int(layer) for layer in list(opt.VGGLayers)]
            self.VGGLayers.sort()
            if self.VGGLayers[0] < 0 or self.VGGLayers[-1] > 3:
                raise Exception("Only support VGG Loss on Layers 0~3")       
            self.VGG_model = networks.Vgg16(requires_grad=False).cuda()
            # GPU_list = list(map(int,opt.GPUs))
            # if len(GPU_list) > 1:
            #     self.VGG_model    = nn.DataParallel(self.VGG_model,device_ids=GPU_list)
            #     self.VGG_model    = self.VGG_model.cuda()

        self.epsilon = torch.Tensor([0.01]).float().cuda()
        # self.device = torch.device("cuda" if use_cuda else "cpu")

    def VGG_loss(self,input1,input2,criterion_L1,VGG_loss_value):
        input1_r = input1.repeat(1,3,1,1)
        input2_r = input2.repeat(1,3,1,1)
        input1_r = utils.normalize_ImageNet_stats(input1_r)
        input2_r = utils.normalize_ImageNet_stats(input2_r)
        ### extract VGG features
        self.features_input_1 = self.VGG_model(input1_r, self.VGGLayers[-1])
        self.features_input_2 = self.VGG_model(input2_r, self.VGGLayers[-1])
        VGG_loss_all = []
        for l in self.VGGLayers:
            VGG_loss_all.append(criterion_L1(self.features_input_1[l], self.features_input_2[l]) )
            
        VGG_loss_value += self.opt.w_VGG * sum(VGG_loss_all)

        return VGG_loss_value
    
    def gradient_loss(self,input1,input2,criterion_L1,gradient_loss_value):

        sobel_filter_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).reshape((1, 1, 3, 3))
        sobel_filter_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).reshape((1, 1, 3, 3))
        sobel_filter_X = torch.from_numpy(sobel_filter_X).float().cuda()
        sobel_filter_Y = torch.from_numpy(sobel_filter_Y).float().cuda()
       

        input1_grad_X = F.conv2d(input1, sobel_filter_X, bias=None, stride=1, padding=1).cuda()
        input1_grad_Y = F.conv2d(input1, sobel_filter_Y, bias=None, stride=1, padding=1).cuda()

        input2_grad_X = F.conv2d(input2, sobel_filter_X, bias=None, stride=1, padding=1).cuda()
        input2_grad_Y = F.conv2d(input2, sobel_filter_Y, bias=None, stride=1, padding=1).cuda()
        
        gradient_loss_value  = criterion_L1(input1_grad_X,input2_grad_X)+criterion_L1(input1_grad_Y,input2_grad_Y)+self.epsilon

        return gradient_loss_value

    def TVLoss(self,x,tv_loss_weight=1):
        def tensor_size(t):
            return t.size()[1] * t.size()[2] * t.size()[3]
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = tensor_size(x[:, :, 1:, :])
        count_w = tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
    
    def ganloss(self,input_fake,input_real,model_D,Net='G'):
        self.cri_gan = GANLoss(self.opt.gan_type, 1.0, 0.0).cuda()#three type loss

        # self.l_gan_w = self.opt.gan_weight
       
        if self.opt.gan_type == 'wgan-gp':
            self.random_pt = torch.Tensor(1, 1, 1, 1).cuda()
            # gradient penalty loss
            self.cri_gp = GradientPenaltyLoss().cuda()
            self.loss_gp_w = self.opt.gp_weight

        if Net=='G':
            pred_g_fake = model_D(input_fake)
            pred_d_real = model_D(input_real).detach()
            # g_loss = cri_gan(pred_g_fake,True)
            g_loss =  (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                                      self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
            return g_loss
        if Net=='D':
            loss_d_total=0
            pred_d_real = model_D(input_real)            
            pred_d_fake = model_D(input_fake.detach())

            d_real_loss = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
            d_fake_loss = self.cri_gan(pred_d_fake - torch.mean(pred_d_real),False)

            d_loss= (d_real_loss + d_fake_loss)/2
            loss_d_total+=d_loss
            if self.opt.gan_type == 'wgan-gp':
                batch_size = input_real.size(0)
                if self.random_pt.size(0) != batch_size:
                    self.random_pt.resize_(batch_size, 1, 1, 1)
                self.random_pt.uniform_()    
                interp = self.random_pt * input_fake + (1 - self.random_pt) * input_real
                # interp.requires_grad = True
                interp = Variable(interp, requires_grad=True)
                interp_crit = model_D(interp)
                loss_d_gp = self.loss_gp_w * self.cri_gp(interp, interp_crit) 
                loss_d_total +=loss_d_gp

            return loss_d_total



# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss().cuda() #no need to add Sigmoid(),if use nn.BCELoss(),add Sigmoid() before it
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss().cuda()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss        


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.cuda()

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp, \
            grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss

    


