import functools
import torch
import torch.nn as nn
from torch.nn import init

####################
# initialize
####################
class model_init(object):
    def __init__(self,scale=0, std=0.02):
        
        self.scale=scale
        self.std=std

    def weights_init_normal(self,m, std=0.02):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                m.bias.data.zero_()
        if classname.find('ConvTranspose2d') != -1:
            init.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            init.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
            init.constant_(m.bias.data, 0.0)


    def weights_init_kaiming(self,m, scale=0):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.kaiming_normal_(m.weight.data, a=scale, mode='fan_in')
            # m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('ConvTranspose2d') != -1:
            init.kaiming_normal_(m.weight.data, a=scale, mode='fan_in')
            # m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            init.kaiming_normal_(m.weight.data, a=scale, mode='fan_in')
            # m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm2d') != -1:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)


    def weights_init_orthogonal(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        if classname.find('ConvTranspose2d') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm2d') != -1:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)


    def init_weights(self,init_type='kaiming',net=None):
        # scale for 'kaiming', std for 'normal'.
        print('initialization method [{:s}]'.format(init_type))
        if init_type == 'normal':
            weights_init_normal_ = functools.partial(self.weights_init_normal, std=self.std)
            net.apply(weights_init_normal_)
        elif init_type == 'kaiming':
            weights_init_kaiming_ = functools.partial(self.weights_init_kaiming, scale=self.scale)
            net.apply(weights_init_kaiming_)
        elif init_type == 'orthogonal':
            net.apply(self.weights_init_orthogonal)
        else:
            raise NotImplementedError('initialization method [{:s}] not implemented'.format(self.init_type))

