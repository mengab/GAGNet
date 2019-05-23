import math
import torch.nn.functional as F
from torch import nn
import GFNet.block_ref as B
import GFNet.spectral_norm as SN
from GFNet.model_init import *
import numpy as np
#it is better to use nn.Sequential() to set Discriminator

####################
# Discriminator
####################

# VGG style Discriminator with input size 128*128
class Discriminator_VGG_128(nn.Module):
	def __init__(self, in_nc, base_nf, norm_type='instance', act_type='relu', mode='CNA'):
		super(Discriminator_VGG_128, self).__init__()
		# features
		# hxw, c
		# 128, 64
		conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
			mode=mode)
		conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		# 64, 64
		conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		# 32, 128
		conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		# 16, 256
		conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		# 8, 512
		conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		# 4, 512
		self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
			conv9)

		# classifier
		self.classifier = nn.Sequential(
			nn.Linear(512 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

	def forward(self, x):
		x = self.features(x)
		# print(x.shape)
		x = x.view(x.size(0), -1)
		# print(x.shape)
		x = self.classifier(x)
		return x


# VGG style Discriminator with input size 128*128, Spectral Normalization
class Discriminator_VGG_128_SN(nn.Module):
	def __init__(self):
		super(Discriminator_VGG_128_SN, self).__init__()
		# features
		# hxw, c
		# 128, 64
		self.lrelu = nn.LeakyReLU(0.2, True)

		self.conv0 = SN.spectral_norm(nn.Conv2d(3, 64, 3, 1, 1))
		self.conv1 = SN.spectral_norm(nn.Conv2d(64, 64, 4, 2, 1))
		# 64, 64
		self.conv2 = SN.spectral_norm(nn.Conv2d(64, 128, 3, 1, 1))
		self.conv3 = SN.spectral_norm(nn.Conv2d(128, 128, 4, 2, 1))
		# 32, 128
		self.conv4 = SN.spectral_norm(nn.Conv2d(128, 256, 3, 1, 1))
		self.conv5 = SN.spectral_norm(nn.Conv2d(256, 256, 4, 2, 1))
		# 16, 256
		self.conv6 = SN.spectral_norm(nn.Conv2d(256, 512, 3, 1, 1))
		self.conv7 = SN.spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
		# 8, 512
		self.conv8 = SN.spectral_norm(nn.Conv2d(512, 512, 3, 1, 1))
		self.conv9 = SN.spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
		# 4, 512

		# classifier
		self.linear0 = SN.spectral_norm(nn.Linear(512 * 4 * 4, 100))
		self.linear1 = SN.spectral_norm(nn.Linear(100, 1))

	def forward(self, x):
		x = self.lrelu(self.conv0(x))
		x = self.lrelu(self.conv1(x))
		x = self.lrelu(self.conv2(x))
		x = self.lrelu(self.conv3(x))
		x = self.lrelu(self.conv4(x))
		x = self.lrelu(self.conv5(x))
		x = self.lrelu(self.conv6(x))
		x = self.lrelu(self.conv7(x))
		x = self.lrelu(self.conv8(x))
		x = self.lrelu(self.conv9(x))
		x = x.view(x.size(0), -1)
		x = self.lrelu(self.linear0(x))
		x = self.linear1(x)
		return x


class Discriminator_VGG_96(nn.Module):
	def __init__(self, in_nc, base_nf, norm_type='instance', act_type='leakyrelu', mode='CNA'):
		super(Discriminator_VGG_96, self).__init__()
		# features
		# hxw, c
		# 96, 64
		conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
			mode=mode)
		conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		# 48, 64
		conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		# 24, 128
		conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		# 12, 256
		conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		# 6, 512
		conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		# 3, 512
		self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
			conv9)

		# classifier
		self.classifier = nn.Sequential(
			nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x


class Discriminator_VGG_192(nn.Module):
	def __init__(self, in_nc, base_nf, norm_type='instance', act_type='leakyrelu', mode='CNA'):
		super(Discriminator_VGG_192, self).__init__()
		# features
		# hxw, c
		# 192, 64
		conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
			mode=mode)
		conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		# 96, 64
		conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		# 48, 128
		conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		# 24, 256
		conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		# 12, 512
		conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		# 6, 512
		conv10 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		conv11 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
			act_type=act_type, mode=mode)
		# 3, 512
		self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
			conv9, conv10, conv11)

		# classifier
		self.classifier = nn.Sequential(
			nn.Linear(512 * 2 * 2, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

	def forward(self, x):
		x = self.features(x)
		
		x = x.view(x.size(0), -1)
		# print(x.shape,'.................')
		x = self.classifier(x)
		return x

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
	def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
		super(NLayerDiscriminator, self).__init__()
		# self.gpu_ids = gpu_ids
		# self.use_parallel = use_parallel
		# if type(norm_layer) == functools.partial:
		# 	use_bias = norm_layer.func == nn.InstanceNorm2d
		# else:
		# 	use_bias = norm_layer == nn.InstanceNorm2d
		if norm_layer:
			use_bias=False
		else:
			use_bias=True

		kw = 5
		padw = int(np.ceil((kw-1)/2))
		sequence = [
			nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
			nn.LeakyReLU(0.2, True)
		]

		nf_mult = 1
		nf_mult_prev = 1
		for n in range(1, n_layers):
			nf_mult_prev = nf_mult
			nf_mult = min(2**n, 8)
			sequence += [
				nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,kernel_size=kw, stride=2, padding=padw, bias=use_bias),
				norm_layer(ndf * nf_mult),
				nn.LeakyReLU(0.2, True)
			]

		nf_mult_prev = nf_mult
		nf_mult = min(2**n_layers, 8)
		sequence += [
			nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,kernel_size=kw, stride=1, padding=padw, bias=use_bias),
			norm_layer(ndf * nf_mult),
			nn.LeakyReLU(0.2, True)
		]

		sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

		# sequence += [nn.Sigmoid()]

		self.model = nn.Sequential(*sequence)

	def forward(self, input):
		output = self.model(input)#N*1*16*16

		return output

def define_D(opt):
	
	which_model = opt.D_type
	if which_model == 'patch_GAN':
		netD = NLayerDiscriminator(input_nc=1, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d)

	elif which_model == 'discriminator_vgg_128':
		netD =Discriminator_VGG_128(in_nc=opt.in_nc, base_nf=opt.nf, \
			norm_type=opt.norm_type, mode=opt.mode, act_type=opt.act_type)

	elif which_model == 'discriminator_vgg_96':
		netD = Discriminator_VGG_96(in_nc=opt.in_nc, base_nf=opt.nf, \
			norm_type=opt.norm_type, mode=opt.mode, act_type=opt.act_type)
	elif which_model == 'discriminator_vgg_192':
		netD = Discriminator_VGG_192(in_nc=opt.in_nc, base_nf=opt.nf, \
			norm_type=opt.norm_type, mode=opt.mode, act_type=opt.act_type)
	elif which_model == 'discriminator_vgg_128_SN':
		netD = Discriminator_VGG_128_SN()
	else:
		raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))

	IW = model_init()
	IW.init_weights(init_type='kaiming',net=netD)
   
	return netD
