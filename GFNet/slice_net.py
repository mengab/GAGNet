import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
from torch.nn.init import kaiming_normal_
from LSTM.BiConvLSTM import BiConvLSTM

def conv(wn, in_planes, out_planes, kernel_size=3, stride=1):
	if wn!=None:
		return nn.Sequential(
			wn(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False)),
			nn.ReLU(inplace=True)
		)
	else:
		return nn.Sequential(
			nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
			nn.ReLU(inplace=True)
		)

def conv_no_lrelu(wn, in_planes, out_planes, kernel_size=3, stride=1):
	if wn!=None:
		return nn.Sequential(
			wn(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False))
			
		)
	else:
		return nn.Sequential(
			nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True)
		)

class res_block(nn.Module):
	def __init__(self, wn):
		super(res_block, self).__init__()
		self.wn = wn
		self.conv3_1 = conv(self.wn, in_planes=64, out_planes=48, kernel_size=3, stride=1)

		self.conv3_2 = conv(self.wn, in_planes=48, out_planes=48, kernel_size=3, stride=1)

		self.conv3_3 = conv(self.wn, in_planes=32, out_planes=48, kernel_size=3, stride=1)

		self.conv3_4 = conv(self.wn, in_planes=48, out_planes=64, kernel_size=3, stride=1)

		self.conv3_5 = conv(self.wn, in_planes=48, out_planes=48, kernel_size=3, stride=1)

		self.conv3_6 = conv(self.wn, in_planes=48, out_planes=96, kernel_size=3, stride=1)

		self.down    = conv(self.wn, in_planes=96, out_planes=64, kernel_size=1, stride=1)
	def forward(self, input):
		split1    = input
		split1_1 = input
		conv3_1 = self.conv3_1(split1)
		conv3_2 = self.conv3_2(conv3_1)
		slice1_1, slice1_2 = torch.split(conv3_2, [16,32], dim=1)
		conv3_3 = self.conv3_3(slice1_2)
		conv3_4 = self.conv3_4(conv3_3)
		slice2_1,slice2_2 = torch.split(conv3_4, [16,48], dim=1)
		conv3_5 = self.conv3_5(slice2_2)
		conv3_6 = self.conv3_6(conv3_5)
		concat1= torch.cat([split1_1,slice1_1,slice2_1],dim=1)#64+16+16
		sum1 = concat1 + conv3_6
		down1 = self.down(sum1)
		return down1

class Slice_Blocks(nn.Module):
   
	def __init__(self, num_sliceBlock,wn):
		super(Slice_Blocks, self).__init__()

		self.ResBlocks = nn.ModuleList([res_block(wn=wn) for i in range(num_sliceBlock)])
	
	def forward(self,input):
		x = input
		for layer in self.ResBlocks:
			x = layer(x)
		
		return x


class DeBlockNet(nn.Module):
	def __init__(self,wn=None, num_sliceBlock=3,is_train=True,input_size=256,use_LSTM=True,use_mask =False,mask_num_sliceBlock=2):
		super(DeBlockNet, self).__init__()

		self.num_sliceBlock = num_sliceBlock
		self.mask_num_sliceBlock= mask_num_sliceBlock
		self.wn   = wn
		self.LSTM = use_LSTM
		self.is_train = is_train
		self.use_mask = use_mask
		self.pre_conv1   = conv(self.wn, 1, 64,  kernel_size=3, stride=1)
		self.pre_conv1_1 = conv(self.wn, 64, 64, kernel_size=3, stride=1)

		self.pre_conv2   = conv(self.wn, 1, 64,  kernel_size=3, stride=1)
		self.pre_conv2_1 = conv(self.wn, 64, 64, kernel_size=3, stride=1)

		self.pre_conv3   = conv(self.wn, 1, 64,  kernel_size=3, stride=1)
		self.pre_conv3_1 = conv(self.wn, 64, 64, kernel_size=3, stride=1)
		if self.LSTM:
			print("===> used BiLSTM")
			self.biconvlstm  = BiConvLSTM(input_size=(input_size, input_size), input_dim=64, hidden_dim=64,kernel_size=(3, 3), num_layers=1)
			self.LSTM_out = conv(self.wn,64,64,  kernel_size=3,  stride=1)
		else:
			self.pre_conv_pre_cur_1     = conv(self.wn,128,64,kernel_size=1,stride=1)
			self.pre_conv_pre_cur_2     = conv(self.wn,64,64,kernel_size=3,stride=1)

			self.pre_conv_cur_aft_1     = conv(self.wn, 128, 64, kernel_size=1, stride=1)
			self.pre_conv_cur_aft_2     = conv(self.wn, 64, 64, kernel_size=3, stride=1)

			self.pre_conv_pre_cur_aft_1 = conv(self.wn, 128, 64, kernel_size=1, stride=1)
			self.pre_conv_pre_cur_aft_2 = conv(self.wn, 64, 64, kernel_size=3, stride=1)

		if use_mask:
			self.mask_conv1   = conv(self.wn, 1, 64,  kernel_size=3, stride=1)
			self.mask_conv1_1 = conv(self.wn, 64, 64, kernel_size=3, stride=1)
			self.mask_Blocks = Slice_Blocks(num_sliceBlock= self.mask_num_sliceBlock ,wn=self.wn)
			self.mask_output_conv1 =  conv(self.wn, 64, 64, kernel_size=3, stride=1)

			self.mask_out = conv_no_lrelu(self.wn, 64, 1,kernel_size=5,stride =1)


		self.Blocks     = Slice_Blocks( num_sliceBlock=self.num_sliceBlock,wn=self.wn)

		self.output_conv1 =  conv(self.wn,64, 64, kernel_size=3, stride=1)
		
		self.output_conv2 =  conv(self.wn, 64, 64, kernel_size=7, stride=1)
		self.output_conv3 =  conv(self.wn, 64, 64, kernel_size=5, stride=1)
		self.output_conv4 =  conv_no_lrelu(self.wn, 64, 1,  kernel_size=5, stride=1)

		self.sigmoid = nn.Sigmoid()
		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				kaiming_normal_(m.weight.data)
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def forward(self,data1,data2,data3,mask=None):

		CNN_seq = []
		pre_conv1 = self.pre_conv1(data1)
		pre_conv1_1 = self.pre_conv1_1(pre_conv1)
		CNN_seq.append(pre_conv1_1)

		pre_conv2  = self.pre_conv2(data2)
		pre_conv2_1 = self.pre_conv2_1(pre_conv2)
		CNN_seq.append(pre_conv2_1)

		pre_conv3 = self.pre_conv2(data3)
		pre_conv3_1 = self.pre_conv3_1(pre_conv3)
		CNN_seq.append(pre_conv3_1)
		if self.LSTM:
			CNN_seq_out      = torch.stack(CNN_seq, dim=1)
			CNN_seq_feature_maps = self.biconvlstm(CNN_seq_out)
			LSTM_out         = self.LSTM_out(CNN_seq_feature_maps[:, 1, ...])
			block_in         = LSTM_out
		else:
			pre_conv_pre_cur_concat = torch.cat([pre_conv1_1,pre_conv2_1],dim=1)
			pre_conv_cur_aft_concat = torch.cat([pre_conv3_1,pre_conv2_1],dim=1)

			pre_conv_pre_cur_1 = self.pre_conv_pre_cur_1(pre_conv_pre_cur_concat)
			pre_conv_pre_cur_2 = self.pre_conv_pre_cur_2(pre_conv_pre_cur_1)

			pre_conv_cur_aft_1 = self.pre_conv_cur_aft_1(pre_conv_cur_aft_concat)
			pre_conv_cur_aft_2 = self.pre_conv_cur_aft_2(pre_conv_cur_aft_1)

			pre_conv_pre_cur_aft_concat = torch.cat([pre_conv_pre_cur_2,pre_conv_cur_aft_2],dim=1)

			pre_conv_pre_cur_aft_1 = self.pre_conv_pre_cur_aft_1(pre_conv_pre_cur_aft_concat)
			pre_conv_pre_cur_aft_2 = self.pre_conv_pre_cur_aft_2(pre_conv_pre_cur_aft_1)
			block_in = pre_conv_pre_cur_aft_2
		
		Blocks_out = self.Blocks(block_in)
		
		output_conv1 = self.output_conv1(Blocks_out)


		if self.use_mask:
			mask_pre   = self.mask_conv1(mask)
			mask_pre_1 = self.mask_conv1_1(mask_pre)
			mask_out   = self.Blocks(mask_pre_1)      
			mask_out   =   self.mask_output_conv1(mask_out)
		if self.use_mask:
			
			mask_out     = self.mask_out(mask_out)
			output_conv1 = output_conv1+ mask_out 

		output_conv2 = self.output_conv2(output_conv1)
		output_conv3 = self.output_conv3(output_conv2)
		output_conv4 = self.output_conv4(output_conv3)
		if self.use_mask:
			output = data2+output_conv4
		else:
			output =  data2+output_conv4
		# output =  self.sigmoid(output)
		if self.is_train==False:
			output = torch.clamp(output,0,1)
		return output
