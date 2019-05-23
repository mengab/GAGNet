import argparse

parser = argparse.ArgumentParser(description='Deblock_Net')
# Hardware specifications
parser.add_argument('--n_threads', type=int, default=1,help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',help='use cpu only')
parser.add_argument('--gpu_id', type=int, default=0,help='start id of GPU')
parser.add_argument('--GPUs', type=str, default='0123',help='the id of GPUs')
parser.add_argument('--seed', type=int, default=1,help='random seed')

# Data specifications

parser.add_argument('--rgb_range', type=int, default=1,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=1,
                    help='number of color channels to use')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

#path paramter
parser.add_argument('--input_data_path', type=str, default='QP32',help='path of the enhanced result')
parser.add_argument('--save_txt_path', type=str, default='./Test_result/xxx.txt',help='txt document path')
parser.add_argument('--save_result_path', type=str, default='./Rec_frame_xxx/',help='path of the enhanced result')
# Model specifications
parser.add_argument('--pre_model_DB_path', type=str, default='./pretrained_models/model_DB_LD37.pth',help='DB_model pre-trained model directory')
parser.add_argument('--test_model_DB_path', type=str, default='./checkpoint/model.pth',help='DB_model test model directory')
parser.add_argument('--pre_model_MC_path', type=str, default='./pretrained_models/FlowNet2_checkpoint.pth.tar',help='MC_optical_flow_model  model directory')
parser.add_argument('--pre_model_PWC_path', type=str, default='./pretrained_models/pwc_net.pth.tar',help='MC_optical_flow_model  model directory')
parser.add_argument('--flow_type', type=str, default='flownet2.0',choices=['pwc_net','flownet2.0'],help='two choices to calculate the optical flow')
parser.add_argument('--mode_type', type=str, default='LD',choices=['AI','LD'],help='To choose the coding mode')


# Training specifications
parser.add_argument('--reset', action='store_true',help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,help='do test per every N batches')
parser.add_argument('--nEpochs', type=int, default=30,help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,help='input batch size for training')
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate is divided 10 every 10 epochs")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
parser.add_argument("--patch_size", type=int, default=128, help=" size of the input image. Default=256")
parser.add_argument("--num_sliceBlock", type=int, default=4, help=" SliceBlock's number . Default8")
parser.add_argument("--wn", type=bool, default=True, help="use WeightNorm?. Default=True")
parser.add_argument("--use_LSTM", type=bool, default=True , help="use BiLSTM?. Default=True")
parser.add_argument("--Muti_frame", type=bool, default=True, help="use Muti_frame?. Default=True")
parser.add_argument("--Single_frame", type=bool, default=False, help="use Single_frame?. Default=False")
parser.add_argument("--use_Warp", type=bool, default=True, help="use optical flow to Warp?. Default=True")
parser.add_argument('--loss_type', type=str, default='MSE_loss',choices=['MSE_loss','lap_l2'],help='two choices to calculate the loss function')
parser.add_argument("--use_GAN", type=bool, default=True, help="whether train with GAN. Default=True")

#test specifications
parser.add_argument("--use_key_frame", type=bool, default=True, help="use LR(Left Right ) key frames as reference  ?. Default=True")
parser.add_argument("--test_patch_size", type=int, default=128, help=" size of the input image to test. Default=192")
parser.add_argument("--test_numframes", type=int, default=25, help=" how many frames to test. Default=50")
parser.add_argument("--nums_every_video", type=int, default=20, help=" how many frames to test. Default=30")

# Optimization specifications
parser.add_argument('-solver',   type=str,   default="ADAM", choices=["SGD", "ADAM"], 
                    help="optimizer")
parser.add_argument('-momentum', type=float, default=0.9,   
                    help='momentum for SGD')
parser.add_argument('--lr_init', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=200,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')

#GAN specification
parser.add_argument('--D_lr_init', type=float, default=0.0001,
                    help='learning rate')
# parser.add_argument('-gan_weight', type=float,default=1,   
#                     help='weight for GAN loss ')
parser.add_argument('-gp_weight', type=float,default=10,   
                    help='weight for GAN loss ')
parser.add_argument('-gan_type',type=str, default='wgan-gp',choices=['vanilla','lsgan','wgan-gp'],
                     help='type of the GAN ')
parser.add_argument('-D_type',type=str, default='patch_GAN',choices=['patch_GAN','discriminator_vgg_128,96,196','dis_acd','discriminator_vgg_128_SN'],
                     help='type of the Distrimintor ')
parser.add_argument('-in_nc', type=int, default=1,   help='in channels')
parser.add_argument('-nf',    type=int,   default=64,   help='out channels')
parser.add_argument('-norm_type', type=str,default='instance', choices=['batch','instance']  , help='norm type')
parser.add_argument('-mode', type=str,default='CNA', choices=['CNA', 'NAC', 'CNAC'] ,
                     help='compose mode of block')
parser.add_argument('-act_type', type=str,default='leakyrelu', choices=['relu', 'leakyrelu', ] , 
                    help='type of the activate function ')   

 ### loss optinos
parser.add_argument('--VGGLayers',type=str,     default="0123",help="VGG layers for perceptual loss, combinations of ,0,1,2,3")
parser.add_argument('--loss',     type=str,     default="L1",  help="optimizer [Options: SGD, ADAM]")
parser.add_argument('--w_ST',     type=float,   default=100,   help='weight for short-term temporal loss')
parser.add_argument('--w_LT',     type=float,   default=100,   help='weight for long-term temporal loss')
parser.add_argument('--w_VGG',    type=float,   default=0,     help='weight for VGG perceptual loss')
parser.add_argument('--w_Grad',    type=float,   default=10,   help='weight for gradient penalty loss ')
parser.add_argument('-w_GAN',    type=float,   default=0.0001,     help='weight for GAN loss')

# Log specifications
parser.add_argument('--save', type=str, default='save_path',
                    help='file name to save')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')
parser.add_argument('--resume', action='store_true',
                    help='resume from the latest if true')
opt = parser.parse_args()
