import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataloader.h5_dataset_T import DatasetFromHdf5_3_data
from option import opt
from GFNet.slice_net import DeBlockNet
from GFNet.MPSN_net import MPSN
from GFNet.Discriminator import define_D
import time
import numpy as np
import BiFlowNet
import loss
import torchvision.transforms as transforms

def main():
    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)

    torch.cuda.manual_seed_all(opt.seed)
    # if opt.use_GAN:
    #     trans = transforms.Compose(transforms = [   
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))
    #     ])
    # else:
    #      trans = transforms.Compose(transforms = [   
    #         transforms.ToTensor()
    #     ])

    print("===> Loading datasets")
    data_path1 = "./data/train_xx1.h5"
    data_path2 = "./data/train_xx2.h5"
    data_path3 = "./data/train_xx3.h5"
    train_set  = DatasetFromHdf5_3_data(data_root_1= data_path1,data_root_2=data_path2,data_root_3=data_path3)

    training_data_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=opt.batch_size, shuffle=True,drop_last=True)
    print("===> Building model")
    wn=None
    if opt.wn==True:
        print('===>use weightNorm')
        wn = lambda x: torch.nn.utils.weight_norm(x)
        model_DB = DeBlockNet(wn=wn, num_sliceBlock=6,is_train=True,input_size=128,use_LSTM=True,use_mask =True,mask_num_sliceBlock=2)
    else:
        model_DB = DeBlockNet(wn=None, num_sliceBlock=6,is_train=True,input_size=128,use_LSTM=True,use_mask =True,mask_num_sliceBlock=2).cuda()
    if opt.use_Warp:
        if opt.flow_type=='flownet2.0':
            print('===>use optical flow(flownet2.0) to warp')                
            model_MC = BiFlowNet.FlowNet2(opt, requires_grad=False)                   
            MC_model_path =opt.pre_model_MC_path
            print("===> Load %s" %MC_model_path)
            checkpoint = torch.load(MC_model_path,map_location=lambda storage, loc: storage)
            model_MC.load_state_dict(checkpoint['state_dict'])
        elif opt.flow_type=='pwc_net':
            print('===>use optical flow(pwc_net) to warp') 
            model_MC = BiFlowNet.PWCNet(requires_grad=False,is_train=False)
            MC_model_path = opt.pre_model_PWC_path
            print("===> Load %s" %MC_model_path)
            data = torch.load(MC_model_path,map_location=lambda storage, loc: storage)
            if 'state_dict' in data.keys():
                model_MC.load_state_dict(data['state_dict'])
            else:
                model_MC.load_state_dict(data)
            model_MC.eval()

    model_DB_path = './pretrained_models/pretrained_model_LD/pretrained_model_LD32_GAN.pth'
    model_DB.load_state_dict(torch.load(model_DB_path,map_location=lambda storage, loc: storage))
    print("===> Load %s" % model_DB_path)

    #G_set_up
    model_MPSN = MPSN(args=opt,motionCompensator=model_MC,U_net=model_DB,use_Warp=opt.use_Warp)
    criterion = nn.MSELoss()
    criterion_L1 = nn.L1Loss()
    print("===> Setting GPU")
    GPU_list = list(map(int,opt.GPUs))
    if len(GPU_list) > 1:
        model_MPSN        = nn.DataParallel(model_MPSN,device_ids=GPU_list)
        model_MPSN        = model_MPSN.cuda()
        criterion = criterion.cuda()
        criterion_L1=criterion_L1.cuda()
    else:
        model_MPSN = model_MPSN.cuda()
        criterion = criterion.cuda()
        criterion_L1=criterion_L1.cuda()
    print("===> Setting Optimizer")
    if opt.solver == 'SGD':
        param_2 = {'params':model_DB.parameters(),'lr':opt.lr_init}
        param_groups = [param_2]
        optimizer = optim.SGD(param_groups)
    elif opt.solver == 'ADAM':
        param_2 = {'params':model_DB.parameters(),'lr':opt.lr_init}
        param_groups = [param_2]
        optimizer = optim.Adam(param_groups)
    else:
        raise Exception("Not supported solver (%s)" %opt.solver)

    #D_set_up
    model_D = None
    if opt.use_GAN:
        print('===> use GAN')
        model_D = define_D(opt)
        if len(GPU_list) > 1:
            model_D = nn.DataParallel(model_D,device_ids=GPU_list).cuda()
        else:
            model_D=model_D.cuda()
            
        if opt.solver == 'SGD':
            param_D = {'params':model_D.parameters(),'lr':opt.D_lr_init}   
            optimizer_D = optim.SGD([param_D])
        elif opt.solver == 'ADAM':
            param_D = {'params':model_D.parameters(),'lr':opt.D_lr_init}
            optimizer_D = optim.Adam([param_D])
        else:
            raise Exception("Not supported solver (%s)" %opt.solver)

    print("===>start  Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        lr,lr_D = adjust_learning_rate(epoch-1)
        optimizer.param_groups[0]['lr']  = lr
        optimizer_D.param_groups[0]['lr'] = lr_D
        if opt.use_GAN:
            print("Epoch = {}, lr = {} lr_D={}".format(epoch, optimizer.param_groups[0]["lr"],optimizer_D.param_groups[0]["lr"]))
        else:
            print("Epoch = {}, lr = {} ".format(epoch, optimizer.param_groups[0]["lr"]))
        total_count=0
        if opt.use_GAN:
            total_count=train_GAN(training_data_loader, optimizer, model_MPSN,optimizer_D, model_D, criterion,criterion_L1, epoch,total_count)
        else:
            # total_count=train(training_data_loader, optimizer, model_MPSN, criterion,criterion_L1, epoch,total_count)
            raise Exception("Not supported GAN" )
        save_checkpoint(model_D , model_DB,epoch)
        
def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by 10 every num(opt.step) epochs"""
    lr  = opt.lr_init * (0.1 ** (epoch // opt.step))
    lr_D= opt.D_lr_init * (0.1 ** (epoch // opt.step))
    return lr,lr_D

# def train(training_data_loader, optimizer, model_G, criterion,criterion_L1, epoch,total_count):
#     model_G.train()
#     for iteration, batch in enumerate(training_data_loader, 0):
#         total_count+=1
#         time_start = time.time()
#         input1,input2,input3, target = Variable(batch[0]),Variable(batch[1]),Variable(batch[2]), Variable(batch[3])
#         # print(input1.shape)
        
#         input1 = input1.cuda()
#         input2 = input2.cuda()
#         input3 = input3.cuda()
#         target = target.cuda()

#         enhanced_image = model_G(input1,input2,input3)
#         loss = criterion(enhanced_image , target)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         time_end = time.time()
#         time_step = time_end - time_start
#         if iteration%10 == 0:
#             loss_L1 = criterion_L1(enhanced_image , target)*255.0
#             total_time = (time_step*len(training_data_loader))*(opt.nEpochs-epoch)/3600
#             print("===> Epoch[{}/{}]({}/{}): lr:{} Loss: {:.10f}  Loss_L1:{:.10f} time_step:{:.10f} total_time:{:.4f}".format(epoch,opt.nEpochs, total_count, len(training_data_loader),optimizer.param_groups[0]["lr"], loss.item(),loss_L1.item(),time_step, total_time))
#     return total_count

def train_GAN(training_data_loader, optimizer, model_G,optimizer_D, model_D, criterion,criterion_L1, epoch,total_count):
    model_G.train()
    model_D.train()
    Base_Loss = loss.Basic_Loss(opt)
    for iteration, data in enumerate(training_data_loader, 0):
        total_count+=1
        time_start = time.time()
        # input1,input2,input3, target = Variable(batch[0]),Variable(batch[1]),Variable(batch[2]), Variable(batch[3])
        data_pre_value = Variable(data[0]).cuda()
        data_cur_value = Variable(data[1]).cuda()
        data_aft_value = Variable(data[2]).cuda()
        data_mask_value = Variable(data[3]).cuda()
        data_label_value = Variable(data[4]).cuda()
        # print(input1.shape)
        

        #train_G
        for i in range(1):
            optimizer.zero_grad()
            loss_total_G = 0
            loss_pix =0
            loss_feature=0
            loss_gNet =0
            enhanced_image_fake = model_G(data_pre_value, data_cur_value, data_aft_value, data_mask_value)
            loss_pix = criterion(enhanced_image_fake, data_label_value)
            if opt.w_VGG>0:
                loss_feature = Base_Loss.VGG_loss(enhanced_image_fake,data_label_value,criterion_L1,loss_feature)
            if opt.use_GAN and opt.w_GAN>0:
                loss_gNet = Base_Loss.ganloss(input_fake =enhanced_image_fake,input_real=data_cur_value,model_D=model_D,Net='G')

            loss_total_G = loss_pix +  0.0001*loss_gNet #no perceptual loss
            # loss_total_G=loss_gNet
            loss_total_G.backward()
            optimizer.step()
        #train_D
        if opt.use_GAN:
            optimizer_D.zero_grad()
            loss_D = Base_Loss.ganloss(input_fake =enhanced_image_fake,input_real=data_label_value,model_D=model_D,Net='D')
            
            loss_D.backward(retain_graph=True)
            optimizer_D.step()

        time_end = time.time()
        time_step = time_end - time_start
        if iteration%10 == 0:
            loss_L1 = criterion_L1(enhanced_image_fake , data_label_value)*255.0
            total_time = (time_step*len(training_data_loader))*(opt.nEpochs-epoch)/3600
            if opt.w_VGG>0 and opt.w_GAN>0:
                print("===> Epoch[{}/{}]({}/{}): lr:{} lr_D:{} Loss_L2: {:.4f}  Loss_L1:{:.4f} loss_VGG:{:.4f} loss_gNet:{:.4f} loss_D:{:.4f} time_step:{:.4f} total_time:{:.4f}"\
                            .format(epoch,opt.nEpochs, total_count, len(training_data_loader),optimizer.param_groups[0]["lr"],optimizer_D.param_groups[0]["lr"],\
                             loss_pix.item(),loss_L1.item(),loss_feature,loss_gNet,loss_D,time_step, total_time))
            elif opt.w_VGG>0 and opt.w_GAN==0:
                print("===> Epoch[{}/{}]({}/{}): lr:{} lr_D:{} Loss_L2: {:.4f}  Loss_L1:{:.4f} loss_VGG:{:.4f}  time_step:{:.4f} total_time:{:.4f}"\
                        .format(epoch,opt.nEpochs, total_count, len(training_data_loader),optimizer.param_groups[0]["lr"],optimizer_D.param_groups[0]["lr"],\
                            loss_pix.item(),loss_L1.item(),loss_feature,time_step, total_time))
            elif opt.w_VGG==0 and opt.w_GAN>0:
                print("===> Epoch[{}/{}]({}/{}): lr:{} lr_D:{} Loss_L2: {:.4f}  Loss_L1:{:.4f} loss_gNet:{:.4f} loss_D:{:.4f} time_step:{:.4f} total_time:{:.4f}"\
                            .format(epoch,opt.nEpochs, total_count, len(training_data_loader),optimizer.param_groups[0]["lr"],optimizer_D.param_groups[0]["lr"],\
                             loss_pix.item(),loss_L1.item(),loss_gNet,loss_D,time_step, total_time))
            elif opt.w_VGG==0 and opt.w_GAN==0:
                print("===> Epoch[{}/{}]({}/{}): lr:{} Loss: {:.4f}  Loss_L1:{:.4f} time_step:{:.4f} total_time:{:.4f}"\
                .format(epoch,opt.nEpochs, total_count, len(training_data_loader),optimizer.param_groups[0]["lr"], \
                loss_pix.item(),loss_L1.item(),time_step, total_time))
            
            else:
                raise NotImplementedError('error in weight')
    return total_count

def save_checkpoint(model_1=None,model_2=None, epoch=None):
    if(model_1):
        model_1_out_path = "./pretrained_models/D_GAN_0.0001/" + "model_D_epoch_{}.pth".format(epoch)
        os.makedirs("./pretrained_models/D_GAN_0.0001/",exist_ok=True)
        torch.save(model_1.state_dict(), model_1_out_path)
    if(model_2):
        model_2_out_path = "./pretrained_models/G_GAN_0.0001/" + "model_G_epoch_{}.pth".format(epoch)
        os.makedirs("./pretrained_models/G_GAN_0.0001/",exist_ok=True)
        torch.save(model_2.state_dict(), model_2_out_path)
    
if __name__ == "__main__":
    print(list(map(int,opt.GPUs)))
    opt.gpu_id = list(map(int,opt.GPUs))[0]
    torch.cuda.set_device(opt.gpu_id)
    cudnn.benchmark = True
    opt.rgb_max = 1.0
    opt.fp16 = False
    # net_graph()
    main()
