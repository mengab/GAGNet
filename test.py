#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from numpy import *
# from scipy.misc import imresize
from skimage.measure import compare_ssim
import torch.backends.cudnn as cudnn
import cv2
import glob
import os
import argparse
from GFNet.slice_net import DeBlockNet
from GFNet.MPSN_net import *
from GFNet.multiscaleloss import multiscaleEPE
from option import opt
import BiFlowNet
import torch
import copy


def test_batch(data_Y,mask_Y, label_Y,start, batch_size=1):
    # print(data_Y[0].shape)
    if opt.mode_type=='AI':
        pre= start - 1
        aft =start + 1
    elif opt.mode_type=='LD':
        pre = ((start)//4)*4
        aft = ((start) // 4) * 4 + 4
    data_pre = (data_Y[pre])/255.0
    data_cur = (data_Y[start])/255.0
    data_aft =(data_Y[aft])/255.0

    mask    = (mask_Y[start])/255.0

    label    = (label_Y[start])

    start+=1
    return  data_pre,data_cur,data_aft,mask,label,start


def get_data(data_path,mask_path,label_path,video_index,num_frame):
    data_Y  = []
    mask_Y  = []
    label_Y = []

    comp_data = np.sort(glob.glob(data_path[video_index]+'/*.npy'))
    comp_mask = np.sort(glob.glob(mask_path[video_index]+'/*.npy'))
    label_data= np.sort(glob.glob(label_path[video_index]+'/*.npy'))

    for i in range(len(comp_data)):
        data = np.load(comp_data[i],'r')
        mask = np.load(comp_mask[i],'r')
        data_label = np.load(label_data[i],'r')
        data_Y.append(data)
        mask_Y.append(mask)
        label_Y.append(data_label)


    return data_Y,mask_Y,label_Y

def PSNR(img1, img2):
    mse = np.mean( (img1.astype(np.float32) - img2.astype(np.float32)) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def patch_test(data_path,mask_path,label_path,net_G,patch_size=(128,128),f_txt=None):
    video_num=4
    ave_diff_psnr    =0.
    ave_diff_ssim    =0.
    ave_psnr_pre_gt  =0.
    ave_psnr_data_gt =0.
    ave_ssim_pre_gt  =0.
    ave_ssim_data_gt =0.
    for video_index in range(video_num):

        data_Y,mask_Y,label_Y = get_data(data_path=data_path,mask_path=mask_path,label_path=label_path, video_index=video_index,num_frame=opt.test_numframes)

        start =1
        patch_size = patch_size
        psnr_diff_sum = 0
        psnr_pre_gt_sum=0
        psnr_data_gt_sum=0

        ssim_diff_sum = 0
        ssim_pre_gt_sum=0
        ssim_data_gt_sum =0
        nums_every_video =opt.nums_every_video
        nums=0
        for itr in range(0, nums_every_video):
            if opt.mode_type == 'LD':
                if (start) % 4 != 0:
                    nums+=1
                    if opt.use_key_frame:
                        data_pre, data_cur, data_aft,mask, label, start = test_batch(data_Y=data_Y,mask_Y=mask_Y,label_Y=label_Y, start=start, batch_size=1)
                    height = data_pre.shape[2]
                    width = data_pre.shape[3]
                    image_synthesis = np.zeros((height , width ), np.float32)
                    image_mask = np.zeros((height , width ), np.float32)
                    num_rows = height // patch_size[0]
                    num_cols = width // patch_size[1]
                    row_redundant = int(height % patch_size[0])
                    col_redundant = int(width % patch_size[1])
                    result_row_redundant = int((height ) % (patch_size[0] ))
                    result_col_redundant = int((width ) % (patch_size[1] ))
                    row_start = 0
                    result_row_start = 0
                    patch_step = 0
                    for row in range(num_rows + 1):
                        col_start = 0
                        result_col_start = 0
                        for col in range(num_cols + 1):
                            patch_step += 1
                            data_pre_value_patch  = torch.from_numpy(data_pre[:, :,  row_start:row_start + patch_size[0],col_start:col_start + patch_size[1]]).float().cuda()
                            data_cur_value_patch  = torch.from_numpy(data_cur[:, :,  row_start:row_start + patch_size[0],col_start:col_start + patch_size[1]]).float().cuda()
                            data_aft_value_patch  = torch.from_numpy(data_aft[:, :,  row_start:row_start + patch_size[0],col_start:col_start + patch_size[1]]).float().cuda()
                            data_mask_value_patch = torch.from_numpy(mask[:,:, row_start:row_start + patch_size[0],col_start:col_start + patch_size[1]]).float().cuda()
                            if opt.use_Warp:
                                start_time = time.time()
                                fake_image = net_G(data_pre_value_patch, data_cur_value_patch, data_aft_value_patch, data_mask_value_patch)
                                end_time   = time.time()
                            else:
                                start_time = time.time()
                                fake_image= net_G(data_pre_value_patch, data_cur_value_patch, data_aft_value_patch,data_mask_value_patch)
                                end_time=time.time()
                            if  patch_step==3 and itr==1:
                                print('%dx%d time: %04f'%(patch_size[0],patch_size[1],end_time-start_time))
                            fake_image_numpy = fake_image.detach().cpu().numpy()
                            fake_image_numpy = np.squeeze(fake_image_numpy)
                            image_synthesis[result_row_start:result_row_start + patch_size[0] ,result_col_start:result_col_start + patch_size[1] ] += copy.copy(np.array(fake_image_numpy))
                            image_mask[result_row_start:result_row_start + patch_size[0] ,result_col_start:result_col_start + patch_size[1] ] += 1.0
                            if col == 0:
                                col_start = col_start + col_redundant
                                result_col_start = result_col_start + result_col_redundant
                            else:
                                col_start += patch_size[1]
                                result_col_start += patch_size[1]
                        if row == 0:
                            row_start += row_redundant
                            result_row_start = result_row_start + result_row_redundant
                        else:
                            row_start += patch_size[0]
                            result_row_start += patch_size[0]
                    finally_image = ((((image_synthesis / image_mask)*255.0))).astype(np.float32)
                    finally_image=np.squeeze(finally_image)
                    os.makedirs(opt.save_result_path+'/%02d/%02d'%(video_index+1,itr+2),exist_ok = True)
                    cv2.imwrite(opt.save_result_path+'/%02d/%02d/Enhanced_%02d.png'%(video_index+1,itr+2,itr+2),finally_image.astype(np.uint8))
                    data_cur_image = (np.squeeze(data_cur)*255.0).astype(np.float32)
                    label = np.squeeze(label).astype(np.float32)
                    cv2.imwrite(opt.save_result_path+'/%02d/%02d/Label_%02d.png'%(video_index+1,itr+2,itr+2),label.astype(np.uint8))
                    cv2.imwrite(opt.save_result_path+'/%02d/%02d/HEVC_%02d.png'%(video_index+1,itr+2,itr+2),data_cur_image.astype(np.uint8))
                    psnr_pre_gt  = PSNR(finally_image, label)
                    psnr_data_gt = PSNR(data_cur_image, label)
                    psnr_diff = psnr_pre_gt - psnr_data_gt
                    psnr_diff_sum +=psnr_diff
                    psnr_pre_gt_sum+=psnr_pre_gt
                    psnr_data_gt_sum+=psnr_data_gt
                    ssim_pre_gt = compare_ssim(finally_image.astype(np.uint8), label.astype(np.uint8))
                    ssim_data_gt = compare_ssim(data_cur_image.astype(np.uint8), label.astype(np.uint8))
                    ssim_diff = ssim_pre_gt - ssim_data_gt
                    ssim_diff_sum += ssim_diff
                    ssim_pre_gt_sum+=ssim_pre_gt
                    ssim_data_gt_sum+=ssim_data_gt
                    print('psnr_pre_gt:{:.04f} psnr_data_gt:{:.04f}  psnr_diff:{:.04f} ssim_pre_gt:{:.04f} ssim_data_gt:{:.04f}  ssim_diff:{:.04f}'.format(psnr_pre_gt,psnr_data_gt,psnr_diff,ssim_pre_gt,ssim_data_gt,ssim_diff),file=f_txt)
                else:
                    start+=1
            elif opt.mode_type == 'AI':
                if (start)%1 == 0:
                    nums+=1
                    patch_size=128
                    if opt.use_key_frame:
                        data_pre, data_cur, data_aft,mask, label, start = test_batch(data_Y=data_Y,mask_Y=mask_Y,label_Y=label_Y, start=start, batch_size=1)
                    height = data_pre.shape[2]
                    width = data_pre.shape[3]
                    image_synthesis = np.zeros((height , width ), np.float32)
                    image_mask = np.zeros((height , width ), np.float32)
                    num_rows = height // patch_size
                    num_cols = width // patch_size
                    row_redundant = int(height % patch_size)
                    col_redundant = int(width % patch_size)
                    result_row_redundant = int((height ) % (patch_size ))
                    result_col_redundant = int((width ) % (patch_size ))
                    row_start = 0
                    result_row_start = 0
                    patch_step = 0
                    for row in range(num_rows + 1):
                        col_start = 0
                        result_col_start = 0
                        for col in range(num_cols + 1):
                            patch_step += 1
                            data_pre_value_patch  = torch.from_numpy(data_pre[:, :,  row_start:row_start + patch_size,col_start:col_start + patch_size]).float().cuda()
                            data_cur_value_patch  = torch.from_numpy(data_cur[:, :,  row_start:row_start + patch_size,col_start:col_start + patch_size]).float().cuda()
                            data_aft_value_patch  = torch.from_numpy(data_aft[:, :,  row_start:row_start + patch_size,col_start:col_start + patch_size]).float().cuda()
                            data_mask_value_patch = torch.from_numpy(mask[:,:, row_start:row_start + patch_size,col_start:col_start + patch_size]).float().cuda()
                            if opt.use_Warp:
                                start_time = time.time()
                                fake_image = net_G(data_pre_value_patch, data_cur_value_patch, data_aft_value_patch, data_mask_value_patch)
                                end_time   = time.time()
                            else:
                                start_time = time.time()
                                fake_image= net_G(data_pre_value_patch, data_cur_value_patch, data_aft_value_patch,data_mask_value_patch)
                                end_time=time.time()
                            if  patch_step==3 and itr==1:
                                print('%dx%d time: %04f'%(opt.test_patch_size,opt.test_patch_size,end_time-start_time))
                            fake_image_numpy = fake_image.detach().cpu().numpy()
                            fake_image_numpy = np.squeeze(fake_image_numpy)
                            image_synthesis[result_row_start:result_row_start + patch_size ,result_col_start:result_col_start + patch_size ] += copy.copy(np.array(fake_image_numpy))
                            image_mask[result_row_start:result_row_start + patch_size ,result_col_start:result_col_start + patch_size ] += 1.0
                            if col == 0:
                                col_start = col_start + col_redundant
                                result_col_start = result_col_start + result_col_redundant
                            else:
                                col_start += patch_size
                                result_col_start += patch_size
                        if row == 0:
                            row_start += row_redundant
                            result_row_start = result_row_start + result_row_redundant
                        else:
                            row_start += patch_size
                            result_row_start += patch_size
                    finally_image = ((((image_synthesis / image_mask)*255.0))).astype(np.float32)
                    finally_image=np.squeeze(finally_image)
                    os.makedirs(opt.save_result_path+'/%02d/%02d'%(video_index+1,itr+2),exist_ok = True)
                    cv2.imwrite(opt.save_result_path+'/%02d/%02d/Enhanced_%02d.png'%(video_index+1,itr+2,itr+2),finally_image.astype(np.uint8))
                    data_cur_image = (np.squeeze(data_cur)*255.0).astype(np.float32)
                    label = np.squeeze(label).astype(np.float32)
                    cv2.imwrite(opt.save_result_path+'/%02d/%02d/Label_%02d.png'%(video_index+1,itr+2,itr+2),label.astype(np.uint8))
                    cv2.imwrite(opt.save_result_path+'/%02d/%02d/HEVC_%02d.png'%(video_index+1,itr+2,itr+2),data_cur_image.astype(np.uint8))
                    psnr_pre_gt  = PSNR(finally_image, label)
                    psnr_data_gt = PSNR(data_cur_image, label)
                    psnr_diff = psnr_pre_gt - psnr_data_gt
                    psnr_diff_sum +=psnr_diff#
                    psnr_pre_gt_sum+=psnr_pre_gt
                    psnr_data_gt_sum+=psnr_data_gt
                    ssim_pre_gt = compare_ssim(finally_image.astype(np.uint8), label.astype(np.uint8))
                    ssim_data_gt = compare_ssim(data_cur_image.astype(np.uint8), label.astype(np.uint8))
                    ssim_diff = ssim_pre_gt - ssim_data_gt
                    ssim_diff_sum += ssim_diff
                    ssim_pre_gt_sum+=ssim_pre_gt
                    ssim_data_gt_sum+=ssim_data_gt
                    print('psnr_pre_gt:{:.04f} psnr_data_gt:{:.04f}  psnr_diff:{:.04f} ssim_pre_gt:{:.04f} ssim_data_gt:{:.04f}  ssim_diff:{:.04f}'.format(psnr_pre_gt,psnr_data_gt,psnr_diff,ssim_pre_gt,ssim_data_gt,ssim_diff),file=f_txt)
                else:
                    start+=1

        print( label_path[video_index],'----',"video_index:%02d,psnr_ave:%.04f,ssim_ave:%.04f"%(video_index,psnr_diff_sum/nums,ssim_diff_sum/nums))
        print(' video_index:{:2d} psnr_pre_gt_ave:{:.04f} psnr_data_gt_ave:{:.04f}  psnr_diff_ave:{:.04f} ssim_pre_gt_ave:{:.04f} ssim_data_gt_ave:{:.04f} ssim_diff_ave:{:.04f}'.format(video_index,psnr_pre_gt_sum/nums,psnr_data_gt_sum/nums,psnr_diff_sum/nums,ssim_pre_gt_sum/nums,ssim_data_gt_sum/nums,ssim_diff_sum/nums),file=f_txt)
        print('{}'.format(label_path[video_index]),file=f_txt)
        f_txt.write('\r\n')
        ave_diff_psnr+=psnr_diff_sum/nums
        ave_diff_ssim+=ssim_diff_sum/nums
        ave_psnr_pre_gt  +=psnr_pre_gt_sum/nums
        ave_psnr_data_gt +=psnr_data_gt_sum/nums
        ave_ssim_pre_gt  +=ssim_pre_gt_sum/nums
        ave_ssim_data_gt +=ssim_data_gt_sum/nums
    # print('ave_psnr_pre_gt:',ave_psnr_pre_gt/video_num,'ave_psnr_data_gt:',ave_psnr_data_gt/video_num,'ave_psnr:',ave_diff_psnr/video_num,'ave_ssim_pre_gt:',ave_ssim_pre_gt/video_num,'ave_ssim_data_gt:',ave_ssim_data_gt /video_index, 'ave_ssim:',ave_diff_ssim/video_num)
    print(' ave_psnr_pre_gt:{:.04f} ave_psnr_data_gt:{:.04f}  psnr_diff_ave:{:0.4f} ave_ssim_pre_gt:{:.04f} ave_ssim_data_gt:{:.04f} ssim_diff_ave:{:.04f}'.format(ave_psnr_pre_gt/video_num,ave_psnr_data_gt/video_num,ave_diff_psnr/video_num,ave_ssim_pre_gt/video_num,ave_ssim_data_gt /video_num,ave_diff_ssim/video_num))
    # print( ave_ssim_pre_gt,ave_ssim_data_gt)
    print(' ave_psnr_pre_gt:{:.04f} ave_psnr_data_gt:{:.04f}  psnr_diff_ave:{:0.4f} ave_ssim_pre_gt:{:.04f} ave_ssim_data_gt:{:.04f} ssim_diff_ave:{:.04f}'.format(ave_psnr_pre_gt/video_num,ave_psnr_data_gt/video_num,ave_diff_psnr/video_num,ave_ssim_pre_gt/video_num,ave_ssim_data_gt /video_num,ave_diff_ssim/video_num), file=f_txt)

if __name__ == "__main__":

    torch.cuda.set_device(opt.gpu_id)
    cudnn.benchmark = True
    opt.rgb_max = 1.0
    opt.fp16 = False
    os.makedirs('./Test_result/',exist_ok=True)
    txt_name = opt.save_txt_path
    if os.path.isfile(txt_name):
        f = open(txt_name, 'w+')
    else:
        os.mknod(txt_name)
        f = open(txt_name, 'w+')

    data_path =  './Input_mask/'+ opt.input_data_path +'/data/'
    mask_path =  './Input_mask/'+ opt.input_data_path +'/mask/'
    label_path = './Input_mask/label/'
    data_path_1  = np.sort(glob.glob(data_path  + '*'))
    mask_path_1  = np.sort(glob.glob(mask_path  + '*'))
    label_path_1 = np.sort(glob.glob(label_path + '*'))
    # print(label_path_1)
    model_MC = None
    model_DB = None

    if opt.use_Warp:
        if opt.flow_type=='flownet2.0':
            print("===> use optical flow(flownet2.0) to warp")
            model_MC = BiFlowNet.FlowNet2(opt, requires_grad=False).cuda()
            model_MC.eval()
        elif opt.flow_type=='pwc_net':
            print("===> use optical flow(pwc_net) to warp")
            model_MC = BiFlowNet.PWCNet(requires_grad=False,is_train=False).cuda()
            model_MC.eval()

    patch_size =opt.test_patch_size
    height = 128
    width  = 128
    input_size = (height,width)
    if opt.wn==True:
        print('===>use weightNorm')
        wn = lambda x: torch.nn.utils.weight_norm(x)
        model_DB = DeBlockNet(wn=wn, num_sliceBlock=6,is_train=False,input_size=128,use_LSTM=True,use_mask =True,mask_num_sliceBlock=2)
        # model_DB = DeBlockNet(wn=wn, num_sliceBlock=6,is_train=True,input_size=64,use_LSTM=True,use_mask =True).cuda()
    else:
        model_DB = DeBlockNet(wn=None, num_sliceBlock=6,is_train=False,input_size=128,use_LSTM=True,use_mask =True).cuda()
    model_DB=model_DB.cuda()
    model_DB.eval()
    model_MPSN = MPSN_test(args=opt,motionCompensator=model_MC,U_net=model_DB,use_Warp=opt.use_Warp)
    model_MPSN=model_MPSN.cuda()
    model_MPSN.eval()
    patch_test(data_path=data_path_1,mask_path=mask_path_1,label_path=label_path_1,net_G=model_MPSN,patch_size=input_size,f_txt = f)
    f.close()
