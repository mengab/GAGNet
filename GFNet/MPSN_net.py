import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import kaiming_normal
import time
from BiFlowNet.resample2d_package.modules.resample2d import Resample2d


def warp( x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        # print(grid.shape,flo.shape,'...')
        vgrid = grid + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)
        
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask

class MPSN(nn.Module):
    def __init__(self, args,motionCompensator,U_net,use_Warp=True):
        
        super(MPSN, self).__init__()
        print("Creating MPSN")  
        self.U_net         = U_net
        self.flow_warping = Resample2d().cuda()
        self.motionCompensator=motionCompensator
        self.use_Warp=use_Warp
        self.flow_type = args.flow_type
    def forward(self, frame1,frame2,frame3,mask):
        W = frame2.shape[3]
        H = frame2.shape[2]
        if self.use_Warp:
            frame1_r = frame1.repeat(1,3,1,1)
            frame3_r = frame3.repeat(1,3,1,1)
            frame2_r = frame2.repeat(1,3,1,1)
            if self.flow_type=='flownet2.0':
                flow1 = self.motionCompensator(frame2_r, frame1_r)
                flow2 = self.motionCompensator(frame2_r, frame3_r)
                frame1_compensated = self.flow_warping(frame1, flow1)
                frame3_compensated = self.flow_warping(frame3, flow2)
            elif self.flow_type == 'pwc_net':
                flow1 = self.motionCompensator(frame2_r, frame1_r)*20
                flow2 = self.motionCompensator(frame2_r, frame3_r)*20
                # print('before warp',flow1.shape,flow2.shape)
                flow1 = F.upsample(flow1,(W,H),mode='nearest')
                flow2 = F.upsample(flow2,(W,H),mode='nearest')
                # print('after warp')
                frame1_compensated = self.flow_warping(frame1, flow1)
                frame3_compensated = self.flow_warping(frame3, flow2)
            # print('................')
            finally_enhanced_image = self.U_net(frame1_compensated, frame2, frame3_compensated,mask)
        else:
            finally_enhanced_image = self.U_net(frame1, frame2, frame3,mask)
        
        return finally_enhanced_image


class MPSN_test(nn.Module):
    def __init__(self, args,motionCompensator,U_net,use_Warp=True):
        
        super(MPSN_test, self).__init__()
        self.flow_warping      = Resample2d().cuda()
        self.motionCompensator = motionCompensator 
        self.U_net         = U_net
        self.use_Warp=use_Warp
        self.flow_type = args.flow_type
        if self.use_Warp:
            if args.flow_type=='flownet2.0':
                MC_model_filename = args.pre_model_MC_path
                print("===> Load %s" %MC_model_filename)
                checkpoint = torch.load(MC_model_filename,map_location=lambda storage, loc: storage.cuda(args.gpu_id))
                self.motionCompensator.load_state_dict(checkpoint['state_dict'])
            elif args.flow_type =='pwc_net':
                MC_model_filename = args.pre_model_PWC_path
                print("===> Load %s" %MC_model_filename)
                data = torch.load(MC_model_filename,map_location=lambda storage, loc: storage.cuda(args.gpu_id))
                if 'state_dict' in data.keys():
                    self.motionCompensator.load_state_dict(data['state_dict'])
                else:
                    self.motionCompensator.load_state_dict(data)
                # checkpoint = torch.load(MC_model_filename,map_location=lambda storage, loc: storage.cuda(args.gpu_id))
                # self.motionCompensator.load_state_dict(checkpoint['state_dict'])

        test_DB_model_path=args.test_model_DB_path
        self.U_net.load_state_dict(torch.load(test_DB_model_path,map_location=lambda storage, loc: storage.cuda(args.gpu_id)))
        print("===> Load %s" %test_DB_model_path)
        print('load finished......')
        self.count=0
    def forward(self, frame1,frame2,frame3,mask):
        W = frame2.shape[3]
        H = frame2.shape[2]
        self.count+=1
        if self.use_Warp:
            flow_start=time.time()
            frame1_r = frame1.repeat(1,3,1,1)
            frame3_r = frame3.repeat(1,3,1,1)
            frame2_r = frame2.repeat(1,3,1,1)
            if self.flow_type=='flownet2.0':
                
                flow1 = self.motionCompensator(frame2_r, frame1_r)
                flow2 = self.motionCompensator(frame2_r, frame3_r)
                flow_end=time.time()

                warp_start=time.time()
                frame1_compensated = self.flow_warping(frame1, flow1)
                frame3_compensated = self.flow_warping(frame3, flow2)
                warp_end=time.time()
                if self.count==5:
                    print('flownet2.0_flow_time:%.04f'%(flow_end-flow_start))
                    print('flownet2.0_warp_time:%.04f'%(warp_end-warp_start))


            elif self.flow_type == 'pwc_net':
                
                flow1 = self.motionCompensator(frame2_r, frame1_r)*20
                flow2 = self.motionCompensator(frame2_r, frame3_r)*20
                
                flow1 = F.upsample(flow1,(W,H),mode='nearest')
                flow2 = F.upsample(flow2,(W,H),mode='nearest')
                flow_end=time.time()
                # frame1_compensated = self.flow_warping(frame1, flow1)
                # frame3_compensated = self.flow_warping(frame3, flow2)
                warp_start=time.time()
                frame1_compensated = self.flow_warping(frame1, flow1)
                frame3_compensated = self.flow_warping(frame3, flow2)
                warp_end=time.time()
                if self.count==5:
                    print('pwc_flow_time:%.04f'%(flow_end-flow_start))
                    print('pwc_warp_time:%.04f'%(warp_end-warp_start))

            U_net_start=time.time()
            finally_enhanced_image = self.U_net(frame1_compensated, frame2, frame3_compensated,mask)
            U_net_end=time.time()
            if self.count==5:
                print('db_net:%.04f'%(U_net_end-U_net_start))

            return finally_enhanced_image
        else:
            U_net_start=time.time()
            # print(frame1.shape, frame2.shape, frame3.shape)
            finally_enhanced_image = self.U_net(frame1, frame2, frame3,mask)
            U_net_end=time.time()
            if self.count==5:
                print('db_net:%.04f'%(U_net_end-U_net_start))

            return finally_enhanced_image