import torch
import numpy as np
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter



class ZeroPad(nn.Module):
    def __init__(self, pad=1, USE_GPU=True):
        super(ZeroPad, self).__init__()
        self.USE_GPU = USE_GPU
        self.pad = pad

    def forward(self, x):
        pad_row = Variable(torch.FloatTensor(x.size(0), x.size(1), self.pad, x.size(3)).zero_()).cuda()
        x = torch.cat((pad_row, x, pad_row), 2)
        pad_col = Variable(torch.FloatTensor(x.size(0), x.size(1), x.size(2), self.pad).zero_()).cuda()
        x = torch.cat((pad_col, x, pad_col), 3)
        return x

class CubePad(nn.Module):
    def __init__(self, USE_GPU = True):
        super(CubePad, self).__init__()
        self.USE_GPU = USE_GPU

    def flip(self, tensor):
        idx = [i for i in range(tensor.size(1)-1, -1, -1)]
        
        if self.USE_GPU:
            idx = Variable(torch.cuda.LongTensor(idx))
        else:
            idx = Variable(torch.LongTensor(idx))

        inverted_tensor = tensor.index_select(1, idx)
        return inverted_tensor

    def forward(self, x):

        f_back = x[0]
        f_down = x[1]
        f_front = x[2]
        f_left = x[3]
        f_right = x[4]
        f_top = x[5]         

    
        #faces order: 123456 = bdflrt (back, bottom, front, left, right, top)
 
        '''
        4 plates:  /  t  /
                  |=====| r
                 l| /   | / 
                  |=====|/
                     d
        '''
        _t12 = torch.cat([torch.unsqueeze(self.flip(f_top[:,0,:]),0), torch.unsqueeze(f_front[:,-1,:],0)],0)
        _t123 = torch.cat([_t12, torch.unsqueeze(f_top[:,-1,:],0)],0)
        _t1234 = torch.cat([_t123, torch.unsqueeze(f_top[:,:,0],0)],0)
        _t12345 = torch.cat([_t1234, torch.unsqueeze(self.flip(f_top[:,:,-1]),0)],0)
        _t123456 = torch.cat([_t12345, torch.unsqueeze(self.flip(f_back[:,0,:]),0)],0)

        _d12 = torch.cat([torch.unsqueeze(self.flip(f_down[:,-1,:]),0), torch.unsqueeze(self.flip(f_back[:,-1,:]),0)],0)
        _d123 = torch.cat([_d12, torch.unsqueeze(f_down[:,0,:],0)],0)
        _d1234 = torch.cat([_d123, torch.unsqueeze(self.flip(f_down[:,:,0]),0)],0)
        _d12345 = torch.cat([_d1234, torch.unsqueeze(f_down[:,:,-1],0) ],0)
        _d123456 = torch.cat([_d12345, torch.unsqueeze(f_front[:,0,:],0)],0)

        _l12 = torch.cat([torch.unsqueeze(f_right[:,:,-1],0), torch.unsqueeze(self.flip(f_left[:,-1,:]),0)],0)
        _l123 = torch.cat([_l12, torch.unsqueeze(f_left[:,:,-1],0)],0)
        _l1234 = torch.cat([_l123, torch.unsqueeze(f_back[:,:,-1],0)],0)
        _l12345 = torch.cat([_l1234, torch.unsqueeze(f_front[:,:,-1],0)],0)
        _l123456 = torch.cat([_l12345, torch.unsqueeze(f_left[:,0,:],0)],0)

        _r12 = torch.cat([torch.unsqueeze(f_left[:,:,0],0), torch.unsqueeze(f_right[:,-1,:],0)],0)
        _r123 = torch.cat([_r12, torch.unsqueeze(f_right[:,:,0],0)],0)
        _r1234 = torch.cat([_r123, torch.unsqueeze(f_front[:,:,0],0)],0)
        _r12345 = torch.cat([_r1234, torch.unsqueeze(f_back[:,:,0],0)],0)
        _r123456 = torch.cat([_r12345, torch.unsqueeze(self.flip(f_right[:,0,:]),0)],0)
        # 6 x c x w for each
        
        p_tr = torch.unsqueeze(torch.mul(torch.add(_t123456[:,:,-1],_r123456[:,:,0]),0.5),2)
        p_tl = torch.unsqueeze(torch.mul(torch.add(_t123456[:,:,0],_l123456[:,:,0]),0.5),2)
        p_dr = torch.unsqueeze(torch.mul(torch.add(_d123456[:,:,-1],_r123456[:,:,-1]),0.5),2)
        p_dl = torch.unsqueeze(torch.mul(torch.add(_d123456[:,:,0],_l123456[:,:,-1]),0.5),2)
        # 6 x c x 1 for each

        _lp123456p = torch.cat([torch.cat([p_tl,_l123456],2), p_dl],2)
        _rp123456p = torch.cat([torch.cat([p_tr,_r123456],2), p_dr],2)

        t_out = torch.cat([torch.unsqueeze(_t123456,2),x],2)
        td_out = torch.cat([t_out,torch.unsqueeze(_d123456,2)],2)
        tdl_out = torch.cat([torch.unsqueeze(_lp123456p,3),td_out],3)
        tdlr_out = torch.cat([tdl_out,torch.unsqueeze(_rp123456p,3)],3)
        return tdlr_out

class CubePadding(nn.Module):
    def __init__(self, USE_GPU = True):
        super(CubePadding, self).__init__()
        self.USE_GPU = USE_GPU
        #self.pad = pad

    def flip(self, tensor, dim):
        idx = [i for i in range(tensor.size(dim)-1, -1, -1)]

        if self.USE_GPU:
            idx = Variable(torch.cuda.LongTensor(idx))
        else:
            idx = Variable(torch.LongTensor(idx))

        inverted_tensor = tensor.index_select(dim, idx)
        return inverted_tensor
    
    def make_cubepad_edge(self,feat_td,feat_lr):

        td_pad = feat_td.size(2)
        lr_pad = feat_lr.size(3)

        if td_pad>lr_pad:
            return feat_lr.repeat(1,1,td_pad,1)
        else:
            return feat_td.repeat(1,1,1,lr_pad)
        #avg_feat = (tile_lr+tile_td)*0.5
        #return avg_feat

    def forward(self, x, lrtd_pad):

        if type(lrtd_pad)==np.int:
            p_l=lrtd_pad
            p_r=lrtd_pad
            p_t=lrtd_pad
            p_d=lrtd_pad
        else:
            [p_l, p_r, p_t, p_d] = lrtd_pad

        f_back = x[0]
        f_down = x[1]
        f_front = x[2]
        f_left = x[3]
        f_right = x[4]
        f_top = x[5]

        #faces order: 123456 = bdflrt (back, bottom, front, left, right, top)

        '''
        4 plates:  /  t  /
                  |=====| r
                 l| /   | / 
                  |=====|/
                     d
        '''

        if p_t != 0:
            _t12 = torch.cat([torch.unsqueeze(self.flip(f_top[:,:p_t,:],2),0), torch.unsqueeze(f_front[:,-p_t:,:],0)],0)
            _t123 = torch.cat([_t12, torch.unsqueeze(f_top[:,-p_t:,:],0)],0)
            _t1234 = torch.cat([_t123, torch.unsqueeze(f_top[:,:,:p_t].permute(0,2,1),0)],0)
            _t12345 = torch.cat([_t1234, torch.unsqueeze(self.flip((f_top[:,:,-p_t:].permute(0,2,1)),2),0)],0)
            _t123456 = torch.cat([_t12345, torch.unsqueeze(self.flip(f_back[:,:p_t,:],2),0)],0)
        if p_d != 0:
            _d12 = torch.cat([torch.unsqueeze(self.flip(f_down[:,-p_d:,:],2),0), torch.unsqueeze(self.flip(f_back[:,-p_d:,:],2),0)],0)
            _d123 = torch.cat([_d12, torch.unsqueeze(f_down[:,:p_d,:],0)],0)
            _d1234 = torch.cat([_d123, torch.unsqueeze(self.flip(f_down[:,:,:p_d].permute(0,2,1),2),0)],0)
            _d12345 = torch.cat([_d1234, torch.unsqueeze(f_down[:,:,-p_d:].permute(0,2,1),0) ],0)
            _d123456 = torch.cat([_d12345, torch.unsqueeze(f_front[:,:p_d,:],0)],0)
        if p_l != 0:
            _l12 = torch.cat([torch.unsqueeze(f_right[:,:,-p_l:],0), torch.unsqueeze(self.flip(f_left[:,-p_l:,:].permute(0,2,1),1),0)],0)
            _l123 = torch.cat([_l12, torch.unsqueeze(f_left[:,:,-p_l:],0)],0)
            _l1234 = torch.cat([_l123, torch.unsqueeze(f_back[:,:,-p_l:],0)],0)
            _l12345 = torch.cat([_l1234, torch.unsqueeze(f_front[:,:,-p_l:],0)],0)
            _l123456 = torch.cat([_l12345, torch.unsqueeze(f_left[:,:p_l,:].permute(0,2,1),0)],0)
        if p_r != 0:
            _r12 = torch.cat([torch.unsqueeze(f_left[:,:,:p_r],0), torch.unsqueeze(f_right[:,-p_r:,:].permute(0,2,1),0)],0)
            _r123 = torch.cat([_r12, torch.unsqueeze(f_right[:,:,:p_r],0)],0)
            _r1234 = torch.cat([_r123, torch.unsqueeze(f_front[:,:,:p_r],0)],0)
            _r12345 = torch.cat([_r1234, torch.unsqueeze(f_back[:,:,:p_r],0)],0)
            _r123456 = torch.cat([_r12345, torch.unsqueeze(self.flip(f_right[:,:p_r,:].permute(0,2,1),1),0)],0)

        # 6 x c x w for each

        # corner cases

        if p_r!=0 and p_t!=0:
            p_tr = self.make_cubepad_edge(_t123456[:,:,-p_t:,-1:],_r123456[:,:,:1,:p_r])
        if p_t!=0 and p_l!=0:
            p_tl = self.make_cubepad_edge(_t123456[:,:,:p_t,:1],_l123456[:,:,:1,:p_l])
        if p_d!=0 and p_r!=0:
            p_dr = self.make_cubepad_edge(_d123456[:,:,-p_d:,-1:],_r123456[:,:,-1:,-p_r:])
        if p_d!=0 and p_l!=0:
            p_dl = self.make_cubepad_edge(_d123456[:,:,:p_d,:1],_l123456[:,:,-1:,-p_l:])

        # 6 x c x 1 for each

        if p_r!=0:
            _rp123456p = _r123456
            if 'p_tr' in locals():
                _rp123456 = torch.cat([p_tr,_r123456],2)
            else:
                _rp123456 = _r123456

            if 'p_dr' in locals():
                _rp123456p = torch.cat([_rp123456, p_dr],2)
            else:
                _rp123456p = _rp123456

        if p_l!=0:
            _lp123456p = _l123456
            if 'p_tl' in locals():
                _lp123456 = torch.cat([p_tl,_l123456],2)
            else:
                _lp123456 = _l123456
            if 'p_dl' in locals():
                _lp123456p = torch.cat([_lp123456, p_dl],2)
            else:
                _lp123456p = _lp123456

        if p_t != 0:
            t_out = torch.cat([_t123456,x],2)
        else:
            t_out = x
        if p_d != 0:
            td_out = torch.cat([t_out,_d123456],2)
        else:
            td_out = t_out
        if p_l != 0:
            tdl_out = torch.cat([_lp123456p,td_out],3)
        else:
            tdl_out = td_out
        if p_r != 0:
            tdlr_out = torch.cat([tdl_out,_rp123456p],3)
        else:
            tdlr_out = tdl_out
        return tdlr_out

