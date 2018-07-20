import pdb
import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
# Define some constants
KERNEL_SIZE = 3
#PADDING = KERNEL_SIZE // 2
PADDING = 0

CP = True
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


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size):
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Conv1 = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, KERNEL_SIZE, padding=PADDING)
        self.Conv2 = nn.Conv2d(4 * hidden_size, 4 * hidden_size, KERNEL_SIZE, padding=PADDING)
        self.Relu = nn.ReLU(inplace=True)
        self.Gates = nn.Conv2d(4 * hidden_size, 4 * hidden_size, KERNEL_SIZE, padding=PADDING)
        self.LSoftMax = nn.LogSoftmax()
        self._initialize_weights()
        if CP:
            self.pad = CubePad()
        else:
            self.pad = ZeroPad()

    def forward(self, input_, prev_state=None):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)),
                Variable(torch.zeros(state_size))
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        out = self.pad(stacked_inputs)
        out = self.Conv1(out)
        out = self.Relu(out)
        out = self.pad(out)
        out = self.Conv2(out)
        out = self.Relu(out)
        out = self.pad(out)
        gates = self.Gates(out)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = f.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * f.tanh(cell)
        hidden_softmax = self.LSoftMax(hidden)
        return hidden, cell

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    # homemade pre-trained model loading function according to order only :)
    def load_pretrained_model_seq(self, pretrained_state_dict):

        custom_state_dict = self.state_dict()
        # cname cparam pname pparam
        for name, param in zip(custom_state_dict.keys(), pretrained_state_dict.values()):

            if isinstance(param, Parameter):
                param = param.data

            try:
                custom_state_dict[name].copy_(param)
            except:
                print("skip loading key '{}' due to inconsistent size".format(name))

        self.load_state_dict(custom_state_dict)
