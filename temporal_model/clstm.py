import pdb
import os, sys
import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
sys.path.append('..')
from utils.CubePad import CubePad
# Define some constants
KERNEL_SIZE = 3
#PADDING = KERNEL_SIZE // 2
PADDING = 0

CP = True

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
