import torch.nn as nn
import math
import pdb
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.nn.parameter import Parameter

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

CP = True


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

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

class CubePadding(nn.Module):
    def __init__(self, USE_GPU = True):
        super(CubePadding, self).__init__()
        self.USE_GPU = USE_GPU

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

        #pdb.set_trace()
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

'''
class CubePad(nn.Module):
    def __init__(self, pad=1, USE_GPU = True):
        super(CubePad, self).__init__()
        self.USE_GPU = USE_GPU
        self.pad = pad

    def flip(self, tensor, dim):
        idx = [i for i in range(tensor.size(dim)-1, -1, -1)]

        if self.USE_GPU:
            idx = Variable(torch.cuda.LongTensor(idx))
        else:
            idx = Variable(torch.LongTensor(idx))

        inverted_tensor = tensor.index_select(dim, idx)
        return inverted_tensor
    
    def make_cubepad_edge(self,feat_td,feat_lr,pad):

        tile_lr = feat_lr.repeat(1,1,pad,1)
        tile_td = feat_td.repeat(1,1,1,pad)
        #avg_feat = torch.mul(torch.add(tile_lr,tile_td),0.5)
        avg_feat = (tile_lr+tile_td)*0.5
        return avg_feat
        #return tile_lr


    def forward(self, x):

        f_back = x[0]
        f_down = x[1]
        f_front = x[2]
        f_left = x[3]
        f_right = x[4]
        f_top = x[5]
        pad = self.pad


        #faces order: 123456 = bdflrt (back, bottom, front, left, right, top)
'''
'''
        4 plates:  /  t  /
                  |=====| r
                 l| /   | / 
                  |=====|/
                     d
'''
'''
        _t12 = torch.cat([torch.unsqueeze(self.flip(f_top[:,:pad,:],2),0), torch.unsqueeze(f_front[:,-pad:,:],0)],0)
        _t123 = torch.cat([_t12, torch.unsqueeze(f_top[:,-pad:,:],0)],0)
        _t1234 = torch.cat([_t123, torch.unsqueeze(f_top[:,:,:pad].permute(0,2,1),0)],0)
        _t12345 = torch.cat([_t1234, torch.unsqueeze(self.flip((f_top[:,:,-pad:].permute(0,2,1)),2),0)],0)
        _t123456 = torch.cat([_t12345, torch.unsqueeze(self.flip(f_back[:,:pad,:],2),0)],0)

        _d12 = torch.cat([torch.unsqueeze(self.flip(f_down[:,-pad:,:],2),0), torch.unsqueeze(self.flip(f_back[:,-pad:,:],2),0)],0)
        _d123 = torch.cat([_d12, torch.unsqueeze(f_down[:,:pad,:],0)],0)
        _d1234 = torch.cat([_d123, torch.unsqueeze(self.flip(f_down[:,:,:pad].permute(0,2,1),2),0)],0)
        _d12345 = torch.cat([_d1234, torch.unsqueeze(f_down[:,:,-pad:].permute(0,2,1),0) ],0)
        _d123456 = torch.cat([_d12345, torch.unsqueeze(f_front[:,:pad,:],0)],0)

        _l12 = torch.cat([torch.unsqueeze(f_right[:,:,-pad:],0), torch.unsqueeze(self.flip(f_left[:,-pad:,:].permute(0,2,1),1),0)],0)
        _l123 = torch.cat([_l12, torch.unsqueeze(f_left[:,:,-pad:],0)],0)
        _l1234 = torch.cat([_l123, torch.unsqueeze(f_back[:,:,-pad:],0)],0)
        _l12345 = torch.cat([_l1234, torch.unsqueeze(f_front[:,:,-pad:],0)],0)
        _l123456 = torch.cat([_l12345, torch.unsqueeze(f_left[:,:pad,:].permute(0,2,1),0)],0)

        _r12 = torch.cat([torch.unsqueeze(f_left[:,:,:pad],0), torch.unsqueeze(f_right[:,-pad:,:].permute(0,2,1),0)],0)
        _r123 = torch.cat([_r12, torch.unsqueeze(f_right[:,:,:pad],0)],0)
        _r1234 = torch.cat([_r123, torch.unsqueeze(f_front[:,:,:pad],0)],0)
        _r12345 = torch.cat([_r1234, torch.unsqueeze(f_back[:,:,:pad],0)],0)
        _r123456 = torch.cat([_r12345, torch.unsqueeze(self.flip(f_right[:,:pad,:].permute(0,2,1),1),0)],0)
        # 6 x c x w for each

        p_tr = self.make_cubepad_edge(_t123456[:,:,-pad:,-1:],_r123456[:,:,:1,:pad],pad)
        p_tl = self.make_cubepad_edge(_t123456[:,:,:pad,:1],_l123456[:,:,:1,:pad],pad)
        p_dr = self.make_cubepad_edge(_d123456[:,:,-pad:,-1:],_r123456[:,:,-1:,-pad:],pad)
        p_dl = self.make_cubepad_edge(_d123456[:,:,:pad,:1],_l123456[:,:,-1:,-pad:],pad)
        # 6 x c x 1 for each
        _lp123456p = torch.cat([torch.cat([p_tl,_l123456],2), p_dl],2)
        _rp123456p = torch.cat([torch.cat([p_tr,_r123456],2), p_dr],2)

        t_out = torch.cat([_t123456,x],2)
        td_out = torch.cat([t_out,_d123456],2)
        tdl_out = torch.cat([_lp123456p,td_out],3)
        tdlr_out = torch.cat([tdl_out,_rp123456p],3)
        
        
        #pdb.set_trace()
        return tdlr_out
'''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        if CP:
            self.pad = CubePadding()
        else:
            self.pad = ZeroPad()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.pad(out,1)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if CP:
            #self.pad3 = CubePad(pad=3)
            self.cp = CubePadding()
            #self.pad1 = CubePad(pad=1)
        else:
            self.pad3 = ZeroPad(pad=3)
            self.pad1 = ZeroPad(pad=1)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0,
                               bias=False) # padding original value = 3
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0) # padding original value = 1
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7) #512: 16, #224: 7 #800:25
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.cp(x,3)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.cp(x,1)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    # homemade pre-trained model loading function :)
    def load_pretrained_model(self, pretrained_state_dict):

        custom_state_dict = self.state_dict()

        for name, param in pretrained_state_dict.items():

            if name not in custom_state_dict:
                raise KeyError("unexpected key '{}' in state_dict".format(name))

            if isinstance(param, Parameter):
                param = param.data

            try:
                custom_state_dict[name].copy_(param)
            except:
                print("skip loading key '{}' due to inconsistent size".format(name))

        self.load_state_dict(custom_state_dict)


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_pretrained_model(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_pretrained_model(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_pretrained_model(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_pretrained_model(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_pretrained_model(model_zoo.load_url(model_urls['resnet152']))
    return model
