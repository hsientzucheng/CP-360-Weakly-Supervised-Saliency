import math
import torch
import matplotlib.pyplot as plt
import pdb

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from torch.autograd import Variable
from torch.nn.parameter import Parameter

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

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



class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()

        '''
	self.bn1 = nn.BatchNorm2d(64)
	self.bn2 = nn.BatchNorm2d(128)
	self.bn3 = nn.BatchNorm2d(256)
	self.bn4 = nn.BatchNorm2d(512)
	self.bn5 = nn.BatchNorm2d(512)

        self.conv1_1 = nn.Conv2d(3, 64, 3)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2_1 = nn.Conv2d(64, 128, 3)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3_1 = nn.Conv2d(128, 256, 3)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv4_1 = nn.Conv2d(256, 512, 3)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.conv5_1 = nn.Conv2d(512, 512, 3)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3)
        self.relu5_3 = nn.ReLU(inplace=True)

        self.pad = CubePad()
        '''
        self.features = features
        self.camconv = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(True),
        )
        self.avgpool = nn.AvgPool2d((28,56)) #14
        self.classifier = nn.Linear(1024, num_classes)
        '''
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        '''
        self._initialize_weights()
    '''
    def forward(self, x):
        x = self.features(x)
        x = self.camconv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    '''
    
    def forward(self, x):
        '''
        #vgg features
        x = self.pad(x)
        x = self.conv1_1(x)
	x = self.bn1(x)
        x = self.relu1_1(x)
        x = self.pad(x)
        x = self.conv1_2(x)
	x = self.bn1(x)
        x = self.relu1_2(x)
        x = self.pool1(x)
        x = self.pad(x)
        x = self.conv2_1(x)
	x = self.bn2(x)
        x = self.relu2_1(x)
        x = self.pad(x)
        x = self.conv2_2(x) 
	x = self.bn2(x)
        x = self.relu2_2(x)
        x = self.pool2(x)
        x = self.pad(x)
        x = self.conv3_1(x)
	x = self.bn3(x)
        x = self.relu3_1(x)
        x = self.pad(x)
        x = self.conv3_2(x)
	x = self.bn3(x)
        x = self.relu3_2(x)
        x = self.pad(x)
        x = self.conv3_3(x) 
	x = self.bn3(x)
        x = self.relu3_3(x)
        x = self.pool3(x)
        x = self.pad(x)
        x = self.conv4_1(x)
	x = self.bn4(x)
        x = self.relu4_1(x)
        x = self.pad(x)
        x = self.conv4_2(x)
	x = self.bn4(x)
        x = self.relu4_2(x)
        x = self.pad(x)
        x = self.conv4_3(x)
	x = self.bn4(x)
        x = self.relu4_3(x)
        x = self.pool4(x)
        x = self.pad(x)
        x = self.conv5_1(x)
	x = self.bn5(x)
        x = self.relu5_1(x)
        x = self.pad(x)
        x = self.conv5_2(x)
	x = self.bn5(x)
        x = self.relu5_2(x)
        x = self.pad(x)
        x = self.conv5_3(x)
	x = self.bn5(x)
        x = self.relu5_3(x)
        '''
        #cam layers
        x = self.features(x)
        x = self.camconv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

        


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
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



    # homemade pre-trained model loading function :)
    def load_pretrained_model(self, pretrained_state_dict):

        custom_state_dict = self.state_dict()
        for name, param in pretrained_state_dict.items():

            if name not in custom_state_dict:
                print("skip loading key '{}' due to absence in state_dict".format(name))

            if isinstance(param, Parameter):
                param = param.data

            try:
                custom_state_dict[name].copy_(param)
            except:
                print("skip loading key '{}' due to inconsistent size".format(name))

        self.load_state_dict(custom_state_dict)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1) 
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                #layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
                #layers += [conv2d, nn.ReLU(inplace=True)]

            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_pretrained_model(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_pretrained_model(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_pretrained_model(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_pretrained_model(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_pretrained_model(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        #
        #model.load_pretrained_model_seq(model_zoo.load_url(model_urls['vgg16_bn']))
        model.load_pretrained_model_seq(torch.load('cam_vgg16_ep4.pth'))
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_pretrained_model(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_pretrained_model(model_zoo.load_url(model_urls['vgg19_bn']))
    return model
