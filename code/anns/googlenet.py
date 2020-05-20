import torch
from torch import nn
import torch.nn.functional as F
#from torchsummary import summary
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import re

class BasicConv2d(nn.Module):
    """Some Information about BasicConv2d"""
    def __init__(self, in_filters, out_filters, kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=False):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_filters, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        x = self.conv(x)
        return F.relu(self.bn(x))
    
class Inception(nn.Module):
    """Some Information about Inception"""
    def __init__(self, st_bconv_in_filters,b1_bconv_out_filters, b2_bconv1_out_filters,b2_bconv2_out_filters,b3_bconv1_out_filters,b3_bconv2_out_filters,b4_bconv_out_filters):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_filters=st_bconv_in_filters,out_filters=b1_bconv_out_filters, kernel_size=(1,1), stride=(1,1), bias=False)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_filters=st_bconv_in_filters,out_filters=b2_bconv1_out_filters, kernel_size=(1,1), stride=(1,1), bias=False),
            BasicConv2d(in_filters=b2_bconv1_out_filters,out_filters=b2_bconv2_out_filters, kernel_size=(3,3), stride=(1,1),padding=(1,1),bias=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_filters=st_bconv_in_filters,out_filters=b3_bconv1_out_filters, kernel_size=(1,1), stride=(1,1), bias=False),
            BasicConv2d(in_filters=b3_bconv1_out_filters,out_filters=b3_bconv2_out_filters, kernel_size=(3,3), stride=(1,1),padding=(1,1),bias=False)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True),
            BasicConv2d(in_filters=st_bconv_in_filters,out_filters=b4_bconv_out_filters, kernel_size=(1,1), stride=(1,1), bias=False)
        )

    def forward(self, x):
        return torch.cat((self.branch1(x), self.branch2(x),self.branch3(x), self.branch4(x)),1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()

        self.conv1 = BasicConv2d(in_filters=3,out_filters=64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0,dilation=1,ceil_mode=True)

        self.conv2 = BasicConv2d(in_filters=64,out_filters=64, kernel_size=(1,1), stride=(1,1), bias=False)
        self.conv3 = BasicConv2d(in_filters=64,out_filters=192, kernel_size=(3,3), stride=(1,1),padding=(1,1), bias=False)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0,dilation=1,ceil_mode=True)

        self.inception3a = Inception(st_bconv_in_filters=192,b1_bconv_out_filters=64,b2_bconv1_out_filters=96,b2_bconv2_out_filters=128,b3_bconv1_out_filters=16,b3_bconv2_out_filters=32,b4_bconv_out_filters=32)
        self.inception3b = Inception(st_bconv_in_filters=256,b1_bconv_out_filters=128,b2_bconv1_out_filters=128,b2_bconv2_out_filters=192,b3_bconv1_out_filters=32,b3_bconv2_out_filters=96,b4_bconv_out_filters=64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)

        self.inception4a = Inception(st_bconv_in_filters=480,b1_bconv_out_filters=192,b2_bconv1_out_filters=96,b2_bconv2_out_filters=208,b3_bconv1_out_filters=16,b3_bconv2_out_filters=48,b4_bconv_out_filters=64)
        self.inception4b = Inception(st_bconv_in_filters=512,b1_bconv_out_filters=160,b2_bconv1_out_filters=112,b2_bconv2_out_filters=224,b3_bconv1_out_filters=24,b3_bconv2_out_filters=64,b4_bconv_out_filters=64)
        self.inception4c = Inception(st_bconv_in_filters=512,b1_bconv_out_filters=128,b2_bconv1_out_filters=128,b2_bconv2_out_filters=256,b3_bconv1_out_filters=24,b3_bconv2_out_filters=64,b4_bconv_out_filters=64)
        self.inception4d = Inception(st_bconv_in_filters=512,b1_bconv_out_filters=112,b2_bconv1_out_filters=144,b2_bconv2_out_filters=288,b3_bconv1_out_filters=32,b3_bconv2_out_filters=64,b4_bconv_out_filters=64)
        self.inception4e = Inception(st_bconv_in_filters=528,b1_bconv_out_filters=256,b2_bconv1_out_filters=160,b2_bconv2_out_filters=320,b3_bconv1_out_filters=32,b3_bconv2_out_filters=128,b4_bconv_out_filters=128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)

        self.inception5a = Inception(st_bconv_in_filters=832,b1_bconv_out_filters=256,b2_bconv1_out_filters=160,b2_bconv2_out_filters=320,b3_bconv1_out_filters=32,b3_bconv2_out_filters=128,b4_bconv_out_filters=128)
        self.inception5b = Inception(st_bconv_in_filters=832,b1_bconv_out_filters=384,b2_bconv1_out_filters=192,b2_bconv2_out_filters=384,b3_bconv1_out_filters=48,b3_bconv2_out_filters=128,b4_bconv_out_filters=128)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.2,inplace=False)
        self.fc = nn.Linear(in_features=1024, out_features=1000, bias=True)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        return self.fc(x)

model_urls = {
    # GoogLeNet ported from TensorFlow
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
}

def googlenet(pretrained=False, progress=True, **kwargs):
    r"""GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = GoogLeNet()

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['googlenet'],
                                              progress=progress)
        helper_state_dict = state_dict.copy()
        for key in helper_state_dict.keys():
            if re.match(r"aux*",key):
                del state_dict[key]

        #state_dict["conv1.conv.weight"]=state_dict["conv1.conv.weight"][:,:2,:,:]
        state_dict["fc.weight"] = _initialize_weights(model.fc.weight)
        state_dict["fc.bias"] = _initialize_weights(model.fc.bias)
        model.load_state_dict(state_dict)

    
    return model

def _initialize_weights(layer):
    import scipy.stats as stats
    X = stats.truncnorm(-2, 2, scale=0.01)
    values = torch.as_tensor(X.rvs(layer.numel()), dtype=layer.dtype)
    values = values.view(layer.size())
    with torch.no_grad():
        layer.copy_(values)
    return values

# def main():
#     model=googlenet(pretrained=True).cuda()
#     #summary(model,(3,224,224))

# if __name__=="__main__":
#     main()
