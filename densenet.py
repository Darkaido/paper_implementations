# Learned from torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class _Transition(nn.Sequential):
    def __init__(self,num_input_features,num_output_features):
        super(_Transition,self).__init__()
        self.add_module('norm',nn.BatchNorm2d(num_input_features))
        self.add_module('relu',nn.ReLU(inplace=True))
        self.add_module('conv',nn.Conv2d(num_input_features,num_output_features,kernel_size=1,stride=1,padding=1,bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class _DenseLayer(nn.Module):
    def __init__(self,num_input_features,growth_rate,bn_size, drop_rate,memory_efficient=False):
        super(_DenseLayer,self).__init__()
        self.add_module('norm1',nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features,bn_size * growth_rate, kernel_size=1,stride=1,bias=False)),

        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size* growth_rate,growth_rate,kernel_size=3,stride=1,padding=1,bias=False)),

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self,input):
        "bottleneck function"
        # type: (List[Tensor]) -- >Tensor
        concat_feature = torch.cat(input,1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concat_feature)))
        return bottleneck_output
    
    def forward(self,input_image):
        if isinstance(input_image, Tensor):
            prev_feature = [input_image]
        else:
            prev_feature = input_image

        bn_output = self.bn_function(prev_feature)
        new_feature = self.conv2(self.relu2(self.norm2(bn_output)))

        if self.drop_rate > 0:
            new_feature =  F.dropout(new_feature, p = self.drop_rate, training=self.training)

        return new_feature

class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self,num_layers,num_input_features,bn_size, growth_rate, drop_rate):
        super(_DenseBlock,self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features+ i * growth_rate,
                                growth_rate=growth_rate,
                                bn_size = bn_size,
                                drop_rate=drop_rate,)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self,init_features):
        features = [init_features]
        for name,layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features,1)
    
class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):

        super(DenseNet, self).__init__()

        # Convolution and pooling part from table-1
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Add multiple denseblocks based on config 
        # for densenet-121 config: [6,12,24,16]
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                # add transition layer between denseblocks to 
                # downsample
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    

def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    return model

def densenet121(pretrained=False, progress=True, **kwargs):
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)


if __name__ == "__main__":
    x = torch.randn((2,3,224,224))
    print(densenet121(x))