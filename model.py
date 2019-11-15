import torch
# import encoding ## pip install torch-encoding . For synchnonized Batch norm in pytorch 1.0.0
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch import Tensor

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features
    
class _DenseBlock(nn.Module):
    __constants__ = ['layers']

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        self.layers = nn.ModuleDict()
        self.num_input_features = num_input_features
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.layers['denselayer%d' % (i + 1)] = layer

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.layers.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)
        
class _MDenseNet_STEM(nn.Module):
    def __init__(self,first_channel=32,first_kernel = (3,3),scale=3,kl = [],drop_rate = 0.1,bn_size=4):
        super(_MDenseNet_STEM,self).__init__()
        self.first_channel = 32
        self.first_kernel = first_kernel
        self.scale = scale
        self.kl = kl
        
        self.first_conv = nn.Conv2d(2,first_channel,first_kernel)
        self.downsample_layer = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.upsample_layers = nn.ModuleList()
        self.dense_padding = nn.ModuleList()
        self.dense_layers = nn.ModuleList()
        self.channels = [self.first_channel]
        ## [_,d1,...,ds,ds+1,u1,...,us]
        for k,l in kl[:scale+1]:
            self.dense_layers.append(_DenseBlock( 
                l, self.channels[-1], bn_size, k, drop_rate))
            self.channels.append(self.channels[-1]+k*l)
        
        
        for i,(k, l) in enumerate(kl[scale+1:]):
            self.upsample_layers.append(nn.ConvTranspose2d(self.channels[-1],self.channels[-1], kernel_size=2, stride=2))
            self.channels.append(self.channels[-1]+self.channels[scale-i])
            self.dense_layers.append(_DenseBlock(
                l, self.channels[-1], bn_size, k, drop_rate))
            self.channels.append(self.channels[-1]+k*l)
    
    def _pad(self,x,target):
        if x.shape != target.shape:
            padding_1 = target.shape[2] - x.shape[2]
            padding_2 = target.shape[3] - x.shape[3]
        return F.pad(x,(padding_2,0,padding_1,0),'replicate')
    
    def forward(self,input):
        ## stem
        output = self.first_conv(input)
        dense_outputs = []
        
        ## downsample way
        for i in range(self.scale):
            output = self.dense_layers[i](output)
            dense_outputs.append(output)
            output = self.downsample_layer(output) ## downsample

        ## upsample way
        output = self.dense_layers[self.scale](output)
        for i in range(self.scale):
            output = self.upsample_layers[i](output)
            output = self._pad(output,dense_outputs[-(i+1)])
            output = torch.cat([output,dense_outputs[-(i+1)]],dim = 1)
            output = self.dense_layers[self.scale+1+i](output)
        output = self._pad(output,input)
        return output
    
class MMDenseNet(nn.Module):
    def __init__(self,drop_rate = 0.1,bn_size=4,k=10,l=3):
        super(MMDenseNet,self).__init__()
        
        kl_low = [(k,l),(k,l),(k,l),(k,l),(k,l),(k,l),(k,l)]
        kl_high = [(k,l),(k,l),(k,l),(k,l),(k,l),(k,l),(k,l)]
        kl_full = [(k,l),(k,l),(k,l),(k,l),(k,l),(k,l),(k,l)]
        self.lowNet = _MDenseNet_STEM(first_channel=32,first_kernel = (4,3),scale=3,kl = kl_low,drop_rate = drop_rate,bn_size=bn_size)
        self.highNet = _MDenseNet_STEM(first_channel=32,first_kernel = (3,3),scale=3,kl = kl_high,drop_rate = drop_rate,bn_size=bn_size)
        self.fullNet = _MDenseNet_STEM(first_channel=32,first_kernel = (4,3),scale=3,kl = kl_full,drop_rate = drop_rate,bn_size=bn_size)
        
        last_channel = self.lowNet.channels[-1] + self.fullNet.channels[-1]
        self.out = nn.Sequential(
            _DenseBlock( 
                2, last_channel, bn_size, 4, drop_rate),
            nn.Conv2d(last_channel+8,2,1)
        )
        
    def forward(self,input):
#         print(input.shape)
        B,C,F,T = input.shape
        low_input = input[:,:,:F//2,:]
        high_input = input[:,:,F//2:,:]
        
        output = torch.cat([self.lowNet(low_input),self.highNet(high_input)],2)##Frequency 방향
        full_output = self.fullNet(input)
        output = torch.cat([output,full_output],1) ## Channel 방향
        output = self.out(output)
        
        return output
