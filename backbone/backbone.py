import torch
import torch.nn as nn
import pickle
import math
from collections import OrderedDict
import torch.nn.functional as F




class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,padding=dilation, bias=False, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
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
    """ Adapted from torchvision.models.resnet """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False, dilation=dilation)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

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


###################################################################################
# class Classification_weather(nn.Module):
#     def __init__(self, num_features_in, num_classes=7, feature_size=256):
#         super(Classification_weather, self).__init__()

#         self.num_classes = num_classes
#         self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
#         self.act1 = nn.ReLU()
#         self.mp = nn.MaxPool2d(2)
#         self.fc = nn.Linear(56320, num_classes)
#         # self.bn1 = nn.BatchNorm2d(32)

#     def forward(self, x):
#         in_size = x.size(0)
#         # x = x.float()
#         first_layer=self.conv1(x)
#         x = F.relu(self.mp(first_layer))
#         # x = F.relu(self.mp(self.conv2(x)))
#         x = x.view(in_size, -1)  # flatten the tensor
#         x = self.fc(x)
#         # prob=nn.Softmax(x)
#         # print(prob.data[0])
#         prob=F.log_softmax(x)
#         # prob=F.softmax(x)
#         # prob=prob.long()
#         return prob


class sunshine_classification(nn.Module):
    def __init__(self, num_features_in, num_classes=4, feature_size=256):
        super(sunshine_classification, self).__init__()

        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(20736, num_classes)
        # self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        in_size = x.size(0)
        # x = x.float()
        first_layer=self.conv1(x)
        x = F.relu(self.mp(first_layer))
        # x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        # prob=nn.Softmax(x)
        # print(prob.data[0])
        prob=F.log_softmax(x)
        # prob=F.softmax(x)
        # prob=prob.long()
        return prob

## Class added to implement the classification for Road Type
# class Classification_roadtype(nn.Module):
#     def __init__(self, num_features_in, num_classes=7, feature_size=256):
#         super(Classification_roadtype, self).__init__()

#         self.num_classes = num_classes
#         self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
#         self.act1 = nn.ReLU()
#         self.mp = nn.MaxPool2d(2)
#         self.fc = nn.Linear(56320, num_classes)
#         # self.bn1 = nn.BatchNorm2d(32)

#     def forward(self, x):
#         in_size = x.size(0)
#         # x = x.float()
#         first_layer=self.conv1(x)
#         x = F.relu(self.mp(first_layer))
#         # x = F.relu(self.mp(self.conv2(x)))
#         x = x.view(in_size, -1)  # flatten the tensor
#         x = self.fc(x)
#         # prob=nn.Softmax(x)
#         # print(prob.data[0])
#         prob=F.log_softmax(x)
#         # prob=F.softmax(x)
#         # prob=prob.long()
#         return prob
###################################################################################

class ResNetBackbone(nn.Module):
    """ Adapted from torchvision.models.resnet """

    # def __init__(self, weather_num_classes, daytime_num_classes, roadtype_num_classes, layers, atrous_layers=[], block=Bottleneck, norm_layer=nn.BatchNorm2d):
    def __init__(self, layers, atrous_layers=[], block=Bottleneck, norm_layer=nn.BatchNorm2d):
        super().__init__()

        # These will be populated by _make_layer
        self.num_base_layers = len(layers)
        self.layers = nn.ModuleList()
        self.channels = []
        self.norm_layer = norm_layer
        self.dilation = 1
        self.atrous_layers = atrous_layers

        # From torchvision.models.resnet.Resnet
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.Class_weather = Classification_weather(2048, num_classes=weather_num_classes)
        self.Class_daytime = sunshine_classification(2048)
        # self.Class_roadtype = Classification_roadtype(2048, num_classes=roadtype_num_classes)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # This contains every module that should be initialized by loading in pretrained weights.
        # Any extra layers added onto this that won't be initialized by init_backbone will not be
        # in this list. That way, Yolact::init_weights knows which backbone weights to initialize
        # with xavier, and which ones to leave alone.
        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]
        
    
    def _make_layer(self, block, planes, blocks, stride=1):
        """ Here one layer means a string of n Bottleneck blocks. """
        downsample = None

        # This is actually just to create the connection between layers, and not necessarily to
        # downsample. Even if the second condition is met, it only downsamples when stride != 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if len(self.layers) in self.atrous_layers:
                self.dilation += 1
                stride = 1
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False,
                          dilation=self.dilation),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.norm_layer, self.dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self.norm_layer))

        layer = nn.Sequential(*layers)

        self.channels.append(planes * block.expansion)
        self.layers.append(layer)

        return layer

    def forward(self, x):
        """ Returns a list of convouts for each layer. """
        if self.training:
            x, daytime_target = x[0], x[1]
        else:
            x = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print('printing printing', daytime_val)

        outs = []
        # for layer in self.layers:
        #     x = layer(x)
        #     outs.append(x)
        x1 = self.layer1(x)
        outs.append(x1)
        x2 = self.layer2(x1)
        outs.append(x2)
        x3 = self.layer3(x2)
        outs.append(x3)
        x4 = self.layer4(x3)
        outs.append(x4)

        #######################################
        # weather_prob_output=self.Class_weather(x4)
        daytime_prob_output=self.Class_daytime(x4)
        # # adding road type classification layer
        # roadtype_prob_output=self.Class_roadtype(x4)
        if self.training:
            daytime_class_loss = F.nll_loss(daytime_prob_output, torch.tensor(daytime_target))

            return tuple(outs), daytime_class_loss
        else:
            return tuple(outs), daytime_prob_output
        # print('Classification loss:', daytime_class_loss)
        #######################################


    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        state_dict = torch.load(path, map_location={'cuda:0': 'cpu'})

        # Replace layer1 -> layers.0 etc.
        keys = list(state_dict)
        for key in keys:
            if key.startswith('layer'):
                idx = int(key[5])
                new_key = 'layers.' + str(idx-1) + key[6:]
                state_dict[new_key] = state_dict.pop(key)

        # Note: Using strict=False is berry scary. Triple check this.
        self.load_state_dict(state_dict, strict=False)

    def add_layer(self, conv_channels=1024, downsample=2, depth=1, block=Bottleneck):
        """ Add a downsample layer to the backbone as per what SSD does. """
        self._make_layer(block, conv_channels // block.expansion, blocks=depth, stride=downsample)




class ResNetBackboneGN(ResNetBackbone):

    def __init__(self, layers, num_groups=32):
        super().__init__(layers, norm_layer=lambda x: nn.GroupNorm(num_groups, x))

    def init_backbone(self, path):
        """ The path here comes from detectron. So we load it differently. """
        with open(path, 'rb') as f:
            state_dict = pickle.load(f, encoding='latin1') # From the detectron source
            state_dict = state_dict['blobs']
        
        our_state_dict_keys = list(self.state_dict().keys())
        new_state_dict = {}
    
        gn_trans     = lambda x: ('gn_s' if x == 'weight' else 'gn_b')
        layeridx2res = lambda x: 'res' + str(int(x)+2)
        block2branch = lambda x: 'branch2' + ('a', 'b', 'c')[int(x[-1:])-1]

        # Transcribe each Detectron weights name to a Yolact weights name
        for key in our_state_dict_keys:
            parts = key.split('.')
            transcribed_key = ''

            if (parts[0] == 'conv1'):
                transcribed_key = 'conv1_w'
            elif (parts[0] == 'bn1'):
                transcribed_key = 'conv1_' + gn_trans(parts[1])
            elif (parts[0] == 'layers'):
                if int(parts[1]) >= self.num_base_layers: continue

                transcribed_key = layeridx2res(parts[1])
                transcribed_key += '_' + parts[2] + '_'

                if parts[3] == 'downsample':
                    transcribed_key += 'branch1_'
                    
                    if parts[4] == '0':
                        transcribed_key += 'w'
                    else:
                        transcribed_key += gn_trans(parts[5])
                else:
                    transcribed_key += block2branch(parts[3]) + '_'

                    if 'conv' in parts[3]:
                        transcribed_key += 'w'
                    else:
                        transcribed_key += gn_trans(parts[4])

            new_state_dict[key] = torch.Tensor(state_dict[transcribed_key])
        
        # strict=False because we may have extra unitialized layers at this point
        self.load_state_dict(new_state_dict, strict=False)







def darknetconvlayer(in_channels, out_channels, *args, **kwdargs):
    """
    Implements a conv, activation, then batch norm.
    Arguments are passed into the conv layer.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, *args, **kwdargs, bias=False),
        nn.BatchNorm2d(out_channels),
        # Darknet uses 0.1 here.
        # See https://github.com/pjreddie/darknet/blob/680d3bde1924c8ee2d1c1dea54d3e56a05ca9a26/src/activations.h#L39
        nn.LeakyReLU(0.1, inplace=True)
    )

class DarkNetBlock(nn.Module):
    """ Note: channels is the lesser of the two. The output will be expansion * channels. """

    expansion = 2

    def __init__(self, in_channels, channels):
        super().__init__()

        self.conv1 = darknetconvlayer(in_channels, channels,                  kernel_size=1)
        self.conv2 = darknetconvlayer(channels,    channels * self.expansion, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + x




class DarkNetBackbone(nn.Module):
    """
    An implementation of YOLOv3's Darnet53 in
    https://pjreddie.com/media/files/papers/YOLOv3.pdf

    This is based off of the implementation of Resnet above.
    """

    def __init__(self, layers=[1, 2, 8, 8, 4], block=DarkNetBlock):
        super().__init__()

        # These will be populated by _make_layer
        self.num_base_layers = len(layers)
        self.layers = nn.ModuleList()
        self.channels = []
        
        self._preconv = darknetconvlayer(3, 32, kernel_size=3, padding=1)
        self.in_channels = 32
        
        self._make_layer(block, 32,  layers[0])
        self._make_layer(block, 64,  layers[1])
        self._make_layer(block, 128, layers[2])
        self._make_layer(block, 256, layers[3])
        self._make_layer(block, 512, layers[4])

        # This contains every module that should be initialized by loading in pretrained weights.
        # Any extra layers added onto this that won't be initialized by init_backbone will not be
        # in this list. That way, Yolact::init_weights knows which backbone weights to initialize
        # with xavier, and which ones to leave alone.
        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]
    
    def _make_layer(self, block, channels, num_blocks, stride=2):
        """ Here one layer means a string of n blocks. """
        layer_list = []

        # The downsample layer
        layer_list.append(
            darknetconvlayer(self.in_channels, channels * block.expansion,
                             kernel_size=3, padding=1, stride=stride))

        # Each block inputs channels and outputs channels * expansion
        self.in_channels = channels * block.expansion
        layer_list += [block(self.in_channels, channels) for _ in range(num_blocks)]

        self.channels.append(self.in_channels)
        self.layers.append(nn.Sequential(*layer_list))

    def forward(self, x):
        """ Returns a list of convouts for each layer. """

        x = self._preconv(x)

        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)

        return tuple(outs)

    def add_layer(self, conv_channels=1024, stride=2, depth=1, block=DarkNetBlock):
        """ Add a downsample layer to the backbone as per what SSD does. """
        self._make_layer(block, conv_channels // block.expansion, num_blocks=depth, stride=stride)
    
    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        # Note: Using strict=False is berry scary. Triple check this.
        self.load_state_dict(torch.load(path), strict=False)





class VGGBackbone(nn.Module):
    """
    Args:
        - cfg: A list of layers given as lists. Layers can be either 'M' signifying
                a max pooling layer, a number signifying that many feature maps in
                a conv layer, or a tuple of 'M' or a number and a kwdargs dict to pass
                into the function that creates the layer (e.g. nn.MaxPool2d for 'M').
        - extra_args: A list of lists of arguments to pass into add_layer.
        - norm_layers: Layers indices that need to pass through an l2norm layer.
    """

    def __init__(self, cfg, extra_args=[], norm_layers=[]):
        super().__init__()
        
        self.channels = []
        self.layers = nn.ModuleList()
        self.in_channels = 3
        self.extra_args = list(reversed(extra_args)) # So I can use it as a stack

        # Keeps track of what the corresponding key will be in the state dict of the
        # pretrained model. For instance, layers.0.2 for us is 2 for the pretrained
        # model but layers.1.1 is 5.
        self.total_layer_count = 0
        self.state_dict_lookup = {}

        for idx, layer_cfg in enumerate(cfg):
            self._make_layer(layer_cfg)

        self.norms = nn.ModuleList([nn.BatchNorm2d(self.channels[l]) for l in norm_layers])
        self.norm_lookup = {l: idx for idx, l in enumerate(norm_layers)}

        # These modules will be initialized by init_backbone,
        # so don't overwrite their initialization later.
        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

    def _make_layer(self, cfg):
        """
        Each layer is a sequence of conv layers usually preceded by a max pooling.
        Adapted from torchvision.models.vgg.make_layers.
        """

        layers = []

        for v in cfg:
            # VGG in SSD requires some special layers, so allow layers to be tuples of
            # (<M or num_features>, kwdargs dict)
            args = None
            if isinstance(v, tuple):
                args = v[1]
                v = v[0]

            # v should be either M or a number
            if v == 'M':
                # Set default arguments
                if args is None:
                    args = {'kernel_size': 2, 'stride': 2}

                layers.append(nn.MaxPool2d(**args))
            else:
                # See the comment in __init__ for an explanation of this
                cur_layer_idx = self.total_layer_count + len(layers)
                self.state_dict_lookup[cur_layer_idx] = '%d.%d' % (len(self.layers), len(layers))

                # Set default arguments
                if args is None:
                    args = {'kernel_size': 3, 'padding': 1}

                # Add the layers
                layers.append(nn.Conv2d(self.in_channels, v, **args))
                layers.append(nn.ReLU(inplace=True))
                self.in_channels = v
        
        self.total_layer_count += len(layers)
        self.channels.append(self.in_channels)
        self.layers.append(nn.Sequential(*layers))

    def forward(self, x):
        """ Returns a list of convouts for each layer. """
        outs = []

        for idx, layer in enumerate(self.layers):
            x = layer(x)
            
            # Apply an l2norm module to the selected layers
            # Note that this differs from the original implemenetation
            if idx in self.norm_lookup:
                x = self.norms[self.norm_lookup[idx]](x)
            outs.append(x)
        
        return tuple(outs)

    def transform_key(self, k):
        """ Transform e.g. features.24.bias to layers.4.1.bias """
        vals = k.split('.')
        layerIdx = self.state_dict_lookup[int(vals[0])]
        return 'layers.%s.%s' % (layerIdx, vals[1])

    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        state_dict = torch.load(path)
        state_dict = OrderedDict([(self.transform_key(k), v) for k,v in state_dict.items()])

        self.load_state_dict(state_dict, strict=False)

    def add_layer(self, conv_channels=128, downsample=2):
        """ Add a downsample layer to the backbone as per what SSD does. """
        if len(self.extra_args) > 0:
            conv_channels, downsample = self.extra_args.pop()
        
        padding = 1 if downsample > 1 else 0
        
        layer = nn.Sequential(
            nn.Conv2d(self.in_channels, conv_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_channels, conv_channels*2, kernel_size=3, stride=downsample, padding=padding),
            nn.ReLU(inplace=True)
        )

        self.in_channels = conv_channels*2
        self.channels.append(self.in_channels)
        self.layers.append(layer)



 
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

    
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)       
          
class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2], # useful
            [6, 32, 3, 2], # useful
            [6, 64, 4, 2],
            [6, 96, 3, 1], # useful
            [6, 160, 3, 2],
            [6, 320, 1, 1], # useful
        ]

        # Select the output layers
        # idx-1 => 56 * 56 * 24
        # idx-2 => 28 * 28 * 32
        # idx-4 => 14 * 14 * 96
        # idx-6 => 7 * 7 * 320
        select_idxs = [1, 2, 4, 6]


        # building first layer
        input_channel = int(input_channel)
        self.conv2d = conv_bn(3, input_channel, 2)
        self.layers = nn.ModuleList()
        self.channels = []

        # building inverted residual blocks
        features = []
        for idx, (t, c, n, s) in enumerate(interverted_residual_setting):
            output_channel = int(c)

            for i in range(n):
                if i == 0:
                    features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

            if idx in select_idxs:
                self.layers.append(nn.Sequential(*features))
                self.channels.append(output_channel)
                features = []

        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

        # random initial
        self._initialize_weights()

    def forward(self, x):
        x = self.conv2d(x)

        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        # print("out",outs)
        return tuple(outs)

    def _initialize_weights(self):
        modules = self.modules()
        for m in modules:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        state_dict = torch.load(path)

        # Replace featuresXXX -> layers.x.xx(/conv2d.x) etc.
        keys = list(state_dict)
        for key in keys:
            if key.startswith('features'):
                idx = int(key.split('.')[1])
                if idx <= 17:
                    if idx == 0:
                        new_key = 'conv2d.' + key[11:]
                    elif (idx >= 1 and idx <= 3):
                        new_key = "layers.0." + str(idx - 1) + key[10:]
                    elif (idx >= 4 and idx <=6):
                        new_key = "layers.1." + str(idx - 4) + key[10:]
                    elif (idx >= 7 and idx <= 13):
                        if idx <= 9:
                            new_key = "layers.2." + str(idx - 7) + key[10:]
                        else:
                            new_key = "layers.2." + str(idx - 7) + key[11:]
                    else:
                        new_key = "layers.3." + str(idx - 14) + key[11:]

                    state_dict[new_key] = state_dict.pop(key)
                else:
                    state_dict.pop(key)
            else:
                state_dict.pop(key)


        # Note: Using strict=False is berry scary. Triple check this.nn.ModuleList()
        self.load_state_dict(state_dict, strict=False)      


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        print(inp,oup,stride)
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=1000, inverted_residual=InvertedResidual):
        
        stages_repeats = [4, 8, 4]
        stages_out_channels =  [24, 48, 96, 192, 1024]
        

        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels
        self.layers = nn.ModuleList()
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels
        print("input channels",input_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layers = []
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            name_block = nn.Sequential(*seq)
            input_channels = output_channels
            self.layers.append(name_block)
            # self.layers.append(self.stage3)
            # self.layers.append(self.stage4)

        # print(self.layers)
        # print(layers)

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        # self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        # x = self.stage2(x)
        # x = self.stage3(x)
        # x = self.stage4(x)
        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)

        x = self.conv5(outs[-1])
        # x = x.mean([2, 3])  # globalpool
        # x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        state_dict = torch.load(path)
        keys = list(state_dict)
        for key in keys:
            if key.startswith('stage'):
                old_layer = key.split('.',1)[0]
                if old_layer == 'stage2':
                    new_key = 'layers.0.'
                elif old_layer == 'stage3':
                    new_key = 'layers.1.'
                if old_layer == 'stage4':
                    new_key = 'layers.2.'
                print(new_key+key.split('.',1)[1],key)
                state_dict[new_key+key.split('.',1)[1]] = state_dict.pop(key)
        self.load_state_dict(state_dict, strict=False)

def construct_backbone(cfg):
    """ Constructs a backbone given a backbone config object (see config.py). """
    print(cfg.name)
    if cfg.type.__name__ == 'ResNetBackbone':
        if cfg.name == 'ResNet18' or cfg.name == 'ResNet34':
            backbone = cfg.type(*cfg.args,block=BasicBlock)
        else:   
            backbone = cfg.type(*cfg.args,block=Bottleneck)
    elif cfg.type.__name__ == 'ShuffleNetV2':
        backbone = cfg.type()
    else:
        backbone = cfg.type(*cfg.args)
    # Add downsampling layers until we reach the number we need
    num_layers = max(cfg.selected_layers) + 1

    while len(backbone.layers) < num_layers:
        backbone.add_layer()

    return backbone
