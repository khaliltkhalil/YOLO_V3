import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from helper import *
import cv2
import matplotlib.pyplot as plt

def get_test_input():
    img = cv2.imread("imgs/dog.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img,x,y = resize_image(img, 416)
    plt.imshow(img)
    img = img.transpose((2,0,1))  # C X H X W
    img = img/255
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)
    return img


def parse_cfg(cfg_file):
    """
    parse the configuration file and return list of  dictionaries containing blocks of the network
    :param cfg_file:  the config file containing the network architecture
    :return: list of dictionaries containing the blocks
    """

    file = open(cfg_file, "r")
    lines = file.read().split("\n")                 # store the lines in  a list
    lines = [x for x in lines if len(x) > 0]        # remove empty lines
    lines = [x for x in lines if x[0] != "#"]       # remove comment
    lines = [x.lstrip().rstrip() for x in lines]   # remove whitespace before and after the string

    block = {}                                      # dictionary to store the info of each block
    blocks = []

    for line in lines:
        if line[0] == "[":                          # check if it is the beginning of new block
            if len(block) != 0:                     # if the blocks not empty, add the previous block to blocks and
                blocks.append(block)                # re-initialize
                block = {}
            block["type"] = line[1:-1].rstrip()              # store the type of the block
        else:                                       # store the parameter and its value in block dict
            parameter, value = line.split("=")
            block[parameter.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def create_modules(blocks):
    """
    use the list of blocks created from config file to create list of module
    :param blocks: list of dicts containing the net module info
    :return: tuple of net info (dict) and list of modules(nn.ModuleList)
    """
    net_info = blocks[0]            # get the net info from the first dict in blocks
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        if x["type"] == "convolutional":
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            kernel_size = int(x["size"])
            padding = int(x["pad"])
            stride = int(x["stride"])
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters,filters,kernel_size,stride,pad,bias= bias)
            module.add_module("conv_{0}".format(index), conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index),bn)

            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace= True)
                module.add_module("leaky_{0}".format(index), activn)

        elif (x["type"] == "upsample"):
            stride = x["stride"]
            upsample = nn.Upsample(scale_factor= stride, mode="bilinear",align_corners=True)
            module.add_module("upsample_{0}".format(index), upsample)

        elif x["type"] == "route":
            layers = x["layers"].split(",")
            layers = [int(i) if int(i) > 0 else index+int(i) for i in layers]
            if len(layers) == 1:
                filters = output_filters[layers[0]]
            else:
                filters = output_filters[layers[0]] + output_filters[layers[1]]
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{0}".format(index), shortcut)

        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(i) for i in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(i) for i in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0,len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("detection_{0}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)

class Darknet(nn.Module):
    def __init__(self, cfg_file):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfg_file)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}

        write = 0
        for index, module in enumerate(modules):
            module_type = module["type"]
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[index](x)

            elif module_type == "route":
                layers = module["layers"].split(",")
                layers = [int(i) if int(i) > 0 else int(i)+index for i in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                else:
                    map1 = outputs[layers[0]]
                    map2 = outputs[layers[1]]

                    x = torch.cat((map1, map2), 1)

            elif module_type == "shortcut":
                from_layer = module["from"]
                from_layer = int(from_layer) if int(from_layer) > 0 else int(from_layer) + index
                x = outputs[index-1] + outputs[from_layer]

            elif module_type == "yolo":
                anchors = self.module_list[index][0].anchors
                inp_dim = int(self.net_info["height"])
                num_classes = int(module["classes"])

                #x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[index] = x

        return detections

    def load_weights(self, weight_file):
        file = open(weight_file, "rb")
        header = np.fromfile(file, dtype=np.int32, count = 5)
        self.header = torch.from_numpy(header)

        weights = np.fromfile(file, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]["type"]

            if module_type == "convolutional":
                module = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = module[0]

                if batch_normalize:
                    bn = module[1]

                    num_bn_biases = bn.bias.numel()
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    num_biases = conv.bias.numel()
                    conv_biases = torch.from_numpy(weights[ptr:ptr + num_biases])
                    ptr += num_biases

                    conv_biases = conv_biases.view_as(conv.bias.data)

                    conv.bias.data.copy_(conv_biases)

                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr += num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)










        













