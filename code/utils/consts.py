import torchvision.models as models
import time
from datetime import datetime

import torch

from code.anns.googlenet import googlenet
from code.anns.inception_v3 import inception_v3



data_compositions = {
    "RGB":3,
    "NIR":1,
    "SLOPE":1,
    "ROUGHNESS":1,
    "NDVI":1,
    "DOM":1,
    "RGB_NIR":4,
    "RGB_SLOPE":4,
    "RGB_NDVI":4,
    "NIR_SLOPE":2,
    "NDVI_SLOPE":2,
    "NDVI_NIR":2,
    "RGB_NIR_SLOPE":5,
    "NDVI_NIR_SLOPE":3,
    "RGB_NIR_NDVI_SLOPE":6,
}

model_dict = \
{
    "resnet18":models.resnet18,
    "resnet50":models.resnet50,
    "resnet101":models.resnet101,
    "alexnet":models.alexnet,
    "vgg16":models.vgg16,
    "densnet":models.densenet161,
    "inception":inception_v3,
    "googlenet":googlenet,
    "shufflenet":models.shufflenet_v2_x1_0,
    "mobilenet":models.mobilenet_v2,
    "resnext50_32x4d":models.resnext50_32x4d,
    "resnext101_32x8d":models.resnext101_32x8d,
    "wide_resnet50_2":models.wide_resnet50_2,
}

learning_rate = 1e-3
nr_of_classes = 2

time_stamp  = datetime.utcfromtimestamp(int(time.time())).strftime("%Y%m%d%H%M%S")

optimizer_dict = {
    "Adadelta":torch.optim.Adadelta,
    "Adagrad":torch.optim.Adagrad,
    "Adam":torch.optim.Adam,
    "AdamW":torch.optim.AdamW,
    "Adamax":torch.optim.Adamax,
    "ASGD":torch.optim.ASGD,
    "RMSprop":torch.optim.RMSprop,
    "Rprop":torch.optim.Rprop,
    "SGD":torch.optim.SGD
}

loss_dict = {
    "BCELoss":torch.nn.BCELoss,
    "MSELoss":torch.nn.MSELoss
}