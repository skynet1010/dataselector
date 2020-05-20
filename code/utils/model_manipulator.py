from torch import nn
from code.utils.consts import nr_of_classes, model_dict


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def manipulateModel(model_name, is_feature_extraction,dim):
    model = model_dict[model_name](pretrained=is_feature_extraction)
    set_parameter_requires_grad(model, True)
    #output layer
    if model_name == "resnet18" or \
        model_name == "resnet50" or \
        model_name == "resnet101" or \
        model_name == "inception" or \
        model_name == "googlenet" or \
        model_name == "shufflenet" or \
        model_name == "resnext50_32x4d" or \
        model_name == "resnext101_32x8d" or \
        model_name == "wide_resnet50_2":
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, nr_of_classes)
    elif model_name =="alexnet" or model_name == "vgg16" or model_name=="mobilenet":
        layer_number = 0
        if model_name == "alexnet" or model_name == "vgg16":
            layer_number = 6
        elif model_name == "mobilenet":
            layer_number = 1
        num_ftrs = model.classifier[layer_number].in_features
        model.classifier[layer_number] = nn.Linear(num_ftrs,nr_of_classes)
    elif model_name == "densnet":
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs,nr_of_classes)    
    
    #input layer:
    if dim!=3:
        if model_name =="alexnet" or model_name =="vgg16":
            layer_index = 0
            model.features[layer_index] = nn.Conv2d(dim,64,kernel_size=(11,11),stride=(4,4),padding=(2,2))
        elif model_name == "resnet18" or model_name=="resnet50" or model_name=="resnet101" or model_name == "googlenet" or model_name=="resnext50_32x4d" or model_name=="resnext101_32x8d" or model_name == "wide_resnet50_2":
             if model_name == "googlenet":
                 model.conv1.conv = nn.Conv2d(dim,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
             else:
                 model.conv1 = nn.Conv2d(dim,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
                
        elif model_name == "densnet":
            model.features.conv0 = nn.Conv2d(dim,96,kernel_size=(7,7),stride=(2,2),padding=(3,3), bias=False)
        elif model_name == "inception":
            model.Conv2d_1a_3x3.conv = nn.Conv2d(dim,32,kernel_size=(3,3),stride=(2,2),bias=False)
        elif model_name =="shufflenet":
            model.conv1[0] = nn.Conv2d(dim,24,kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)
        elif model_name == "mobilenet":
            model.features[0][0] = nn.Conv2d(dim,32,kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)



    return model.cuda()
