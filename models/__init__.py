# from .vavanilla_resnet_cifar import vanilla_resnet20
from .vanilla_models.vanilla_resnet_imagenet import resnet18
from .quan_resnet_imagenet import resnet18_quan, resnet34_quan
from .quan_alexnet_imagenet import alexnet_quan


############## ResNet for CIFAR-10 ###########
from .vanilla_models.vanilla_resnet_cifar import vanilla_resnet20
from .quan_resnet_cifar import resnet20_quan, resnet32_quan
from .bin_resnet_cifar import resnet20_bin

############## VGG for CIFAR #############

from .vanilla_models.vanilla_vgg_cifar import vgg11_bn, vgg11
from .quan_vgg_cifar import vgg11_bn_quan, vgg11_quan, vgg16
from .bin_vgg_cifar import vgg11_bn_bin


############# Mobilenet for ImageNet #######
from .vanilla_models.vanilla_mobilenet_imagenet import mobilenet_v2
from .quan_mobilenet_imagenet import mobilenet_v2_quan


############### etc for CIFAR 100 ############~~~
from .googlenet_cifar100 import googlenet
from .densenet_cifar100 import densenet121
from .quan_googlenet_cifar100 import googlenet_quan
from .quan_densenet_cifar100 import densenet121_quan, densenet169_quan, densenet201_quan, densenet161_quan
from .quan_shufflenetv2_cifar100 import shufflenetv2_quan
from .quan_mobilenetv2_cifar100 import mobilenetv2_quan
from .quan_squeezenet_cifar100 import squeezenet_quan

############# NEW CIFAR 10 ########
from .quan_wideresnet import wideresnet_quan

################ NEW CIFAR 100 ######
#from .quan_swinmldecoder import swinmldecoder_quan
from .effnet_l2 import effnet_l2
from .quan_vit import vit_quan, vit
from .quan_wyze import wyze_resnet20_quan
