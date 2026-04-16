############## ResNet for CIFAR-10 ###########
from .quan_resnet_cifar import resnet20_quan, resnet32_quan
from .bin_resnet_cifar import resnet20_bin

############## VGG for CIFAR #############
from .vanilla_models.vanilla_vgg_cifar import vgg11_bn, vgg11
from .quan_vgg_cifar import vgg11_bn_quan, vgg11_quan, vgg16
from .bin_vgg_cifar import vgg11_bn_bin

############# NEW CIFAR 10 ########
from .quan_wideresnet import wideresnet_quan

################ NEW CIFAR 100 ######
#from .quan_swinmldecoder import swinmldecoder_quan
from .quan_vit import vit_quan, vit
