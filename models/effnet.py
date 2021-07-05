from efficientnet_pytorch import EfficientNet
from torch import nn
import torch

efn_pretrained = {
    'efficientnet-b0': '../pretrained/efficientnet-b0-355c32eb.pth',
    3: '../pretrained/efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
    4: '../pretrained/efficientnet-b4_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
    5: '../pretrained/efficientnet-b5_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
    6: '../pretrained/efficientnet-b6_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
    7: '../pretrained/efficientnet-b7_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
}

class EffNet(nn.Module):
    def __init__(self, efname='efficientnet-b0'):
        super(EffNet, self).__init__()
        self.model = EfficientNet.from_name(efname)
        pretrain_model_fp = efn_pretrained[efname]
        print(f'Load pretrain model from {pretrain_model_fp}')
        self.model.load_state_dict(torch.load(pretrain_model_fp))

    def forward(self, data):
        bs = data.shape[0]
        feat = self.model.extract_features(data)
        feat = nn.functional.adaptive_avg_pool2d(feat, output_size=(1))
        feat = feat.view(bs, -1)
        return feat
