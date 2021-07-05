import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence
from models.tabnet.tab_network import TabNet

from .fusion import BilinearFusion
from copy import deepcopy

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class MILNet(nn.Module):
    def __init__(self, input_dim=512, attention_dim=128, attention_out_dim=1, if_freeze_param=False, dropout_p=0.5,
                 instance_attention_layers=1, instance_attention_dim=128, feature_attention_layers=1,
                 feature_represent_dim=512,
                 feature_represent_layers=1,
                 feature_attention_dim=256):
        """

        Args:
            input_dim: number of instance feature
            attention_dim:
            attention_out_dim: number of features after attention
            if_freeze_param:
        """
        super(MILNet, self).__init__()  # super(subClass, instance).method(args), python 2.x
        self.L = input_dim
        self.D = attention_dim
        self.K = attention_out_dim
        self.if_freeze_param = if_freeze_param

        """
        instance attention layers
        """
        attention_list = [
            nn.Linear(feature_represent_dim, instance_attention_dim),
            nn.LeakyReLU(),
        ]
        for _ in range(instance_attention_layers):
            attention_list.extend([
                nn.Dropout(dropout_p),
                nn.Linear(instance_attention_dim, instance_attention_dim),
                nn.LeakyReLU(),

            ])
        attention_list.extend([
            nn.Linear(instance_attention_dim, self.K)
        ])
        self.attention = nn.Sequential(*attention_list)

        """
        feature represent layers
        """
        feature_represent_layer_list = [
            nn.Linear(self.L, feature_represent_dim),
            nn.LeakyReLU(),
        ]
        for _ in range(feature_represent_layers):
            feature_represent_layer_list.extend([
                nn.Linear(feature_represent_dim, feature_represent_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout_p),
            ])
        self.feature_represent = nn.Sequential(*feature_represent_layer_list)

        """
        feature attention layers
        """
        feature_attention_list = [
            nn.Linear(feature_represent_dim, feature_attention_dim),
            Swish(),
        ]
        for _ in range(feature_attention_layers):
            feature_attention_list.extend([
                nn.Linear(feature_attention_dim, feature_attention_dim),
                nn.LeakyReLU(),
            ])
        feature_attention_list.extend([
            nn.Linear(feature_attention_dim, feature_represent_dim)
        ])
        self.feature_attention = nn.Sequential(*feature_attention_list)

        """
        final classifier
        """
        self.classifier = nn.Sequential(
            nn.Linear(feature_represent_dim * self.K, feature_represent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(feature_represent_dim, 1)
        )

    def get_instance_attention_parameters(self):
        params = []
        for param in self.attention.parameters():
            params.append(param)
        return params

    def get_feature_attention_parameters(self):
        params = []
        for param in self.feature_attention.parameters():
            params.append(param)
        return params

    def get_classifier_parameters(self):
        params = []
        for param in self.classifier.parameters():
            params.append(param)
        return params

    def forward(self, batch_data: torch.Tensor):
        # H: NxL embedding
        # if len(batch_data.size()) == 5:
        # 1 num_instance num_channel, w, h
        if len(batch_data.size()) == 3:
            # 1 #instance #feat
            batch_data = batch_data.squeeze(0)

        bag = batch_data

        bag = self.feature_represent(bag)
        # instance attention
        A = self.attention(bag)  # NxK attentions
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        # feature attention
        feature_attention = self.feature_attention(bag)
        feature_attention = torch.sigmoid(feature_attention)
        bag = bag * feature_attention
        M = torch.mm(A, bag)  # KxL

        Y_prob = self.classifier(M)
        return Y_prob, A


class MILNetWithTabFeat(nn.Module):
    def __init__(self, input_dim=512, attention_dim=128, attention_out_dim=1, if_freeze_param=False, dropout_p=0.5,
                 instance_attention_layers=1, instance_attention_dim=128, feature_attention_layers=1,
                 feature_represent_dim=512,
                 feature_represent_layers=1,
                 feature_attention_dim=256,
                 tabfeat_dim=32):
        """

        Args:
            input_dim: number of instance feature
            attention_dim:
            attention_out_dim: number of features after attention
            if_freeze_param:
        """
        super(MILNetWithTabFeat, self).__init__()  # super(subClass, instance).method(args), python 2.x
        self.L = input_dim
        self.D = attention_dim
        self.K = attention_out_dim
        self.if_freeze_param = if_freeze_param

        """
        instance attention layers
        """
        attention_list = [
            nn.Linear(feature_represent_dim, instance_attention_dim),
            nn.LeakyReLU(),
        ]
        for _ in range(instance_attention_layers):
            attention_list.extend([
                nn.Dropout(dropout_p),
                nn.Linear(instance_attention_dim, instance_attention_dim),
                nn.LeakyReLU(),
            ])
        attention_list.extend([
            nn.Linear(instance_attention_dim, self.K)
        ])
        self.attention = nn.Sequential(*attention_list)

        """
        feature represent layers
        """
        feature_represent_layer_list = [
            nn.Linear(self.L, feature_represent_dim),
            nn.LeakyReLU(),
        ]
        for _ in range(feature_represent_layers):
            feature_represent_layer_list.extend([
                nn.Linear(feature_represent_dim, feature_represent_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout_p),
            ])
        self.feature_represent = nn.Sequential(*feature_represent_layer_list)

        """
        feature attention layers
        """
        feature_attention_list = [
            nn.Linear(feature_represent_dim, feature_attention_dim),
            Swish(),
        ]
        for _ in range(feature_attention_layers):
            feature_attention_list.extend([
                nn.Linear(feature_attention_dim, feature_attention_dim),
                nn.LeakyReLU(),
            ])
        feature_attention_list.extend([
            nn.Linear(feature_attention_dim, feature_represent_dim)
        ])
        self.feature_attention = nn.Sequential(*feature_attention_list)

        """
        final classifier
        """
        self.classifier = nn.Sequential(
            nn.Linear(feature_represent_dim * self.K + tabfeat_dim, feature_represent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(feature_represent_dim, 1)
        )

    def get_instance_attention_parameters(self):
        params = []
        for param in self.attention.parameters():
            params.append(param)
        return params

    def get_feature_attention_parameters(self):
        params = []
        for param in self.feature_attention.parameters():
            params.append(param)
        return params

    def get_classifier_parameters(self):
        params = []
        for param in self.classifier.parameters():
            params.append(param)
        return params

    def forward(self, batch_data: torch.Tensor, tab_feat: torch.Tensor):
        # H: NxL embedding
        # if len(batch_data.size()) == 5:
        # 1 num_instance num_channel, w, h
        if len(batch_data.size()) == 3:
            # 1 #instance #feat
            batch_data = batch_data.squeeze(0)

        bag = batch_data

        bag = self.feature_represent(bag)
        # instance attention
        A = self.attention(bag)  # NxK attentions
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        # feature attention
        feature_attention = self.feature_attention(bag)
        feature_attention = torch.sigmoid(feature_attention)
        bag = bag * feature_attention
        M = torch.mm(A, bag)  # KxL

        M = M.view(1, -1)
        tab_feat = tab_feat.view(1, -1)

        merge_feat = torch.cat([M, tab_feat], dim=1)
        Y_prob = self.classifier(merge_feat)

        return Y_prob


class MMTMBi(nn.Module):
    """
    bi moludal fusion
    """

    def __init__(self, dim_tab, dim_img, ratio=4):
        """

        Parameters
        ----------
        dim_tab: feature dimension of tabular data
        dim_img: feature dimension of MIL image modal
        ratio
        """
        super(MMTMBi, self).__init__()
        dim = dim_tab + dim_img
        dim_out = int(2 * dim / ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)

        self.fc_tab = nn.Linear(dim_out, dim_tab)
        self.fc_img = nn.Linear(dim_out, dim_img)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, tab_feat, img_feat) -> Sequence[torch.Tensor]:
        """

        Parameters
        ----------
        tab_feat: b * c
        skeleton: b * c

        Returns
            表格数据加权结果
            WSI 全局特征加权结果
            WSI 全局特征加权权重
        -------

        """

        squeeze = torch.cat([tab_feat, img_feat], dim=1)

        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)

        tab_out = self.fc_tab(excitation)
        img_out = self.fc_img(excitation)

        tab_out = self.sigmoid(tab_out)
        img_out = self.sigmoid(img_out)

        return tab_feat * tab_out, img_feat * img_out, img_out

class MMTMTri(nn.Module):
    """
    tri-modal fusion
    """

    def __init__(self, dim_img, ratio=4):
        """

        Parameters
        ----------
        dim_tab: feature dimension of tabular data
        dim_img: feature dimension of MIL model
        ratio
        """
        super(MMTMTri, self).__init__()
        dim = dim_img * 3
        dim_out = int(2 * dim / ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)


        self.fc_img_scale1 = nn.Linear(dim_out, dim_img)
        self.fc_img_scale2 = nn.Linear(dim_out, dim_img)
        self.fc_img_scale3 = nn.Linear(dim_out, dim_img)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_feat_scale1, img_feat_scale2, img_feat_scale3) -> Sequence[torch.Tensor]:
        """

        Parameters
        ----------
        tab_feat: b * c
        skeleton: b * c

        Returns
        -------

        """

        squeeze = torch.cat([img_feat_scale1, img_feat_scale2, img_feat_scale3], dim=1)

        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)


        img_out_scale1 = self.fc_img_scale1(excitation)
        img_out_scale2 = self.fc_img_scale2(excitation)
        img_out_scale3 = self.fc_img_scale3(excitation)

        img_out_scale1 = self.sigmoid(img_out_scale1)
        img_out_scale2 = self.sigmoid(img_out_scale2)
        img_out_scale3 = self.sigmoid(img_out_scale3)

        return img_feat_scale1 * img_out_scale1, img_out_scale1, img_feat_scale2 * img_out_scale2, img_out_scale2,  img_feat_scale2 * img_out_scale3, img_out_scale3

class MMTMQuad(nn.Module):
    """
    quad modal fusion
    """

    def __init__(self, dim_tab, dim_img, ratio=4):
        """

        Parameters
        ----------
        dim_tab: feature dimension of tabular data
        dim_img: feature dimension of MIL model
        ratio
        """
        super(MMTMQuad, self).__init__()
        dim = dim_tab + dim_img * 3
        dim_out = int(2 * dim / ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)

        self.fc_tab = nn.Linear(dim_out, dim_tab)

        self.fc_img_scale1 = nn.Linear(dim_out, dim_img)
        self.fc_img_scale2 = nn.Linear(dim_out, dim_img)
        self.fc_img_scale3 = nn.Linear(dim_out, dim_img)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, tab_feat, img_feat_scale1, img_feat_scale2, img_feat_scale3) -> Sequence[torch.Tensor]:
        """

        Parameters
        ----------
        tab_feat: b * c
        skeleton: b * c

        Returns
        -------

        """

        squeeze = torch.cat([tab_feat, img_feat_scale1, img_feat_scale2, img_feat_scale3], dim=1)

        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)

        tab_out = self.fc_tab(excitation)
        img_out_scale1 = self.fc_img_scale1(excitation)
        img_out_scale2 = self.fc_img_scale2(excitation)
        img_out_scale3 = self.fc_img_scale3(excitation)

        tab_out = self.sigmoid(tab_out)
        img_out_scale1 = self.sigmoid(img_out_scale1)
        img_out_scale2 = self.sigmoid(img_out_scale2)
        img_out_scale3 = self.sigmoid(img_out_scale3)

        return tab_feat * tab_out, img_feat_scale1 * img_out_scale1, img_out_scale1, img_feat_scale2 * img_out_scale2, img_out_scale2,  img_feat_scale2 * img_out_scale3, img_out_scale3


class InstanceAttentionGate(nn.Module):
    def __init__(self, feat_dim):
        super(InstanceAttentionGate, self).__init__()
        self.trans = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LeakyReLU(),
            nn.Linear(feat_dim, 1),
        )

    def forward(self, instance_feature, global_feature):
        feat = torch.cat([instance_feature, global_feature], dim=1)
        attention = self.trans(feat)
        return attention


class MILFusion(nn.Module):
    def __init__(self, img_feat_input_dim=512, tab_feat_input_dim=32,
                 img_feat_rep_layers=4,
                 num_modal=2,
                 use_tabnet=False,
                 tab_indim=0,
                 local_rank=0,
                 cat_idxs=None,
                 cat_dims=None,
                 lambda_sparse=1e-3,
                 fusion='mmtm',
                 use_k_agg=False,
                 k_agg=10,
                 ):
        super(MILFusion, self).__init__()
        self.num_modal = num_modal
        self.local_rank = local_rank
        self.use_tabnet = use_tabnet
        self.tab_indim = tab_indim
        self.lambda_sparse = lambda_sparse
        # define K mean agg
        self.use_k_agg = use_k_agg
        self.k_agg = k_agg

        self.fusion_method = fusion
        if self.use_tabnet:
            self.tabnet = TabNet(input_dim=tab_indim, output_dim=1,
                                 n_d=32, n_a=32, n_steps=5,
                                 gamma=1.5, n_independent=2, n_shared=2,
                                 momentum=0.3,
                                 cat_idxs=cat_idxs, cat_dims=cat_dims)
        else:
            self.tabnet = None

        if self.use_tabnet and num_modal == 1:
            self.only_tabnet = True
        else:
            self.only_tabnet = False

        """
        Control tabnet
        """
        if self.only_tabnet:
            self.feature_fine_tuning = None
        else:
            """pretrained feature fine tune"""
            feature_fine_tuning_layers = []
            for _ in range(img_feat_rep_layers):
                feature_fine_tuning_layers.extend([
                    nn.Linear(img_feat_input_dim, img_feat_input_dim),
                    nn.LeakyReLU(),
                ])
            self.feature_fine_tuning = nn.Sequential(*feature_fine_tuning_layers)

        # 3 means three scales of images as modals
        if self.num_modal == 4 or self.num_modal == 3:
            self.feature_fine_tuning2 = nn.Sequential(*feature_fine_tuning_layers)
            self.feature_fine_tuning3 = nn.Sequential(*feature_fine_tuning_layers)
        else:
            self.feature_fine_tuning2 = None
            self.feature_fine_tuning3 = None

        if self.only_tabnet or self.num_modal == 3:
            self.table_feature_ft = None
        else:
            """tab feature fine tuning"""
            self.table_feature_ft = nn.Sequential(
                nn.Linear(tab_feat_input_dim, tab_feat_input_dim)
            )

        # k agg score
        self.score_fc = nn.ModuleList()
        if self.use_k_agg:
            for _ in range(self.num_modal - 1):
                self.score_fc.append(
                    nn.Sequential(
                        nn.Linear(img_feat_input_dim, img_feat_input_dim),
                        nn.LeakyReLU(),
                        nn.Linear(img_feat_input_dim, img_feat_input_dim),
                        nn.LeakyReLU(),
                        nn.Linear(img_feat_input_dim, 1),
                        nn.Sigmoid()
                    )
                )


        """modal fusion"""
        self.wsi_select_gate = None
        # define different fusion methods and related output feature dimension and fusion module
        if self.only_tabnet:
            self.mmtm = None
        elif self.fusion_method == 'concat':
            self.fusion_out_dim = tab_feat_input_dim + img_feat_input_dim
            self.wsi_select_gate = nn.Sequential(
                nn.Linear(img_feat_input_dim, 1),
                nn.Sigmoid()
            )
            self.mmtm = nn.Linear(self.fusion_out_dim, self.fusion_out_dim)
        elif self.fusion_method == 'bilinear':
            self.wsi_select_gate = nn.Sequential(
                nn.Linear(img_feat_input_dim, 1),
                nn.Sigmoid()
            )
            self.fusion_out_dim = tab_feat_input_dim + img_feat_input_dim
            self.mmtm = nn.Bilinear(tab_feat_input_dim, img_feat_input_dim, self.fusion_out_dim)
        elif self.fusion_method == 'add':
            self.wsi_select_gate = nn.Sequential(
                nn.Linear(img_feat_input_dim, 1),
                nn.Sigmoid()
            )
            self.fusion_out_dim = tab_feat_input_dim
            self.mmtm = nn.Linear(img_feat_input_dim * (num_modal - 1), tab_feat_input_dim)
        elif self.fusion_method == 'gate':
            self.wsi_select_gate = nn.Sequential(
                nn.Linear(img_feat_input_dim, 1),
                nn.Sigmoid()
            )
            self.fusion_out_dim = 96
            self.mmtm = BilinearFusion(dim1=tab_feat_input_dim, dim2=img_feat_input_dim, mmhid=self.fusion_out_dim)

        elif self.num_modal == 2 and self.fusion_method == 'mmtm':
            self.fusion_out_dim = (img_feat_input_dim * 2) * (num_modal - 1) + tab_feat_input_dim
            self.mmtm = MMTMBi(dim_tab=tab_feat_input_dim, dim_img=img_feat_input_dim)
        elif self.num_modal == 3 and self.fusion_method == 'mmtm':
            self.fusion_out_dim = (img_feat_input_dim * 2) * 3
            self.mmtm = MMTMTri(dim_img=img_feat_input_dim)
        elif self.num_modal == 4 and self.fusion_method == 'mmtm':
            self.fusion_out_dim = (img_feat_input_dim * 2) * (num_modal - 1) + tab_feat_input_dim
            self.mmtm = MMTMQuad(dim_tab=tab_feat_input_dim, dim_img=img_feat_input_dim)
        else:
            raise NotImplementedError(f'num_modal {num_modal} not implemented')

        """instance selection"""
        if self.only_tabnet or self.fusion_method in ['concat', 'add', 'bilinear', 'gate']:
            self.instance_gate1 = None
        else:
            self.instance_gate1 = InstanceAttentionGate(img_feat_input_dim)

        if (self.num_modal == 4 or self.num_modal == 3)and self.fusion_method == 'mmtm':
            self.instance_gate2 = InstanceAttentionGate(img_feat_input_dim)
            self.instance_gate3 = InstanceAttentionGate(img_feat_input_dim)
        else:
            self.instance_gate2 = None
            self.instance_gate3 = None

        """classifier layer"""
        if self.only_tabnet:
            self.classifier = None
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.fusion_out_dim, self.fusion_out_dim),
                nn.Dropout(0.5),
                nn.Linear(self.fusion_out_dim, 1)
            )

    def agg_k_cluster_by_score(self, data: torch.Tensor, score_fc: nn.Module):
        num_elements = data.shape[0]
        score = score_fc(data)

        """
        >>> score = torch.rand(4,1)
        >>> top_score, top_idx = torch.topk(score, k=num_elements, dim=0)
        >>> top_score, top_idx
        (tensor([[0.3963],
         [0.0856],
         [0.0704],
         [0.0247]]),
         tensor([[1],
                 [0],
                 [3],
                 [2]]))
        """
        top_score, top_idx = torch.topk(score, k=num_elements, dim=0)
        """
        >>> data
        tensor([[0.0672, 0.9001, 0.5660, 0.0522, 0.1543],
        [0.1965, 0.7711, 0.9737, 0.5269, 0.9255],
        [0.6761, 0.5801, 0.4687, 0.1683, 0.8136],
        [0.2091, 0.9620, 0.8105, 0.8210, 0.3391]])
        >>> top_idx[:, 0]
        tensor([1, 0, 3, 2])
        >>> data_sorted
        tensor([[0.1965, 0.7711, 0.9737, 0.5269, 0.9255],
        [0.0672, 0.9001, 0.5660, 0.0522, 0.1543],
        [0.2091, 0.9620, 0.8105, 0.8210, 0.3391],
        [0.6761, 0.5801, 0.4687, 0.1683, 0.8136]])
        """
        data_sorted = torch.zeros_like(data)
        data_sorted.index_copy_(dim=0, index=top_idx[:, 0], source=data)

        # set Batch with feature dim
        data_sorted = torch.transpose(data_sorted, 1, 0)
        data_sorted = data_sorted.unsqueeze(1)


        agg_result = nn.functional.adaptive_max_pool1d(data_sorted, self.k_agg)

        agg_result = agg_result.squeeze(1)
        agg_result = torch.transpose(agg_result, 1, 0)

        return agg_result

    def forward(self, data):

        attention_weight_out_list = []
        if self.use_tabnet:
            if torch.cuda.is_available():
                tab_data = data['tab_data'].cuda(self.local_rank)
            else:
                tab_data = data['tab_data']
            if self.only_tabnet:
                tab_logit, M_loss = self.tabnet(tab_data)
            else:
                tab_logit, tab_feat, M_loss = self.tabnet(tab_data)

            tab_loss_weight = 1.
        else:
            tab_feat = data['tab_feat'].cuda(self.local_rank)
            tab_logit = torch.zeros((1, 1)).cuda(self.local_rank)
            M_loss = 0.
            tab_loss_weight = 0.

        if torch.cuda.is_available():
            y = data['label'].cuda(self.local_rank)
            wsi_feat_scale1 = data['wsi_feat_scale1'].cuda(self.local_rank)
        else:
            y = data['label']
            wsi_feat_scale1 = data['wsi_feat_scale1']
        if len(wsi_feat_scale1.size()) == 3:
            # 1 #instance #feat
            wsi_feat_scale1 = wsi_feat_scale1.squeeze(0)
        scale1_bs = wsi_feat_scale1.shape[0]

        if self.only_tabnet:
            out = tab_logit
            tab_loss_weight = 0.
        elif self.fusion_method in ['concat', 'add', 'bilinear', 'gate']:
            tab_feat = self.table_feature_ft(tab_feat)
            # fusion first
            feat_list = []
            for scale, ft_fc in zip(range(3), [self.feature_fine_tuning, self.feature_fine_tuning2, self.feature_fine_tuning3]):
                if ft_fc is None:
                    break
                wsi_feat = data[f'wsi_feat_scale{scale+1}'].cuda(self.local_rank)
                wsi_feat = wsi_feat.squeeze(0)
                wsi_feat = ft_fc(wsi_feat)

                feat_list.append(wsi_feat)

            wsi_feat_concat = torch.cat(feat_list, dim=0)

            global_wsi_feat_weight = self.wsi_select_gate(wsi_feat_concat)
            global_wsi_feat = torch.sum(wsi_feat_concat * global_wsi_feat_weight, dim=0, keepdim=True)

            if self.fusion_method == 'concat':
                fusion_feat = self.mmtm(torch.cat([tab_feat, global_wsi_feat], dim=1))
            elif self.fusion_method == 'add':
                fusion_feat = self.mmtm(global_wsi_feat) + tab_feat
            elif self.fusion_method == 'bilinear':
                fusion_feat = self.mmtm(tab_feat, global_wsi_feat)
            elif self.fusion_method == 'gate':
                fusion_feat = self.mmtm(tab_feat, global_wsi_feat)

            out = self.classifier(fusion_feat)


        elif self.num_modal == 2:
            wsi_feat_scale1 = self.feature_fine_tuning(wsi_feat_scale1)
            wsi_feat_scale1_gloabl = torch.mean(wsi_feat_scale1, dim=0, keepdim=True)  # instance level mean

            tab_feat_mmtm, wsi_feat1_gloabl, wsi_feat_scale1_gate = self.mmtm(tab_feat, wsi_feat_scale1_gloabl)

            # table feature calculate once more
            tab_feat_ft = self.table_feature_ft(tab_feat_mmtm)

            # weight on feature level
            wsi_feat_scale1 = wsi_feat_scale1 * wsi_feat_scale1_gate

            wsi_feat1_gloabl_repeat = wsi_feat1_gloabl.detach().repeat(scale1_bs, 1)

            # N * 1
            instance_attention_weight = self.instance_gate1(wsi_feat_scale1, wsi_feat1_gloabl_repeat)
            # 1 * N
            instance_attention_weight = torch.transpose(instance_attention_weight, 1, 0)

            instance_attention_weight = torch.softmax(instance_attention_weight, dim=1)

            attention_weight_out_list.append(instance_attention_weight.detach().clone())

            # 1 * N
            wsi_feat_agg_scale1 = torch.mm(instance_attention_weight, wsi_feat_scale1)

            final_feat = torch.cat([tab_feat_ft, wsi_feat_agg_scale1, wsi_feat1_gloabl], dim=1)

            out = self.classifier(final_feat)

        elif self.num_modal == 3:
            """
            Fuse 3 modalities
            """
            wsi_feat_scale2 = data['wsi_feat_scale2'].cuda(self.local_rank)
            wsi_feat_scale3 = data['wsi_feat_scale3'].cuda(self.local_rank)
            if len(wsi_feat_scale2.size()) == 3:
                # 1 #instance #feat
                wsi_feat_scale2 = wsi_feat_scale2.squeeze(0)
            if len(wsi_feat_scale3.size()) == 3:
                # 1 #instance #feat
                wsi_feat_scale3 = wsi_feat_scale3.squeeze(0)

            """fine-tuning on 3 scales"""
            wsi_ft_feat_list = []
            for ft_conv, wsi_feat in zip(
                    [self.feature_fine_tuning, self.feature_fine_tuning2, self.feature_fine_tuning3],
                    [wsi_feat_scale1, wsi_feat_scale2, wsi_feat_scale3],
            ):
                wsi_ft_feat_list.append(ft_conv(wsi_feat))

            if self.use_k_agg:
                agg_feat_list = []
                for data_feat, score_fc in zip(wsi_ft_feat_list, self.score_fc):
                    agg_feat_list.append(self.agg_k_cluster_by_score(data_feat, score_fc))
                wsi_ft_feat_list = agg_feat_list

                wsi_feat_scale_gloabl_list = []
                for data_feat, score_fc in zip(agg_feat_list, self.score_fc):
                    feat_score = score_fc(data_feat)
                    feat_attention = torch.softmax(feat_score, dim=0)
                    global_feat = torch.sum(data_feat * feat_attention, dim=0, keepdim=True)
                    wsi_feat_scale_gloabl_list.append(global_feat)

            else:
                """global representation of 3 scales features"""
                wsi_feat_scale_gloabl_list = []
                for feat in wsi_ft_feat_list:
                    wsi_feat_scale_gloabl_list.append(torch.mean(feat, dim=0, keepdim=True))

            """mmtm"""
            wsi_feat1_gloabl, wsi_feat_scale1_gate, wsi_feat2_gloabl, wsi_feat_scale2_gate, wsi_feat3_gloabl, wsi_feat_scale3_gate = self.mmtm(
                *wsi_feat_scale_gloabl_list)

            """instance selection on 3 scales"""
            wsi_feat_agg_list = []
            for wsi_feat_at_scale, wsi_feat_gate_at_scale, wsi_global_rep, instance_gate in zip(
                    wsi_ft_feat_list,
                    [wsi_feat_scale1_gate, wsi_feat_scale2_gate, wsi_feat_scale3_gate],
                    wsi_feat_scale_gloabl_list,
                    [self.instance_gate1, self.instance_gate2, self.instance_gate3]
            ):
                #
                bs_at_scale = wsi_feat_at_scale.shape[0]
                wsi_feat_at_scale = wsi_feat_at_scale * wsi_feat_gate_at_scale
                wsi_global_rep_repeat = wsi_feat_gate_at_scale.detach().repeat(bs_at_scale, 1)

                # N * 1
                instance_attention_weight = instance_gate(wsi_feat_at_scale, wsi_global_rep_repeat)
                # 1 * N
                instance_attention_weight = torch.transpose(instance_attention_weight, 1, 0)
                instance_attention_weight = torch.softmax(instance_attention_weight, dim=1)

                # instance aggregate
                wsi_feat_agg = torch.mm(instance_attention_weight, wsi_feat_at_scale)
                wsi_feat_agg_list.append(wsi_feat_agg)


            final_feat = torch.cat(
                [*wsi_feat_agg_list, wsi_feat1_gloabl, wsi_feat2_gloabl, wsi_feat3_gloabl], dim=1)

            out = self.classifier(final_feat)

        elif self.num_modal == 4:
            """
            Fuse 4 modalities
            """
            if torch.cuda.is_available():
                wsi_feat_scale2 = data['wsi_feat_scale2'].cuda(self.local_rank)
                wsi_feat_scale3 = data['wsi_feat_scale3'].cuda(self.local_rank)
            else:
                wsi_feat_scale2 = data['wsi_feat_scale2']
                wsi_feat_scale3 = data['wsi_feat_scale3']
            if len(wsi_feat_scale2.size()) == 3:
                # 1 #instance #feat
                wsi_feat_scale2 = wsi_feat_scale2.squeeze(0)
            if len(wsi_feat_scale3.size()) == 3:
                # 1 #instance #feat
                wsi_feat_scale3 = wsi_feat_scale3.squeeze(0)

            if self.use_k_agg:
                if wsi_feat_scale1.shape[0] < self.k_agg:
                    pad_size = self.k_agg - wsi_feat_scale1.shape[0]
                    zero_size = (pad_size, *wsi_feat_scale1.shape[1:])
                    pad_tensor = torch.zeros(zero_size).to(wsi_feat_scale1.device)
                    wsi_feat_scale1 = torch.cat([wsi_feat_scale1, pad_tensor])
                if wsi_feat_scale2.shape[0] < self.k_agg:
                    pad_size = self.k_agg - wsi_feat_scale2.shape[0]
                    zero_size = (pad_size, *wsi_feat_scale2.shape[1:])
                    pad_tensor = torch.zeros(zero_size).to(wsi_feat_scale2.device)
                    wsi_feat_scale2 = torch.cat([wsi_feat_scale2, pad_tensor])
                if wsi_feat_scale3.shape[0] < self.k_agg:
                    pad_size = self.k_agg - wsi_feat_scale3.shape[0]
                    zero_size = (pad_size, *wsi_feat_scale3.shape[1:])
                    pad_tensor = torch.zeros(zero_size).to(wsi_feat_scale3.device)
                    wsi_feat_scale3 = torch.cat([wsi_feat_scale3, pad_tensor])

            """fine-tuning 3 scales features"""
            wsi_ft_feat_list = []
            for ft_conv, wsi_feat in zip(
                    [self.feature_fine_tuning, self.feature_fine_tuning2, self.feature_fine_tuning3],
                    [wsi_feat_scale1, wsi_feat_scale2, wsi_feat_scale3],
                ):
                wsi_ft_feat_list.append(ft_conv(wsi_feat))

            if self.use_k_agg:
                agg_feat_list = []
                for data_feat, score_fc in zip(wsi_ft_feat_list, self.score_fc):
                    agg_feat_list.append(self.agg_k_cluster_by_score(data_feat, score_fc))
                wsi_ft_feat_list = agg_feat_list

                wsi_feat_scale_gloabl_list = []
                for data_feat, score_fc in zip(agg_feat_list, self.score_fc):
                    feat_score = score_fc(data_feat)
                    feat_attention = torch.sigmoid(feat_score)

                    attention_weight_out_list.append(feat_attention.detach().clone())
                    global_feat = torch.sum(data_feat * feat_attention, dim=0, keepdim=True)
                    wsi_feat_scale_gloabl_list.append(global_feat)
            else:
                """global representation of 3 scales features"""
                wsi_feat_scale_gloabl_list = []
                for feat in wsi_ft_feat_list:
                    wsi_feat_scale_gloabl_list.append(torch.mean(feat, dim=0, keepdim=True))


            """mmtm"""
            tab_feat_mmtm, wsi_feat1_gloabl, wsi_feat_scale1_gate, wsi_feat2_gloabl, wsi_feat_scale2_gate, wsi_feat3_gloabl, wsi_feat_scale3_gate = self.mmtm(tab_feat, *wsi_feat_scale_gloabl_list)

            """instance selection of 3 scales"""
            wsi_feat_agg_list = []
            for wsi_feat_at_scale, wsi_feat_gate_at_scale, wsi_global_rep, instance_gate in zip(
                        wsi_ft_feat_list,
                        [wsi_feat_scale1_gate, wsi_feat_scale2_gate, wsi_feat_scale3_gate],
                        wsi_feat_scale_gloabl_list,
                        [self.instance_gate1, self.instance_gate2, self.instance_gate3]
                ):
                #
                bs_at_scale = wsi_feat_at_scale.shape[0]
                wsi_feat_at_scale = wsi_feat_at_scale * wsi_feat_gate_at_scale
                wsi_global_rep_repeat = wsi_feat_gate_at_scale.detach().repeat(bs_at_scale, 1)

                # N * 1
                instance_attention_weight = instance_gate(wsi_feat_at_scale, wsi_global_rep_repeat)
                # 1 * N
                instance_attention_weight = torch.transpose(instance_attention_weight, 1, 0)

                instance_attention_weight = torch.softmax(instance_attention_weight, dim=1)


                # instance aggregate
                wsi_feat_agg = torch.mm(instance_attention_weight, wsi_feat_at_scale)

                attention_weight_out_list.append(instance_attention_weight.detach().clone())
                wsi_feat_agg_list.append(wsi_feat_agg)

            """tab feat ft"""
            tab_feat_ft = self.table_feature_ft(tab_feat_mmtm)

            final_feat = torch.cat([tab_feat_ft, *wsi_feat_agg_list, wsi_feat1_gloabl, wsi_feat2_gloabl, wsi_feat3_gloabl], dim=1)

            out = self.classifier(final_feat)


            pass
        y = y.view(-1, 1).float()
        loss = F.binary_cross_entropy_with_logits(out, y) + \
               tab_loss_weight * F.binary_cross_entropy_with_logits(tab_logit, y) - \
               self.lambda_sparse * M_loss

        return out, loss, attention_weight_out_list

    def get_params(self, base_lr):
        ret = []

        if self.tabnet is not None:
            tabnet_params = []
            for param in self.tabnet.parameters():
                tabnet_params.append(param)
            ret.append({
                'params': tabnet_params,
                'lr': base_lr
            })

        cls_learning_rate_rate=100
        if self.classifier is not None:
            classifier_params = []
            for param in self.classifier.parameters():
                classifier_params.append(param)
            ret.append({
                'params': classifier_params,
                'lr': base_lr / cls_learning_rate_rate,
            })


        tab_learning_rate_rate = 100
        if self.table_feature_ft is not None:
            misc_params = []
            for param in self.table_feature_ft.parameters():
                misc_params.append(param)
            ret.append({
                'params': misc_params,
                'lr': base_lr / tab_learning_rate_rate,
            })

        mil_learning_rate_rate = 1000
        misc_params = []
        for part in [self.feature_fine_tuning, self.feature_fine_tuning2, self.feature_fine_tuning3,
                     self.instance_gate1, self.instance_gate2, self.instance_gate3,
                     self.wsi_select_gate,
                     self.score_fc]:
            if part is not None:
                for param in part.parameters():
                    misc_params.append(param)
        ret.append({
            'params': misc_params,
            'lr': base_lr / mil_learning_rate_rate,
        })

        misc_learning_rate_rate = 100
        misc_params = []
        for part in [self.mmtm, ]:
            if part is not None:
                for param in part.parameters():
                    misc_params.append(param)
        ret.append({
            'params': misc_params,
            'lr': base_lr / misc_learning_rate_rate,
        })

        return ret



"""
M3D part models

"""
# textnet inner concat number
neure_num =  [23, 32, 32, 32, 32, 32, 16, 1]

class TextNet(nn.Module):
    def __init__(self, neure_num):
        super(TextNet, self).__init__()
        self.encoder = make_layers(neure_num[:3])
        self.feature = make_layers(neure_num[2:-1])
        self.fc = nn.Linear(neure_num[-2], neure_num[-1])
        self.sig = nn.Sigmoid()
        self._initialize_weights()

    def forward(self, x):
        encoder = self.encoder(x)
        fea = self.feature(encoder)
        y = self.fc(fea)
        y = self.sig(y)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
                m.bias.data.zero_()


def make_layers(cfg):
    layers = []
    n = len(cfg)
    input_dim = cfg[0]
    for i in range(1, n):
        output_dim = cfg[i]
        if i < n - 1:
            layers += [nn.Linear(input_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU(inplace=True)]
        else:
            layers += [nn.Linear(input_dim, output_dim), nn.ReLU(inplace=True)]
        input_dim = output_dim
    return nn.Sequential(*layers)



class M3D(nn.Module):
    def __init__(self, img_feat_input_dim=512, tab_feat_input_dim=32,
                 img_feat_rep_layers=4,
                 num_modal=2,
                 use_tabnet=False,
                 tab_indim=0,
                 local_rank=0,
                 cat_idxs=None,
                 cat_dims=None,
                 lambda_sparse=1e-3,
                 fusion='mmtm',
                 use_k_agg=False,
                 k_agg=10,
                 ):
        super(M3D, self).__init__()
        self.num_modal = num_modal
        self.local_rank = local_rank
        self.use_tabnet = use_tabnet
        self.tab_indim = tab_indim
        self.lambda_sparse = lambda_sparse
        # define mean agg
        self.use_k_agg = use_k_agg
        self.k_agg = k_agg

        self.fusion_method = fusion

        feature_fine_tuning_layers = []
        for _ in range(img_feat_rep_layers):
            feature_fine_tuning_layers.extend([
                nn.Linear(img_feat_input_dim, img_feat_input_dim),
                nn.LeakyReLU(),
            ])

        self.feature_fine_tuning = nn.Sequential(*deepcopy(feature_fine_tuning_layers))
        self.feature_fine_tuning2 = nn.Sequential(*deepcopy(feature_fine_tuning_layers))
        self.feature_fine_tuning3 = nn.Sequential(*deepcopy(feature_fine_tuning_layers))

        self.fc1 = nn.Sequential(
            nn.Linear(img_feat_input_dim, 1),
            nn.Sigmoid(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(img_feat_input_dim, 1),
            nn.Sigmoid(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(img_feat_input_dim, 1),
            nn.Sigmoid(),
        )

        self.text_net = TextNet(neure_num)

    def forward(self, data):
        tab_data = data['tab_data'].cuda(self.local_rank)
        wsi_feat_scale1 = data['wsi_feat_scale1'].cuda(self.local_rank)
        wsi_feat_scale2 = data['wsi_feat_scale2'].cuda(self.local_rank)
        wsi_feat_scale3 = data['wsi_feat_scale3'].cuda(self.local_rank)
        y = data['label'].cuda(self.local_rank)
        y = y.view(-1, 1).float()

        wsi_feat1 = self.feature_fine_tuning(wsi_feat_scale1)
        wsi_feat2 = self.feature_fine_tuning2(wsi_feat_scale2)
        wsi_feat3 = self.feature_fine_tuning3(wsi_feat_scale3)

        wsi_instance_predict1 = self.fc1(wsi_feat1)
        wsi_instance_predict2 = self.fc2(wsi_feat2)
        wsi_instance_predict3 = self.fc3(wsi_feat3)

        text_predict = self.text_net(tab_data)

        bag_predict = torch.max(wsi_instance_predict1) + torch.max(wsi_instance_predict2) \
                      + torch.max(wsi_instance_predict3) + torch.max(text_predict)

        for_debug_predict = (torch.mean(wsi_instance_predict1) + torch.mean(wsi_instance_predict2) + torch.mean(wsi_instance_predict3)) / 3.

        bag_predict = bag_predict / 4.
        bag_predict = bag_predict.view(-1, 1)
        debug_loss =  F.binary_cross_entropy(for_debug_predict.view(-1, 1), y)
        loss = F.binary_cross_entropy(bag_predict, y) + 1e-10 * debug_loss
        return bag_predict, loss

    def get_params(self, base_lr):
        ret = []


        misc_params = []
        for part in [self.feature_fine_tuning, self.feature_fine_tuning2, self.feature_fine_tuning3,
                     self.fc1, self.fc2, self.fc3]:
            if part is not None:
                for param in part.parameters():
                    misc_params.append(param)
        ret.append({
            'params': misc_params,
            'lr': base_lr / 100,
        })

        misc_params = []
        for part in [self.text_net, ]:
            if part is not None:
                for param in part.parameters():
                    misc_params.append(param)
        ret.append({
            'params': misc_params,
            'lr': base_lr / 100,
        })

        return ret





class MILFusionAppend(nn.Module):
    def __init__(self, img_feat_input_dim=512, tab_feat_input_dim=32,
                 img_feat_rep_layers=4,
                 num_modal=2,
                 use_tabnet=False,
                 tab_indim=0,
                 local_rank=0,
                 cat_idxs=None,
                 cat_dims=None,
                 lambda_sparse=1e-3,
                 fusion='mmtm',
                 use_k_agg=False,
                 k_agg=10,
                 ):
        super(MILFusionAppend, self).__init__()
        self.num_modal = num_modal
        self.local_rank = local_rank
        self.use_tabnet = use_tabnet
        self.tab_indim = tab_indim
        self.lambda_sparse = lambda_sparse
        # define K mean agg
        self.use_k_agg = use_k_agg
        self.k_agg = k_agg

        self.fusion_method = fusion
        if self.use_tabnet:
            self.tabnet = TabNet(input_dim=tab_indim, output_dim=1,
                                 n_d=32, n_a=32, n_steps=5,
                                 gamma=1.5, n_independent=2, n_shared=2,
                                 momentum=0.3,
                                 cat_idxs=cat_idxs, cat_dims=cat_dims)
        else:
            self.tabnet = None

        if self.use_tabnet and num_modal == 1:
            self.only_tabnet = True
        else:
            self.only_tabnet = False

        """
        Control tabnet
        """
        if self.only_tabnet:
            self.feature_fine_tuning = None
        else:
            """pretrained feature fine tune"""
            feature_fine_tuning_layers = []
            for _ in range(img_feat_rep_layers):
                feature_fine_tuning_layers.extend([
                    nn.Linear(img_feat_input_dim, img_feat_input_dim),
                    nn.LeakyReLU(),
                ])
            self.feature_fine_tuning = nn.Sequential(*feature_fine_tuning_layers)

        # 3 is the 3 scales of image
        if self.num_modal == 4 or self.num_modal == 3:
            self.feature_fine_tuning2 = nn.Sequential(*feature_fine_tuning_layers)
            self.feature_fine_tuning3 = nn.Sequential(*feature_fine_tuning_layers)
        else:
            self.feature_fine_tuning2 = None
            self.feature_fine_tuning3 = None

        if self.only_tabnet or self.num_modal == 3:
            self.table_feature_ft = None
        else:
            """tab feature fine tuning"""
            self.table_feature_ft = nn.Sequential(
                nn.Linear(tab_feat_input_dim, tab_feat_input_dim)
            )

        # k agg score
        self.score_fc = nn.ModuleList()
        if self.use_k_agg:
            for _ in range(self.num_modal - 1):
                self.score_fc.append(
                    nn.Sequential(
                        nn.Linear(img_feat_input_dim, img_feat_input_dim),
                        nn.LeakyReLU(),
                        nn.Linear(img_feat_input_dim, img_feat_input_dim),
                        nn.LeakyReLU(),
                        nn.Linear(img_feat_input_dim, 1),
                        nn.Sigmoid()
                    )
                )


        """modal fusion"""
        self.wsi_select_gate = None

        if self.only_tabnet:
            self.mmtm = None
        elif self.fusion_method == 'concat':
            self.fusion_out_dim = tab_feat_input_dim + img_feat_input_dim
            self.wsi_select_gate = nn.Sequential(
                nn.Linear(img_feat_input_dim, 1),
                nn.Sigmoid()
            )
            self.mmtm = nn.Linear(self.fusion_out_dim, self.fusion_out_dim)
        elif self.fusion_method == 'bilinear':
            self.wsi_select_gate = nn.Sequential(
                nn.Linear(img_feat_input_dim, 1),
                nn.Sigmoid()
            )
            self.fusion_out_dim = tab_feat_input_dim + img_feat_input_dim
            self.mmtm = nn.Bilinear(tab_feat_input_dim, img_feat_input_dim, self.fusion_out_dim)
        elif self.fusion_method == 'add':
            self.wsi_select_gate = nn.Sequential(
                nn.Linear(img_feat_input_dim, 1),
                nn.Sigmoid()
            )
            self.fusion_out_dim = tab_feat_input_dim
            self.mmtm = nn.Linear(img_feat_input_dim * (num_modal - 1), tab_feat_input_dim)
        elif self.fusion_method == 'gate':
            self.wsi_select_gate = nn.Sequential(
                nn.Linear(img_feat_input_dim, 1),
                nn.Sigmoid()
            )
            self.fusion_out_dim = 96
            self.mmtm = BilinearFusion(dim1=tab_feat_input_dim, dim2=img_feat_input_dim, mmhid=self.fusion_out_dim)
        elif self.num_modal == 2 and self.fusion_method == 'mmtm':
            self.fusion_out_dim = (img_feat_input_dim * 2) * (num_modal - 1) + tab_feat_input_dim
            self.mmtm = MMTMBi(dim_tab=tab_feat_input_dim, dim_img=img_feat_input_dim)
        elif self.num_modal == 3 and self.fusion_method == 'mmtm':
            self.fusion_out_dim = (img_feat_input_dim * 2) * 3
            self.mmtm = MMTMTri(dim_img=img_feat_input_dim)
        elif self.num_modal == 4 and self.fusion_method == 'mmtm':
            self.fusion_out_dim = (img_feat_input_dim * 2) * (num_modal - 1) + tab_feat_input_dim
            self.mmtm = MMTMQuad(dim_tab=tab_feat_input_dim, dim_img=img_feat_input_dim)
        else:
            raise NotImplementedError(f'num_modal {num_modal} not implemented')

        """instance selection"""
        if self.only_tabnet or self.fusion_method in ['concat', 'add', 'bilinear', 'gate']:
            self.instance_gate1 = None
        else:
            self.instance_gate1 = InstanceAttentionGate(img_feat_input_dim)

        if (self.num_modal == 4 or self.num_modal == 3)and self.fusion_method == 'mmtm':
            self.instance_gate2 = InstanceAttentionGate(img_feat_input_dim)
            self.instance_gate3 = InstanceAttentionGate(img_feat_input_dim)
        else:
            self.instance_gate2 = None
            self.instance_gate3 = None

        """classifier layer"""
        if self.only_tabnet:
            self.classifier = None
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.fusion_out_dim, self.fusion_out_dim),
                nn.Dropout(0.5),
                nn.Linear(self.fusion_out_dim, 1)
            )

    def agg_k_cluster_by_score(self, data: torch.Tensor, score_fc: nn.Module):
        num_elements = data.shape[0]
        score = score_fc(data)

        """
        >>> score = torch.rand(4,1)
        >>> top_score, top_idx = torch.topk(score, k=num_elements, dim=0)
        >>> top_score, top_idx
        (tensor([[0.3963],
         [0.0856],
         [0.0704],
         [0.0247]]),
         tensor([[1],
                 [0],
                 [3],
                 [2]]))
        """
        top_score, top_idx = torch.topk(score, k=num_elements, dim=0)
        """
        >>> data
        tensor([[0.0672, 0.9001, 0.5660, 0.0522, 0.1543],
        [0.1965, 0.7711, 0.9737, 0.5269, 0.9255],
        [0.6761, 0.5801, 0.4687, 0.1683, 0.8136],
        [0.2091, 0.9620, 0.8105, 0.8210, 0.3391]])
        >>> top_idx[:, 0]
        tensor([1, 0, 3, 2])
        >>> data_sorted
        tensor([[0.1965, 0.7711, 0.9737, 0.5269, 0.9255],
        [0.0672, 0.9001, 0.5660, 0.0522, 0.1543],
        [0.2091, 0.9620, 0.8105, 0.8210, 0.3391],
        [0.6761, 0.5801, 0.4687, 0.1683, 0.8136]])
        """
        data_sorted = torch.zeros_like(data)
        data_sorted.index_copy_(dim=0, index=top_idx[:, 0], source=data)

        # Batch set as feature dim
        data_sorted = torch.transpose(data_sorted, 1, 0)
        data_sorted = data_sorted.unsqueeze(1)

        agg_result = nn.functional.adaptive_max_pool1d(data_sorted, self.k_agg)

        agg_result = agg_result.squeeze(1)
        agg_result = torch.transpose(agg_result, 1, 0)
        return agg_result

    def forward(self, data):
        attention_weight_out_list = []
        if self.use_tabnet:
            tab_data = data['tab_data'].cuda(self.local_rank)
            if self.only_tabnet:
                tab_logit, M_loss = self.tabnet(tab_data)
            else:
                tab_logit, tab_feat, M_loss = self.tabnet(tab_data)

            tab_loss_weight = 1.
        else:
            tab_feat = data['tab_feat'].cuda(self.local_rank)
            tab_logit = torch.zeros((1, 1)).cuda(self.local_rank)
            M_loss = 0.
            tab_loss_weight = 0.

        y = data['label'].cuda(self.local_rank)
        wsi_feat_scale1 = data['wsi_feat_scale1'].cuda(self.local_rank)
        if len(wsi_feat_scale1.size()) == 3:
            # 1 #instance #feat
            wsi_feat_scale1 = wsi_feat_scale1.squeeze(0)
        scale1_bs = wsi_feat_scale1.shape[0]

        # get the instance weight during the first forward
        #
        """
        Fuse 4 modalities
        """
        wsi_feat_scale2 = data['wsi_feat_scale2'].cuda(self.local_rank)
        wsi_feat_scale3 = data['wsi_feat_scale3'].cuda(self.local_rank)
        if len(wsi_feat_scale2.size()) == 3:
            # 1 #instance #feat
            wsi_feat_scale2 = wsi_feat_scale2.squeeze(0)
        if len(wsi_feat_scale3.size()) == 3:
            # 1 #instance #feat
            wsi_feat_scale3 = wsi_feat_scale3.squeeze(0)

        with torch.no_grad():
            if self.use_k_agg:
                if wsi_feat_scale1.shape[0] < self.k_agg:
                    pad_size = self.k_agg - wsi_feat_scale1.shape[0]
                    zero_size = (pad_size, *wsi_feat_scale1.shape[1:])
                    pad_tensor = torch.zeros(zero_size).to(wsi_feat_scale1.device)
                    wsi_feat_scale1 = torch.cat([wsi_feat_scale1, pad_tensor])
                if wsi_feat_scale2.shape[0] < self.k_agg:
                    pad_size = self.k_agg - wsi_feat_scale2.shape[0]
                    zero_size = (pad_size, *wsi_feat_scale2.shape[1:])
                    pad_tensor = torch.zeros(zero_size).to(wsi_feat_scale2.device)
                    wsi_feat_scale2 = torch.cat([wsi_feat_scale2, pad_tensor])
                if wsi_feat_scale3.shape[0] < self.k_agg:
                    pad_size = self.k_agg - wsi_feat_scale3.shape[0]
                    zero_size = (pad_size, *wsi_feat_scale3.shape[1:])
                    pad_tensor = torch.zeros(zero_size).to(wsi_feat_scale3.device)
                    wsi_feat_scale3 = torch.cat([wsi_feat_scale3, pad_tensor])

            """fine-tuning 3 scales"""
            wsi_ft_feat_list = []
            for ft_conv, wsi_feat in zip(
                    [self.feature_fine_tuning, self.feature_fine_tuning2, self.feature_fine_tuning3],
                    [wsi_feat_scale1, wsi_feat_scale2, wsi_feat_scale3],
                ):
                wsi_ft_feat_list.append(ft_conv(wsi_feat))

            if self.use_k_agg:
                agg_feat_list = []
                for data_feat, score_fc in zip(wsi_ft_feat_list, self.score_fc):
                    agg_feat_list.append(self.agg_k_cluster_by_score(data_feat, score_fc))
                wsi_ft_feat_list = agg_feat_list

                wsi_feat_scale_gloabl_list = []
                for data_feat, score_fc in zip(agg_feat_list, self.score_fc):
                    feat_score = score_fc(data_feat)
                    feat_attention = torch.sigmoid(feat_score)

                    attention_weight_out_list.append(feat_attention.detach().clone())
                    global_feat = torch.sum(data_feat * feat_attention, dim=0, keepdim=True)
                    wsi_feat_scale_gloabl_list.append(global_feat)
            else:
                """global representation of 3 scale images"""
                wsi_feat_scale_gloabl_list = []
                for feat in wsi_ft_feat_list:
                    wsi_feat_scale_gloabl_list.append(torch.mean(feat, dim=0, keepdim=True))


            """mmtm"""
            tab_feat_mmtm, wsi_feat1_gloabl, wsi_feat_scale1_gate, wsi_feat2_gloabl, wsi_feat_scale2_gate, wsi_feat3_gloabl, wsi_feat_scale3_gate = self.mmtm(tab_feat, *wsi_feat_scale_gloabl_list)

            """instance selection of 3 scales"""
            wsi_feat_agg_list = []
            for wsi_feat_at_scale, wsi_feat_gate_at_scale, wsi_global_rep, instance_gate in zip(
                        wsi_ft_feat_list,
                        [wsi_feat_scale1_gate, wsi_feat_scale2_gate, wsi_feat_scale3_gate],
                        wsi_feat_scale_gloabl_list,
                        [self.instance_gate1, self.instance_gate2, self.instance_gate3]
                ):
                #
                bs_at_scale = wsi_feat_at_scale.shape[0]
                wsi_feat_at_scale = wsi_feat_at_scale * wsi_feat_gate_at_scale
                wsi_global_rep_repeat = wsi_feat_gate_at_scale.detach().repeat(bs_at_scale, 1)

                # N * 1
                instance_attention_weight = instance_gate(wsi_feat_at_scale, wsi_global_rep_repeat)
                # 1 * N
                instance_attention_weight = torch.transpose(instance_attention_weight, 1, 0)

                instance_attention_weight = torch.softmax(instance_attention_weight, dim=1)


                # instance aggregate
                wsi_feat_agg = torch.mm(instance_attention_weight, wsi_feat_at_scale)
                attention_weight_out_list.append(instance_attention_weight.detach().clone())

        # second time forward
        if self.use_k_agg:
            if wsi_feat_scale1.shape[0] < self.k_agg:
                pad_size = self.k_agg - wsi_feat_scale1.shape[0]
                zero_size = (pad_size, *wsi_feat_scale1.shape[1:])
                pad_tensor = torch.zeros(zero_size).to(wsi_feat_scale1.device)
                wsi_feat_scale1 = torch.cat([wsi_feat_scale1, pad_tensor])
            if wsi_feat_scale2.shape[0] < self.k_agg:
                pad_size = self.k_agg - wsi_feat_scale2.shape[0]
                zero_size = (pad_size, *wsi_feat_scale2.shape[1:])
                pad_tensor = torch.zeros(zero_size).to(wsi_feat_scale2.device)
                wsi_feat_scale2 = torch.cat([wsi_feat_scale2, pad_tensor])
            if wsi_feat_scale3.shape[0] < self.k_agg:
                pad_size = self.k_agg - wsi_feat_scale3.shape[0]
                zero_size = (pad_size, *wsi_feat_scale3.shape[1:])
                pad_tensor = torch.zeros(zero_size).to(wsi_feat_scale3.device)
                wsi_feat_scale3 = torch.cat([wsi_feat_scale3, pad_tensor])

        """fine-tuning 3 scales"""
        wsi_ft_feat_list = []
        for ft_conv, wsi_feat in zip(
                [self.feature_fine_tuning, self.feature_fine_tuning2, self.feature_fine_tuning3],
                [wsi_feat_scale1, wsi_feat_scale2, wsi_feat_scale3],
        ):
            wsi_ft_feat_list.append(ft_conv(wsi_feat))

        if self.use_k_agg:
            agg_feat_list = []
            for data_feat, score_fc in zip(wsi_ft_feat_list, self.score_fc):
                agg_feat_list.append(self.agg_k_cluster_by_score(data_feat, score_fc))
            wsi_ft_feat_list = agg_feat_list

            wsi_feat_scale_gloabl_list = []
            for data_feat, score_fc in zip(agg_feat_list, self.score_fc):
                feat_score = score_fc(data_feat)
                feat_attention = torch.sigmoid(feat_score)

                attention_weight_out_list.append(feat_attention.detach().clone())
                global_feat = torch.sum(data_feat * feat_attention, dim=0, keepdim=True)
                wsi_feat_scale_gloabl_list.append(global_feat)
        else:
            """global representation of 3 scales"""
            wsi_feat_scale_gloabl_list = []
            for idx, feat in enumerate(wsi_ft_feat_list):
                current_global_feature = torch.mm(attention_weight_out_list[idx], feat)
                wsi_feat_scale_gloabl_list.append(current_global_feature)

        attention_weight_out_list = []
        """mmtm"""
        tab_feat_mmtm, wsi_feat1_gloabl, wsi_feat_scale1_gate, wsi_feat2_gloabl, wsi_feat_scale2_gate, wsi_feat3_gloabl, wsi_feat_scale3_gate = self.mmtm(
            tab_feat, *wsi_feat_scale_gloabl_list)

        """instance selection of 3 scales"""
        wsi_feat_agg_list = []
        for wsi_feat_at_scale, wsi_feat_gate_at_scale, wsi_global_rep, instance_gate in zip(
                wsi_ft_feat_list,
                [wsi_feat_scale1_gate, wsi_feat_scale2_gate, wsi_feat_scale3_gate],
                wsi_feat_scale_gloabl_list,
                [self.instance_gate1, self.instance_gate2, self.instance_gate3]
        ):
            #
            bs_at_scale = wsi_feat_at_scale.shape[0]
            wsi_feat_at_scale = wsi_feat_at_scale * wsi_feat_gate_at_scale
            wsi_global_rep_repeat = wsi_feat_gate_at_scale.detach().repeat(bs_at_scale, 1)

            # N * 1
            instance_attention_weight = instance_gate(wsi_feat_at_scale, wsi_global_rep_repeat)
            # 1 * N
            instance_attention_weight = torch.transpose(instance_attention_weight, 1, 0)

            instance_attention_weight = torch.softmax(instance_attention_weight, dim=1)

            # instance aggregate
            wsi_feat_agg = torch.mm(instance_attention_weight, wsi_feat_at_scale)

            attention_weight_out_list.append(instance_attention_weight.detach().clone())
            wsi_feat_agg_list.append(wsi_feat_agg)

        """tab feat ft"""
        tab_feat_ft = self.table_feature_ft(tab_feat_mmtm)

        final_feat = torch.cat([tab_feat_ft, *wsi_feat_agg_list, wsi_feat1_gloabl, wsi_feat2_gloabl, wsi_feat3_gloabl], dim=1)

        out = self.classifier(final_feat)


        pass
        y = y.view(-1, 1).float()
        loss = F.binary_cross_entropy_with_logits(out, y) + \
               tab_loss_weight * F.binary_cross_entropy_with_logits(tab_logit, y) - \
               self.lambda_sparse * M_loss

        return out, loss, attention_weight_out_list

    def get_params(self, base_lr):
        ret = []

        if self.tabnet is not None:
            tabnet_params = []
            for param in self.tabnet.parameters():
                tabnet_params.append(param)
            ret.append({
                'params': tabnet_params,
                'lr': base_lr
            })

        cls_learning_rate_rate=100
        if self.classifier is not None:
            classifier_params = []
            for param in self.classifier.parameters():
                classifier_params.append(param)
            ret.append({
                'params': classifier_params,
                'lr': base_lr / cls_learning_rate_rate,
            })


        tab_learning_rate_rate = 100
        if self.table_feature_ft is not None:
            misc_params = []
            for param in self.table_feature_ft.parameters():
                misc_params.append(param)
            ret.append({
                'params': misc_params,
                'lr': base_lr / tab_learning_rate_rate,
            })

        mil_learning_rate_rate = 1000
        misc_params = []
        for part in [self.feature_fine_tuning, self.feature_fine_tuning2, self.feature_fine_tuning3,
                     self.instance_gate1, self.instance_gate2, self.instance_gate3,
                     self.wsi_select_gate,
                     self.score_fc]:
            if part is not None:
                for param in part.parameters():
                    misc_params.append(param)
        ret.append({
            'params': misc_params,
            'lr': base_lr / mil_learning_rate_rate,
        })

        misc_learning_rate_rate = 100
        misc_params = []
        for part in [self.mmtm, ]:
            if part is not None:
                for param in part.parameters():
                    misc_params.append(param)
        ret.append({
            'params': misc_params,
            'lr': base_lr / misc_learning_rate_rate,
        })

        return ret



class InstanceAttentionGateAdd(nn.Module):
    def __init__(self, feat_dim):
        super(InstanceAttentionGateAdd, self).__init__()
        self.trans = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LeakyReLU(),
            nn.Linear(feat_dim, 1),
        )

    def forward(self, instance_feature, global_feature):

        feat = instance_feature + global_feature
        attention = self.trans(feat)
        return attention


class MILFusionAdd(nn.Module):
    def __init__(self, img_feat_input_dim=512, tab_feat_input_dim=32,
                 img_feat_rep_layers=4,
                 num_modal=2,
                 use_tabnet=False,
                 tab_indim=0,
                 local_rank=0,
                 cat_idxs=None,
                 cat_dims=None,
                 lambda_sparse=1e-3,
                 fusion='mmtm',
                 use_k_agg=False,
                 k_agg=10,
                 ):
        super(MILFusionAdd, self).__init__()
        self.num_modal = num_modal
        self.local_rank = local_rank
        self.use_tabnet = use_tabnet
        self.tab_indim = tab_indim
        self.lambda_sparse = lambda_sparse
        # define K mean agg
        self.use_k_agg = use_k_agg
        self.k_agg = k_agg

        self.fusion_method = fusion
        if self.use_tabnet:
            self.tabnet = TabNet(input_dim=tab_indim, output_dim=1,
                                 n_d=32, n_a=32, n_steps=5,
                                 gamma=1.5, n_independent=2, n_shared=2,
                                 momentum=0.3,
                                 cat_idxs=cat_idxs, cat_dims=cat_dims)
        else:
            self.tabnet = None

        if self.use_tabnet and num_modal == 1:
            self.only_tabnet = True
        else:
            self.only_tabnet = False

        """
        Control tabnet
        """
        if self.only_tabnet:
            self.feature_fine_tuning = None
        else:
            """pretrained feature fine tune"""
            feature_fine_tuning_layers = []
            for _ in range(img_feat_rep_layers):
                feature_fine_tuning_layers.extend([
                    nn.Linear(img_feat_input_dim, img_feat_input_dim),
                    nn.LeakyReLU(),
                ])
            self.feature_fine_tuning = nn.Sequential(*feature_fine_tuning_layers)

        # 3 为三个图像模态
        if self.num_modal == 4 or self.num_modal == 3:
            self.feature_fine_tuning2 = nn.Sequential(*feature_fine_tuning_layers)
            self.feature_fine_tuning3 = nn.Sequential(*feature_fine_tuning_layers)
        else:
            self.feature_fine_tuning2 = None
            self.feature_fine_tuning3 = None

        if self.only_tabnet or self.num_modal == 3:
            self.table_feature_ft = None
        else:
            """tab feature fine tuning"""
            self.table_feature_ft = nn.Sequential(
                nn.Linear(tab_feat_input_dim, tab_feat_input_dim)
            )

        # k agg score
        self.score_fc = nn.ModuleList()
        if self.use_k_agg:
            for _ in range(self.num_modal - 1):
                self.score_fc.append(
                    nn.Sequential(
                        nn.Linear(img_feat_input_dim, img_feat_input_dim),
                        nn.LeakyReLU(),
                        nn.Linear(img_feat_input_dim, img_feat_input_dim),
                        nn.LeakyReLU(),
                        nn.Linear(img_feat_input_dim, 1),
                        nn.Sigmoid()
                    )
                )


        """modal fusion"""
        self.wsi_select_gate = None
        # define modal fusion related output feature dimension and fusion module
        if self.only_tabnet:
            self.mmtm = None
        elif self.fusion_method == 'concat':
            self.fusion_out_dim = tab_feat_input_dim + img_feat_input_dim
            self.wsi_select_gate = nn.Sequential(
                nn.Linear(img_feat_input_dim, 1),
                nn.Sigmoid()
            )
            self.mmtm = nn.Linear(self.fusion_out_dim, self.fusion_out_dim)
        elif self.fusion_method == 'bilinear':
            self.wsi_select_gate = nn.Sequential(
                nn.Linear(img_feat_input_dim, 1),
                nn.Sigmoid()
            )
            self.fusion_out_dim = tab_feat_input_dim + img_feat_input_dim
            self.mmtm = nn.Bilinear(tab_feat_input_dim, img_feat_input_dim, self.fusion_out_dim)
        elif self.fusion_method == 'add':
            self.wsi_select_gate = nn.Sequential(
                nn.Linear(img_feat_input_dim, 1),
                nn.Sigmoid()
            )
            self.fusion_out_dim = tab_feat_input_dim
            self.mmtm = nn.Linear(img_feat_input_dim * (num_modal - 1), tab_feat_input_dim)
        elif self.fusion_method == 'gate':
            self.wsi_select_gate = nn.Sequential(
                nn.Linear(img_feat_input_dim, 1),
                nn.Sigmoid()
            )
            self.fusion_out_dim = 96
            self.mmtm = BilinearFusion(dim1=tab_feat_input_dim, dim2=img_feat_input_dim, mmhid=self.fusion_out_dim)
        elif self.num_modal == 2 and self.fusion_method == 'mmtm':
            self.fusion_out_dim = (img_feat_input_dim * 2) * (num_modal - 1) + tab_feat_input_dim
            self.mmtm = MMTMBi(dim_tab=tab_feat_input_dim, dim_img=img_feat_input_dim)
        elif self.num_modal == 3 and self.fusion_method == 'mmtm':
            self.fusion_out_dim = (img_feat_input_dim * 2) * 3
            self.mmtm = MMTMTri(dim_img=img_feat_input_dim)
        elif self.num_modal == 4 and self.fusion_method == 'mmtm':
            self.fusion_out_dim = (img_feat_input_dim * 2) * (num_modal - 1) + tab_feat_input_dim
            self.mmtm = MMTMQuad(dim_tab=tab_feat_input_dim, dim_img=img_feat_input_dim)
        else:
            raise NotImplementedError(f'num_modal {num_modal} not implemented')

        """instance selection"""
        if self.only_tabnet or self.fusion_method in ['concat', 'add', 'bilinear', 'gate']:
            self.instance_gate1 = None
        else:
            self.instance_gate1 = InstanceAttentionGateAdd(img_feat_input_dim)

        if (self.num_modal == 4 or self.num_modal == 3)and self.fusion_method == 'mmtm':
            self.instance_gate2 = InstanceAttentionGateAdd(img_feat_input_dim)
            self.instance_gate3 = InstanceAttentionGateAdd(img_feat_input_dim)
        else:
            self.instance_gate2 = None
            self.instance_gate3 = None

        """classifier layer"""
        if self.only_tabnet:
            self.classifier = None
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.fusion_out_dim, self.fusion_out_dim),
                nn.Dropout(0.5),
                nn.Linear(self.fusion_out_dim, 1)
            )

    def agg_k_cluster_by_score(self, data: torch.Tensor, score_fc: nn.Module):
        num_elements = data.shape[0]
        score = score_fc(data)

        """
        >>> score = torch.rand(4,1)
        >>> top_score, top_idx = torch.topk(score, k=num_elements, dim=0)
        >>> top_score, top_idx
        (tensor([[0.3963],
         [0.0856],
         [0.0704],
         [0.0247]]),
         tensor([[1],
                 [0],
                 [3],
                 [2]]))
        """
        top_score, top_idx = torch.topk(score, k=num_elements, dim=0)
        """
        >>> data
        tensor([[0.0672, 0.9001, 0.5660, 0.0522, 0.1543],
        [0.1965, 0.7711, 0.9737, 0.5269, 0.9255],
        [0.6761, 0.5801, 0.4687, 0.1683, 0.8136],
        [0.2091, 0.9620, 0.8105, 0.8210, 0.3391]])
        >>> top_idx[:, 0]
        tensor([1, 0, 3, 2])
        >>> data_sorted
        tensor([[0.1965, 0.7711, 0.9737, 0.5269, 0.9255],
        [0.0672, 0.9001, 0.5660, 0.0522, 0.1543],
        [0.2091, 0.9620, 0.8105, 0.8210, 0.3391],
        [0.6761, 0.5801, 0.4687, 0.1683, 0.8136]])
        """
        data_sorted = torch.zeros_like(data)
        data_sorted.index_copy_(dim=0, index=top_idx[:, 0], source=data)

        # Batch set as feature dim
        data_sorted = torch.transpose(data_sorted, 1, 0)
        data_sorted = data_sorted.unsqueeze(1)

        agg_result = nn.functional.adaptive_max_pool1d(data_sorted, self.k_agg)

        agg_result = agg_result.squeeze(1)
        agg_result = torch.transpose(agg_result, 1, 0)
        return agg_result

    def forward(self, data):

        attention_weight_out_list = []
        if self.use_tabnet:
            tab_data = data['tab_data'].cuda(self.local_rank)
            if self.only_tabnet:
                tab_logit, M_loss = self.tabnet(tab_data)
            else:
                tab_logit, tab_feat, M_loss = self.tabnet(tab_data)

            tab_loss_weight = 1.
        else:
            tab_feat = data['tab_feat'].cuda(self.local_rank)
            tab_logit = torch.zeros((1, 1)).cuda(self.local_rank)
            M_loss = 0.
            tab_loss_weight = 0.

        y = data['label'].cuda(self.local_rank)
        wsi_feat_scale1 = data['wsi_feat_scale1'].cuda(self.local_rank)
        if len(wsi_feat_scale1.size()) == 3:
            # 1 #instance #feat
            wsi_feat_scale1 = wsi_feat_scale1.squeeze(0)
        scale1_bs = wsi_feat_scale1.shape[0]

        if True:
            """
            Fusion of 4 modalities
            """
            wsi_feat_scale2 = data['wsi_feat_scale2'].cuda(self.local_rank)
            wsi_feat_scale3 = data['wsi_feat_scale3'].cuda(self.local_rank)
            if len(wsi_feat_scale2.size()) == 3:
                # 1 #instance #feat
                wsi_feat_scale2 = wsi_feat_scale2.squeeze(0)
            if len(wsi_feat_scale3.size()) == 3:
                # 1 #instance #feat
                wsi_feat_scale3 = wsi_feat_scale3.squeeze(0)

            if self.use_k_agg:
                if wsi_feat_scale1.shape[0] < self.k_agg:
                    pad_size = self.k_agg - wsi_feat_scale1.shape[0]
                    zero_size = (pad_size, *wsi_feat_scale1.shape[1:])
                    pad_tensor = torch.zeros(zero_size).to(wsi_feat_scale1.device)
                    wsi_feat_scale1 = torch.cat([wsi_feat_scale1, pad_tensor])
                if wsi_feat_scale2.shape[0] < self.k_agg:
                    pad_size = self.k_agg - wsi_feat_scale2.shape[0]
                    zero_size = (pad_size, *wsi_feat_scale2.shape[1:])
                    pad_tensor = torch.zeros(zero_size).to(wsi_feat_scale2.device)
                    wsi_feat_scale2 = torch.cat([wsi_feat_scale2, pad_tensor])
                if wsi_feat_scale3.shape[0] < self.k_agg:
                    pad_size = self.k_agg - wsi_feat_scale3.shape[0]
                    zero_size = (pad_size, *wsi_feat_scale3.shape[1:])
                    pad_tensor = torch.zeros(zero_size).to(wsi_feat_scale3.device)
                    wsi_feat_scale3 = torch.cat([wsi_feat_scale3, pad_tensor])

            """fine-tuning 3 scales features"""
            wsi_ft_feat_list = []
            for ft_conv, wsi_feat in zip(
                    [self.feature_fine_tuning, self.feature_fine_tuning2, self.feature_fine_tuning3],
                    [wsi_feat_scale1, wsi_feat_scale2, wsi_feat_scale3],
                ):
                wsi_ft_feat_list.append(ft_conv(wsi_feat))

            if self.use_k_agg:
                agg_feat_list = []
                for data_feat, score_fc in zip(wsi_ft_feat_list, self.score_fc):
                    agg_feat_list.append(self.agg_k_cluster_by_score(data_feat, score_fc))
                wsi_ft_feat_list = agg_feat_list

                wsi_feat_scale_gloabl_list = []
                for data_feat, score_fc in zip(agg_feat_list, self.score_fc):
                    feat_score = score_fc(data_feat)
                    # feat_attention = torch.softmax(feat_score, dim=0)
                    feat_attention = torch.sigmoid(feat_score)

                    attention_weight_out_list.append(feat_attention.detach().clone())
                    global_feat = torch.sum(data_feat * feat_attention, dim=0, keepdim=True)
                    wsi_feat_scale_gloabl_list.append(global_feat)
            else:
                """global representation of 3 scales"""
                wsi_feat_scale_gloabl_list = []
                for feat in wsi_ft_feat_list:
                    wsi_feat_scale_gloabl_list.append(torch.mean(feat, dim=0, keepdim=True))


            """mmtm"""
            tab_feat_mmtm, wsi_feat1_gloabl, wsi_feat_scale1_gate, wsi_feat2_gloabl, wsi_feat_scale2_gate, wsi_feat3_gloabl, wsi_feat_scale3_gate = self.mmtm(tab_feat, *wsi_feat_scale_gloabl_list)

            """instance selection of 3 scales"""
            wsi_feat_agg_list = []
            for wsi_feat_at_scale, wsi_feat_gate_at_scale, wsi_global_rep, instance_gate in zip(
                        wsi_ft_feat_list,
                        [wsi_feat_scale1_gate, wsi_feat_scale2_gate, wsi_feat_scale3_gate],
                        wsi_feat_scale_gloabl_list,
                        [self.instance_gate1, self.instance_gate2, self.instance_gate3]
                ):
                #
                bs_at_scale = wsi_feat_at_scale.shape[0]
                wsi_feat_at_scale = wsi_feat_at_scale * wsi_feat_gate_at_scale
                wsi_global_rep_repeat = wsi_feat_gate_at_scale.detach().repeat(bs_at_scale, 1)

                # N * 1
                instance_attention_weight = instance_gate(wsi_feat_at_scale, wsi_global_rep_repeat)
                # 1 * N
                instance_attention_weight = torch.transpose(instance_attention_weight, 1, 0)

                instance_attention_weight = torch.softmax(instance_attention_weight, dim=1)


                # instance aggregate
                wsi_feat_agg = torch.mm(instance_attention_weight, wsi_feat_at_scale)
                attention_weight_out_list.append(instance_attention_weight.detach().clone())
                wsi_feat_agg_list.append(wsi_feat_agg)

            """tab feat ft"""
            tab_feat_ft = self.table_feature_ft(tab_feat_mmtm)

            final_feat = torch.cat([tab_feat_ft, *wsi_feat_agg_list, wsi_feat1_gloabl, wsi_feat2_gloabl, wsi_feat3_gloabl], dim=1)

            out = self.classifier(final_feat)


            pass
        y = y.view(-1, 1).float()
        loss = F.binary_cross_entropy_with_logits(out, y) + \
               tab_loss_weight * F.binary_cross_entropy_with_logits(tab_logit, y) - \
               self.lambda_sparse * M_loss

        return out, loss, attention_weight_out_list

    def get_params(self, base_lr):
        ret = []

        if self.tabnet is not None:
            tabnet_params = []
            for param in self.tabnet.parameters():
                tabnet_params.append(param)
            ret.append({
                'params': tabnet_params,
                'lr': base_lr
            })

        cls_learning_rate_rate=100
        if self.classifier is not None:
            classifier_params = []
            for param in self.classifier.parameters():
                classifier_params.append(param)
            ret.append({
                'params': classifier_params,
                'lr': base_lr / cls_learning_rate_rate,
            })

        tab_learning_rate_rate = 100
        if self.table_feature_ft is not None:
            misc_params = []
            for param in self.table_feature_ft.parameters():
                misc_params.append(param)
            ret.append({
                'params': misc_params,
                'lr': base_lr / tab_learning_rate_rate,
            })

        mil_learning_rate_rate = 1000
        misc_params = []
        for part in [self.feature_fine_tuning, self.feature_fine_tuning2, self.feature_fine_tuning3,
                     self.instance_gate1, self.instance_gate2, self.instance_gate3,
                     self.wsi_select_gate,
                     self.score_fc]:
            if part is not None:
                for param in part.parameters():
                    misc_params.append(param)
        ret.append({
            'params': misc_params,
            'lr': base_lr / mil_learning_rate_rate,
        })

        misc_learning_rate_rate = 100
        misc_params = []
        for part in [self.mmtm, ]:
            if part is not None:
                for param in part.parameters():
                    misc_params.append(param)
        ret.append({
            'params': misc_params,
            'lr': base_lr / misc_learning_rate_rate,
        })

        return ret








