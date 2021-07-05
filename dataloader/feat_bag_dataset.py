import os.path as osp
import pickle
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
from rich import print


class FeatBagDataset(data_utils.Dataset):
    def __init__(self, bag_feat_root: str, bag_feat_aug_root: str, df: pd.DataFrame, cfg, shuffle_bag=False,
                 is_train=False, local_rank=0) -> None:
        self.pids = df['pid'].values.tolist()
        self.bag_feat_root = bag_feat_root
        self.bag_feat_aug_root = bag_feat_aug_root
        self.is_train = is_train

        miss_cnt = 0

        targets = df['target'].values.tolist()

        exist_targets = []
        exist_pids = []

        for idx, pid in enumerate(self.pids):
            bag_fp = osp.join(self.bag_feat_aug_root, f'{pid}.pkl')
            if osp.exists(bag_fp):  # and len(os.listdir(bag_fp)) > 0:
                exist_pids.append(pid)
                exist_targets.append(targets[idx])
            else:
                miss_cnt += 1

        if cfg.local_rank == 0:
            print(f'Total : {len(self.pids)}, found {len(exist_pids)}, miss {miss_cnt}')
        self.pids = exist_pids
        self.targets = exist_targets
        self.shuffle_bag = shuffle_bag
        self.if_shuffled = False
        self.local_rank = local_rank

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx) -> Dict:
        if (not self.if_shuffled) and self.is_train:
            np.random.seed(self.local_rank)
            p = np.random.permutation(len(self.pids))
            self.pids = np.array(self.pids)[p]
            self.targets = np.array(self.targets)[p]
            self.if_shuffled = True

        c_pid = self.pids[idx]
        label = self.targets[idx]

        bag_fp = osp.join(self.bag_feat_aug_root, f'{c_pid}.pkl')
        pid_dir = osp.join(self.bag_feat_aug_root, c_pid)

        with open(bag_fp, 'rb') as infile:
            bag_feat_list_obj = pickle.load(infile)

        bag_feat = []
        for aug_feat_dict in bag_feat_list_obj:
            if self.is_train:
                aug_feat = aug_feat_dict['tr']
                aug_feat = np.vstack([aug_feat, np.expand_dims(aug_feat_dict['val'], 0)])
                random_row = np.random.randint(0, aug_feat.shape[0])
                choice_feat = aug_feat[random_row]
                bag_feat.append(choice_feat)
            else:
                aug_feat = aug_feat_dict['val']
                bag_feat.append(aug_feat)

        bag_feat = np.vstack(bag_feat)

        if self.is_train and bag_feat.shape[0] <= 1:
            # print(f'Only one instance in {bag_fp}')
            rand_choose_idx = np.random.randint(0, len(self))
            return self[rand_choose_idx]

        if self.shuffle_bag:
            instance_size = bag_feat.shape[0]
            shuffle_idx = np.random.permutation(instance_size)
            bag_feat = bag_feat[shuffle_idx]

            num_of_drop_columns = np.random.randint(0, 10)
            for _ in range(num_of_drop_columns):
                random_drop_column = np.random.randint(0, bag_feat.shape[1])
                bag_feat[:, random_drop_column] = 0

            if np.random.rand() < 0.3:
                noise = np.random.normal(loc=0, scale=0.05, size=bag_feat.shape)
                bag_feat += noise

        return {
            'data': torch.from_numpy(bag_feat).float(),
            'name': c_pid,
            'label': torch.tensor(label).float(),
            'pid': c_pid
        }


class FeatBagTabFeatDataset(data_utils.Dataset):
    """
    同时加载WSI的特征和Tabnetj计算后的特征
    """

    def __init__(self,
                 bag_feat_root: str, bag_feat_aug_root: str,
                 df: pd.DataFrame, cfg, shuffle_bag=False,
                 is_train=False, local_rank=0) -> None:
        """

        Parameters
        ----------
        bag_feat_root
        bag_feat_aug_root: 整合后的离线特征路径
        df:
        cfg:
        shuffle_bag:
        is_train:
        local_rank:
        """
        self.pids = df['pid'].values.tolist()
        self.bag_feat_root = bag_feat_root
        self.bag_feat_aug_root = bag_feat_aug_root
        self.is_train = is_train

        self.tab_feat_df = pd.read_csv(cfg.dataset.tab_feat_df_fp)
        self.tab_feat_cols = [x for x in self.tab_feat_df.columns if x.startswith('feat')]

        miss_cnt = 0

        targets = df['target'].values.tolist()

        exist_targets = []
        exist_pids = []

        for idx, pid in enumerate(self.pids):
            bag_fp = osp.join(self.bag_feat_aug_root, f'{pid}.pkl')
            if osp.exists(bag_fp):  # and len(os.listdir(bag_fp)) > 0:
                exist_pids.append(pid)
                exist_targets.append(targets[idx])
            else:
                miss_cnt += 1

        if cfg.local_rank == 0:
            print(f'Tab feat : {len(self.tab_feat_cols)}')
            print(f'Total : {len(self.pids)}, found {len(exist_pids)}, miss {miss_cnt}')
        self.pids = exist_pids
        self.targets = exist_targets
        self.shuffle_bag = shuffle_bag
        self.if_shuffled = False
        self.local_rank = local_rank

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx) -> Dict:
        if (not self.if_shuffled) and self.is_train:
            np.random.seed(self.local_rank)
            p = np.random.permutation(len(self.pids))
            self.pids = np.array(self.pids)[p]
            self.targets = np.array(self.targets)[p]
            self.if_shuffled = True

        c_pid = self.pids[idx]
        label = self.targets[idx]

        bag_fp = osp.join(self.bag_feat_aug_root, f'{c_pid}.pkl')

        with open(bag_fp, 'rb') as infile:
            bag_feat_list_obj = pickle.load(infile)

        bag_feat = []
        for aug_feat_dict in bag_feat_list_obj:
            if self.is_train:
                aug_feat = aug_feat_dict['tr']
                aug_feat = np.vstack([aug_feat, np.expand_dims(aug_feat_dict['val'], 0)])
                random_row = np.random.randint(0, aug_feat.shape[0])
                choice_feat = aug_feat[random_row]
                bag_feat.append(choice_feat)
            else:
                aug_feat = aug_feat_dict['val']
                bag_feat.append(aug_feat)

        bag_feat = np.vstack(bag_feat)

        if self.is_train and bag_feat.shape[0] <= 1:

            rand_choose_idx = np.random.randint(0, len(self))
            return self[rand_choose_idx]

        if self.shuffle_bag:
            instance_size = bag_feat.shape[0]
            shuffle_idx = np.random.permutation(instance_size)
            bag_feat = bag_feat[shuffle_idx]

            num_of_drop_columns = np.random.randint(0, 10)
            for _ in range(num_of_drop_columns):
                random_drop_column = np.random.randint(0, bag_feat.shape[1])
                bag_feat[:, random_drop_column] = 0

            if np.random.rand() < 0.3:
                noise = np.random.normal(loc=0, scale=0.05, size=bag_feat.shape)
                bag_feat += noise

        tab_feat = self.tab_feat_df[self.tab_feat_df.pid == c_pid][self.tab_feat_cols].values[0]

        return {
            'data': torch.from_numpy(bag_feat).float(),
            'name': c_pid,
            'tab_feat': torch.from_numpy(tab_feat).float(),
            'label': torch.tensor(label).float(),
            'pid': c_pid
        }



class ModalFusionDataset(data_utils.Dataset):
    """"
    Load multi-modal data
    """
    def __init__(self,
                 cli_feat: pd.DataFrame,
                 cli_data: pd.DataFrame,
                 scale1_feat_root, scale2_feat_root: None, scale3_feat_root: None,
                 select_scale: int,
                 cfg: Any,
                 shuffle_bag=False, is_train=False):
        """

        Args:
            cli_feat:
            cli_data:
            scale1_feat_root:
            scale2_feat_root:
            scale3_feat_root:
            select_scale:
            cfg:
            shuffle_bag:
            is_train:
        """
        super(ModalFusionDataset, self).__init__()

        self.pids = cli_feat.pid.values.tolist()
        self.targets = cli_feat.target.values.tolist()

        self.cli_feat_df = cli_feat
        self.cli_feat_cols = [x for x in cli_feat.columns if x.startswith('feat')]

        self.cli_data_df = cli_data
        self.cli_data_cols = [x for x in cli_data.columns if x not in ['pid', 'split', 'target']]

        self.scale1_feat_root = scale1_feat_root
        self.scale2_feat_root = scale2_feat_root
        self.scale3_feat_root = scale3_feat_root
        self.select_scale = select_scale

        self.cfg = cfg
        self.shuffle_bag = shuffle_bag
        self.is_train = is_train

        exist_targets = []
        exist_pids = []
        miss_cnt = 0
        for idx, pid in enumerate(self.pids):
            bag_fp = osp.join(self.scale1_feat_root, f'{pid}.pkl')
            if osp.exists(bag_fp):
                exist_pids.append(pid)
                exist_targets.append(self.targets[idx])
            else:
                miss_cnt += 1

        if cfg.local_rank == 0:
            print(f'Tab feat : {len(self.cli_feat_cols)}')
            print(f'Total : {len(self.pids)}, found {len(exist_pids)}, miss {miss_cnt}')

        self.pids = exist_pids
        self.targets = exist_targets

    @property
    def tab_data_shape(self):
        return len(self.cli_data_cols)

    def __len__(self):
        return len(self.pids)

    def load_feat_and_aug(self, bag_fp) -> np.ndarray:
        """
        Load WSI feature bag
        Args:
        bag_fp:

        Returns:
        """
        with open(bag_fp, 'rb') as infile:
            bag_feat_list_obj = pickle.load(infile)

        bag_feat = []
        feat_names = []
        for aug_feat_dict in bag_feat_list_obj:
            if self.is_train:
                aug_feat = aug_feat_dict['tr']
                aug_feat = np.vstack([aug_feat, np.expand_dims(aug_feat_dict['val'], 0)])
                random_row = np.random.randint(0, aug_feat.shape[0])
                choice_feat = aug_feat[random_row]
                bag_feat.append(choice_feat)
            else:
                aug_feat = aug_feat_dict['val']
                bag_feat.append(aug_feat)

            feat_names.append(aug_feat_dict['feat_name'])

        del bag_feat_list_obj
        bag_feat = np.vstack(bag_feat)



        if self.is_train:
            if np.random.rand() < 0.5:
                num_of_drop_columns = np.random.randint(0, 100)
                for _ in range(num_of_drop_columns):
                    random_drop_column = np.random.randint(0, bag_feat.shape[1])
                    bag_feat[:, random_drop_column] = 0
            if np.random.rand() < 0.5:
                noise = np.random.normal(loc=0, scale=0.01, size=bag_feat.shape)
                bag_feat += noise
        if self.shuffle_bag:
            instance_size = bag_feat.shape[0]
            shuffle_idx = np.random.permutation(instance_size)
            bag_feat = bag_feat[shuffle_idx]

        return bag_feat, feat_names

    def __getitem__(self, idx) -> Dict:
        c_pid = self.pids[idx]
        label = self.targets[idx]
        ret = {}

        if self.select_scale == 0:
            for idx, feat_root in enumerate([self.scale1_feat_root, self.scale2_feat_root, self.scale3_feat_root]):
                bag_fp = osp.join(feat_root, f'{c_pid}.pkl')
                if osp.exists(bag_fp):
                    bag_feat, feat_name = self.load_feat_and_aug(bag_fp)
                else:
                    bag_feat = np.zeros((1, 1280))
                    feat_name = []
                k = f'wsi_feat_scale{idx+1}'
                ret[k] = torch.from_numpy(bag_feat).float()
                ret[k+'_feat_name'] = feat_name
        else:
            if self.select_scale == 1:
                feat_root = self.scale1_feat_root
            elif self.select_scale == 2:
                feat_root = self.scale2_feat_root
            elif self.select_scale == 3:
                feat_root = self.scale3_feat_root

            bag_fp = osp.join(feat_root, f'{c_pid}.pkl')
            if osp.exists(bag_fp):
                bag_feat, feat_name = self.load_feat_and_aug(bag_fp)
            else:
                bag_feat = np.zeros((1, 1280))
                feat_name = []
            ret['wsi_feat_scale1'] = torch.from_numpy(bag_feat).float()
            ret['wsi_feat_scale1_feat_name'] = feat_name

        tab_feat = self.cli_feat_df[self.cli_feat_df.pid == c_pid][self.cli_feat_cols].values[0]
        tab_data = self.cli_data_df[self.cli_data_df.pid == c_pid][self.cli_data_cols].values[0]
        ret['name'] = c_pid
        ret['tab_feat'] = torch.from_numpy(tab_feat).float()
        ret['tab_data'] = torch.from_numpy(tab_data).float()
        ret['label'] = torch.tensor(label).float()
        ret['pid'] = c_pid
        return ret


def mixup_data(x, alpha=1.0, use_cuda=False):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.shape[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x

class EMPOBJ:
    def __init__(self):
        self.local_rank = 0

if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('/path/to/your/table')
    bag_feat_root = "path/to/your/bag"
    cfg = EMPOBJ()
    from rich.progress import track
    cfg.local_rank = 0
    ds = ModalFusionDataset(
        cli_feat=df,
        scale1_feat_root='path/to/your/scale1/features',
        scale2_feat_root='path/to/your/scale2/features',
        scale3_feat_root='path/to/your/scale3/features',
        select_scale=0,
        cfg=cfg,
        shuffle_bag=True,
        is_train=True
    )
    local_rank = 0
    dl = data_utils.DataLoader(ds, num_workers=4)
    for data in track(dl):
        tab_feat = data['tab_feat'].cuda(local_rank)
        wsi_feat_scale1 = data['wsi_feat_scale1'].cuda(local_rank)
