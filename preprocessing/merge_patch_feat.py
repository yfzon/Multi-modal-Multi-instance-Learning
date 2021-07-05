# -*- coding: utf-8 -*-


import os
import os.path as osp
import pickle
from glob import glob
from multiprocessing import Pool

from rich import print
from rich.progress import track

"""
extract_patch_with_tta.py Multiple features will be extracted for each Patch via TTA
- WSI id1
    - patch1 feat.pkl
    - patch2 feat.pkl
- WSI id2
    - patch1 feat.pkl
    - patch2 feat.pkl
This file combines the features extracted by offline within the same WSI into a single file
- WSI id1.pkl
- WSI id2.pkl
"""

feat_save_dirx10 = 'path_to_extracted_features_at_x5'
merge_feat_save_dirx10 = 'path_to_combined_features_at_x5'

feat_save_dirx20 = 'path_to_extracted_features_at_x20'
merge_feat_save_dirx20 = 'path_to_combined_features_at_x20'

feat_save_dirx5 = 'path_to_extracted_features_at_x5'
merge_feat_save_dirx5 = 'path_to_combined_features_at_x5'


def merge_wsi_feat(wsi_feat_dir) -> None:
    """
    Args:
        wsi_feat_dir:

    Returns:

    """


    files = glob(osp.join(wsi_feat_dir, '*.pkl'))

    save_obj = []
    for fp in files:
        #
        try:
            with open(fp, 'rb') as infile:
                obj = pickle.load(infile)

            # add patch file name
            obj['feat_name'] = osp.basename(fp).rsplit('.', 1)[0]
            save_obj.append(obj)
        except Exception as e:
            print(f'Error in {fp} as {e}')
            continue

    bname = osp.basename(wsi_feat_dir).lower()  # wsi id
    save_fp = osp.join(merge_feat_save_dir, f'{bname}.pkl')
    with open(save_fp, 'wb') as outfile:
        pickle.dump(save_obj, outfile)


if __name__ == '__main__':
    for feat_save_dir, merge_feat_save_dir in zip(
            [feat_save_dirx20, feat_save_dirx10, feat_save_dirx5, ],
            [merge_feat_save_dirx20, merge_feat_save_dirx10, merge_feat_save_dirx5]
    ):
        print(f'Save to {merge_feat_save_dir}')
        os.makedirs(merge_feat_save_dir, exist_ok=True)
        wsi_dirs = glob(osp.join(feat_save_dir, '*'))

        with Pool(160) as p:
            for _ in track(p.imap_unordered(merge_wsi_feat, wsi_dirs), total=len(wsi_dirs)):
                pass
