import init_path
import os
import os.path as osp
from lib.dataset import voc12_sgan


def get_dataset(dataset_name, args):
    if dataset_name == "voc12_sgan":
        train_dataset = voc12_sgan.VOC12ClsSalDataset(voc12_root=args.dataset_root,
                                                    cue_file=args.cue_file,
                                                    max_iters=args.max_iter * args.batch_size,
                                                    new_size=args.resize,
                                                    sal_subdir=args.sal_subdir,
                                                    sal_thresh=args.sal_thresh,
                                                    seed_subdir=args.seed_subdir,
                                                    mean=args.mean,
                                                    std=args.std)
        val_dataset = voc12_sgan.VOC12ClsSalDataset(voc12_root=args.dataset_root,
                                                    cue_file=args.cue_file,
                                                    max_iters=100,
                                                    new_size=args.resize,
                                                    sal_subdir=args.sal_subdir,
                                                    sal_thresh=args.sal_thresh,
                                                    seed_subdir=args.seed_subdir,
                                                    mean=args.mean,
                                                    std=args.std,
                                                    mirror=False)
    else:
        raise KeyError

    return train_dataset, val_dataset
    