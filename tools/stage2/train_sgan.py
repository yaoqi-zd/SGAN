import os, time, random
import os.path as osp
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import init_path
# from lib.dataset import voc12_sgan
from lib.dataset.get_dataset import get_dataset
from lib.utils import pyutils
from lib.network import sgan
import argparse
from ipdb import set_trace
from torchsummary import summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", default=None, type=str)
    args = parser.parse_args()
    args = pyutils.read_yaml2cls(args.cfg_file)
    
    return args

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** power)


def adjust_learning_rate(optimizer, iter, args):
    """adjust the learning rate by poly decay"""
    lr = lr_poly(base_lr=args.lr, iter=iter, max_iter=args.max_iter, power=args.power)
    optimizer.param_groups[0]["lr"] = lr
    optimizer.param_groups[1]["lr"] = lr * 2
    optimizer.param_groups[2]["lr"] = lr * 10
    optimizer.param_groups[3]["lr"] = lr * 20


def filter_pretrained_weights(weight_dict, model_dict):
    # filter the parameters that exist in the pretrained model
    pretrained_dict = {k:v for k, v in weight_dict.items() if k in model_dict}
    for k in sorted(pretrained_dict.keys()):
        print("loading pretrained weights: {}".format(k))
    model_dict.update(pretrained_dict)
    return model_dict


if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda:0")
    cur_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    pyutils.Logger(osp.join("log", cur_time + ".log"))
    for key in sorted(args.keys()):
        print("{:<25s}:{}".format(key, args[key]))

    # set randomness
    np.random.seed(7) # numpy
    torch.manual_seed(7) # cpu
    random.seed(7) # python
    torch.cuda.manual_seed_all(7) # gpu
    torch.backends.cudnn.deterministic = True # cudnn
    
    # dataset
    train_dataset, _ = get_dataset(args.dataset_name, args)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                   shuffle=True, pin_memory=True, num_workers=args.num_worker, drop_last=True)

    # set network
    model = sgan.SGAN(backbone_name=args.backbone)

    # initialize network
    model.initialize(args.pretrain)

    # set optimizer
    param_groups = model.get_param_groups()
    optimizer = optim.SGD([
        {"params": param_groups[0], "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": param_groups[1], "lr": 2 * args.lr, "weight_decay": 0},
        {"params": param_groups[2], "lr": 10 * args.lr, "weight_decay": args.weight_decay},
        {"params": param_groups[3], "lr": 20 * args.lr, "weight_decay": 0}
    ], lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)

    model = torch.nn.DataParallel(model).cuda()
    model.train()
    
    checkpoint_path = osp.join(args.save_model_path, args.cfg_name)
    if not osp.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # training loop
    criterion = torch.nn.CrossEntropyLoss(ignore_index=20)
    avg_loss = {'ce': 0, "seed":0} if args.use_sal else {'ce': 0}
    for num, pack in enumerate(train_data_loader):
        adjust_learning_rate(optimizer, num, args)

        img, label = pack[1].to(device, dtype=torch.float32), pack[2].to(device, dtype=torch.float32)

        if args.use_sal:
            fg_sim = pack[3].to(device, dtype=torch.float32)
            bg_sim = pack[4].to(device, dtype=torch.float32)
            seed = pack[5].to(device, dtype=torch.long)
            out = model(img, fg_sim, bg_sim)

            cls_loss = F.multilabel_soft_margin_loss(out[0], label)
            seed_loss = criterion(out[1], seed)

            loss = cls_loss + args.seed_loss_ratio * seed_loss
            avg_loss['ce'] += cls_loss.item()
            avg_loss['seed'] += seed_loss.item()
        else:
            out = model(img)
            loss = F.multilabel_soft_margin_loss(out[0], label)
            avg_loss['ce'] += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (num + 1) % args.display_step == 0:
            loss_str = ["%s:%.4f" % (loss_name, avg_loss[loss_name] / args.display_step) for loss_name in avg_loss.keys()]
            print("Iter:[%4d/%4d]" % (num + 1, args.max_iter),
                  *loss_str,
                  "lr:%.6f" % optimizer.param_groups[0]["lr"],
                  "time:%s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), flush=True)

            for key in avg_loss.keys():
                avg_loss[key] = 0

        if (num + 1) % 2000 == 0:
            torch.save(model.module.state_dict(), osp.join(checkpoint_path, "model_iter_" + str(num + 1) + ".pth"))

        if (num + 1) >= args.max_iter:
            break
