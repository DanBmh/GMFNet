import sys
from utils import h36motion3d as datasets
from model import AttModel
from utils.opt import Options
from utils import util
from utils import log

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import torch.optim as optim
import tqdm

# ==================================================================================================

sys.path.append("/PoseForecasters/")
import utils_pipeline

datamode = "gt-gt"
# datamode = "pred-pred"

sconfig = {
    "item_step": 2,
    "window_step": 2,
    # "item_step": 1,
    # "window_step": 1,
    "select_joints": [
        "hip_right",
        "hip_left",
        "knee_right",
        "knee_left",
        "ankle_right",
        "ankle_left",
        "nose",
        "shoulder_right",
        "shoulder_left",
        "elbow_right",
        "elbow_left",
        "wrist_right",
        "wrist_left",
    ],
    "input_n": 50,
    "output_n": 25,
}

datasets_train = [
    "/datasets/preprocessed/human36m/train_forecast_rpt.json",
]

dataset_eval_test = "/datasets/preprocessed/human36m/{}_forecast_rpt.json"

njoints = len(sconfig["select_joints"])

# ==================================================================================================


def main(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    print('>>> create models')
    in_features = opt.in_features
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    dev = opt.dev
    net_pred = AttModel.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n)
    print(torch.cuda.is_available())
    net_pred.to(dev)

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))
    print('>>> loading datasets')

    # Load preprocessed datasets
    print("Loading datasets ...")
    dataset_train, dlen_train = [], 0
    for dp in datasets_train:
        ds, dlen = utils_pipeline.load_dataset(dp, "train", sconfig)
        dataset_train.extend(ds["sequences"])
        dlen_train += dlen
    dataset_eval, dlen_eval = utils_pipeline.load_dataset(
        dataset_eval_test.format("eval"), "eval", sconfig
    )
    dataset_eval = dataset_eval["sequences"]
    data_loader = (dataset_train, dlen_train, opt.batch_size)
    valid_loader = (dataset_eval, dlen_eval, opt.batch_size)

    # training
    if not opt.is_eval:
        err_best = 1000
        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            # if epo % opt.lr_decay == 0:
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))
            print('>>> training epoch: {:d}'.format(epo))
            ret_train = run_model(net_pred, optimizer, is_train=0, data_loader=data_loader, epo=epo, opt=opt)
            print('train error: {:.3f}'.format(ret_train['m_p3d_h36']))
            ret_valid = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt, epo=epo)
            print('validation error: {:.3f}'.format(ret_valid['m_p3d_h36']))
            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            for k in ret_valid.keys():
                ret_log = np.append(ret_log, [ret_valid[k]])
                head = np.append(head, ['valid_' + k])
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
            if ret_valid['m_p3d_h36'] < err_best:
                err_best = ret_valid['m_p3d_h36']
                is_best = True
            log.save_ckpt({'epoch': epo,
                           'lr': lr_now,
                           'err': ret_valid['m_p3d_h36'],
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          is_best=is_best, opt=opt)


def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None):

    dev = opt.dev

    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()

    l_p3d = 0
    if is_train <= 1:
        m_p3d_h36 = 0
    else:
        titles = np.array(range(opt.output_n)) + 1
        m_p3d_h36 = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    seq_in = opt.kernel_size
    itera = 1

    label_gen = utils_pipeline.create_labels_generator(data_loader[0], sconfig)

    nbatch = data_loader[2]
    for batch in tqdm.tqdm(
        utils_pipeline.batch_iterate(label_gen, batch_size=nbatch),
        total=int(data_loader[1] / nbatch),
    ):
        batch_size = nbatch
        # when only one sample in this batch
        if batch_size == 1 and is_train == 0:
            continue

        n += batch_size

        sequences_train = utils_pipeline.make_input_sequence(
            batch, "input", datamode
        )
        sequences_gt = utils_pipeline.make_input_sequence(batch, "target", datamode)

        # Convert to millimeters
        sequences_train = sequences_train * 1000
        sequences_gt = sequences_gt * 1000

        # Add empty joints to pad the number of joints to the original number of joints
        s1 = sequences_train.shape
        s2 = sequences_gt.shape
        s3 = len(sconfig["select_joints"])
        sequences_train = np.concatenate(
            (sequences_train, np.zeros((s1[0], s1[1], 22 - s3, 3))), axis=2
        )
        sequences_gt = np.concatenate(
            (sequences_gt, np.zeros((s2[0], s2[1], 22 - s3, 3))), axis=2
        )

        # Merge joints and coordinates to a single dimension
        sequences_train = sequences_train.reshape(
            [nbatch, sequences_train.shape[1], -1]
        )
        sequences_gt = sequences_gt.reshape([nbatch, sequences_gt.shape[1], -1])

        sequences_train = torch.from_numpy(sequences_train.astype(np.float32)).to("cuda")
        sequences_gt = torch.from_numpy(sequences_gt.astype(np.float32)).to("cuda")

        seq_all = torch.cat([sequences_train, sequences_gt], dim=1)

        p3d_out_all = net_pred(seq_all, input_n=in_n, output_n=out_n, itera=itera)

        p3d_out = p3d_out_all[:, seq_in:, 0]
        p3d_out = p3d_out.reshape([-1, out_n, p3d_out_all.shape[-1] // 3, 3])

        p3d_h36 = seq_all.reshape([-1, in_n + out_n, seq_all.shape[-1] // 3, 3])
        p3d_sup = p3d_h36[:, -out_n - seq_in :, :, :]

        p3d_out_all = p3d_out_all.reshape(
            [batch_size, seq_in + out_n, itera, p3d_out_all.shape[-1] // 3, 3]
        )

        # 2d joint loss:
        grad_norm = 0
        if is_train == 0:
            loss_p3d = torch.mean(torch.norm(p3d_out_all[:, :, 0] - p3d_sup, dim=3))
            loss_all = loss_p3d
            optimizer.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
            optimizer.step()
            # update log values
            l_p3d += loss_p3d.cpu().data.numpy() * batch_size

        if is_train <= 1:  # if is validation or train simply output the overall mean error
            mpjpe_p3d_h36 = torch.mean(torch.norm(p3d_h36[:, in_n:in_n + out_n] - p3d_out, dim=3))
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size
        else:
            mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()
       
    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n

    if is_train <= 1:
        ret["m_p3d_h36"] = m_p3d_h36 / n
    else:
        m_p3d_h36 = m_p3d_h36 / n
        for j in range(out_n):
            ret["#{:d}".format(titles[j])] = m_p3d_h36[j]
    return ret


if __name__ == '__main__':
    option = Options().parse()

    stime = time.time()
    main(option)

    ftime = time.time()
    print("Training took {} seconds".format(int(ftime - stime)))
