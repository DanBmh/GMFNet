import sys
import time
from utils import h36motion3d as datasets
from model import AttModel
from utils.opt import Options
from torch.utils.data import DataLoader
import torch
import numpy as np
import tqdm

# ==================================================================================================

sys.path.append("/PoseForecasters/")
import utils_pipeline

datamode = "gt-gt"
# datamode = "pred-pred"

sconfig = {
    "item_step": 2,
    "window_step": 2,
    "input_n": 50,
    "output_n": 25,
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
}

dataset_eval_test = "/datasets/preprocessed/human36m/{}_forecast_rpt.json"

viz_action = ""
# viz_action = "walking"

# ==================================================================================================


def prepare_sequences(batch, batch_size: int, split: str, device):
    sequences = utils_pipeline.make_input_sequence(batch, split, "gt-gt")

    # Add empty joints to pad the number of joints to the original number of joints
    s1 = sequences.shape
    s3 = len(sconfig["select_joints"])
    sequences = np.concatenate(
        (sequences, np.zeros((s1[0], s1[1], 22 - s3, 3))), axis=2
    )

    # Merge joints and coordinates to a single dimension
    sequences = sequences.reshape([batch_size, sequences.shape[1], -1])

    # Convert to millimeters
    sequences = sequences * 1000

    sequences = torch.from_numpy(sequences.astype(np.float32)).to(device)

    return sequences


# ==================================================================================================


def main(opt):
    print('>>> create models')
    in_features = opt.in_features
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    dev = opt.dev
    net_pred = AttModel.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n)
    net_pred.to(dev)
    print(">>> loading ckpt len from '{}'".format(opt.ckpt))
    if torch.cuda.is_available() :
        ckpt = torch.load(opt.ckpt, map_location=torch.device(dev))
    else:
        ckpt = torch.load(opt.ckpt, map_location=torch.device('cpu'))
    net_pred.load_state_dict(ckpt['state_dict'])
    net_pred.cuda()
    print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    run_test(net_pred, opt)


def run_test(model, opt):

    model.eval()
    action_losses = []
    itera = 1
    in_n = opt.input_n
    out_n = opt.output_n
    seq_in = opt.kernel_size

    # Load preprocessed datasets
    dataset_test, dlen = utils_pipeline.load_dataset(
        dataset_eval_test.format("test"), "test", sconfig
    )
    dataset_test = dataset_test["sequences"]
    label_gen_test = utils_pipeline.create_labels_generator(dataset_test, sconfig)

    stime = time.time()
    frame_losses = np.zeros([out_n])
    nitems = 0

    with torch.no_grad():
        nbatch = 1
        batch_size = nbatch

        for batch in tqdm.tqdm(label_gen_test, total=dlen):

            if nbatch == 1:
                batch = [batch]

            nitems += nbatch
            sequences_train = prepare_sequences(batch, nbatch, "input", "cuda")
            sequences_gt = prepare_sequences(batch, nbatch, "target", "cuda")
            seq_all = torch.cat([sequences_train, sequences_gt], dim=1)

            p3d_out_all = model(seq_all, input_n=in_n, output_n=out_n, itera=itera)

            sequences_predict = (
                p3d_out_all[:, seq_in:]
                .transpose(1, 2)
                .reshape([batch_size, out_n * itera, -1])[:, :out_n]
            )

            # Remove the padding again
            s2 = sequences_gt.shape
            s3 = len(sconfig["select_joints"])
            s4 = sequences_predict.shape
            sequences_predict = sequences_predict.reshape([s4[0], s4[1], 22, 3])
            sequences_predict = sequences_predict[:, :, :s3, :]
            sequences_gt = sequences_gt.reshape([s2[0], s2[1], 22, 3])
            sequences_gt = sequences_gt[:, :, :s3, :]

            loss = torch.sqrt(
                torch.sum((sequences_predict - sequences_gt) ** 2, dim=-1)
            )
            loss = torch.sum(torch.mean(loss, dim=2), dim=0)
            frame_losses += loss.cpu().data.numpy()

    avg_losses = frame_losses / nitems
    print("Averaged frame losses in mm are:", avg_losses)

    ftime = time.time()
    print("Testing took {} seconds".format(int(ftime - stime)))


if __name__ == '__main__':
    option = Options().parse()
    main(option)
