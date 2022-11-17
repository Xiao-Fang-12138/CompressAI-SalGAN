# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# python3 examples/train.py -m cheng2020-attn -d /media/disk2/jjp/jjp/Dataset/DIV2K/DIV2K_train_HR/ -d_test /media/disk2/jjp/jjp/Dataset/div_after_crop/ -q 4 --lambda 0.001 --batch-size 6 -lr 1e-5 --save --cuda --exp exp_cheng_En_01_only_q4 --checkpoint /home/jjp/CompressAI/pretrained_model/cheng2020-attn/cheng2020_attn-ms-ssim-4-8b2f647e.pth.tar
# python3 examples/train.py -m cheng2020-attn -d /media/disk2/jjp/jjp/Dataset/DIV2K/DIV2K_train_HR/ -d_test /media/disk2/jjp/jjp/Dataset/div_after_crop/ -q 3 --lambda 0.001 --batch-size 6 -lr 1e-4 --save --cuda --exp exp_cheng_En_01_only_q3 --checkpoint /home/jjp/CompressAI/pretrained_model/cheng2020-attn/cheng2020_attn-mse-3-2d07bbdf.pth.tar
# python3 examples/train.py - m cheng2020 - attn - d / media / disk2 / jjp / Dataset / IRN / JPEG / DIV2K_train / -d_test / media / disk2 / jjp / Dataset / IRN / JPEG / DIV2K_valid / --lambda 0.001 --batch-size 32 -lr 1e-5 --save --cuda --exp exp_JPEG_En_25

import os
import argparse
import math
import random
import shutil
import sys
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import cv2

from torch.utils.data import DataLoader
from torchvision import transforms
from collections import OrderedDict

import logging
from utils import util
from compressai.datasets import ImageFolder
from compressai.zoo import image_models
from compressai.models import Generator, Discriminator, Saliency
from torchvision.transforms import ToPILImage
import numpy as np
import PIL.Image as Image
from torchvision.transforms import ToPILImage
from torch.utils.tensorboard import SummaryWriter


def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.cpu().clamp_(0, 1).squeeze())

def file_name(file_dir):
    namelist = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                namelist.append(os.path.splitext(file)[0])
    return namelist

def get_corr(fake_Y, Y):  # 计算两个向量person相关系数
    fake_Y, Y = fake_Y.reshape(-1), Y.reshape(-1)
    fake_Y_mean, Y_mean = torch.mean(fake_Y), torch.mean(Y)
    corr = (torch.sum((fake_Y - fake_Y_mean) * (Y - Y_mean))) / (
            torch.sqrt(torch.sum((fake_Y - fake_Y_mean) ** 2)) * torch.sqrt(torch.sum((Y - Y_mean) ** 2)))
    return corr

def kld(fake_Y, Y):  # 计算两个向量person相关系数
    fake_Y, Y = fake_Y / torch.sum(fake_Y), Y / torch.sum(Y)
    loss = Y * torch.log((Y + 1e-17) / (fake_Y + 1e-17))
    loss = torch.sum(loss)
    return loss


class FlagLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
        super().__init__()
        # self.kld = nn.KLDivLoss(reduction='sum')

    def forward(self, flag, truth):
        out = {}
        out["KL_loss"] = kld(flag, truth)
        out["CC_loss"] = get_corr(flag, truth)
        # out["loss"] = self.mse(x_l, x_enh) + out["mse_loss"]
        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def test(model, criterion, test_dataloader, epoch, logger_test, tb_logger, namelist):
    model.eval()
    device = next(model.parameters()).device
    KL_loss = AverageMeter()
    CC_loss = AverageMeter()
    csvfile = open('./figure/pic/result.csv', 'w',
                   newline='')
    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            img = d["img"].to(device)
            map = d["map"].to(device)
            predict_map = model(img)

            reimage = predict_map
            reimage.squeeze()
            # 出现色块的原因可能是没有强制收敛使得大于1变为0
            reimage[reimage > 1.0] = 1.0
            reimage[reimage < 0] = 0
            reimage = reimage.squeeze(0)
            reimage = transforms.ToPILImage()(reimage)  # PIL格式
            # print(list)
            out_path = "./figure/pic/" + namelist[i] + ".png"
            reimage.save(out_path)
            predict_map = cv2.resize(cv2.imread(out_path,0), (1920, 1080), interpolation=cv2.INTER_AREA)
            cv2.imwrite("./figure/bigpic/" + namelist[i] + ".png", predict_map)
            trans = transforms.ToTensor()
            predict_map = trans(predict_map).to(device)
            loss = criterion(predict_map, map)
            print(loss["KL_loss"])
            print(loss["CC_loss"])
            KL_loss.update(loss["KL_loss"])
            CC_loss.update(loss["CC_loss"])
            writer = csv.writer(csvfile)
            writeRow = [i, '"' + str(namelist[i]) + '"', str(loss["KL_loss"].item()), str(loss["CC_loss"].item())]
            writer.writerow(writeRow)  # final close remember!
        csvfile.close()



def predict(model, img, epoch, path):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        img = Image.open(img).convert("RGB")
        to_tensor = transforms.ToTensor()  # Transforms 0-255 numbers to 0 - 1.0.
        img = to_tensor(img).to(device)
        img = img.unsqueeze(0)
        out = model(img)
        print(out.max())
        map_out = out.cpu().data.squeeze(0)
        new_path = path + str(epoch) + ".png"
        pilTrans = transforms.ToPILImage()
        pilImg = pilTrans(map_out)
        print('Image saved to ', new_path)
        pilImg.save(new_path)

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def load_network(load_path, network, strict=True):
    if isinstance(network, nn.DataParallel):
        network = network.module
    load_net = torch.load(load_path)

    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    network.load_state_dict(load_net_clean, strict=strict)



def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    # parser.add_argument(
    #     "-d_test", "--test_dataset", type=str, required=True, help="Training dataset"
    # )
    parser.add_argument(
        "-e",
        "--epochs",
        default=200,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(128, 128),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        "-exp", "--experiment", type=str, required=True, help="Experiment name"
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    if not os.path.exists(os.path.join('experiments', args.experiment)):
        os.makedirs(os.path.join('experiments', args.experiment))

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(device)
    if device == 'cuda':
        torch.cuda.set_device(args.gpu_id)
    print('temp gpu device number:')
    print(torch.cuda.current_device())

    util.setup_logger('generate', os.path.join('experiments', args.experiment), 'generate_' + args.experiment,
                      level=logging.INFO,
                      screen=True, tofile=True)
    util.setup_logger('discriminate', os.path.join('experiments', args.experiment), 'discriminate_' + args.experiment,
                      level=logging.INFO,
                      screen=True, tofile=True)

    logger_test = logging.getLogger('generate')

    tb_logger = SummaryWriter(log_dir='./tb_logger/' + args.experiment)


    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )
    test_dataset = ImageFolder(args.dataset, transform=test_transforms, image="images256x192", map="map")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    namelist = file_name(os.path.join(args.dataset, "images256x192/"))

    net = Saliency()
    net = net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))

    logger_test.info(args)
    logger_test.info(net)
    criterion = FlagLoss()
    epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)

        checkpoint = os.path.join(args.checkpoint, "net_checkpoint_best_loss.pth.tar")
        checkpoint = torch.load(checkpoint, map_location=device)

        net.load_state_dict(checkpoint["state_dict"])
    loss = test(net, criterion, test_dataloader, epoch, logger_test, tb_logger, namelist)



if __name__ == "__main__":
    main(sys.argv[1:])
