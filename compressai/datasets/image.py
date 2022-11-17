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

from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
import random
from natsort import natsorted


class ImageFolder(Dataset):
    def __init__(self, root, transform=None, image="image", map="map", is_train=True):
        splitdir_image = Path(root) / image
        splitdir_map = Path(root) / map


        if not splitdir_map.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples_image = natsorted([f for f in splitdir_image.iterdir() if f.is_file()])
        self.samples_map = natsorted([f for f in splitdir_map.iterdir() if f.is_file()])
        self.transform = transform
        self.is_train = is_train

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples_image[index]).convert("RGB")
        map = Image.open(self.samples_map[index]).convert("L")


        # rnd_w = random.randint(0, max(0, map.size[0] - 256))
        # rnd_h = random.randint(0, max(0, map.size[1] - 192))
        # box = (rnd_w, rnd_h, rnd_w + 256, rnd_h + 192)
        # img = img.crop(box)
        # map = map.crop(box)

        # img = img[rnd_w:rnd_w + 128, rnd_h:rnd_h + 128, :]
        # img_qp = img_qp[rnd_w:rnd_w + 128, rnd_h:rnd_h + 128, :]

        if self.transform:
            img = self.transform(img)
            map = self.transform(map)
            return {"img": img, "map": map}
        else:
            return {"img": img, "map": map}

    def __len__(self):
        return len(self.samples_image)


class CodecFolder(Dataset):

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        print(self.samples)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)