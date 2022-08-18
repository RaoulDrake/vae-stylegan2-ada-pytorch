from typing import Dict
import os
import json
import numpy as np
import pandas as pd
import torch
import dnnlib
from training.dataset import ImageFolderDataset
from torch_utils import misc
# import projector


def load():
  batch_size = 16
  num_gpus = 1
  data_loader_kwargs = {}  # dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)
  random_seed = 0

  data_path = "E:\\Data\\CVDL_Datasets\\tsinghua_dogs_bnd_box_crop__style_gan_2_ada.zip"
  tsinghua_dataset = ImageFolderDataset(path=data_path, use_labels=True, max_size=None, xflip=False)

  labels_one_hot = [tsinghua_dataset.get_label(l) for l in range(len(tsinghua_dataset))]
  labels = [np.argmax(tsinghua_dataset.get_label(l)) for l in range(len(tsinghua_dataset))]
  print(
    f"Min num of imgs per class: {np.unique(labels, return_counts=True)[1].min()}\n"
    f"Min num of imgs per class: {np.unique(labels, return_counts=True)[1].max()}\n"
    f"Mean num of imgs per class: {np.unique(labels, return_counts=True)[1].mean()}\n"
  )

  # num_imgs_per_label = [0] * 130
  # for img, label_one_hot in tsinghua_dataset:
  #   label_int = int(np.argmax(label_one_hot))
  #   num_imgs_per_label[label_int] += 1

  labels_json_path = "E:\\Data\\CVDL_Datasets\\Tsinghua Dogs\\labels.json"
  with open(os.path.join(labels_json_path), "r", encoding="utf8") as f:
    labels_to_targets: Dict[str, int] = json.loads(f.read())
  targets_to_labels: Dict[int, str] = {v: k for k, v in labels_to_targets.items()}

  labels = list(range(130))
  label_names = [""] * 130
  num_imgs = [0] * 130
  for img, label_one_hot in tsinghua_dataset:
    label_int = int(np.argmax(label_one_hot))
    num_imgs[label_int] += 1
    label_names[label_int] = targets_to_labels[label_int]

  df = pd.DataFrame.from_dict({
    'label': labels,
    'label_name': label_names,
    'num_imgs': num_imgs
  })
  df_out_path = "C:\\Users\\Johannes\\OneDrive\\OneDrive - campus.lmu.de\\Studium\\SS 22\\CVDL\\Project\\"
  df_out_filepath = df_out_path + "num_imgs_per_class.csv"
  df.to_csv(df_out_filepath, index=False)

  training_set_sampler = misc.InfiniteSampler(
    dataset=tsinghua_dataset, rank=0, num_replicas=num_gpus, seed=random_seed
  )
  training_set_iterator = iter(
    torch.utils.data.DataLoader(
      dataset=tsinghua_dataset, sampler=training_set_sampler,
      batch_size=batch_size//num_gpus, **data_loader_kwargs
    )
  )
  phase_real_img, phase_real_c = next(training_set_iterator)
  print()


if __name__ == '__main__':
    load()
