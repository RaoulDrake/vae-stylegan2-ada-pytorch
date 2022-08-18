from typing import Optional, Callable, Dict, Union, List, Tuple
import os
import json
import random
import xmltodict
import numpy as np
import scipy.io as sp_io
from PIL import Image
import torch
import torchvision
from torchvision import transforms as T
from torchvision.datasets import utils as ds_utils
from tqdm import tqdm


def _rename(name: str):
  renamed = name[0].upper()
  last_char_parsed = ''
  for char in name[1:]:
    if last_char_parsed not in ['_', '-']:
      if char != '_':
        renamed += char.lower()
      else:
        renamed += ' '
    else:
      renamed += char.upper()
    last_char_parsed = char
  return renamed


def square_crop(img: Image) -> Image:
  # Prepare bounding box arguments
  width, height = img.size
  left, upper, right, lower = (0, 0, width, height)

  if width == height:
    # Fast path if image is already square
    return img
  else:
    # Compute bounds for cropping if the image is not square
    if width > height:
      crop_left = crop_right = (width - height) // 2
      if (crop_left + crop_right) < (width - height):
        crop_left += 1
      assert (crop_left + crop_right) == (width - height), \
        f"Error while trying to compute square bounding box for cropping image of size {width}x{height}"
      left = crop_left
      right = width - crop_right
    else:  # height > width:
      crop_upper = crop_lower = (height - width) // 2
      if (crop_upper + crop_lower) < (height - width):
        crop_upper += 1
      assert (crop_upper + crop_lower) == (height - width), \
        f"Error while trying to compute square bounding box for cropping image of size {width}x{height}"
      upper = crop_upper
      lower = height - crop_lower

    # Crop the image
    crop_box = (left, upper, right, lower)
    img = img.crop(crop_box)

    return img


def square_bounding_box_crop(img: Image, bounding_box: Tuple[int, int, int, int]) -> Image:
  # Unpack bounding box and determine width and height
  bnd_box_left, bnd_box_upper, bnd_box_right, bnd_box_lower = bounding_box
  bnd_box_width, bnd_box_height = (bnd_box_right - bnd_box_left), (bnd_box_lower - bnd_box_upper)

  if bnd_box_width == bnd_box_height:
    # Fast path if bounding box is already square
    img = img.crop(bounding_box)
    return img
  else:
    # Determine image width and height
    img_width, img_height = img.size
    if bnd_box_width > bnd_box_height:
      # Check if we can extend height of bounding box
      if (img_height - bnd_box_height) >= (bnd_box_width - bnd_box_height):
        # Try to extend upper and lower bounds as evenly as possible
        extend_upper = extend_lower = (bnd_box_width - bnd_box_height) // 2
        if (extend_upper + extend_lower) < (bnd_box_width - bnd_box_height):
          extend_lower += 1
        bnd_box_upper -= extend_upper
        bnd_box_lower += extend_lower
        # Check if we reach out of bounds with the intended extension and correct if necessary
        if bnd_box_upper < 0:
          # since bnd_box_upper is negative,
          # we add to bnd_box_lower when we subtract bnd_box_upper from bnd_box_lower,
          # effectively lowering the lower bound by the amount necessary to shift the upper bound to 0
          bnd_box_lower -= bnd_box_upper
          bnd_box_upper = 0
        elif bnd_box_lower > img_height:
          # we subtract the (positive) difference between the bounding box lower bound and the image height
          # from the bounding box upper bound,
          # effectively lifting the upper bound by the amount necessary to shift the lower bound to the image height
          bnd_box_upper -= (bnd_box_lower - img_height)
          bnd_box_lower = img_height
      else:  # We need to crop width of bounding box
        # Extend the height of the bounding box as much as possible (i.e., to the image height)
        bnd_box_upper, bnd_box_lower = (0, img_height)
        # Determine amount by which we must crop width
        crop_left = crop_right = (bnd_box_width - img_height) // 2
        if (crop_left + crop_right) < (bnd_box_width - img_height):
          crop_left += 1
        # Prepare width crop
        bnd_box_left += crop_left
        bnd_box_right -= crop_right

    else:  # bnd_box_height > bnd_box_width
      # Check if we can extend width of bounding box
      if (img_width - bnd_box_width) >= (bnd_box_height - bnd_box_width):
        # Try to extend left and right bounds as evenly as possible
        extend_left = extend_right = (bnd_box_height - bnd_box_width) // 2
        if (extend_left + extend_right) < (bnd_box_height - bnd_box_width):
          extend_left += 1
        bnd_box_left -= extend_left
        bnd_box_right += extend_right
        # Check if we reach out of bounds with the intended extension and correct if necessary
        if bnd_box_left < 0:
          # since bnd_box_left is negative,
          # we add to bnd_box_right when we subtract bnd_box_left from bnd_box_right,
          # effectively extending the right bound rightwards
          # by the amount necessary to shift the left bound to 0
          bnd_box_right -= bnd_box_left
          bnd_box_left = 0
        elif bnd_box_right > img_width:
          # we subtract the (positive) difference between the bounding box right bound and the image width
          # from the bounding box left bound,
          # effectively extending the left bound leftwards
          # by the amount necessary to shift the right bound to the image width
          bnd_box_left -= (bnd_box_right - img_width)
          bnd_box_right = img_width
      else:  # We need to crop height of bounding box
        # Extend the width of the bounding box as much as possible (i.e., to the image width)
        bnd_box_left, bnd_box_right = (0, img_width)
        # Determine amount by which we must crop height
        crop_upper = crop_lower = (bnd_box_height - img_width) // 2
        if (crop_upper + crop_lower) < (bnd_box_height - img_width):
          crop_upper += 1
        # Prepare height crop
        bnd_box_upper += crop_upper
        bnd_box_lower -= crop_lower

    all_ok = (
        ((bnd_box_lower - bnd_box_upper) == (bnd_box_right - bnd_box_left)) and
        (bnd_box_upper >= 0 and bnd_box_lower <= img_height and bnd_box_left >= 0 and bnd_box_right <= img_width)
    )
    assert all_ok, \
      f"Error while trying to compute square bounding box for cropping image " \
      f"to bounding box of size {bnd_box_width}x{bnd_box_height}"

    # Crop the image
    crop_box = (bnd_box_left, bnd_box_upper, bnd_box_right, bnd_box_lower)
    img = img.crop(crop_box)

    return img


class StanfordDogs(torchvision.datasets.VisionDataset):

  images_url: str = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
  images_filename: str = "images.tar"
  annotations_url: str = "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar"
  annotations_filename: str = "annotation.tar"
  lists_url: str = "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar"
  lists_filename: str = "lists.tar"

  def __init__(
      self,
      root: str,
      train: bool = True,
      download: bool = False,
      transforms: Optional[Callable] = None,
      transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None,
      stream_from_disk: bool = True,
      crop_to_bounding_box: bool = False
  ):
    """
    Stanford Dogs Dataset.
    Note: transforms and the combination of transform and target_transform are mutually exclusive.

    :param root: Root directory of dataset.
    :param train: TODO
    :param download: TODO
    :param transforms: A function/transforms that takes in an image and a label
    and returns the transformed versions of both.
    :param transform: A function/transform that takes in an PIL image and returns a transformed version.
    E.g, transforms.RandomCrop
    :param target_transform:  A function/transform that takes in the target and transforms it.
    :param stream_from_disk: TODO
    :param crop_to_bounding_box: TODO
    """
    assert (
        all([transforms is None, transform is None, target_transform is None]) or
        (transforms is not None and transform is None and target_transform is None) or
        (transforms is None and (transform is not None or target_transform is not None))
    ), "transforms and the combination of transform and target_transform are mutually exclusive."
    super(StanfordDogs, self).__init__(
      root=root, transforms=transforms, transform=transform, target_transform=target_transform
    )

    self.train: bool = train
    self.stream_from_disk: bool = stream_from_disk
    self.crop_to_bounding_box: bool = crop_to_bounding_box

    if download:
      self.download()

    if not self._is_downloaded():
      raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

    if self.train:
      self.downloaded_list: Dict[str, Union[bytes, str, list, np.ndarray]] = sp_io.loadmat(
        os.path.join(self.root, "train_list.mat")
      )
    else:
      self.downloaded_list: Dict[str, Union[bytes, str, list, np.ndarray]] = sp_io.loadmat(
        os.path.join(self.root, "test_list.mat")
      )

    self.data: List[Union[str, np.ndarray]] = []
    self.targets: List[np.uint8] = []
    self.annotations: List[Dict] = []
    self.classes: List[str] = [""] * np.max(self.downloaded_list['labels'])

    for i in range(len(self.downloaded_list['file_list'])):
      data_path: str = self.downloaded_list['file_list'][i][0][0]
      data_path = os.path.join(self.root, 'Images', data_path)
      if self.stream_from_disk:
        self.data.append(data_path)
      else:
        with Image.open(data_path) as data_img:
          if data_img.mode == "RGB":
            img_arr = np.array(data_img)
          else:
            img_arr = np.array(data_img.convert('RGB'))
          self.data.append(img_arr)

      target: np.uint8 = np.uint8(self.downloaded_list['labels'][i][0]) - 1
      self.targets.append(target)

      annotation_path: str = self.downloaded_list['annotation_list'][i][0][0]
      annotation_path = os.path.join(self.root, 'Annotation', annotation_path)
      with open(annotation_path, 'r') as annotation_file:
        annotation_content = annotation_file.read()
        annotation_dict = xmltodict.parse(annotation_content)
        assert len(annotation_dict) == 1, f"found annotation file with multiple annotations"
        annotation_dict = annotation_dict['annotation']
        self.annotations.append(annotation_dict)

      # if not isinstance(annotation_dict['object'], dict):
      #   names = [o['name'] for o in annotation_dict['object']]
      #   num_classes = len(set(names))
      #   if num_classes > 1:
      #     print(f"Found example with multiple objects of different class. "
      #           f"# objects: {len(annotation_dict['object'])}, # classes: {num_classes}")

      if not self.classes[int(target)] and isinstance(annotation_dict['object'], dict):
        self.classes[int(target)] = _rename(annotation_dict['object']['name'])

    self.targets = np.array(self.targets)
    if self.stream_from_disk:
      self.data = np.array(self.data)

  def _is_downloaded(self) -> bool:
    """
    Checks if the dataset is already downloaded.

    :return: bool value: True if dataset is already downloaded, False if not.
    """
    # Images
    img_filepath = os.path.join(self.root, self.images_filename)
    images_are_downloaded = os.path.isfile(img_filepath)

    # Annotations
    annotations_filepath = os.path.join(self.root, self.annotations_filename)
    annotations_are_downloaded = os.path.isfile(annotations_filepath)

    # Train/test lists
    train_test_lists_filepath = os.path.join(self.root, self.lists_filename)
    train_test_lists_are_downloaded = os.path.isfile(train_test_lists_filepath)

    is_downloaded = (
        images_are_downloaded and
        annotations_are_downloaded and
        train_test_lists_are_downloaded
    )

    return is_downloaded

  def download(self):
    """
    Downloads the dataset if not already downloaded.
    """
    if not self._is_downloaded():
      ds_utils.download_and_extract_archive(self.images_url, self.root, filename=self.images_filename)
      ds_utils.download_and_extract_archive(self.annotations_url, self.root, filename=self.annotations_filename)
      ds_utils.download_and_extract_archive(self.lists_url, self.root, filename=self.lists_filename)

  def to_style_gan_2_ada_dataset(self, out_path: str):
    """
    Crops the dataset to square images and saves it in the specified out_path
    so that it is compatible with StyleGan2-ADA.

    Class labels are stored in a file called 'dataset.json' that is stored at
    the dataset root folder.  This file has the following structure:

    {
        "labels": [
            ["00000/img00000000.png",6],
            ["00000/img00000001.png",9],
            ... repeated for every image in the dataset
            ["00049/img00049999.png",1]
        ]
    }

    :param out_path: Path to the output folder.
    """
    print("\nConverting Stanford Dogs Dataset to ensure compatibility with StyleGan2-ADA...")
    # Create directory if necessary
    if not os.path.exists(out_path):
      os.makedirs(out_path)
    # Initialize label list for the dataset.json output file
    dataset_json_labels = []

    for i in tqdm(range(len(self.downloaded_list['file_list']))):
      target: np.uint8 = np.uint8(self.downloaded_list['labels'][i][0]) - 1
      class_name: str = self.classes[int(target)]

      annotation_path: str = self.downloaded_list['annotation_list'][i][0][0]
      annotation_path = os.path.join(self.root, 'Annotation', annotation_path)
      with open(annotation_path, 'r') as annotation_file:
        annotation_content = annotation_file.read()
        annotation_dict = xmltodict.parse(annotation_content)
        assert len(annotation_dict) == 1, f"found annotation file with multiple annotations"
        annotation_dict = annotation_dict['annotation']
        annotation_object = annotation_dict['object']

      # Only include images that contain only one dog / exclude images with multiple dogs in them
      # (a first train run of StyleGan2-ADA had the generator generate images that contain
      # two dog heads on one body, etc., which could potentially be due to the train images with multiple dogs in them)
      if isinstance(annotation_object, dict):
        # Create output path if it does not exist
        if not os.path.exists(os.path.join(out_path, f"{int(target)}_{class_name.replace(' ', '_')}")):
          os.makedirs(os.path.join(out_path, f"{int(target)}_{class_name.replace(' ', '_')}"))
        # Load image
        img_path: str = self.downloaded_list['file_list'][i][0][0]
        img_filename, img_file_extension = os.path.splitext(os.path.split(img_path)[-1])
        img_path = os.path.join(self.root, 'Images', img_path)
        with Image.open(img_path) as img_file:
          img = img_file
          if img_file.mode != "RGB":
            # There is one case of an RGBA image in the dataset, so we need to convert it to RGB
            img = img_file.convert('RGB')

          if not self.crop_to_bounding_box:
            # Crop to square image
            img = square_crop(img)

            # Save image
            img_out_path = f"{int(target)}_{class_name.replace(' ', '_')}/{img_filename}.png"
            img.save(os.path.join(out_path, img_out_path))

            # Add StyleGan2-Ada friendly label to dataset_json_labels
            json_label = [img_out_path, int(target)]
            dataset_json_labels.append(json_label)

          else:
            # Crop to bounding box while ensuring a square image
            bounding_box: Dict[str, int] = annotation_object['bndbox']
            bounding_box: Tuple[int, int, int, int] = (
              int(bounding_box['xmin']), int(bounding_box['ymin']),
              int(bounding_box['xmax']), int(bounding_box['ymax'])
            )
            # Crop to square image
            img = square_bounding_box_crop(img, bounding_box)

            # Save image
            img_out_path = f"{int(target)}_{class_name.replace(' ', '_')}/{img_filename}.png"
            img.save(os.path.join(out_path, img_out_path))

            # Add StyleGan2-Ada friendly label to dataset_json_labels
            assert self.classes[int(target)] == _rename(annotation_object["name"]), \
              f"Mismatch between target class label ({self.classes[int(target)]}) " \
              f"and annotation object label ({_rename(annotation_object['name'])})"
            json_label = [img_out_path, int(target)]
            dataset_json_labels.append(json_label)

    # Convert label list to json and save it as dataset.json
    dataset_json = {"labels": dataset_json_labels}
    with open(os.path.join(out_path, 'dataset.json'), "w", encoding="utf8") as dataset_json_file:
      dataset_json_file.write(json.dumps(dataset_json, indent=4))

    print(f"\nDone converting Stanford Dogs Dataset. Output can be found in {out_path}.")

  def __len__(self):
    return self.targets.shape[0]  # len(self.data)

  def __getitem__(self, item):
    img, target = self.data[item], self.targets[item]

    if self.stream_from_disk:
      with Image.open(img) as img_file:
        if img_file.mode == "RGB":
          img = np.array(img_file)
        else:
          img = np.array(img_file.convert('RGB'))

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.fromarray(img)

    if self.crop_to_bounding_box:
      annotation_object = self.annotations[item]['object']
      if not isinstance(annotation_object, dict):
        annotation_object = random.choice(annotation_object)
      bounding_box = annotation_object['bndbox']
      # box – The crop rectangle, as a (left, upper, right, lower)-tuple.
      img = img.crop(box=(
        int(bounding_box['xmin']),
        int(bounding_box['ymin']),
        int(bounding_box['xmax']),
        int(bounding_box['ymax'])
      ))

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target


class TsinghuaDogs(torchvision.datasets.VisionDataset):

  def __init__(
      self,
      root: str,
      train: bool = True,
      transforms: Optional[Callable] = None,
      transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None,
      stream_from_disk: bool = True,
      crop_to_bounding_box: bool = False,
      bounding_box_type: str = "body"
  ):
    """
    Stanford Dogs Dataset.
    Note: transforms and the combination of transform and target_transform are mutually exclusive.

    :param root: Root directory of dataset.
    :param train: TODO
    :param transforms: A function/transforms that takes in an image and a label
    and returns the transformed versions of both.
    :param transform: A function/transform that takes in an PIL image and returns a transformed version.
    E.g, transforms.RandomCrop
    :param target_transform:  A function/transform that takes in the target and transforms it.
    :param stream_from_disk: TODO
    :param crop_to_bounding_box: TODO
    :param bounding_box_type: TODO
    """
    assert (
        all([transforms is None, transform is None, target_transform is None]) or
        (transforms is not None and transform is None and target_transform is None) or
        (transforms is None and (transform is not None or target_transform is not None))
    ), "transforms and the combination of transform and target_transform are mutually exclusive."
    assert bounding_box_type in ["body", "head"], 'bounding_box must be either "body" or "head"'
    if not os.path.exists(root):
      raise RuntimeError("Dataset not found or corrupted.")
    super(TsinghuaDogs, self).__init__(
      root=root, transforms=transforms, transform=transform, target_transform=target_transform
    )

    self.train: bool = train
    self.stream_from_disk: bool = stream_from_disk
    self.crop_to_bounding_box: bool = crop_to_bounding_box
    self.bounding_box_type = bounding_box_type

    if self.train:
      list_path = os.path.join(self.root, "TrainValSplit", "TrainAndValList", "train.lst")
    else:
      list_path = os.path.join(self.root, "TrainValSplit", "TrainAndValList", "validation.lst")
    with open(list_path, "r", encoding="utf-8-sig") as f:
      self.data_list: List[str] = f.read().splitlines()

    if not os.path.isfile(os.path.join(self.root, "labels.json")):
      print(f"Creating labels.json file in {self.root}")
      # TODO: create mapping from class names to targets and save it
      labels = {
        _rename(sub_folder.split('-')[-1]): i
        for i, sub_folder in enumerate(os.listdir(os.path.join(self.root, 'high-resolution')))
        if os.path.isdir(os.path.join(os.path.join(self.root, 'high-resolution'), sub_folder))
      }
      with open(os.path.join(self.root, "labels.json"), "w", encoding="utf8") as f:
        f.write(json.dumps(labels, indent=4))

    with open(os.path.join(self.root, "labels.json"), "r", encoding="utf8") as f:
      self.labels_to_targets: Dict[str, int] = json.loads(f.read())
    self.targets_to_labels: Dict[int, str] = {v: k for k, v in self.labels_to_targets.items()}

    self.img_paths: List[str] = []
    self.img_arrays: List[np.ndarray] = []
    self.targets: List[np.uint8] = []
    self.annotations: List[Dict] = []

    for i in tqdm(range(len(self.data_list)), desc="Loading Tsinghua Dogs data"):
      data_path: str = self.data_list[i]
      img_path = os.path.join(self.root, 'high-resolution', data_path[3:])
      self.img_paths.append(img_path)
      if not self.stream_from_disk:
        with Image.open(img_path) as data_img:
          if data_img.mode == "RGB":
            img_arr = np.array(data_img)
          else:
            img_arr = np.array(data_img.convert('RGB'))
          self.img_arrays.append(img_arr)

      annotation_path = os.path.join(self.root, 'high-annotations', data_path[3:] + '.xml')
      with open(annotation_path, 'r', encoding="utf-8-sig") as annotation_file:
        annotation = annotation_file.read()
        annotation = xmltodict.parse(annotation)
        assert len(annotation) == 1, f"found annotation file with multiple annotations"
        annotation = annotation['annotation']
        self.annotations.append(annotation)

        if not isinstance(annotation['object'], dict):
          names = [o['name'] for o in annotation['object']]
          num_classes = len(set(names))
          if num_classes > 1:
            print(f"Found example with multiple objects of different class. "
                  f"# objects: {len(annotation['object'])}, # classes: {num_classes}")

      label: str = annotation['object']['name']
      label = _rename(label)
      if label not in self.labels_to_targets:
        raise RuntimeError(f"Encountered inconsistent label {label}")
      target: np.uint8 = np.uint8(self.labels_to_targets[label])
      self.targets.append(target)

    self.targets = np.array(self.targets)
    self.img_paths = np.array(self.img_paths)

  def __len__(self):
    return self.targets.shape[0]

  def __getitem__(self, item):
    target = self.targets[item]

    if self.stream_from_disk:
      with Image.open(self.img_paths[item]) as img_file:
        if img_file.mode == "RGB":
          img_arr = np.array(img_file)
        else:
          img_arr = np.array(img_file.convert('RGB'))
    else:
      img_arr = self.img_arrays[item]

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.fromarray(img_arr)

    if self.crop_to_bounding_box:
      annotation_object = self.annotations[item]['object']
      if not isinstance(annotation_object, dict):
        annotation_object = random.choice(annotation_object)
      bounding_box = annotation_object[f'{self.bounding_box_type}bndbox']
      # box – The crop rectangle, as a (left, upper, right, lower)-tuple.
      img = img.crop(box=(
        int(bounding_box['xmin']),
        int(bounding_box['ymin']),
        int(bounding_box['xmax']),
        int(bounding_box['ymax'])
      ))

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def to_style_gan_2_ada_dataset(self, out_path: str):
    """
    Crops the dataset to square images and saves it in the specified out_path
    so that it is compatible with StyleGan2-ADA.

    Class labels are stored in a file called 'dataset.json' that is stored at
    the dataset root folder.  This file has the following structure:

    {
        "labels": [
            ["00000/img00000000.png",6],
            ["00000/img00000001.png",9],
            ... repeated for every image in the dataset
            ["00049/img00049999.png",1]
        ]
    }

    :param out_path: Path to the output folder.
    """
    print("\nConverting Tsinghua Dogs Dataset to ensure compatibility with StyleGan2-ADA...")
    # Create directory if necessary
    if not os.path.exists(out_path):
      os.makedirs(out_path)
    # Initialize label list for the dataset.json output file
    dataset_json_labels = []

    for i in tqdm(range(len(self.img_paths))):
      img_path: str = self.img_paths[i]
      target: np.uint8 = self.targets[i]
      label: str = self.targets_to_labels[self.targets[i]]
      annotation: Dict = self.annotations[i]
      annotation_object: Dict = annotation['object']

      # Only include images that contain only one dog / exclude images with multiple dogs in them
      # (a first train run of StyleGan2-ADA had the generator generate images that contain
      # two dog heads on one body, etc., which could potentially be due to the train images with multiple dogs in them)
      if isinstance(annotation_object, dict):
        # Create output sub-folder for class if it does not exist.
        class_path = f"{int(target)}_{label.replace(' ', '_')}"
        if not os.path.exists(os.path.join(out_path, class_path)):
          os.makedirs(os.path.join(out_path, class_path))
        # Load image
        if self.stream_from_disk:
          with Image.open(img_path) as img_file:
            if img_file.mode == "RGB":
              img_arr = np.array(img_file)
            else:
              img_arr = np.array(img_file.convert('RGB'))
        else:
          img_arr = self.img_arrays[i]
        img: Image = Image.fromarray(img_arr)

        if not self.crop_to_bounding_box:
          # Crop to square image
          img = square_crop(img)
        else:
          # Crop to bounding box while ensuring a square image
          bounding_box: Dict[str, int] = annotation_object[f'{self.bounding_box_type}bndbox']
          bounding_box: Tuple[int, int, int, int] = (
            int(bounding_box['xmin']), int(bounding_box['ymin']),
            int(bounding_box['xmax']), int(bounding_box['ymax'])
          )
          img = square_bounding_box_crop(img, bounding_box)

        # Save image
        img_filename = os.path.splitext(os.path.split(img_path)[-1])[0]
        img_out_path = f"{class_path}/{img_filename}.png"
        img.save(os.path.join(out_path, img_out_path))
        img.close()

        # Add StyleGan2-Ada friendly label to dataset_json_labels
        assert self.targets_to_labels[int(target)] == _rename(annotation_object["name"]), \
          f"Mismatch between target class label ({self.targets_to_labels[int(target)]}) " \
          f"and annotation object label ({_rename(annotation_object['name'])})"
        json_label = [img_out_path, int(target)]
        dataset_json_labels.append(json_label)

    # Convert label list to json and save it as dataset.json
    dataset_json = {"labels": dataset_json_labels}
    with open(os.path.join(out_path, 'dataset.json'), "w", encoding="utf8") as dataset_json_file:
      dataset_json_file.write(json.dumps(dataset_json, indent=4))

    print(f"\nDone converting Tsinghua Dogs Dataset. Output can be found in {out_path}.")


def get_stanford_dogs_loader(
    batch_size: int,
    aug_train: Callable = T.ToTensor(),
    aug_test: Callable = T.ToTensor(),
    root='./data/stanford_dogs',
    drop_last: bool = False,
    num_workers_per_ds: int = 0,
    persistent_workers: bool = False,
    pin_memory: bool = False,
    stream_from_disk: bool = False,
    crop_to_bounding_box: bool = False
):
  # create data directory if not present already
  if not os.path.exists(root):
    os.makedirs(root)
  trainset = StanfordDogs(
    root=root, train=True, download=True, transform=aug_train,
    stream_from_disk=stream_from_disk, crop_to_bounding_box=crop_to_bounding_box
  )
  trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, drop_last=drop_last,
    num_workers=num_workers_per_ds, persistent_workers=persistent_workers, pin_memory=pin_memory
  )

  testset = StanfordDogs(
    root=root, train=False, download=True, transform=aug_test,
    stream_from_disk=stream_from_disk, crop_to_bounding_box=crop_to_bounding_box
  )
  testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, drop_last=drop_last,
    num_workers=num_workers_per_ds, persistent_workers=persistent_workers, pin_memory=pin_memory
  )

  return trainloader, testloader


def get_tsinghua_dogs_loader(
    batch_size: int,
    aug_train: Callable = T.ToTensor(),
    aug_test: Callable = T.ToTensor(),
    root='./data/tsinghua_dogs',
    drop_last: bool = False,
    num_workers_per_ds: int = 0,
    persistent_workers: bool = False,
    pin_memory: bool = False,
    stream_from_disk: bool = False,
    crop_to_bounding_box: bool = False,
    bounding_box_type: str = "body"
):
  # create data directory if not present already
  if not os.path.exists(root):
    os.makedirs(root)
  trainset = TsinghuaDogs(
    root=root, train=True, transform=aug_train,
    stream_from_disk=stream_from_disk, crop_to_bounding_box=crop_to_bounding_box,
    bounding_box_type=bounding_box_type
  )
  trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, drop_last=drop_last,
    num_workers=num_workers_per_ds, persistent_workers=persistent_workers, pin_memory=pin_memory
  )

  testset = TsinghuaDogs(
    root=root, train=False, transform=aug_test,
    stream_from_disk=stream_from_disk, crop_to_bounding_box=crop_to_bounding_box,
    bounding_box_type=bounding_box_type
  )
  testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, drop_last=drop_last,
    num_workers=num_workers_per_ds, persistent_workers=persistent_workers, pin_memory=pin_memory
  )

  return trainloader, testloader


if __name__ == '__main__':
  from datetime import datetime
  import matplotlib.pyplot as plt
  from torchvision.utils import make_grid

  # ds = StanfordDogs('./data/stanford_dogs', download=True)
  # ds = StanfordDogs('./data/stanford_dogs', download=False, stream_from_disk=True, crop_to_bounding_box=True)
  # ds.to_style_gan_2_ada_dataset('./data/stanford_dogs_bnd_box_crop_style_gan_2_ada')

  ds = TsinghuaDogs(
    'E:\\Data\\CVDL_Datasets\\Tsinghua Dogs',
    stream_from_disk=True,
    crop_to_bounding_box=True,
    bounding_box_type="body",
    train=False
  )
  ds.to_style_gan_2_ada_dataset(
    'E:\\Data\\CVDL_Datasets\\tsinghua_dogs_test_bnd_box_crop_style_gan_2_ada'
  )

  # start = datetime.utcnow()
  # RESOLUTION = 256
  #
  # # instantiate the base data loaders
  # trainloader_base, testloader_base = get_tsinghua_dogs_loader(  # get_stanford_dogs_loader(
  #   root="E:\\Data\\CVDL_Datasets\\Tsinghua Dogs",
  #   batch_size=16,
  #   aug_train=T.Compose([
  #     T.ToTensor(),
  #     T.Resize((RESOLUTION, RESOLUTION))
  #   ]),
  #   aug_test=T.Compose([
  #     T.ToTensor(),
  #     T.Resize((RESOLUTION, RESOLUTION))
  #   ]),
  #   stream_from_disk=True,
  #   crop_to_bounding_box=True,
  #   bounding_box_type="body"
  # )
  #
  # end = datetime.utcnow()
  # time_taken = end - start
  # print(f"Time taken (in seconds) for loading train- and test-loaders: {time_taken.total_seconds()}")
  #
  # def show_batch(dl):
  #   for img, lb in dl:
  #       fig, ax = plt.subplots(figsize=(4, 4))
  #       ax.set_xticks([])
  #       ax.set_yticks([])
  #       ax.imshow(make_grid(img.cpu(), nrow=4).permute(1, 2, 0))
  #       break
  #
  # show_batch(trainloader_base)
  # plt.show()