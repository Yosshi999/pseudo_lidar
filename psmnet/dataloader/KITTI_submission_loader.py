import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

  left_fold  = 'image_2/'
  right_fold = 'image_3/'
  calib_fold = 'calib/'
  # left_fold  = 'image_2/data/'
  # right_fold = 'image_3/data/'


  # image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]
  image = [img for img in os.listdir(filepath+left_fold)]
  image = sorted(image)

  image_idx = map(lambda x: ".".join(x.split(".")[:-1]), image)

  left_test  = [filepath+left_fold+img for img in image]
  right_test = [filepath+right_fold+img for img in image]
  calib_test = [filepath+calib_fold+img+".txt" for img in image_idx]

  return left_test, right_test, calib_test
