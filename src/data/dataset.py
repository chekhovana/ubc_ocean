import os
from abc import abstractmethod
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
from pandas import DataFrame
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler

class UbcOceanDataset(Dataset):
    def __init__(self, image_folder: str, df_filename: DataFrame,
                 mode='train', transforms=None):
        self.image_folder = image_folder
        self.df = pd.read_csv(df_filename)
        modes = ['train', 'valid', 'infer']
        assert mode in modes, f'Mode should be one of {modes}'
        self.mode = mode
        self.transforms = transforms

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        filename = annotation.image.filename
        item = {'filename': filename}
        filename = os.path.join(self.image_folder, filename)
        image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        xmin, ymin, width, height = map(int, annotation.bbox)
        xmax, ymax = xmin + width, ymin + height
        image = image[ymin:ymax, xmin:xmax, :]
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        item['features'] = image
        item['annotation'] = annotation.id
        if self.mode != 'infer':
            item['targets'] = self.get_targets(annotation)
        return item
