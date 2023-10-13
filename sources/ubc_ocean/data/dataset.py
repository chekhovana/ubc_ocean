import os

import cv2
import numpy as np
import pandas as pd
from pandas import DataFrame
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
import yaml
import hydra
import random
import torch
from tqdm import tqdm

class UbcTileDataset(Dataset):
    def __init__(self, image_folder: str, annotation_file: str,
                 mode='train', transforms=None, subsample=None):
        self.image_folder = image_folder
        self.df = self.load_df(annotation_file)
        self.subsample = subsample
        modes = ['train', 'valid', 'infer']
        assert mode in modes, f'Mode should be one of {modes}'
        self.mode = mode
        self.transforms = transforms
        labels = np.array(sorted(self.df['label'].unique())).reshape(-1, 1)
        self.label_encoder = OneHotEncoder()
        self.label_encoder.fit(labels)

    def load_df(self, annotation_file):
        return pd.read_csv(annotation_file)

    def __len__(self):
        return len(self.df)

    def get_image_folder(self, image_id):
        return os.path.join(self.image_folder, str(image_id))

    def __getitem__(self, idx):
        df_row = self.df.iloc[idx]
        image_id = df_row['image_id']
        label = self.label_encoder.transform([[df_row['label']]])
        label = torch.tensor(label.toarray()[0], dtype=torch.float32)
        folder = self.get_image_folder(image_id)
        # folder = 'data/tiles/size_224_overlap_10/45630'
        image_files = [os.path.join(folder, fn) for fn in os.listdir(folder)]
        random.shuffle(image_files)
        if self.subsample is not None:
            image_files = image_files[:self.subsample]
        features = []
        for image_file in image_files:
            image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
            image = self.transforms(image=image)['image']
            features.append(image)
        features = torch.stack(features)
        item = dict(features=features, targets=label)
        return item


class UbcTileThumbnailDataset(UbcTileDataset):
    def load_df(self, annotation_file):
        df = super().load_df(annotation_file)
        df = df[df['is_tma'] == 0]
        return df

    def get_image_folder(self, image_id):
        return os.path.join(self.image_folder, str(image_id) + '_thumbnail')


def main():
    df_filename = 'data/original/annotations/train.csv'
    image_folder = 'data/tiles/thumbnails/size_224_overlap_10'
    dataset = UbcTileDataset(image_folder, df_filename)
    item = dataset[0]
    # df = pd.read_csv(df_filename)
    # print(df.head())
    # labels = np.array(sorted(df['label'].unique())).reshape(-1, 1)
    # from sklearn.preprocessing import OneHotEncoder
    # enc = OneHotEncoder()
    # enc.fit(labels)
    # for l in labels:
    #     print(l, enc.transform([l]).toarray())
    # l = 'EC'
    # print(enc.transform([[l]]).toarray())
    # # >> > X = [['Male', 1], ['Female', 3], ['Female', 2]]
    # # >> > enc.fit(X)
    # # OneHotEncoder(handle_unknown='ignore')
    # # >> > enc.categories_
    # # [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    # # >> > enc.transform([['Female', 1], ['Male', 4]]).toarray()
    # # pass


if __name__ == '__main__':
    # print(os.getcwd())
    # print(os.listdir(os.getcwd()))
    with open('configs/train/base.yaml') as f:
        config = yaml.load(f, yaml.Loader)
        data = hydra.utils.instantiate(config['data'])
        dataset = data['loaders']['train'].dataset
        for i in range(len(dataset)):
            item = dataset[i]
            print(i, item['targets'])
    # main()