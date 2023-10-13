import sys

import cv2
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import pyvips


class TileGenerator:
    def __init__(self, input, output, size, overlap, threshold):
        self.input = input
        self.size = size
        self.overlap = overlap
        self.threshold = threshold
        self.output = (f'{output}/size_{size}_overlap_{int(overlap * 100)}_'
                       f'threshold_{int(threshold * 100)}')

    def generate_tiles(self, image):
        height, width = image.shape[:2]
        step = int(self.size * (1 - self.overlap))
        rows = height // step + (1 if height % step > 0 else 0)
        cols = width // step + (1 if width % step > 0 else 0)
        tiles = []
        for row in range(rows):
            for col in range(cols):
                top, left = step * row, step * col
                bottom, right = top + self.size, left + self.size
                tile = image[top:bottom, left:right, :]
                h, w = tile.shape[:2]
                if h < self.size or w < self.size:
                    tile = np.pad(tile,
                                  ((0, self.size - h), (0, self.size - w),
                                   (0, 0)))
                tiles.append((row, col, tile))
        return tiles

    def export_tiles(self, basename, ext, tiles):
        output_folder = f'{self.output}/{basename}'
        os.makedirs(output_folder, exist_ok=True)
        for row, col, tile in tiles:
            cv2.imwrite(f'{output_folder}/{basename}_{row:04d}_{col:04d}{ext}',
                        tile)

    def process_image(self, filename):
        basename, ext = os.path.splitext(os.path.basename(filename))
        assert os.path.exists(filename)
        image = pyvips.Image.new_from_file(filename).numpy()
        # res = pyvips.Image.new_from_file(filename).resize(1 / 20).numpy()
        tiles = self.generate_tiles(image)
        tqdm.write(f'{basename} number of tiles {len(tiles)}, ', end='')
        tiles = self.remove_empty_tiles(tiles)
        tqdm.write(f'{basename} number of non-empty tiles {len(tiles)}')
        self.export_tiles(basename, ext, tiles)

    def is_empty(self, tile):
        if np.sum(tile) == 0:
            return True
        tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        # _, blur = cv2.GaussianBlur(tile, (5, 5), 0)
        _, res = cv2.threshold(tile, 10, 255,
                               cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
        relative_area = res.sum() / 255
        return relative_area < self.threshold

    def remove_empty_tiles(self, tiles):
        tiles = [tile for tile in tiles if not self.is_empty(tile[2])]
        return tiles

    def run(self):
        filenames = sorted(glob(f'{self.input}/*.png'))
        os.makedirs(self.output, exist_ok=True)
        log = f'{self.output}/processed.txt'
        try:
            with open(log) as f:
                processed = f.readlines()
                processed = [p.rstrip() for p in processed]
        except:
            processed = []
        print('processed', processed)
        pbar = tqdm(filenames)
        try:
            for filename in pbar:
                basename = os.path.basename(filename)
                pbar.set_postfix(dict(filename=basename))
                if basename in processed:
                    tqdm.write(f'file already processed, ignoring')
                    continue
                self.process_image(filename)
                processed.append(basename)
        finally:
            with open(log, 'w') as f:
                f.write('\n'.join(processed) + '\n')


def main():
    size = 256
    overlap = 0.1
    threshold = 0.5
    input = 'data/original/train_images'
    output = 'data/tiles/images'
    tg = TileGenerator(input, output, size, overlap, threshold)
    tg.run()


if __name__ == '__main__':
    main()
