import sys

import cv2
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import pyvips

def generate_tiles(image, tile_size=224, overlap=0.1):
    height, width = image.shape[:2]
    step = int(tile_size * (1 - overlap))
    rows = height // step + 1 if height % step > 0 else 0
    cols = width // step + 1 if width % step > 0 else 0
    tiles = []
    for row in range(rows):
        for col in range(cols):
            top, left = step * row, step * col
            bottom, right = top + tile_size, left + tile_size
            tile = image[top:bottom, left:right, :]
            h, w = tile.shape[:2]
            if h < tile_size or w < tile_size:
                tile = np.pad(tile,
                              ((0, tile_size - h), (0, tile_size - w), (0, 0)))
            tiles.append((row, col, tile))
    return tiles


def export_tiles(output_folder, basename, ext, tiles):
    os.makedirs(output_folder, exist_ok=True)
    for row, col, tile in tiles:
        cv2.imwrite(f'{output_folder}/{basename}_{row:04d}_{col:04d}{ext}',
                    tile)


def process_image(filename, output_folder, tile_size=224, overlap=0.1):
    basename, ext = os.path.splitext(os.path.basename(filename))
    assert os.path.exists(filename)
    # image = cv2.imread(filename)
    image = pyvips.Image.new_from_file(filename).numpy()
    # assert image is not None, f'Image {filename} not found'
    # res = pyvips.Image.new_from_file(filename).resize(1 / 20).numpy()
    tiles = generate_tiles(image, tile_size=tile_size, overlap=overlap)
    tqdm.write(f'{basename} number of tiles {len(tiles)}')
    tiles = remove_empty_tiles(tiles)
    tqdm.write(f'{basename} number of non-empty tiles {len(tiles)}')
    folder = (f'{output_folder}/size_{tile_size}_overlap_'
              f'{int(overlap * 100)}/{basename}')
    export_tiles(folder, basename, ext, tiles)


def is_empty(tile, area_threshold=0.05):
    if np.sum(tile) == 0:
        return True
    tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    # _, blur = cv2.GaussianBlur(tile, (5, 5), 0)
    _, res = cv2.threshold(tile, 10, 255,
                           cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
    relative_area = res.sum() / 255
    return relative_area < area_threshold


def remove_empty_tiles(tiles):
    tiles = [tile for tile in tiles if not is_empty(tile[2])]
    return tiles


def process_image_folder(input, output):
    filenames = sorted(glob(f'{input}/*.png'))
    tile_size = 224
    overlap = 0.1
    log = f'processed.txt'
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
            process_image(filename, output, tile_size, overlap)
            processed.append(basename)
    finally:
        with open(log, 'w') as f:
            f.write('\n'.join(processed) + '\n')


def main():
    input = 'data/original/train_thumbnails'
    output = 'data/tiles/thumbnails'
    process_image_folder(input, output)


if __name__ == '__main__':
    main()
