import warnings
import time
warnings.filterwarnings("ignore")
import spatialdata_io
from pathlib import Path
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import numpy as np
import torch
import random
import argparse
import pandas as pd
#import tifffile
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from alignment_data_process import *

def parse_args():
    parser = argparse.ArgumentParser(description='data preprocess checking')
    
    #parameter settings
    parser.add_argument('-sample', type=str, nargs='+', default=['f59'], help='which sample to check.')
    parser.add_argument('-data_file_path', type=str, nargs='+', default=['./Dataset/'], help='the xenium data file path.')
    parser.add_argument('-device', type=str, nargs='+', default=['cpu'], help='the device used for training model: cpu or cuda.')
    
    args = parser.parse_args()
    return args

def seed_random(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    
    #random setting
    seed_random(300)

    #load settings
    args = parse_args()
    sample = args.sample[0]
    data_file_path = args.data_file_path[0]
    device = args.device[0]
    
    #load os
    data_path, mip_ome_tif_data_path, sample, he_img_path = load_os(sample, data_file_path)

    #set os
    image_check_path = './'+sample+'_image_check/'
    if not os.path.exists(image_check_path):
        os.makedirs(image_check_path)
    
    #load mip_ome_tif_image 
    mip_ome_tif_r = tifffile.TiffReader(mip_ome_tif_data_path)
    mip_ome_tif_image = mip_ome_tif_r.pages[0].asarray()
    mip_ome_tif_rows, mip_ome_tif_cols = mip_ome_tif_image.shape[:2]
    mip_ome_tif_image_gray = normalization_min_max_grayscale(mip_ome_tif_image)
    mip_ome_tif_image_gray_rgb = np.expand_dims(mip_ome_tif_image_gray,2).repeat(3,axis=2)
    mip_ome_tif_image_gray_rgb_uint8 = np.uint8(mip_ome_tif_image_gray_rgb)

    #pillow setting
    mip_ome_tif_image_gray_rgb_uint8_pillow = Image.fromarray(mip_ome_tif_image_gray_rgb_uint8)

    #check mode
    if mip_ome_tif_image_gray_rgb_uint8_pillow.mode != 'RGB':
        mip_ome_tif_image_gray_rgb_uint8_pillow = mip_ome_tif_image_gray_rgb_uint8_pillow.convert('RGB')

    #save mip_ome_tif_image
    mip_ome_tif_image_gray_rgb_uint8_pillow.save(image_check_path+sample+'_cell_morphology_image.jpg', "JPEG")

    #load HE init affine
    tif_image_pillow = Image.open(he_img_path)
    rotated_270_image = tif_image_pillow.transpose(Image.ROTATE_270)
    flipped_lr_image = rotated_270_image.transpose(Image.FLIP_LEFT_RIGHT)
    tif_image_init_affine_pillow = flipped_lr_image
    tif_image_init_affine_pillow.save(image_check_path+sample+'_initialied_HE_image.jpg', "JPEG")

    print('Please go to the '+image_check_path+' folder and check whether it is consistent of the image layout between the cell morphology image with black-and-white color and the H&E-stained image!')