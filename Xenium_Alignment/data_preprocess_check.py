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
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def parse_args():
    parser = argparse.ArgumentParser(description='data preprocess checking')
    
    #parameter settings
    parser.add_argument('-sample', type=str, nargs='+', default=['f59'], help='which sample to check.')
    parser.add_argument('-preservation_method', type=str, nargs='+', default=['ff'], help='the preservation method used for tissue section. eg: ff/ffpe')
    parser.add_argument('-data_file_path', type=str, nargs='+', default=['../Dataset/'], help='the xenium data file path.')
    
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
    preservation_method = args.preservation_method[0]
    data_file_path = args.data_file_path[0]
    
    #print('sample',sample)
    #print('preservation_method',preservation_method)
    #print('data_file_path',data_file_path)
    
    #set os
    image_check_path = './'+sample+'_image_check/'
    if not os.path.exists(image_check_path):
        os.makedirs(image_check_path)

    if preservation_method == 'ff':
        #load os
        data_path, mip_ome_tif_data_path, sample, he_img_path = load_os_ff_sample(sample, data_file_path)
    elif preservation_method == 'ffpe':
        if sample == 'kidney_cancer':
            data_file_path = data_file_path+'kidney_cancerprcc_data/'
        elif sample == 'kidney_nondiseased':
            data_file_path = data_file_path+'kidney_nondiseased_data/'
        #load os
        data_path, mip_ome_tif_data_path, sample, he_img_path = load_os_ffpe_sample(sample, data_file_path, image_check_path)
    
    ###load mip_ome_tif_image 
    mip_ome_tif_r = tifffile.TiffReader(mip_ome_tif_data_path)
    mip_ome_tif_image = mip_ome_tif_r.pages[0].asarray()
    mip_ome_tif_rows, mip_ome_tif_cols = mip_ome_tif_image.shape[:2]
    mip_ome_tif_image_gray = normalization_min_max_grayscale(mip_ome_tif_image)
    mip_ome_tif_image_gray_rgb = np.expand_dims(mip_ome_tif_image_gray,2).repeat(3,axis=2)
    mip_ome_tif_image_gray_rgb_uint8 = np.uint8(mip_ome_tif_image_gray_rgb)
    #load shape
    dapi_image_height = mip_ome_tif_image_gray_rgb_uint8.shape[0]
    dapi_image_width = mip_ome_tif_image_gray_rgb_uint8.shape[1]
    #print('dapi_image_height',dapi_image_height)
    #print('dapi_image_width',dapi_image_width)
    #pillow setting
    mip_ome_tif_image_gray_rgb_uint8_pillow = Image.fromarray(mip_ome_tif_image_gray_rgb_uint8)
    #check mode
    if mip_ome_tif_image_gray_rgb_uint8_pillow.mode != 'RGB':
        mip_ome_tif_image_gray_rgb_uint8_pillow = mip_ome_tif_image_gray_rgb_uint8_pillow.convert('RGB')
    #save mip_ome_tif_image
    #mip_ome_tif_image_gray_rgb_uint8_pillow.save(image_check_path+sample+'_cell_morphology_image.jpg', "JPEG")

    ###load HE
    tif_image_pillow = Image.open(he_img_path)
    #check mode
    if tif_image_pillow.mode != 'RGB':
        tif_image_pillow = tif_image_pillow.convert('RGB')
    #load shape
    tif_image_width, tif_image_height = tif_image_pillow.size
    #print('tif_image_width',tif_image_width)
    #print('tif_image_height',tif_image_height)
    if preservation_method == 'ffpe':
        os.remove(he_img_path)
    
    init_rotate_list = []
    #init rotate HE
    if dapi_image_height > dapi_image_width and tif_image_height < tif_image_width:
        tif_image_pillow_init = tif_image_pillow.transpose(Image.ROTATE_90)
        init_rotate_list.append('rotate90')
    elif dapi_image_height < dapi_image_width and tif_image_height > tif_image_width:
        tif_image_pillow_init = tif_image_pillow.transpose(Image.ROTATE_90)
        init_rotate_list.append('rotate90')
    else:
        tif_image_pillow_init = tif_image_pillow
        init_rotate_list.append('rotate0')
    tif_image_init_width, tif_image_init_height = tif_image_pillow_init.size
    #print('tif_image_init_width',tif_image_init_width)
    #print('tif_image_init_height',tif_image_init_height)
    
    ###load resize 
    dapi_image_min = min(dapi_image_width, dapi_image_height)
    tif_image_min = min(tif_image_init_width, tif_image_init_height)
    if dapi_image_min > tif_image_min:
        dapi_image_resize_pillow = mip_ome_tif_image_gray_rgb_uint8_pillow.resize((tif_image_init_width,tif_image_init_height))
        tif_image_pillow_resize = tif_image_pillow_init
    elif dapi_image_min < tif_image_min:
        dapi_image_resize_pillow = mip_ome_tif_image_gray_rgb_uint8_pillow
        tif_image_pillow_resize = tif_image_pillow_init.resize((dapi_image_width,dapi_image_height))
    else:
        dapi_image_resize_pillow = mip_ome_tif_image_gray_rgb_uint8_pillow
        tif_image_pillow_resize = tif_image_pillow_init
    dapi_image_resize_width, dapi_image_resize_height = dapi_image_resize_pillow.size
    tif_image_resize_width, tif_image_resize_height = tif_image_pillow_resize.size
    #print('dapi_image_resize_width',dapi_image_resize_width)
    #print('dapi_image_resize_height',dapi_image_resize_height)
    #print('tif_image_resize_width',tif_image_resize_width)
    #print('tif_image_resize_height',tif_image_resize_height)
    
    ###rotate check
    rotate_list = ['original','flipLR','flipTB','rotate180']
    rotate_mse_list = []
    
    dapi_image_resize_pillow_np = np.array(dapi_image_resize_pillow)
    dapi_image_resize_pillow.save(image_check_path+sample+'_Xenium_DAPI_image.jpg', "JPEG")
    
    for rotate_num in range(len(rotate_list)):
        current_rotate = rotate_list[rotate_num]
        #print('current_rotate',current_rotate)
        if current_rotate == 'original':
            current_tif_image_pillow_resize_rotate = tif_image_pillow_resize
            current_tif_image_pillow_resize_rotate_np = np.array(current_tif_image_pillow_resize_rotate)
            
            #calculate metrics rgb
            current_mse_rgb = compare_mse(current_tif_image_pillow_resize_rotate_np, dapi_image_resize_pillow_np)
            rotate_mse_list.append(current_mse_rgb)
            
            #save tif_image_rotate
            current_tif_image_pillow_resize_rotate.save(image_check_path+sample+'_he_rotate_'+current_rotate+'_image.jpg', "JPEG")
            
        elif current_rotate == 'flipLR':
            current_tif_image_pillow_resize_rotate = tif_image_pillow_resize.transpose(Image.FLIP_LEFT_RIGHT)
            current_tif_image_pillow_resize_rotate_np = np.array(current_tif_image_pillow_resize_rotate)
            
            #calculate metrics rgb
            current_mse_rgb = compare_mse(current_tif_image_pillow_resize_rotate_np, dapi_image_resize_pillow_np)
            rotate_mse_list.append(current_mse_rgb)
    
            #save tif_image_rotate
            current_tif_image_pillow_resize_rotate.save(image_check_path+sample+'_he_rotate_'+current_rotate+'_image.jpg', "JPEG")
    
        elif current_rotate == 'flipTB':
            current_tif_image_pillow_resize_rotate = tif_image_pillow_resize.transpose(Image.FLIP_TOP_BOTTOM)
            current_tif_image_pillow_resize_rotate_np = np.array(current_tif_image_pillow_resize_rotate)
            
            #calculate metrics rgb
            current_mse_rgb = compare_mse(current_tif_image_pillow_resize_rotate_np, dapi_image_resize_pillow_np)
            rotate_mse_list.append(current_mse_rgb)
    
            #save tif_image_rotate
            current_tif_image_pillow_resize_rotate.save(image_check_path+sample+'_he_rotate_'+current_rotate+'_image.jpg', "JPEG")
    
        elif current_rotate == 'rotate180':
            current_tif_image_pillow_resize_rotate = tif_image_pillow_resize.transpose(Image.ROTATE_180)
            current_tif_image_pillow_resize_rotate_np = np.array(current_tif_image_pillow_resize_rotate)
            
            #calculate metrics rgb
            current_mse_rgb = compare_mse(current_tif_image_pillow_resize_rotate_np, dapi_image_resize_pillow_np)
            rotate_mse_list.append(current_mse_rgb)
    
            #save tif_image_rotate
            current_tif_image_pillow_resize_rotate.save(image_check_path+sample+'_he_rotate_'+current_rotate+'_image.jpg', "JPEG")
    
    ###rotate_mse_values_save
    rotate_mse_list_np = np.array(rotate_mse_list).reshape(1,-1)
    row_max_index = np.argmax(rotate_mse_list_np)
    rotate_name = rotate_list[row_max_index]
    
    rotate_full_list = []
    for rotate_num in range(len(rotate_list)):
        current_rotate = rotate_list[rotate_num]
        current_rotate_full = init_rotate_list[0]+'_'+current_rotate
        rotate_full_list.append(current_rotate_full)
        #print('current_rotate',current_rotate)
        if current_rotate == rotate_name:
            current_file_name = image_check_path+sample+'_he_rotate_'+current_rotate+'_image.jpg'
            new_file_name = image_check_path+sample+'_HE_image.jpg'
            os.rename(current_file_name, new_file_name)
        else:
            current_file_name = image_check_path+sample+'_he_rotate_'+current_rotate+'_image.jpg'
            os.remove(current_file_name)
    
    rotate_mse_list_pd = pd.DataFrame(rotate_mse_list_np,index=['mse'],columns=rotate_full_list)
    rotate_mse_list_pd.to_csv(image_check_path+sample+'_he_rotate_mse_values.csv')
    
    print('Please go to the '+image_check_path+' folder and check whether it is consistent of the image layout between the Xenium DAPI image with black-and-white color and the H&E-stained image!')