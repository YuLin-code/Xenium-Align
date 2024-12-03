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
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from alignment_data_process import * 
from util_function import * 
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='the model of scoring and ranking')

    #parameter settings
    parser.add_argument('-pixel_size', type=float, nargs='+', default=[0.2125], help='pixel size in the morphology.ome.tif image file (in µm).')
    parser.add_argument('-crop_radius_pixel', type=int, nargs='+', default=[400], help='the pixel value to crop each nucleus.')
    parser.add_argument('-center_move_pixel', type=int, nargs='+', default=[300], help='the pixel value to move in each cropped patch.')
    parser.add_argument('-crop_image_resize', type=int, nargs='+', default=[224], help='the resize value of the cropped image for checking.')
    parser.add_argument('-sample', type=str, nargs='+', default=['f59'], help='which sample to check.')
    parser.add_argument('-check_cell_num', type=int, nargs='+', default=[10], help='the check cell number.')
    parser.add_argument('-channel_cellpose', type=int, nargs='+', default=[1], help='the channel_cellpose value.')
    parser.add_argument('-min_size', type=int, nargs='+', default=[15], help='the min_size value.')
    parser.add_argument('-flow_threshold', type=float, nargs='+', default=[0.8], help='the flow_threshold value.')
    parser.add_argument('-data_file_path', type=str, nargs='+', default=['./Dataset/'], help='the xenium data file path.')
    parser.add_argument('-save_check_sort_image_num', type=int, nargs='+', default=[5], help='the save check sort image number.')
    parser.add_argument('-graph_source_str', type=str, nargs='+', default=['cell'], help='which mode to build delaunay graph.')
    parser.add_argument('-fig_size', type=int, nargs='+', default=[10], help='the fig size to plot graph.')
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
    pixel_size = args.pixel_size[0]
    crop_radius_pixel = args.crop_radius_pixel[0]
    center_move_pixel = args.center_move_pixel[0]
    crop_image_resize = args.crop_image_resize[0]
    sample = args.sample[0]
    check_cell_num = args.check_cell_num[0]
    channel_cellpose = args.channel_cellpose[0]
    min_size = args.min_size[0]
    flow_threshold = args.flow_threshold[0]
    data_file_path = args.data_file_path[0]
    save_check_sort_image_num = args.save_check_sort_image_num[0]
    graph_source_str = args.graph_source_str[0]
    fig_size = args.fig_size[0]
    device = args.device[0]
    
    #load os
    data_path, mip_ome_tif_data_path, sample, he_img_path = load_os(sample, data_file_path)
    cellpose_save_path = './'+sample+'_cellpose_channel_cellpose'+str(channel_cellpose)+'_flow_threshold'+str(flow_threshold)+'_min_size'+str(min_size)+'/'
    image_check_path = './'+sample+'_image_check/'
    
    #set os
    output_path = './C'+sample+'_CCell'+str(check_cell_num)+'_crop_patch_Rp'+str(crop_radius_pixel)+'_center_move_Rp'+str(center_move_pixel)+'_channel_cellpose'+str(channel_cellpose)+'_flow_threshold'+str(flow_threshold)+'_min_size'+str(min_size)
    score_save_path = output_path+'/score_save/'
    if not os.path.exists(score_save_path):
        os.makedirs(score_save_path)
    value_save_path = output_path+'/value_save/'
    if not os.path.exists(value_save_path):
        os.makedirs(value_save_path)
    crop_save_path = output_path+'/crop_Rp'+str(crop_radius_pixel)+'_save/'
    if not os.path.exists(crop_save_path):
        os.makedirs(crop_save_path)
    keypoints_save_path = output_path+'/keypoints_save/'
    if not os.path.exists(keypoints_save_path):
        os.makedirs(keypoints_save_path)
    graph_save_path = output_path+'/graph_save/'
    if not os.path.exists(graph_save_path):
        os.makedirs(graph_save_path)
    
    #load sdata
    sdata = spatialdata_io.xenium(data_path,cells_as_circles=True)

    #load pillow morphology images
    tif_image_init_affine, segment_label_image_init_affine, xenium_mip_ome_tif_image = initialize_tif_segment_xenium_image(he_img_path, cellpose_save_path, sample, mip_ome_tif_data_path, channel_cellpose, flow_threshold, min_size)
    
    #load values for scoring and ranking
    transform_ratio, segment_nucleus_pixel_values_pd, xenium_nucleus_pixel_values_pd = load_segment_xenium_pixel_value(cellpose_save_path, value_save_path, sample, sdata, pixel_size)
    
    #set check_cell_list
    check_cell_list = random.sample(range(0, segment_nucleus_pixel_values_pd.shape[0]), check_cell_num)
    
    #set learning_name
    learning_name = 'C'+sample+'_CCell'+str(check_cell_num)+'_cellpose_segment'

    #time_computing
    start_time = time.time()
    print("Scoring and Ranking Check. Start Time: %s seconds" %(start_time))
    
    #Scoring and Ranking Functions
    score_and_rank_cellpose_segment_nucleus_multi_angle(sdata, sample, pixel_size, crop_radius_pixel, center_move_pixel, tif_image_init_affine, segment_label_image_init_affine, xenium_mip_ome_tif_image, learning_name, output_path, transform_ratio, crop_image_resize, segment_nucleus_pixel_values_pd, check_cell_list, save_check_sort_image_num)
    check_rank_top5_multi_angle(output_path, check_cell_num, sample, learning_name)
    pixel_values_filtered_by_multi_angles_for_xenium_explorer(sample, value_save_path, cellpose_save_path, keypoints_save_path, learning_name, check_cell_num, check_cell_list, xenium_nucleus_pixel_values_pd)
    filtered_keypoints_num, xenium_segment_explorer_np = pixel_values_filtered_by_multi_angles_and_graph_for_xenium_explorer(sample, graph_save_path, keypoints_save_path, learning_name, graph_source_str, fig_size)
    filtered_keypoints_inverse_num, xenium_segment_explorer_inverse_np = pixel_values_filtered_by_multi_angles_and_graph_inverse_for_xenium_explorer(sample, graph_save_path, keypoints_save_path, learning_name, graph_source_str, fig_size)
    count_row = save_xenium_segment_explorer_keypoints(keypoints_save_path,sample,filtered_keypoints_num,xenium_segment_explorer_np,filtered_keypoints_inverse_num,xenium_segment_explorer_inverse_np,learning_name)
    if count_row <= 30:
        pixel_values_min3_angle_for_xenium_explorer(sample, value_save_path, cellpose_save_path, keypoints_save_path, learning_name, check_cell_num, check_cell_list, xenium_nucleus_pixel_values_pd)
        filtered_keypoints_num, xenium_segment_explorer_np = pixel_values_filtered_by_multi_angles_min3_and_graph_for_xenium_explorer(sample, graph_save_path, keypoints_save_path, learning_name, graph_source_str, fig_size)
        filtered_keypoints_inverse_num, xenium_segment_explorer_inverse_np = pixel_values_filtered_by_multi_angles_min3_and_graph_inverse_for_xenium_explorer(sample, graph_save_path, keypoints_save_path, learning_name, graph_source_str, fig_size)
        save_xenium_segment_by_multi_angles_min3_explorer_keypoints(keypoints_save_path,sample,filtered_keypoints_num,xenium_segment_explorer_np,filtered_keypoints_inverse_num,xenium_segment_explorer_inverse_np,learning_name)
    
    #delete folders
    shutil.rmtree(score_save_path)
    shutil.rmtree(value_save_path)
    shutil.rmtree(crop_save_path)
    shutil.rmtree(graph_save_path)
    shutil.rmtree(image_check_path)
    shutil.rmtree(cellpose_save_path)
    
    #time_computing
    end_time = time.time()
    print("Scoring and Ranking Check. End Time: %s seconds" %(end_time))
    print("Scoring and Ranking Check Done. Total Running Time: %s seconds" %(end_time - start_time))
    
