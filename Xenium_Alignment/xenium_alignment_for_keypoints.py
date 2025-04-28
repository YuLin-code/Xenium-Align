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
from util_function import * 

def parse_args():
    parser = argparse.ArgumentParser(description='the model of xenium-alignment')

    #parameter settings
    parser.add_argument('-pixel_size', type=float, nargs='+', default=[0.2125], help='pixel size in the morphology.ome.tif image file (in µm).')
    parser.add_argument('-crop_radius_pixel', type=int, nargs='+', default=[400], help='the pixel value to crop each nucleus.')
    parser.add_argument('-center_move_pixel', type=int, nargs='+', default=[300], help='the pixel value to move of each cropped patch.')
    parser.add_argument('-crop_image_resize', type=int, nargs='+', default=[224], help='the resize value of the cropped image for checking.')
    parser.add_argument('-sample', type=str, nargs='+', default=['f59'], help='which sample to check.')
    parser.add_argument('-check_cell_num', type=int, nargs='+', default=[30], help='the check cell number in each epoch.')
    parser.add_argument('-channel_cellpose', type=int, nargs='+', default=[1], help='the channel_cellpose value.')
    parser.add_argument('-min_size', type=int, nargs='+', default=[15], help='the min_size value.')
    parser.add_argument('-flow_threshold', type=float, nargs='+', default=[0.8], help='the flow_threshold value.')
    parser.add_argument('-prob_thresh', type=float, nargs='+', default=[0.3], help='the prob_thresh value.')
    parser.add_argument('-data_file_path', type=str, nargs='+', default=['../Dataset/'], help='the xenium data file path.')
    parser.add_argument('-graph_source_str', type=str, nargs='+', default=['cell'], help='which mode to build delaunay triangulation graph.')
    parser.add_argument('-fig_size', type=int, nargs='+', default=[10], help='the fig size to plot graph.')
    parser.add_argument('-device', type=str, nargs='+', default=['cpu'], help='the device used for training model: cpu or cuda.')
    parser.add_argument('-remove_segment_edge', type=int, nargs='+', default=[100], help='the pixel value to remove edge of segmentation image.')
    parser.add_argument('-mip_ome_extract_ratio', type=float, nargs='+', default=[0.125], help='the ratio of the minimum value in width and height of DAPI-stained image to set the radius for search region.')
    parser.add_argument('-mip_ome_extract_min', type=int, nargs='+', default=[20], help='the minimum number of cells in the search region that would be kept.')
    parser.add_argument('-segment_method', type=str, nargs='+', default=['cellpose'], help='the nucleus segmentation model used for H&E image. eg: cellpose/stardist')
    parser.add_argument('-epoch_num', type=int, nargs='+', default=[30], help='the maximum number of epochs used to image alignment.')
    parser.add_argument('-keypoints_min_num', type=int, nargs='+', default=[15], help='the minimum number of keypoints to output.')
    parser.add_argument('-overlap_threshold_ave', type=float, nargs='+', default=[0.9], help='the threshold value of average overlap in nucleus polygon matching.')
    parser.add_argument('-overlap_threshold_min', type=float, nargs='+', default=[0.92], help='the threshold value of minimum overlap in nucleus polygon matching.')
    parser.add_argument('-overlap_type', type=str, nargs='+', default=['overlap_ave'], help='which overlap type is used in nucleus polygon matching.')
    parser.add_argument('-preservation_method', type=str, nargs='+', default=['ff'], help='the preservation method used for tissue section. eg: ff/ffpe')
    
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
    prob_thresh = args.prob_thresh[0]
    data_file_path = args.data_file_path[0]
    graph_source_str = args.graph_source_str[0]
    fig_size = args.fig_size[0]
    device = args.device[0]
    remove_segment_edge = args.remove_segment_edge[0]
    mip_ome_extract_ratio = args.mip_ome_extract_ratio[0]
    mip_ome_extract_min = args.mip_ome_extract_min[0]
    segment_method = args.segment_method[0]
    epoch_num = args.epoch_num[0]
    keypoints_min_num = args.keypoints_min_num[0]
    overlap_threshold_ave = args.overlap_threshold_ave[0]
    overlap_threshold_min = args.overlap_threshold_min[0]
    overlap_type = args.overlap_type[0]
    preservation_method = args.preservation_method[0]

    print('pixel_size',pixel_size)
    print('crop_radius_pixel',crop_radius_pixel)
    print('center_move_pixel',center_move_pixel)
    print('crop_image_resize',crop_image_resize)
    print('sample',sample)
    print('check_cell_num',check_cell_num)
    print('channel_cellpose',channel_cellpose)
    print('min_size',min_size)
    print('flow_threshold',flow_threshold)
    print('prob_thresh',prob_thresh)
    print('data_file_path',data_file_path)
    print('graph_source_str',graph_source_str)
    print('fig_size',fig_size)
    print('device',device)
    print('remove_segment_edge',remove_segment_edge)
    print('mip_ome_extract_ratio',mip_ome_extract_ratio)
    print('mip_ome_extract_min',mip_ome_extract_min)
    print('segment_method',segment_method)
    print('epoch_num',epoch_num)
    print('keypoints_min_num',keypoints_min_num)
    print('overlap_threshold_ave',overlap_threshold_ave)
    print('overlap_threshold_min',overlap_threshold_min)
    print('overlap_type',overlap_type)
    print('preservation_method',preservation_method)
    
    #set os
    if segment_method == 'cellpose':
        segment_save_path = './'+sample+'_cellpose_channel_cellpose'+str(channel_cellpose)+'_flow_threshold'+str(flow_threshold)+'_min_size'+str(min_size)+'/'
        segmentation_label_file = segment_save_path+'channel_cellpose'+str(channel_cellpose)+'_flow_threshold'+str(flow_threshold)+'_min_size'+str(min_size)+'_tif_image_segmented_cellpose.csv'
        output_path = './C'+sample+'_CCell'+str(check_cell_num)+'_crop'+str(crop_radius_pixel)+'_move'+str(center_move_pixel)+'_remove'+str(remove_segment_edge)+'_extract'+str(round(mip_ome_extract_ratio,3))+'_'+str(mip_ome_extract_min)+'_cellpose_channel'+str(channel_cellpose)+'_threshold'+str(flow_threshold)+'_min'+str(min_size)+'_epoch'+str(epoch_num)+'_overA'+str(overlap_threshold_ave)+'_M'+str(overlap_threshold_min)+'_keyMin'+str(keypoints_min_num)+'_'+str(overlap_type)
    elif segment_method == 'stardist':
        segment_save_path = './'+sample+'_stardist_prob_thresh'+str(prob_thresh)+'/'
        segmentation_label_file = segment_save_path+'prob_thresh'+str(prob_thresh)+'_tif_image_segmented_stardist.csv'
        output_path = './C'+sample+'_CCell'+str(check_cell_num)+'_crop'+str(crop_radius_pixel)+'_move'+str(center_move_pixel)+'_remove'+str(remove_segment_edge)+'_extract'+str(round(mip_ome_extract_ratio,3))+'_'+str(mip_ome_extract_min)+'_stardist_prob'+str(prob_thresh)+'_epoch'+str(epoch_num)+'_overA'+str(overlap_threshold_ave)+'_M'+str(overlap_threshold_min)+'_keyMin'+str(keypoints_min_num)+'_'+str(overlap_type)
    
    score_save_path = output_path+'/score_save/'
    if not os.path.exists(score_save_path):
        os.makedirs(score_save_path)
    value_save_path = output_path+'/value_save/'
    if not os.path.exists(value_save_path):
        os.makedirs(value_save_path)
    keypoints_save_path = output_path+'/keypoints_save/'
    if not os.path.exists(keypoints_save_path):
        os.makedirs(keypoints_save_path)
    graph_save_path = output_path+'/graph_save/'
    if not os.path.exists(graph_save_path):
        os.makedirs(graph_save_path)
    image_check_path = './'+sample+'_image_check/'
    
    if preservation_method == 'ff':
        #load os
        data_path, mip_ome_tif_data_path, sample, he_img_path = load_os_ff_sample(sample, data_file_path)
    elif preservation_method == 'ffpe':
        if sample == 'kidney_cancer':
            data_file_path = data_file_path+'kidney_cancerprcc_data/'
        elif sample == 'kidney_nondiseased':
            data_file_path = data_file_path+'kidney_nondiseased_data/'
        #load os
        data_path, mip_ome_tif_data_path, sample, he_img_path = load_os_ffpe_sample(sample, data_file_path, segment_save_path)
    
    #load sdata
    sdata = spatialdata_io.xenium(data_path,cells_as_circles=True)
    #print('sdata_exp shape is',sdata.tables.data['table'].X.A.shape)

    #load morphology images
    tif_image_init_affine, segment_label_image_init_affine, xenium_mip_ome_tif_image, segment_init_affine_image_width, segment_init_affine_image_height, segment_image_width, segment_image_height, xenium_image_width, xenium_image_height = initialize_tif_segment_xenium_image(he_img_path, segment_save_path, segment_method, sample, mip_ome_tif_data_path, channel_cellpose, flow_threshold, min_size, prob_thresh, image_check_path)
    
    #load values for scoring and ranking
    transform_ratio, segment_nucleus_pixel_values_pd, xenium_nucleus_pixel_values_pd, segment_nucleus_pixel_values_num = load_segment_xenium_pixel_value(segment_save_path, segment_method, value_save_path, sample, sdata, pixel_size, segment_image_width, segment_image_height, xenium_image_width, xenium_image_height, remove_segment_edge, crop_radius_pixel, center_move_pixel, image_check_path)
    #print('transform_ratio',transform_ratio)
    
    #set learning_name
    learning_name = 'C'+sample+'_CCell'+str(check_cell_num)+'_'+segment_method+'_segment'
    #print('learning_name', learning_name)

    #time_computing
    start_time = time.time()
    print("Scoring and Ranking Check. Start Time: %s seconds" %(start_time))

    #set check_cell_list
    check_cell_list_init, check_cell_left_list_init = check_cell_list_init_function(segment_nucleus_pixel_values_num, check_cell_num, value_save_path)
    
    #set keypoints_save_list
    keypoints_save_list = []
    current_keypoints_num = 0
    
    keypoints_segment_index_save_list = []
    epoch_num_list = []
    xenium_search_index_list = []
    xenium_barcode_list = []
    xenium_search_crop_list = []

    multi_angles_count_row_list = []
    current_epoch_num_list = []
    
    for current_epoch_num in range(epoch_num):
        
        print('current_epoch_num',current_epoch_num)
        current_epoch_num_list.append('epoch_'+str(current_epoch_num))
        
        #load_xenium_search_region
        check_cell_list, check_cell_list_left = load_xenium_search_region_for_each_segment_nucleus(segment_nucleus_pixel_values_pd, xenium_nucleus_pixel_values_pd, tif_image_init_affine, segment_label_image_init_affine, xenium_mip_ome_tif_image, check_cell_list_init, check_cell_left_list_init, mip_ome_extract_ratio, segment_init_affine_image_width, segment_init_affine_image_height, xenium_image_width, xenium_image_height, transform_ratio, mip_ome_extract_min, output_path, current_epoch_num, image_check_path, sample)
        
        #list update
        check_cell_list_init, check_cell_left_list_init = check_cell_list_update_function(check_cell_list, check_cell_list_left, check_cell_num, value_save_path, current_epoch_num)
        
        ##Scoring and Ranking Functions
        score_and_rank_xenium_segment_nuclei_multi_angle(sdata, sample, pixel_size, crop_radius_pixel, center_move_pixel, tif_image_init_affine, segment_label_image_init_affine, xenium_mip_ome_tif_image, learning_name, output_path, transform_ratio, crop_image_resize, segment_nucleus_pixel_values_pd, check_cell_list, mip_ome_extract_ratio, current_epoch_num, image_check_path, segment_init_affine_image_width, segment_init_affine_image_height)
        filtered_top1_barcode_multi_angle_list = check_rank_top1_multi_angle(output_path, check_cell_num, sample, learning_name, current_epoch_num)
        multi_angles_count_row, xenium_segment_multi_angles_keypoints, rank_filtered_top1_index_list_num, segment_center_pixel_value_original_use_list_np = pixel_values_filtered_by_multi_angles_for_xenium_explorer(sample, value_save_path, segment_save_path, segment_method, keypoints_save_path, learning_name, check_cell_num, check_cell_list, xenium_nucleus_pixel_values_pd, current_epoch_num, remove_segment_edge, crop_radius_pixel, center_move_pixel)
        multi_angles_count_row_list.append(multi_angles_count_row)
        
        #save xenium segment cell index
        segment_center_pixel_value_original_use_list_multi_angles = segment_center_pixel_value_original_use_list_np[rank_filtered_top1_index_list_num]
        #print('segment_center_pixel_value_original_use_list_multi_angles len is!!',len(segment_center_pixel_value_original_use_list_multi_angles))
        
        xenium_search_cell_index_list = list(range(check_cell_num))
        xenium_search_cell_index_np = np.array(xenium_search_cell_index_list)
        #print('xenium_search_cell_index_np',xenium_search_cell_index_np)
        xenium_search_index_current_epoch_np = xenium_search_cell_index_np[rank_filtered_top1_index_list_num]
        #print('xenium_search_index_current_epoch_np',xenium_search_index_current_epoch_np)
        
        check_cell_list_np = np.array(check_cell_list)
        xenium_search_crop_np = check_cell_list_np[rank_filtered_top1_index_list_num]
        
        #graph matching
        if multi_angles_count_row >= 4:
            print('The Delaunay triangulation graph matching is used in current epoch!')
            filtered_keypoints_num, xenium_segment_explorer_np, multi_angles_graph_count_row, graph_xenium_barcode_list = pixel_values_filtered_by_multi_angles_and_graph_for_xenium_explorer(sample, graph_save_path, keypoints_save_path, learning_name, graph_source_str, fig_size, current_epoch_num)
            filtered_keypoints_inverse_num, xenium_segment_explorer_inverse_np, multi_angles_graph_inverse_count_row, graph_inverse_xenium_barcode_list = pixel_values_filtered_by_multi_angles_and_graph_inverse_for_xenium_explorer(sample, graph_save_path, keypoints_save_path, learning_name, graph_source_str, fig_size, current_epoch_num)
            count_row, xenium_segment_explorer_keypoints, xenium_segment_explorer_keypoints_xenium_barcode_list = save_xenium_segment_explorer_keypoints(keypoints_save_path,sample,filtered_keypoints_num,xenium_segment_explorer_np,filtered_keypoints_inverse_num,xenium_segment_explorer_inverse_np, current_epoch_num, graph_xenium_barcode_list, graph_inverse_xenium_barcode_list)
        elif multi_angles_count_row < 4 and multi_angles_count_row > 1:
            print('The Delaunay triangulation graph matching is skipped in current epoch!')
            multi_angles_graph_count_row = multi_angles_count_row
            multi_angles_graph_inverse_count_row = multi_angles_count_row
            graph_xenium_barcode_list = list(range(multi_angles_count_row-1))
            graph_inverse_xenium_barcode_list = graph_xenium_barcode_list
            #print('graph_xenium_barcode_list',graph_xenium_barcode_list)
            #print('graph_inverse_xenium_barcode_list',graph_inverse_xenium_barcode_list)
            count_row, xenium_segment_explorer_keypoints, xenium_segment_explorer_keypoints_xenium_barcode_list = save_xenium_segment_explorer_keypoints(keypoints_save_path,sample,multi_angles_count_row,xenium_segment_multi_angles_keypoints,multi_angles_count_row,xenium_segment_multi_angles_keypoints, current_epoch_num, graph_xenium_barcode_list, graph_inverse_xenium_barcode_list)
        elif multi_angles_count_row == 1:
            print('There is no filtered top1 barcode in multi angles evaluation in current epoch!')
            multi_angles_graph_count_row = 1
            multi_angles_graph_inverse_count_row = 1
            count_row = 1
            kept_count_row = 1
            kept_count_min_row = 1
            
        if multi_angles_count_row > 1:
            #print('xenium_segment_explorer_keypoints_xenium_barcode_list',xenium_segment_explorer_keypoints_xenium_barcode_list)
            current_keypoints_columns_np = xenium_segment_explorer_keypoints[0,:].reshape(1,-1)
            current_xenium_segment_keypoints_np = xenium_segment_explorer_keypoints[1:,:]
            #print('current_xenium_segment_keypoints_np',current_xenium_segment_keypoints_np)
            #current_xenium_segment_keypoints_np = current_xenium_segment_keypoints_np.astype(float)
            current_xenium_segment_keypoints_np = current_xenium_segment_keypoints_np.astype(np.float64)
            #print('current_keypoints_columns_np',current_keypoints_columns_np)
            #print('current_xenium_segment_keypoints_np',current_xenium_segment_keypoints_np)
            
            #index_save_list
            current_segment_center_pixel_value_original_use_np = segment_center_pixel_value_original_use_list_multi_angles[xenium_segment_explorer_keypoints_xenium_barcode_list]
            current_xenium_search_index_current_epoch_np = xenium_search_index_current_epoch_np[xenium_segment_explorer_keypoints_xenium_barcode_list]
            filtered_top1_barcode_multi_angle_list_np = np.array(filtered_top1_barcode_multi_angle_list)
            current_filtered_top1_barcode_multi_angle_list_np = filtered_top1_barcode_multi_angle_list_np[xenium_segment_explorer_keypoints_xenium_barcode_list]
            xenium_search_crop_current_epoch_np = xenium_search_crop_np[xenium_segment_explorer_keypoints_xenium_barcode_list]
            
            ###polygon check
            #coords_list
            xenium_nucleus_exterior_coords_list, segment_nucleus_hull_list, xenium_nucleus_max_min_list, segment_nucleus_max_min_list, xenium_nucleus_barcode_list, segment_nucleus_index_list = xenium_segment_nucleus_coordinate_values(sdata, xenium_nucleus_pixel_values_pd, segment_nucleus_pixel_values_pd, segmentation_label_file, current_xenium_segment_keypoints_np, pixel_size, xenium_segment_explorer_keypoints_xenium_barcode_list, filtered_top1_barcode_multi_angle_list, image_check_path, segment_init_affine_image_width, segment_init_affine_image_height, sample)
            #coords_transform_list
            xenium_geometry_exterior_coords_transform_resize_list, segment_hull_coords_transform_resize_list, xenium_geometry_exterior_coords_transform_resize_patch_list, segment_hull_coords_transform_resize_patch_list = xenium_segment_nucleus_coordinate_transform_relative_resize(xenium_nucleus_exterior_coords_list, segment_nucleus_hull_list, xenium_nucleus_max_min_list, segment_nucleus_max_min_list, xenium_nucleus_barcode_list, segment_nucleus_index_list, output_path, xenium_nucleus_pixel_values_pd, xenium_mip_ome_tif_image, segment_nucleus_pixel_values_pd, segment_label_image_init_affine, crop_image_resize, current_epoch_num, transform_ratio, image_check_path, segment_init_affine_image_width, segment_init_affine_image_height, sample)
            #polygon_check_file
            xenium_segment_kept_index_np = polygon_check_function(xenium_geometry_exterior_coords_transform_resize_list, segment_hull_coords_transform_resize_list, xenium_geometry_exterior_coords_transform_resize_patch_list, segment_hull_coords_transform_resize_patch_list, output_path, value_save_path, learning_name, overlap_threshold_ave, overlap_threshold_min, overlap_type ,current_epoch_num)
            #print('xenium_segment_kept_index_np',xenium_segment_kept_index_np)
            
            if xenium_segment_kept_index_np.shape[0]>0:
                #kept_keypoints
                current_xenium_segment_keypoints_kept_np = current_xenium_segment_keypoints_np[xenium_segment_kept_index_np]
                current_xenium_segment_keypoints_kept_add_columns_np = np.vstack((current_keypoints_columns_np,current_xenium_segment_keypoints_kept_np))
                kept_count_row = save_polygon_kept_explorer_keypoints(keypoints_save_path, sample, current_xenium_segment_keypoints_kept_add_columns_np, current_epoch_num)
                #update_save_list
                current_keypoints_num = current_keypoints_num + (kept_count_row - 1)
                keypoints_save_list.append(current_xenium_segment_keypoints_kept_np)
                #index_save_list
                current_keypoints_segment_index = current_segment_center_pixel_value_original_use_np[xenium_segment_kept_index_np]
                keypoints_segment_index_save_list.append(current_keypoints_segment_index)
                current_epoch_num_np = np.repeat(current_epoch_num, xenium_segment_kept_index_np.shape[0])
                epoch_num_list.append(current_epoch_num_np)
                xenium_search_index_current_epoch_kept_np = current_xenium_search_index_current_epoch_np[xenium_segment_kept_index_np]
                xenium_search_index_list.append(xenium_search_index_current_epoch_kept_np)
                filtered_top1_barcode_multi_angle_kept_list_np = current_filtered_top1_barcode_multi_angle_list_np[xenium_segment_kept_index_np]
                xenium_barcode_list.append(filtered_top1_barcode_multi_angle_kept_list_np)
                xenium_search_crop_current_epoch_kept_np = xenium_search_crop_current_epoch_np[xenium_segment_kept_index_np]
                xenium_search_crop_list.append(xenium_search_crop_current_epoch_kept_np)
            else:
                print('There is no kept keypoints in current epoch!')
                kept_count_row = 0
        
        #save keypoints_count_row and run_time
        keypoints_count_row_np = np.array([multi_angles_count_row, multi_angles_graph_count_row, multi_angles_graph_inverse_count_row, count_row, kept_count_row]).reshape(1,-1)
        keypoints_count_row_np = keypoints_count_row_np - 1
        keypoints_count_row_pd = pd.DataFrame(keypoints_count_row_np,columns=['multi_angles_count_row','multi_angles_graph_count_row','multi_angles_graph_inverse_count_row','count_row_num','kept_count_row_num'], index=['keypoint_count_row'])
        keypoints_count_row_pd.to_csv(value_save_path+learning_name+'_epoch'+str(current_epoch_num)+'_keypoints_count_row_num_save.csv',index=True)
        
        if current_keypoints_num > keypoints_min_num:
            keypoints_save_list_np = np.concatenate(keypoints_save_list,axis=0)
            keypoints_final_np = np.vstack((current_keypoints_columns_np,keypoints_save_list_np))
            save_final_explorer_keypoints(keypoints_save_path, sample, keypoints_final_np)
            print('Current number of keypoints are more than '+str(keypoints_min_num)+' !')
            
            #save index file
            keypoints_segment_index_save_list_np = np.concatenate(keypoints_segment_index_save_list,axis=0).reshape(-1,1)
            epoch_num_list_np = np.concatenate(epoch_num_list,axis=0).reshape(-1,1)
            xenium_search_index_list_np = np.concatenate(xenium_search_index_list,axis=0).reshape(-1,1)
            xenium_barcode_list_np = np.concatenate(xenium_barcode_list,axis=0).reshape(-1,1)
            xenium_search_crop_list_np = np.concatenate(xenium_search_crop_list,axis=0).reshape(-1,1)
            
            keypoints_index_np = np.hstack((keypoints_segment_index_save_list_np,epoch_num_list_np,xenium_search_index_list_np,xenium_search_crop_list_np,xenium_barcode_list_np))
            keypoints_index_pd = pd.DataFrame(keypoints_index_np,columns=['keypoints_segment_index','epoch_num','xenium_search_index','xenium_search_crop_index','xenium_barcode'])
            keypoints_index_pd.to_csv(keypoints_save_path+learning_name+'_keypoints_index_value_save.csv',index=True)
            
            multi_angles_count_row_list_np = np.array(multi_angles_count_row_list).reshape(1,-1)
            multi_angles_count_row_list_np = multi_angles_count_row_list_np - 1
            multi_angles_count_row_ratio_list_np = multi_angles_count_row_list_np/check_cell_num
            multi_angles_count_row_np = np.vstack((multi_angles_count_row_list_np,multi_angles_count_row_ratio_list_np))
            multi_angles_count_row_pd = pd.DataFrame(multi_angles_count_row_np,columns=current_epoch_num_list,index=['multi_angles_count_row','multi_angles_count_row_ratio'])
            multi_angles_count_row_pd.to_csv(keypoints_save_path+sample+'_multi_angles_count_row_value_save.csv')
            
            break

    #time_computing
    end_time = time.time()
    print("Scoring and Ranking Check. End Time: %s seconds" %(end_time))
    print("Scoring and Ranking Check Done. Total Running Time: %s seconds" %(end_time - start_time))
    
    run_time_second = end_time - start_time
    run_time_minute = run_time_second/60
    run_time_hour = run_time_minute/60
    rum_time_np = np.array([run_time_second,run_time_minute,run_time_hour]).reshape(1,-1)
    rum_time_pd = pd.DataFrame(rum_time_np,columns=['run_time_second','run_time_minute','run_time_hour'],index=['rum_time'])
    rum_time_pd.to_csv(value_save_path+learning_name+'_run_time_save.csv',index=True)
    
    ##delete folders
    shutil.rmtree(score_save_path)
    shutil.rmtree(value_save_path)
    shutil.rmtree(graph_save_path)
    shutil.rmtree(segment_save_path)
    shutil.rmtree(image_check_path)
    for delete_epoch_num in range(current_epoch_num):
        #os settings
        crop_xenium_delete_path = output_path+'/crop_epoch'+str(delete_epoch_num)+'_xenium_save/'
        crop_he_delete_path = output_path+'/crop_epoch'+str(delete_epoch_num)+'_he_save/'
        crop_segment_delete_path = output_path+'/crop_epoch'+str(delete_epoch_num)+'_segment_save/'
        shutil.rmtree(crop_xenium_delete_path)
        shutil.rmtree(crop_he_delete_path)
        shutil.rmtree(crop_segment_delete_path)    
    