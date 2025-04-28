import pandas as pd
import os
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from graph_build import *
import csv
import cv2
from scipy.spatial import ConvexHull
from shapely import Polygon
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
import random

def score_and_rank_xenium_segment_nuclei_multi_angle(sdata, sample, pixel_size, crop_radius_pixel, center_move_pixel, tif_image_init_affine, segment_label_image_init_affine, xenium_mip_ome_tif_image, learning_name, output_path, transform_ratio, crop_image_resize, segment_nucleus_pixel_values_pd, check_cell_list, mip_ome_extract_ratio, current_epoch_num, image_check_path, segment_init_affine_image_width, segment_init_affine_image_height):

    #load os
    score_save_path = output_path+'/score_save/'
    value_save_path = output_path+'/value_save/'
    crop_segment_save_path = output_path+'/crop_epoch'+str(current_epoch_num)+'_segment_save/'
    crop_xenium_save_path = output_path+'/crop_epoch'+str(current_epoch_num)+'_xenium_save/'
    
    #load settings
    segment_nucleus_pixel_values_np = segment_nucleus_pixel_values_pd.values
    move_pixel_for_he = center_move_pixel / transform_ratio
    crop_radius_pixel_for_he = crop_radius_pixel / transform_ratio
    
    angle_list = ['center','up','down','left','right']

    #load rotate file
    rotate_mse_list_pd = pd.read_csv(image_check_path+sample+'_he_rotate_mse_values.csv',index_col=0)
    rotate_full_list = rotate_mse_list_pd.columns.tolist()
    
    rotate_mse_list_np = rotate_mse_list_pd.values.reshape(1,-1)
    row_max_index = np.argmax(rotate_mse_list_np)
    rotate_full_name = rotate_full_list[row_max_index]
    rotate_name_parts = rotate_full_name.split('_')
    rotate_name_init = rotate_name_parts[0]
    rotate_name_last = rotate_name_parts[1]

    for angle_num in range(len(angle_list)):
        current_angle_name = angle_list[angle_num]
        
        #score for individual cell
        score_psnr_rgb_list = []
        check_segment_cell_value_list = []
        check_segment_cell_index_list= []
        
        for check_cell_index in range(len(check_cell_list)):
            current_check_cell_index = check_cell_list[check_cell_index]

            if rotate_name_init == 'rotate90':
                if rotate_name_last == 'original':
                    current_pixel_y = segment_init_affine_image_height - segment_nucleus_pixel_values_np[current_check_cell_index,0]
                    current_pixel_x = segment_nucleus_pixel_values_np[current_check_cell_index,1]
                    
                elif rotate_name_last == 'flipLR':
                    current_pixel_y = segment_init_affine_image_height - segment_nucleus_pixel_values_np[current_check_cell_index,0]
                    current_pixel_x = segment_init_affine_image_width - segment_nucleus_pixel_values_np[current_check_cell_index,1]
                    
                elif rotate_name_last == 'flipTB':
                    current_pixel_y = segment_nucleus_pixel_values_np[current_check_cell_index,0]
                    current_pixel_x = segment_nucleus_pixel_values_np[current_check_cell_index,1]
                    
                elif rotate_name_last == 'rotate180':
                    current_pixel_y = segment_nucleus_pixel_values_np[current_check_cell_index,0]
                    current_pixel_x = segment_init_affine_image_width - segment_nucleus_pixel_values_np[current_check_cell_index,1]
            
            elif rotate_name_init == 'rotate0':
                if rotate_name_last == 'original':
                    current_pixel_y = segment_nucleus_pixel_values_np[current_check_cell_index,1]
                    current_pixel_x = segment_nucleus_pixel_values_np[current_check_cell_index,0]
                    
                elif rotate_name_last == 'flipLR':
                    current_pixel_y = segment_nucleus_pixel_values_np[current_check_cell_index,1]
                    current_pixel_x = segment_init_affine_image_width - segment_nucleus_pixel_values_np[current_check_cell_index,0]
                    
                elif rotate_name_last == 'flipTB':
                    current_pixel_y = segment_init_affine_image_height - segment_nucleus_pixel_values_np[current_check_cell_index,1]
                    current_pixel_x = segment_nucleus_pixel_values_np[current_check_cell_index,0]
                    
                elif rotate_name_last == 'rotate180':
                    current_pixel_y = segment_init_affine_image_height - segment_nucleus_pixel_values_np[current_check_cell_index,1]
                    current_pixel_x = segment_init_affine_image_width - segment_nucleus_pixel_values_np[current_check_cell_index,0]
                
            current_pixel_radius = segment_nucleus_pixel_values_np[current_check_cell_index,2]
            current_pixel_value = segment_nucleus_pixel_values_np[current_check_cell_index,:]
            
            check_segment_cell_value_list.append(current_pixel_value.reshape(1,-1))
            check_segment_cell_index_list.append(current_check_cell_index)

            if current_angle_name == 'center':
                current_pixel_y = current_pixel_y
                current_pixel_x = current_pixel_x
            elif current_angle_name == 'up':
                current_pixel_y = current_pixel_y - move_pixel_for_he
                current_pixel_x = current_pixel_x
            elif current_angle_name == 'down':
                current_pixel_y = current_pixel_y + move_pixel_for_he
                current_pixel_x = current_pixel_x
            elif current_angle_name == 'left':
                current_pixel_y = current_pixel_y
                current_pixel_x = current_pixel_x - move_pixel_for_he
            elif current_angle_name == 'right':
                current_pixel_y = current_pixel_y
                current_pixel_x = current_pixel_x + move_pixel_for_he
            
            #crop setting
            imagerow_down  = current_pixel_y - crop_radius_pixel_for_he
            imagerow_up    = current_pixel_y + crop_radius_pixel_for_he
            imagecol_left  = current_pixel_x - crop_radius_pixel_for_he
            imagecol_right = current_pixel_x + crop_radius_pixel_for_he
            
            #segment_label_image_init_affine_crop
            segment_label_image_crop_tile = segment_label_image_init_affine.crop((imagecol_left, imagerow_down, imagecol_right, imagerow_up))
            if segment_label_image_crop_tile.mode != 'RGB':
                #print('segment_label_image_crop_tile mode is',segment_label_image_crop_tile.mode)
                segment_label_image_crop_tile = segment_label_image_crop_tile.convert('RGB')
                #print('segment_label_image_crop_tile mode has been converted!')
        
            segment_label_image_crop_tile_resize = segment_label_image_crop_tile.resize((crop_image_resize,crop_image_resize))
            segment_label_image_crop_tile_resize_np = np.array(segment_label_image_crop_tile_resize)
            segment_label_image_crop_tile_resize.save(crop_segment_save_path+'index'+str(check_cell_index)+'_Rpsave'+str(round(current_pixel_radius,3))+'_Rpcrop'+str(round(crop_radius_pixel_for_he,3))+'_segment_label_image_crop_pillow_'+current_angle_name+'.jpg', "JPEG")
            
            #tif_image_pillow
            tif_image_crop_tile = tif_image_init_affine.crop((imagecol_left, imagerow_down, imagecol_right, imagerow_up))
            if tif_image_crop_tile.mode != 'RGB':
                #print('tif_image_crop_tile mode is',tif_image_crop_tile.mode)
                tif_image_crop_tile = tif_image_crop_tile.convert('RGB')
                #print('tif_image_crop_tile mode has been converted!')
    
            tif_image_crop_tile_resize = tif_image_crop_tile.resize((crop_image_resize,crop_image_resize))
            tif_image_crop_tile_resize_rgb_np = np.array(tif_image_crop_tile_resize)
            tif_image_crop_tile_resize.save(crop_segment_save_path+'index'+str(check_cell_index)+'_Rpsave'+str(round(current_pixel_radius,3))+'_Rpcrop'+str(round(crop_radius_pixel_for_he,3))+'_HE_crop_pillow'+'_'+current_angle_name+'.jpg', "JPEG")
            
            #load xenium search regions
            current_xenium_nucleus_pixel_values_pd = pd.read_csv(crop_xenium_save_path+'check_cell'+str(check_cell_index)+'_index'+str(current_check_cell_index)+'_search_range_extract_ratio'+str(round(mip_ome_extract_ratio,3))+'_xenium_nucleus_pixel_values.csv',index_col=0)
            current_xenium_nucleus_pixel_values_np = current_xenium_nucleus_pixel_values_pd.values
            current_xenium_nucleus_pixel_values_index_list = current_xenium_nucleus_pixel_values_pd.index.tolist()
            #print('current_xenium_nucleus_pixel_values_pd',current_xenium_nucleus_pixel_values_pd)
            
            #set save_check_sort_image_num
            #if current_xenium_nucleus_pixel_values_np.shape[0] < save_check_sort_image_num:
            #    print('check index'+str(check_cell_index)+' segment cell is lack of searched cells in xenium side!')
            #assert (save_check_sort_image_num >= 5)
            
            #set list
            cell_score_search_psnr_rgb_list = []
            cell_all_barcode_list = []
            
            for cell_index in range(current_xenium_nucleus_pixel_values_np.shape[0]):
                #print('cell_index',cell_index)
                #cell_barcode = current_xenium_nucleus_pixel_values_index_list[cell_index]
                #cell_all_barcode_list.append(cell_barcode)
     
                current_xenium_pixel_x = current_xenium_nucleus_pixel_values_np[cell_index,0]
                current_xenium_pixel_y = current_xenium_nucleus_pixel_values_np[cell_index,1]
                
                #set center and crop value
                if current_angle_name == 'center':
                    current_xenium_pixel_x = current_xenium_pixel_x
                    current_xenium_pixel_y = current_xenium_pixel_y
                elif current_angle_name == 'up':
                    current_xenium_pixel_x = current_xenium_pixel_x
                    current_xenium_pixel_y = current_xenium_pixel_y - center_move_pixel
                elif current_angle_name == 'down':
                    current_xenium_pixel_x = current_xenium_pixel_x
                    current_xenium_pixel_y = current_xenium_pixel_y + center_move_pixel
                elif current_angle_name == 'left':
                    current_xenium_pixel_x = current_xenium_pixel_x - center_move_pixel
                    current_xenium_pixel_y = current_xenium_pixel_y
                elif current_angle_name == 'right':
                    current_xenium_pixel_x = current_xenium_pixel_x + center_move_pixel
                    current_xenium_pixel_y = current_xenium_pixel_y

                pixel_center_radius_xenium_used = crop_radius_pixel
                imagerow_down_xenium  = current_xenium_pixel_y - pixel_center_radius_xenium_used
                imagerow_up_xenium    = current_xenium_pixel_y + pixel_center_radius_xenium_used
                imagecol_left_xenium  = current_xenium_pixel_x - pixel_center_radius_xenium_used
                imagecol_right_xenium = current_xenium_pixel_x + pixel_center_radius_xenium_used
                
                crop_nucleus_for_check_image_rgb_tile = xenium_mip_ome_tif_image.crop((imagecol_left_xenium, imagerow_down_xenium, imagecol_right_xenium, imagerow_up_xenium))
                if crop_nucleus_for_check_image_rgb_tile.mode != 'RGB':
                    #print('crop_nucleus_for_check_image_rgb_tile mode is',crop_nucleus_for_check_image_rgb_tile.mode)
                    crop_nucleus_for_check_image_rgb_tile = crop_nucleus_for_check_image_rgb_tile.convert('RGB')
                    #print('crop_nucleus_for_check_image_rgb_tile mode has been converted!')
                crop_nucleus_for_check_image_rgb_tile_resize = crop_nucleus_for_check_image_rgb_tile.resize((crop_image_resize,crop_image_resize))
                crop_nucleus_for_check_image_rgb_tile_resize_np = np.array(crop_nucleus_for_check_image_rgb_tile_resize)
                
                #calculate metrics
                current_psnr_rgb = compare_psnr(segment_label_image_crop_tile_resize_np, crop_nucleus_for_check_image_rgb_tile_resize_np)
                cell_score_search_psnr_rgb_list.append(current_psnr_rgb)
    
            #os setting
            current_sorted_save_path = score_save_path+'check_cell'+str(check_cell_index)+'_epoch'+str(current_epoch_num)+'_'+current_angle_name+'/'
            if not os.path.exists(current_sorted_save_path):
                os.makedirs(current_sorted_save_path)
            
            ###psnr_summary
            cell_score_search_psnr_rgb_list_np = np.array(cell_score_search_psnr_rgb_list).reshape(1,-1)
            score_psnr_rgb_list.append(cell_score_search_psnr_rgb_list_np)
            score_psnr_rgb_list_np = np.array(cell_score_search_psnr_rgb_list_np).reshape(-1,1)
            score_psnr_rgb_list_pd = pd.DataFrame(score_psnr_rgb_list_np,index=current_xenium_nucleus_pixel_values_index_list,columns=['score_psnr_rgb'])
            score_psnr_rgb_list_pd_sorted = score_psnr_rgb_list_pd.sort_values(by='score_psnr_rgb', ascending=False)
            score_psnr_rgb_list_pd_sorted_index_list = score_psnr_rgb_list_pd_sorted.index.tolist()
            #score_psnr_rgb_list_pd_sorted_save = score_psnr_rgb_list_pd_sorted.head(save_check_sort_image_num)
            score_psnr_rgb_list_pd_sorted_save = score_psnr_rgb_list_pd_sorted
            #score_psnr_rgb_list_pd_sorted_save.to_csv(current_sorted_save_path+'check_cell'+str(check_cell_index)+'_sorted'+str(save_check_sort_image_num)+'_score_psnr_rgb_'+current_angle_name+'.csv')
            score_psnr_rgb_list_pd_sorted_save.to_csv(current_sorted_save_path+'check_cell'+str(check_cell_index)+'_score_psnr_rgb_'+current_angle_name+'.csv')
        
        #save check segment cell values
        check_segment_cell_value_list_concatenate = np.concatenate(check_segment_cell_value_list,axis=0)
        #print('check_segment_cell_value_list_concatenate shape is',check_segment_cell_value_list_concatenate.shape)
        check_segment_cell_value_list_concatenate_pd = pd.DataFrame(check_segment_cell_value_list_concatenate, index = check_segment_cell_index_list, columns = ['pixel_center_x_nucleus','pixel_center_y_nucleus','pixel_center_radius_nucleus'])
        check_segment_cell_value_list_concatenate_pd.to_csv(value_save_path+'cell'+str(len(check_cell_list))+'_in_total_epoch'+str(current_epoch_num)+'_segment_nucleus_pixel_values_'+current_angle_name+'.csv')
    
    return print('Current sample psnr checking of epoch'+str(current_epoch_num)+' finished!')
    
def check_rank_top1_multi_angle(output_path, check_cell_num, sample, learning_name, current_epoch_num):

    #load os
    score_save_path = output_path+'/score_save/'
    value_save_path = output_path+'/value_save/'
    
    #set list
    angle_list = ['center','up','down','left','right']
    
    top1_barcode_multi_angle_list = []
    top1_index_multi_angle_list = []
    for angle_num in range(len(angle_list)):
        current_angle_name = angle_list[angle_num]
        
        top1_list = []
        columns_list = []
        first_barcode_list = []
        cell_index_psnr_rgb_list = []
        
        for check_current_cell in range(check_cell_num):
            current_folder = score_save_path+'check_cell'+str(check_current_cell)+'_epoch'+str(current_epoch_num)+'_'+current_angle_name+'/'
            current_top1_psnr_rgb_pd = pd.read_csv(current_folder+'check_cell'+str(check_current_cell)+'_score_psnr_rgb_'+current_angle_name+'.csv')
            current_top1_psnr_rgb_np = current_top1_psnr_rgb_pd.head(1).values
            top1_list.append(current_top1_psnr_rgb_np)
            columns_list.append('cell'+str(check_current_cell)+'_barcode_psnr_rgb')
            columns_list.append('cell'+str(check_current_cell)+'_score_psnr_rgb')
            first_barcode_list.append(current_top1_psnr_rgb_np[0,0])
            cell_index_psnr_rgb_list.append('cell'+str(check_current_cell))
    
        top1_list_np = np.concatenate(top1_list,axis=1)
        top1_list_pd = pd.DataFrame(top1_list_np,columns=columns_list,index=['top1'])
        file_check_name0 = value_save_path+learning_name+'_ranked_top1_psnr_rgb_epoch'+str(current_epoch_num)+'_'+current_angle_name+'.csv'
        if os.path.exists(file_check_name0):
            pass
        else:
            top1_list_pd.to_csv(value_save_path+learning_name+'_ranked_top1_psnr_rgb_epoch'+str(current_epoch_num)+'_'+current_angle_name+'.csv')
        
        first_barcode_list_np = np.array(first_barcode_list).reshape(-1,1)
        top1_barcode_psnr_rgb_pd = pd.DataFrame(first_barcode_list_np,index =cell_index_psnr_rgb_list,columns=['top1'])
        top1_barcode_multi_angle_list.append(first_barcode_list_np)
        top1_index_multi_angle_list.append('top1_'+current_angle_name)
        
        file_check_name1 = value_save_path+learning_name+'_ranked_top1_barcode_psnr_rgb_epoch'+str(current_epoch_num)+'_'+current_angle_name+'.csv'
        if os.path.exists(file_check_name1):
            #print(sample+'_ranked_top1_barcode_psnr_rgb_'+current_angle_name+'.csv is exist!')
            pass
        else:
            #print('Start to generate '+sample+'_ranked_top1_barcode_psnr_rgb_'+current_angle_name+'.csv!')
            top1_barcode_psnr_rgb_pd.to_csv(value_save_path+learning_name+'_ranked_top1_barcode_psnr_rgb_epoch'+str(current_epoch_num)+'_'+current_angle_name+'.csv')

    #save ranked_top1_barcode_psnr_rgb_five_angles
    top1_barcode_multi_angle_list_np = np.concatenate(top1_barcode_multi_angle_list,axis=1)
    top1_barcode_multi_angle_list_pd = pd.DataFrame(top1_barcode_multi_angle_list_np,index=cell_index_psnr_rgb_list,columns=top1_index_multi_angle_list)
    file_check_name2 = value_save_path+learning_name+'_ranked_top1_barcode_psnr_rgb_epoch'+str(current_epoch_num)+'_five_angles.csv'
    if os.path.exists(file_check_name2):
        pass
    else:
        top1_barcode_multi_angle_list_pd.to_csv(value_save_path+learning_name+'_ranked_top1_barcode_psnr_rgb_epoch'+str(current_epoch_num)+'_five_angles.csv')
    
    #save filtered_ranked_top1_barcode_psnr_rgb_five_angles
    filtered_cell_index_list = []
    for check_cell_num in range(top1_barcode_multi_angle_list_np.shape[0]):
        current_check_row = top1_barcode_multi_angle_list_np[check_cell_num,:]
        if len(np.unique(current_check_row)) == 1:
            current_check_index = 'cell'+str(check_cell_num)
            filtered_cell_index_list.append(current_check_index)
    filtered_top1_barcode_multi_angle_list_pd = top1_barcode_multi_angle_list_pd.loc[filtered_cell_index_list]
    file_check_name3 = value_save_path+learning_name+'_filtered_ranked_top1_barcode_psnr_rgb_epoch'+str(current_epoch_num)+'_five_angles.csv'
    if os.path.exists(file_check_name3):
        pass
    else:
        filtered_top1_barcode_multi_angle_list_pd.to_csv(value_save_path+learning_name+'_filtered_ranked_top1_barcode_psnr_rgb_epoch'+str(current_epoch_num)+'_five_angles.csv')
    
    #save filtered_top1_barcode_multi_angle_list_pd barcode
    filtered_top1_barcode_multi_angle_list = filtered_top1_barcode_multi_angle_list_pd.values[:,0].tolist()
    #print('filtered_top1_barcode_multi_angle_list',filtered_top1_barcode_multi_angle_list)
    assert((filtered_top1_barcode_multi_angle_list_pd.values[:,0]==filtered_top1_barcode_multi_angle_list_pd.values[:,1]).all())
    
    print('Current sample multi angles checking of epoch'+str(current_epoch_num)+' finished!')
    
    return filtered_top1_barcode_multi_angle_list
    
def pixel_values_filtered_by_multi_angles_for_xenium_explorer(sample, value_save_path, segment_save_path, segment_method, keypoints_save_path, learning_name, check_cell_num, check_cell_list, xenium_nucleus_pixel_values_pd, current_epoch_num, remove_segment_edge, crop_radius_pixel, center_move_pixel):
    
    #read filterd values
    rank_filtered_top1_index_pd = pd.read_csv(value_save_path+learning_name+'_filtered_ranked_top1_barcode_psnr_rgb_epoch'+str(current_epoch_num)+'_five_angles.csv',index_col=0)
    rank_filtered_top1_index_list = rank_filtered_top1_index_pd.index.tolist()
    rank_filtered_top1_index_list_num = []
    rank_filtered_top1_barcode_list = []
    rank_filtered_top1_index_np = rank_filtered_top1_index_pd.values
    for cell_num in range(len(rank_filtered_top1_index_list)):
        current_cell_str = rank_filtered_top1_index_list[cell_num]
        current_cell_num = int(current_cell_str.split('cell')[1])
        rank_filtered_top1_index_list_num.append(current_cell_num)
        current_cell_barcode = rank_filtered_top1_index_np[cell_num,0]
        rank_filtered_top1_barcode_list.append(current_cell_barcode)
    
    #load segmentation
    #segment_center_pixel_value_filterd_pd = pd.read_csv(cellpose_save_path+sample+'_cellpose_segment_H&E_label_location_filterd_remove_edge_consider_crop_layout_save.csv',index_col=0)
    #segment_center_pixel_value_filterd_pd = pd.read_csv(segment_save_path+sample+'_'+segment_method+'_segment_H&E_label_location_filterd_remove_edge_consider_crop_layout_save.csv',index_col=0)
    #segment_center_pixel_value_filterd_pd = pd.read_csv(segment_save_path+sample+'_'+segment_method+'_segment_H&E_label_final_location_save.csv',index_col=0)
    #segment_center_pixel_value_filterd_pd = pd.read_csv(segment_save_path+sample+'_'+segment_method+'_segment_H&E_label_final_location_save.csv',index_col=0)
    segment_center_pixel_value_filterd_file = segment_save_path+sample+'_'+segment_method+'_segment_H&E_label_remove_edge'+str(remove_segment_edge)+'_consider_crop'+str(crop_radius_pixel)+'_move'+str(center_move_pixel)+'_final_location_save.csv'
    segment_center_pixel_value_filterd_pd = pd.read_csv(segment_center_pixel_value_filterd_file,index_col=0)
    
    #segment_index_save
    segment_center_pixel_value_original_list = segment_center_pixel_value_filterd_pd.index.tolist()
    segment_center_pixel_value_original_list_np = np.array(segment_center_pixel_value_original_list)
    segment_center_pixel_value_original_use_list_np = segment_center_pixel_value_original_list_np[check_cell_list]
    
    #continue
    segment_center_pixel_value_filterd_cell_num_list = list(range(segment_center_pixel_value_filterd_pd.shape[0]))
    segment_center_pixel_value_filterd_pd.index = segment_center_pixel_value_filterd_cell_num_list
    
    #score_cell_list
    #print('segment_center_pixel_value_filterd_pd',segment_center_pixel_value_filterd_pd)
    segment_center_pixel_value_filterd_pd_scored = segment_center_pixel_value_filterd_pd.loc[check_cell_list]
    #print('segment_center_pixel_value_filterd_pd_scored',segment_center_pixel_value_filterd_pd_scored)
    segment_center_pixel_value_filterd_pd_scored.to_csv(value_save_path+learning_name+'_filtered_segment_nucleus_center_pixel_values_index_epoch'+str(current_epoch_num)+'_by_cell_list.csv',index=True)
    cell_num_list = list(range(check_cell_num))
    segment_center_pixel_value_filterd_pd_scored.index = cell_num_list
    segment_center_pixel_value_filterd_pd_scored.to_csv(value_save_path+learning_name+'_filtered_segment_nucleus_center_pixel_values_index_epoch'+str(current_epoch_num)+'_by_cell_num.csv',index=True)
    segment_center_pixel_value_filterd_pd_scored_filtered = segment_center_pixel_value_filterd_pd_scored.loc[rank_filtered_top1_index_list_num]
    segment_center_pixel_value_filterd_scored_filtered_np = segment_center_pixel_value_filterd_pd_scored_filtered[['x_center_pixel','y_center_pixel']].values
    
    #check part
    filterd_check_psnr_pixel_values_center_pd = pd.read_csv(value_save_path+'cell'+str(len(check_cell_list))+'_in_total_epoch'+str(current_epoch_num)+'_segment_nucleus_pixel_values_center.csv',index_col=0)
    filterd_check_psnr_pixel_values_center_index_np = np.array(filterd_check_psnr_pixel_values_center_pd.index.tolist())
    assert((filterd_check_psnr_pixel_values_center_index_np==check_cell_list).all())
    
    #xenium_pixel_nucleus_values
    xenium_nucleus_pixel_values_pd_filtered = xenium_nucleus_pixel_values_pd.loc[rank_filtered_top1_barcode_list]
    xenium_nucleus_pixel_values_pd_filtered_np = xenium_nucleus_pixel_values_pd_filtered[['pixel_center_x_nucleus','pixel_center_y_nucleus']].values
    
    #pixel_value_filtered_for_xenium_explorer
    pixel_value_for_xenium_explorer_filtered_np = np.hstack((xenium_nucleus_pixel_values_pd_filtered_np,segment_center_pixel_value_filterd_scored_filtered_np))
    pixel_value_for_xenium_explorer_filtered_columns_list = ['fixedX','fixedY','alignmentX','alignmentY']
    pixel_value_for_xenium_explorer_filtered_columns_list_np = np.array(pixel_value_for_xenium_explorer_filtered_columns_list).reshape(1,-1)
    pixel_value_for_xenium_explorer_filtered_add_columns_np = np.vstack((pixel_value_for_xenium_explorer_filtered_columns_list_np, pixel_value_for_xenium_explorer_filtered_np))

    with open(keypoints_save_path+learning_name+'_filtered_by_multi_angle_epoch'+str(current_epoch_num)+'_keypoints.csv', 'w', newline='') as file:
        writer = csv.writer(file)  #lineterminator='\n'
        count = 0
        for row_num in range(pixel_value_for_xenium_explorer_filtered_add_columns_np.shape[0]):
            count = count + 1
            if count < pixel_value_for_xenium_explorer_filtered_add_columns_np.shape[0]:
                current_row = pixel_value_for_xenium_explorer_filtered_add_columns_np[row_num,:].tolist()
                writer.writerow(current_row)
            else:
                writer = csv.writer(file,lineterminator='')  #lineterminator='\n'
                current_row = pixel_value_for_xenium_explorer_filtered_add_columns_np[row_num,:].tolist()
                writer.writerow(current_row)
    
    print('The keypoints filtered by multi angles of epoch'+str(current_epoch_num)+' have been generated!')
    
    return count, pixel_value_for_xenium_explorer_filtered_add_columns_np, rank_filtered_top1_index_list_num, segment_center_pixel_value_original_use_list_np
    
def pixel_values_filtered_by_multi_angles_and_graph_for_xenium_explorer(sample, graph_save_path, keypoints_save_path, learning_name, graph_source_str, fig_size, current_epoch_num):

    #os setting
    dataset_root_xenium = graph_save_path+learning_name+'_xenium_explorer_epoch'+str(current_epoch_num)+'/'
    dataset_root_segment = graph_save_path+learning_name+'_segment_explorer_epoch'+str(current_epoch_num)+'/'
    
    nx_graph_root_xenium = dataset_root_xenium+"/graph/"
    fig_save_root_xenium = dataset_root_xenium+"/fig/"
    os.makedirs(nx_graph_root_xenium, exist_ok=True)
    os.makedirs(fig_save_root_xenium, exist_ok=True)
    
    nx_graph_root_segment = dataset_root_segment+"/graph/"
    fig_save_root_segment = dataset_root_segment+"/fig/"
    os.makedirs(nx_graph_root_segment, exist_ok=True)
    os.makedirs(fig_save_root_segment, exist_ok=True)

    region_id_xenium = sample+'_xenium'
    region_id_segment = sample+'_segment'
    
    #load initial keypoints file
    current_filtered_by_multi_angles_file = keypoints_save_path+learning_name+'_filtered_by_multi_angle_epoch'+str(current_epoch_num)+'_keypoints.csv'
    xenium_segment_coords_filtered_by_multi_angles_pd = pd.read_csv(current_filtered_by_multi_angles_file)
    
    xenium_coords_for_graph_pd = xenium_segment_coords_filtered_by_multi_angles_pd[['fixedX','fixedY']]
    segment_coords_for_graph_pd = xenium_segment_coords_filtered_by_multi_angles_pd[['alignmentY','alignmentX']]
    
    cell_index_list = list(range(0,xenium_segment_coords_filtered_by_multi_angles_pd.shape[0]))
    current_check_cell_num = len(cell_index_list)
    cell_columns = ['X','Y']
    xenium_coords_for_graph_pd.columns = cell_columns
    segment_coords_for_graph_pd.columns = cell_columns
    xenium_coords_for_graph_pd.index = cell_index_list
    segment_coords_for_graph_pd.index = cell_index_list
    
    count = 0
    xenium_coords_for_graph_pd.to_csv(dataset_root_xenium+learning_name+'_xenium_for_delaunay_pixel_values_count'+str(count)+'.csv',index=True)
    segment_coords_for_graph_pd.to_csv(dataset_root_segment+learning_name+'_segment_for_delaunay_pixel_values_count'+str(count)+'.csv',index=True)
    
    for check_num in range(current_check_cell_num):
        
        #init file name
        cell_coords_file_xenium = dataset_root_xenium+learning_name+'_xenium_for_delaunay_pixel_values_count'+str(count)+'.csv'
        voronoi_polygon_img_output_xenium = fig_save_root_xenium+learning_name+'_subgraph_voronoi_fig'+str(fig_size)+'_filter_count'+str(count)+'_xenium.jpg'
        graph_img_output_xenium = fig_save_root_xenium+learning_name+'_subgraph_delaunay_fig'+str(fig_size)+'_filter_count'+str(count)+'_xenium.jpg'
        graph_output_xenium = nx_graph_root_xenium+learning_name+'_subgraph_cell_filter_count'+str(count)+'_xenium.gpkl'
        
        cell_coords_file_segment = dataset_root_segment+learning_name+'_segment_for_delaunay_pixel_values_count'+str(count)+'.csv'
        voronoi_polygon_img_output_segment = fig_save_root_segment+learning_name+'_subgraph_voronoi_fig'+str(fig_size)+'_filter_count'+str(count)+'_segment.jpg'
        graph_img_output_segment = fig_save_root_segment+learning_name+'_subgraph_delaunay_fig'+str(fig_size)+'_filter_count'+str(count)+'_segment.jpg'
        graph_output_segment = nx_graph_root_segment+learning_name+'_subgraph_cell_filter_count'+str(count)+'_segment.gpkl'
        
        current_total_cell_num = pd.read_csv(cell_coords_file_xenium,index_col=0).shape[0]
        assert(current_total_cell_num==pd.read_csv(cell_coords_file_segment,index_col=0).shape[0])
        
        #build graph
        G_xenium = construct_graph_for_region(
            region_id_xenium,
            cell_coords_file=cell_coords_file_xenium,
            cell_types_file=None,
            cell_biomarker_expression_file=None,
            #cell_biomarker_expression_file=cell_biomarker_expression_file,
            cell_features_file=None,
            voronoi_file=None,
            graph_source=graph_source_str,
            graph_output=None,                                               #graph_output_xenium,
            voronoi_polygon_img_output=None,                            #voronoi_polygon_img_output_xenium,
            graph_img_output=None,                                      #graph_img_output_xenium,
            figsize=fig_size)
    
        G_segment = construct_graph_for_region(
            region_id_segment,
            cell_coords_file=cell_coords_file_segment,
            cell_types_file=None,
            cell_biomarker_expression_file=None,
            #cell_biomarker_expression_file=cell_biomarker_expression_file,
            cell_features_file=None,
            voronoi_file=None,
            graph_source=graph_source_str,
            graph_output=None,                                               #graph_output_xenium,
            voronoi_polygon_img_output=None,                            #voronoi_polygon_img_output_xenium,
            graph_img_output=None,                                      #graph_img_output_xenium,
            figsize=fig_size)
        
        #filterd outlier
        G_xenium_edge_list = list(G_xenium.edges)
        G_segment_edge_list = list(G_segment.edges)

        for filter_list_index in range(len(G_xenium_edge_list)):
            current_G_xenium_node_edge = G_xenium_edge_list[filter_list_index]
            current_G_segment_node_edge = G_segment_edge_list[filter_list_index]
            if current_G_xenium_node_edge != current_G_segment_node_edge:
                current_G_xenium_node_edge_source = current_G_xenium_node_edge[0]
                current_G_segment_node_edge_source = current_G_segment_node_edge[0]
                current_G_segment_node_edge_remove = min(current_G_xenium_node_edge_source,current_G_segment_node_edge_source)
                break
        
        if filter_list_index == len(G_xenium_edge_list)-1:
            #print('All graph edges are matched between xenium and segment !!!')
            break
        
        if current_G_segment_node_edge_remove >= 0:
            #update pixel values
            current_search_cell_list_use_coords_file_xenium_pd = pd.read_csv(cell_coords_file_xenium,index_col=0)
            current_search_cell_list_use_coords_file_segment_pd = pd.read_csv(cell_coords_file_segment,index_col=0)
            
            current_search_cell_list_use_coords_file_xenium_index = current_search_cell_list_use_coords_file_xenium_pd.index.tolist()
            current_search_cell_list_use_coords_file_segment_index = current_search_cell_list_use_coords_file_segment_pd.index.tolist()
            
            del current_search_cell_list_use_coords_file_xenium_index[current_G_segment_node_edge_remove]
            del current_search_cell_list_use_coords_file_segment_index[current_G_segment_node_edge_remove]
            
            update_search_cell_list_use_coords_file_xenium_pd = current_search_cell_list_use_coords_file_xenium_pd.loc[current_search_cell_list_use_coords_file_xenium_index]
            update_search_cell_list_use_coords_file_segment_pd = current_search_cell_list_use_coords_file_segment_pd.loc[current_search_cell_list_use_coords_file_segment_index]
            
            count = count + 1
            
            update_search_cell_list_use_coords_file_xenium_pd.to_csv(dataset_root_xenium+learning_name+'_xenium_for_delaunay_pixel_values_count'+str(count)+'.csv',index=True)
            update_search_cell_list_use_coords_file_segment_pd.to_csv(dataset_root_segment+learning_name+'_segment_for_delaunay_pixel_values_count'+str(count)+'.csv',index=True)
        
        current_G_segment_node_edge_remove = 'prepare to update!'
    
    if count == 0:
        update_search_cell_list_use_coords_file_xenium_pd = xenium_coords_for_graph_pd
        update_search_cell_list_use_coords_file_segment_pd = segment_coords_for_graph_pd
    
    update_search_cell_list_use_coords_file_segment_pd_explorer = update_search_cell_list_use_coords_file_segment_pd[['Y','X']]
    update_search_cell_list_use_coords_file_xenium_pd_explorer = update_search_cell_list_use_coords_file_xenium_pd
    
    update_search_cell_list_use_coords_file_xenium_explorer_np = update_search_cell_list_use_coords_file_xenium_pd_explorer.values
    update_search_cell_list_use_coords_file_segment_explorer_np = update_search_cell_list_use_coords_file_segment_pd_explorer.values
    
    update_search_cell_list_use_coords_file_xenium_segment_explorer_np = np.hstack((update_search_cell_list_use_coords_file_xenium_explorer_np,update_search_cell_list_use_coords_file_segment_explorer_np))
    update_xenium_segment_explorer_columns_np = np.array(['fixedX','fixedY','alignmentX','alignmentY']).reshape(1,-1)
    update_xenium_segment_explorer_np = np.vstack((update_xenium_segment_explorer_columns_np,update_search_cell_list_use_coords_file_xenium_segment_explorer_np))
    
    ###with open(d+learning_name+'_filtered_by_multi_angle_and_graph_epoch'+str(current_epoch_num)+'_keypoints.csv', 'w', newline='') as file:
    ###    writer = csv.writer(file)  #lineterminator='\n'
    ###    count_row = 0
    ###    for row_num in range(update_xenium_segment_explorer_np.shape[0]):
    ###        count_row = count_row + 1
    ###        if count_row < update_xenium_segment_explorer_np.shape[0]:
    ###            current_row = update_xenium_segment_explorer_np[row_num,:].tolist()
    ###            writer.writerow(current_row)
    ###        else:
    ###            writer = csv.writer(file,lineterminator='')
    ###            current_row = update_xenium_segment_explorer_np[row_num,:].tolist()
    ###            writer.writerow(current_row)
    
    count_row = update_xenium_segment_explorer_np.shape[0]   #add
    
    filtered_keypoints_num = update_xenium_segment_explorer_np.shape[0]
    
    #save xenium barcode list
    graph_xenium_barcode_list = update_search_cell_list_use_coords_file_xenium_pd_explorer.index.tolist()
    #print('graph_xenium_barcode_list',graph_xenium_barcode_list)
    
    return filtered_keypoints_num, update_xenium_segment_explorer_np, count_row, graph_xenium_barcode_list
    
def pixel_values_filtered_by_multi_angles_and_graph_inverse_for_xenium_explorer(sample, graph_save_path, keypoints_save_path, learning_name, graph_source_str, fig_size, current_epoch_num):

    #os setting
    dataset_root_xenium = graph_save_path+learning_name+'_inverse_xenium_explorer_epoch'+str(current_epoch_num)+'/'
    dataset_root_segment = graph_save_path+learning_name+'_inverse_segment_explorer_epoch'+str(current_epoch_num)+'/'
    
    nx_graph_root_xenium = dataset_root_xenium+"/graph/"
    fig_save_root_xenium = dataset_root_xenium+"/fig/"
    os.makedirs(nx_graph_root_xenium, exist_ok=True)
    os.makedirs(fig_save_root_xenium, exist_ok=True)
    
    nx_graph_root_segment = dataset_root_segment+"/graph/"
    fig_save_root_segment = dataset_root_segment+"/fig/"
    os.makedirs(nx_graph_root_segment, exist_ok=True)
    os.makedirs(fig_save_root_segment, exist_ok=True)

    region_id_xenium = sample+'_xenium'
    region_id_segment = sample+'_segment'
    

    #load initial keypoints file
    current_filtered_by_multi_angles_file = keypoints_save_path+learning_name+'_filtered_by_multi_angle_epoch'+str(current_epoch_num)+'_keypoints.csv'
    xenium_segment_coords_filtered_by_multi_angles_pd = pd.read_csv(current_filtered_by_multi_angles_file)
    #delete initial keypoints file
    os.remove(current_filtered_by_multi_angles_file)

    xenium_coords_for_graph_pd = xenium_segment_coords_filtered_by_multi_angles_pd[['fixedX','fixedY']]
    segment_coords_for_graph_pd = xenium_segment_coords_filtered_by_multi_angles_pd[['alignmentY','alignmentX']]
    
    cell_index_list = list(range(0,xenium_segment_coords_filtered_by_multi_angles_pd.shape[0]))
    current_check_cell_num = len(cell_index_list)
    cell_columns = ['X','Y']
    xenium_coords_for_graph_pd.columns = cell_columns
    segment_coords_for_graph_pd.columns = cell_columns
    xenium_coords_for_graph_pd.index = cell_index_list
    segment_coords_for_graph_pd.index = cell_index_list

    #inverse
    xenium_coords_for_graph_pd_inverse0 = xenium_coords_for_graph_pd.iloc[::-1]
    xenium_coords_for_graph_pd_inverse1 = xenium_coords_for_graph_pd.reindex(reversed(xenium_coords_for_graph_pd.index))
    assert((np.array(xenium_coords_for_graph_pd_inverse0.index.tolist())==np.array(xenium_coords_for_graph_pd_inverse1.index.tolist())).all())
    
    segment_coords_for_graph_pd_inverse0 = segment_coords_for_graph_pd.iloc[::-1]
    segment_coords_for_graph_pd_inverse1 = segment_coords_for_graph_pd.reindex(reversed(segment_coords_for_graph_pd.index))
    assert((np.array(segment_coords_for_graph_pd_inverse0.index.tolist())==np.array(segment_coords_for_graph_pd_inverse1.index.tolist())).all())
    
    count = 0
    xenium_coords_for_graph_pd_inverse0.to_csv(dataset_root_xenium+learning_name+'_inverse_xenium_for_delaunay_pixel_values_count'+str(count)+'.csv',index=True)
    segment_coords_for_graph_pd_inverse0.to_csv(dataset_root_segment+learning_name+'_inverse_segment_for_delaunay_pixel_values_count'+str(count)+'.csv',index=True)
    
    for check_num in range(current_check_cell_num):
        
        #init file name
        cell_coords_file_xenium = dataset_root_xenium+learning_name+'_inverse_xenium_for_delaunay_pixel_values_count'+str(count)+'.csv'
        voronoi_polygon_img_output_xenium = fig_save_root_xenium+learning_name+'_subgraph_voronoi_fig'+str(fig_size)+'_filter_count'+str(count)+'_xenium.jpg'
        graph_img_output_xenium = fig_save_root_xenium+learning_name+'_subgraph_delaunay_fig'+str(fig_size)+'_filter_count'+str(count)+'_xenium.jpg'
        graph_output_xenium = nx_graph_root_xenium+learning_name+'_subgraph_cell_filter_count'+str(count)+'_xenium.gpkl'
        
        cell_coords_file_segment = dataset_root_segment+learning_name+'_inverse_segment_for_delaunay_pixel_values_count'+str(count)+'.csv'
        voronoi_polygon_img_output_segment = fig_save_root_segment+learning_name+'_subgraph_voronoi_fig'+str(fig_size)+'_filter_count'+str(count)+'_segment.jpg'
        graph_img_output_segment = fig_save_root_segment+learning_name+'_subgraph_delaunay_fig'+str(fig_size)+'_filter_count'+str(count)+'_segment.jpg'
        graph_output_segment = nx_graph_root_segment+learning_name+'_subgraph_cell_filter_count'+str(count)+'_segment.gpkl'
        
        current_total_cell_num = pd.read_csv(cell_coords_file_xenium,index_col=0).shape[0]
        assert(current_total_cell_num==pd.read_csv(cell_coords_file_segment,index_col=0).shape[0])
        
        #build graph
        G_xenium = construct_graph_for_region(
            region_id_xenium,
            cell_coords_file=cell_coords_file_xenium,
            cell_types_file=None,
            cell_biomarker_expression_file=None,
            #cell_biomarker_expression_file=cell_biomarker_expression_file,
            cell_features_file=None,
            voronoi_file=None,
            graph_source=graph_source_str,
            graph_output=None,                                               #graph_output_xenium,
            voronoi_polygon_img_output=None,                            #voronoi_polygon_img_output_xenium,
            graph_img_output=None,                                      #graph_img_output_xenium,
            figsize=fig_size)
    
        G_segment = construct_graph_for_region(
            region_id_segment,
            cell_coords_file=cell_coords_file_segment,
            cell_types_file=None,
            cell_biomarker_expression_file=None,
            #cell_biomarker_expression_file=cell_biomarker_expression_file,
            cell_features_file=None,
            voronoi_file=None,
            graph_source=graph_source_str,
            graph_output=None,                                               #graph_output_xenium,
            voronoi_polygon_img_output=None,                            #voronoi_polygon_img_output_xenium,
            graph_img_output=None,                                      #graph_img_output_xenium,
            figsize=fig_size)
        
        #filterd outlier
        G_xenium_edge_list = list(G_xenium.edges)
        G_segment_edge_list = list(G_segment.edges)

        for filter_list_index in range(len(G_xenium_edge_list)):
            current_G_xenium_node_edge = G_xenium_edge_list[filter_list_index]
            current_G_segment_node_edge = G_segment_edge_list[filter_list_index]
            if current_G_xenium_node_edge != current_G_segment_node_edge:
                current_G_xenium_node_edge_source = current_G_xenium_node_edge[0]
                current_G_segment_node_edge_source = current_G_segment_node_edge[0]
                current_G_segment_node_edge_remove = min(current_G_xenium_node_edge_source,current_G_segment_node_edge_source)
                break
        
        if filter_list_index == len(G_xenium_edge_list)-1:
            #print('All graph edges are matched between xenium and segment !!!')
            break
        
        if current_G_segment_node_edge_remove >= 0:
            #update pixel values
            current_search_cell_list_use_coords_file_xenium_pd = pd.read_csv(cell_coords_file_xenium,index_col=0)
            current_search_cell_list_use_coords_file_segment_pd = pd.read_csv(cell_coords_file_segment,index_col=0)
            
            current_search_cell_list_use_coords_file_xenium_index = current_search_cell_list_use_coords_file_xenium_pd.index.tolist()
            current_search_cell_list_use_coords_file_segment_index = current_search_cell_list_use_coords_file_segment_pd.index.tolist()
            
            del current_search_cell_list_use_coords_file_xenium_index[current_G_segment_node_edge_remove]
            del current_search_cell_list_use_coords_file_segment_index[current_G_segment_node_edge_remove]
            
            update_search_cell_list_use_coords_file_xenium_pd = current_search_cell_list_use_coords_file_xenium_pd.loc[current_search_cell_list_use_coords_file_xenium_index]
            update_search_cell_list_use_coords_file_segment_pd = current_search_cell_list_use_coords_file_segment_pd.loc[current_search_cell_list_use_coords_file_segment_index]
            
            count = count + 1
            
            update_search_cell_list_use_coords_file_xenium_pd.to_csv(dataset_root_xenium+learning_name+'_inverse_xenium_for_delaunay_pixel_values_count'+str(count)+'.csv',index=True)
            update_search_cell_list_use_coords_file_segment_pd.to_csv(dataset_root_segment+learning_name+'_inverse_segment_for_delaunay_pixel_values_count'+str(count)+'.csv',index=True)
        
        current_G_segment_node_edge_remove = 'prepare to update!'
    
    if count == 0:
        update_search_cell_list_use_coords_file_xenium_pd = xenium_coords_for_graph_pd_inverse0
        update_search_cell_list_use_coords_file_segment_pd = segment_coords_for_graph_pd_inverse0
    
    update_search_cell_list_use_coords_file_segment_pd_explorer = update_search_cell_list_use_coords_file_segment_pd[['Y','X']]
    update_search_cell_list_use_coords_file_xenium_pd_explorer = update_search_cell_list_use_coords_file_xenium_pd

    #inverse
    update_search_cell_list_use_coords_file_segment_pd_explorer_inverse0 = update_search_cell_list_use_coords_file_segment_pd_explorer.iloc[::-1]
    update_search_cell_list_use_coords_file_segment_pd_explorer_inverse1 = update_search_cell_list_use_coords_file_segment_pd_explorer.reindex(reversed(update_search_cell_list_use_coords_file_segment_pd_explorer.index))
    assert((np.array(update_search_cell_list_use_coords_file_segment_pd_explorer_inverse0.index.tolist())==np.array(update_search_cell_list_use_coords_file_segment_pd_explorer_inverse1.index.tolist())).all())
    
    update_search_cell_list_use_coords_file_xenium_pd_explorer_inverse0 = update_search_cell_list_use_coords_file_xenium_pd_explorer.iloc[::-1]
    update_search_cell_list_use_coords_file_xenium_pd_explorer_inverse1 = update_search_cell_list_use_coords_file_xenium_pd_explorer.reindex(reversed(update_search_cell_list_use_coords_file_xenium_pd_explorer.index))
    assert((np.array(update_search_cell_list_use_coords_file_xenium_pd_explorer_inverse0.index.tolist())==np.array(update_search_cell_list_use_coords_file_xenium_pd_explorer_inverse1.index.tolist())).all())

    update_search_cell_list_use_coords_file_xenium_explorer_np = update_search_cell_list_use_coords_file_xenium_pd_explorer_inverse0.values
    update_search_cell_list_use_coords_file_segment_explorer_np = update_search_cell_list_use_coords_file_segment_pd_explorer_inverse0.values
    
    update_search_cell_list_use_coords_file_xenium_segment_explorer_np = np.hstack((update_search_cell_list_use_coords_file_xenium_explorer_np,update_search_cell_list_use_coords_file_segment_explorer_np))
    update_xenium_segment_explorer_columns_np = np.array(['fixedX','fixedY','alignmentX','alignmentY']).reshape(1,-1)
    update_xenium_segment_explorer_np = np.vstack((update_xenium_segment_explorer_columns_np,update_search_cell_list_use_coords_file_xenium_segment_explorer_np))
    
    ###with open(keypoints_save_path+learning_name+'_filtered_by_multi_angle_and_graph_inverse_epoch'+str(current_epoch_num)+'_keypoints.csv', 'w', newline='') as file:
    ###    writer = csv.writer(file)  #lineterminator='\n'
    ###    count_row = 0
    ###    for row_num in range(update_xenium_segment_explorer_np.shape[0]):
    ###        count_row = count_row + 1
    ###        if count_row < update_xenium_segment_explorer_np.shape[0]:
    ###            current_row = update_xenium_segment_explorer_np[row_num,:].tolist()
    ###            writer.writerow(current_row)
    ###        else:
    ###            writer = csv.writer(file,lineterminator='')
    ###            current_row = update_xenium_segment_explorer_np[row_num,:].tolist()
    ###            writer.writerow(current_row)
    
    count_row = update_xenium_segment_explorer_np.shape[0]   #add

    filtered_keypoints_num = update_xenium_segment_explorer_np.shape[0]

    #save xenium barcode list
    graph_inverse_xenium_barcode_list = update_search_cell_list_use_coords_file_xenium_pd_explorer_inverse0.index.tolist()
    #print('graph_inverse_xenium_barcode_list',graph_inverse_xenium_barcode_list)

    return filtered_keypoints_num, update_xenium_segment_explorer_np, count_row, graph_inverse_xenium_barcode_list

def save_xenium_segment_explorer_keypoints(keypoints_save_path,sample,filtered_keypoints_num,xenium_segment_explorer_np,filtered_keypoints_inverse_num,xenium_segment_explorer_inverse_np, current_epoch_num, graph_xenium_barcode_list, graph_inverse_xenium_barcode_list):
    
    #set xenium_segment_explorer_keypoints
    if filtered_keypoints_num >= filtered_keypoints_inverse_num:
        xenium_segment_explorer_keypoints = xenium_segment_explorer_np
        xenium_segment_explorer_keypoints_xenium_barcode_list = graph_xenium_barcode_list
    else:
        xenium_segment_explorer_keypoints = xenium_segment_explorer_inverse_np
        xenium_segment_explorer_keypoints_xenium_barcode_list = graph_inverse_xenium_barcode_list
    
    print('The keypoints filtered by multi angles and Delaunay triangulation graph of epoch'+str(current_epoch_num)+' have been generated!')
    
    ###with open(keypoints_save_path+sample+'_epoch'+str(current_epoch_num)+'_keypoints.csv', 'w', newline='') as file:
    ###    writer = csv.writer(file)  #lineterminator='\n'
    ###    count_row = 0
    ###    for row_num in range(xenium_segment_explorer_keypoints.shape[0]):
    ###        count_row = count_row + 1
    ###        if count_row < xenium_segment_explorer_keypoints.shape[0]:
    ###            current_row = xenium_segment_explorer_keypoints[row_num,:].tolist()
    ###            writer.writerow(current_row)
    ###        else:
    ###            writer = csv.writer(file,lineterminator='')
    ###            current_row = xenium_segment_explorer_keypoints[row_num,:].tolist()
    ###            writer.writerow(current_row)
    
    count_row = xenium_segment_explorer_keypoints.shape[0]   #add
    
    return count_row, xenium_segment_explorer_keypoints, xenium_segment_explorer_keypoints_xenium_barcode_list

def xenium_segment_nucleus_coordinate_values(sdata, xenium_nucleus_pixel_values_pd, segment_nucleus_pixel_values_pd, segmentation_label_file, current_xenium_segment_keypoints_np, pixel_size, xenium_segment_explorer_keypoints_xenium_barcode_list, filtered_top1_barcode_multi_angle_list, image_check_path, segment_init_affine_image_width, segment_init_affine_image_height, sample):
    
    #data preparation
    sdata_shapes = sdata.shapes['nucleus_boundaries']['geometry']
    keypoints_num = current_xenium_segment_keypoints_np.shape[0]
    
    #data settings
    segmentation_label_pd = pd.read_csv(segmentation_label_file,index_col=0)
    segmentation_label_np = segmentation_label_pd.values
    
    segment_nucleus_hull_list = []
    xenium_nucleus_exterior_coords_list = []
    segment_nucleus_max_min_list = []
    xenium_nucleus_max_min_list = []
    segment_nucleus_index_list = []
    xenium_nucleus_barcode_list = []
    
    #load rotate file
    rotate_mse_list_pd = pd.read_csv(image_check_path+sample+'_he_rotate_mse_values.csv',index_col=0)
    rotate_full_list = rotate_mse_list_pd.columns.tolist()
    
    rotate_mse_list_np = rotate_mse_list_pd.values.reshape(1,-1)
    row_max_index = np.argmax(rotate_mse_list_np)
    rotate_full_name = rotate_full_list[row_max_index]
    rotate_name_parts = rotate_full_name.split('_')
    rotate_name_init = rotate_name_parts[0]
    rotate_name_last = rotate_name_parts[1]
    
    count = 0
    for keypoints_num_index in range(keypoints_num):
        count = count + 1
        current_segment_nucleus_pixel_values_pd = segment_nucleus_pixel_values_pd[(segment_nucleus_pixel_values_pd['x_center_pixel']==current_xenium_segment_keypoints_np[keypoints_num_index,2])&(segment_nucleus_pixel_values_pd['y_center_pixel']==current_xenium_segment_keypoints_np[keypoints_num_index,3])]
        #print('current_segment_nucleus_pixel_values_pd',current_segment_nucleus_pixel_values_pd)
        current_segment_nucleus_label = current_segment_nucleus_pixel_values_pd.index.tolist()[0]+1
        #print('current_segment_nucleus_label',current_segment_nucleus_label)
        segment_nucleus_index_list.append(current_segment_nucleus_pixel_values_pd.index.tolist()[0])
        
        labels_coords_current = np.where(segmentation_label_np==current_segment_nucleus_label)
        assert(labels_coords_current[0].shape[0]==labels_coords_current[1].shape[0])
        
        #rotate
        if rotate_name_init == 'rotate90':
            if rotate_name_last == 'original':
                current_y_pixel = segment_init_affine_image_height - labels_coords_current[1].reshape(-1,1)
                current_x_pixel = labels_coords_current[0].reshape(-1,1)
                
            elif rotate_name_last == 'flipLR':
                current_y_pixel = segment_init_affine_image_height - labels_coords_current[1].reshape(-1,1)
                current_x_pixel = segment_init_affine_image_width - labels_coords_current[0].reshape(-1,1)
            
            elif rotate_name_last == 'flipTB':
                current_y_pixel = labels_coords_current[1].reshape(-1,1)
                current_x_pixel = labels_coords_current[0].reshape(-1,1)
                
            elif rotate_name_last == 'rotate180':
                current_y_pixel = labels_coords_current[1].reshape(-1,1)
                current_x_pixel = segment_init_affine_image_width - labels_coords_current[0].reshape(-1,1)
        
        elif rotate_name_init == 'rotate0':
            if rotate_name_last == 'original':
                current_y_pixel = labels_coords_current[0].reshape(-1,1)
                current_x_pixel = labels_coords_current[1].reshape(-1,1)
                
            elif rotate_name_last == 'flipLR':
                current_y_pixel = labels_coords_current[0].reshape(-1,1)
                current_x_pixel = segment_init_affine_image_width - labels_coords_current[1].reshape(-1,1)
                
            elif rotate_name_last == 'flipTB':
                current_y_pixel = segment_init_affine_image_height - labels_coords_current[0].reshape(-1,1)
                current_x_pixel = labels_coords_current[1].reshape(-1,1)
                
            elif rotate_name_last == 'rotate180':
                current_y_pixel = segment_init_affine_image_height - labels_coords_current[0].reshape(-1,1)
                current_x_pixel = segment_init_affine_image_width - labels_coords_current[1].reshape(-1,1)
        
        current_xy_pixel = np.hstack((current_x_pixel,current_y_pixel))
        current_hull = ConvexHull(current_xy_pixel)
        #print('current_hull',current_hull)
        current_hull_pixel = current_xy_pixel[current_hull.vertices]
        #print('current_hull_pixel',current_hull_pixel)
        current_hull_pixel_start_end = current_hull_pixel[0,:].reshape(1,-1)
        current_hull_pixel_used = np.vstack((current_hull_pixel,current_hull_pixel_start_end))
        
        current_segment_hull_coords_list = []
        for hull_index in range(current_hull_pixel_used.shape[0]):
            current_hull_tuple = (current_hull_pixel_used[hull_index,0], current_hull_pixel_used[hull_index,1])
            current_segment_hull_coords_list.append(current_hull_tuple)
        #print('current_segment_hull_coords_list',current_segment_hull_coords_list)
        segment_nucleus_hull_list.append(current_segment_hull_coords_list)
        
        #rotate
        if rotate_name_init == 'rotate90':
            if rotate_name_last == 'original':
                xmin_segment_nucleus_pixel = np.min(labels_coords_current[0])
                xmax_segment_nucleus_pixel = np.max(labels_coords_current[0])
                ymin_segment_nucleus_pixel = segment_init_affine_image_height - np.min(labels_coords_current[1])
                ymax_segment_nucleus_pixel = segment_init_affine_image_height - np.max(labels_coords_current[1])
                
            elif rotate_name_last == 'flipLR':
                xmin_segment_nucleus_pixel = segment_init_affine_image_width - np.min(labels_coords_current[0])
                xmax_segment_nucleus_pixel = segment_init_affine_image_width - np.max(labels_coords_current[0])
                ymin_segment_nucleus_pixel = segment_init_affine_image_height - np.min(labels_coords_current[1])
                ymax_segment_nucleus_pixel = segment_init_affine_image_height - np.max(labels_coords_current[1])
            
            elif rotate_name_last == 'flipTB':
                xmin_segment_nucleus_pixel = np.min(labels_coords_current[0])
                xmax_segment_nucleus_pixel = np.max(labels_coords_current[0])
                ymin_segment_nucleus_pixel = np.min(labels_coords_current[1])
                ymax_segment_nucleus_pixel = np.max(labels_coords_current[1])
                
            elif rotate_name_last == 'rotate180':
                xmin_segment_nucleus_pixel = segment_init_affine_image_width - np.min(labels_coords_current[0])
                xmax_segment_nucleus_pixel = segment_init_affine_image_width - np.max(labels_coords_current[0])
                ymin_segment_nucleus_pixel = np.min(labels_coords_current[1])
                ymax_segment_nucleus_pixel = np.max(labels_coords_current[1])
        
        elif rotate_name_init == 'rotate0':
            if rotate_name_last == 'original':
                xmin_segment_nucleus_pixel = np.min(labels_coords_current[1])
                xmax_segment_nucleus_pixel = np.max(labels_coords_current[1])
                ymin_segment_nucleus_pixel = np.min(labels_coords_current[0])
                ymax_segment_nucleus_pixel = np.max(labels_coords_current[0])
                
            elif rotate_name_last == 'flipLR':
                xmin_segment_nucleus_pixel = segment_init_affine_image_width - np.min(labels_coords_current[1])
                xmax_segment_nucleus_pixel = segment_init_affine_image_width - np.max(labels_coords_current[1])
                ymin_segment_nucleus_pixel = np.min(labels_coords_current[0])
                ymax_segment_nucleus_pixel = np.max(labels_coords_current[0])
                
            elif rotate_name_last == 'flipTB':
                xmin_segment_nucleus_pixel = np.min(labels_coords_current[1])
                xmax_segment_nucleus_pixel = np.max(labels_coords_current[1])
                ymin_segment_nucleus_pixel = segment_init_affine_image_height - np.min(labels_coords_current[0])
                ymax_segment_nucleus_pixel = segment_init_affine_image_height - np.max(labels_coords_current[0])
                
            elif rotate_name_last == 'rotate180':
                xmin_segment_nucleus_pixel = segment_init_affine_image_width - np.min(labels_coords_current[1])
                xmax_segment_nucleus_pixel = segment_init_affine_image_width - np.max(labels_coords_current[1])
                ymin_segment_nucleus_pixel = segment_init_affine_image_height - np.min(labels_coords_current[0])
                ymax_segment_nucleus_pixel = segment_init_affine_image_height - np.max(labels_coords_current[0])
        
        current_segment_nucleus_max_min_list = [xmin_segment_nucleus_pixel, xmax_segment_nucleus_pixel, ymin_segment_nucleus_pixel, ymax_segment_nucleus_pixel]
        #print('current_segment_nucleus_max_min_list',current_segment_nucleus_max_min_list)
        segment_nucleus_max_min_list.append(current_segment_nucleus_max_min_list)
        
        #current_xenium_nucleus_label by list 
        xenium_segment_explorer_keypoints_xenium_barcode_index = xenium_segment_explorer_keypoints_xenium_barcode_list[keypoints_num_index]
        current_xenium_nucleus_label = filtered_top1_barcode_multi_angle_list[xenium_segment_explorer_keypoints_xenium_barcode_index]
        #print('current_xenium_nucleus_label',current_xenium_nucleus_label)

        xenium_nucleus_barcode_list.append(current_xenium_nucleus_label)
        current_xenium_geometry = sdata_shapes.loc[current_xenium_nucleus_label]
        #print('current_xenium_geometry',current_xenium_geometry)
        current_xenium_geometry_exterior_coords_list = list(current_xenium_geometry.exterior.coords)
        
        current_xenium_geometry_exterior_coords_pixel_list = []
        for xenium_geometry_exterior_coords_index in range(len(current_xenium_geometry_exterior_coords_list)):
            current_xenium_geometry_exterior_coords_tuple = current_xenium_geometry_exterior_coords_list[xenium_geometry_exterior_coords_index]
            current_xenium_geometry_exterior_coords_pixel_tuple = (current_xenium_geometry_exterior_coords_tuple[0]/pixel_size , current_xenium_geometry_exterior_coords_tuple[1]/pixel_size)
            current_xenium_geometry_exterior_coords_pixel_list.append(current_xenium_geometry_exterior_coords_pixel_tuple)
        
        #print('current_xenium_geometry_exterior_coords_pixel_list',current_xenium_geometry_exterior_coords_pixel_list)
        xenium_nucleus_exterior_coords_list.append(current_xenium_geometry_exterior_coords_pixel_list)
        xmin_xenium_nucleus_micron, ymin_xenium_nucleus_micron, xmax_xenium_nucleus_micron, ymax_xenium_nucleus_micron = current_xenium_geometry.bounds
        current_xenium_nucleus_max_min_list = [xmin_xenium_nucleus_micron/pixel_size, xmax_xenium_nucleus_micron/pixel_size, ymin_xenium_nucleus_micron/pixel_size, ymax_xenium_nucleus_micron/pixel_size]
        #print('current_xenium_nucleus_max_min_list',current_xenium_nucleus_max_min_list)
        xenium_nucleus_max_min_list.append(current_xenium_nucleus_max_min_list)
    
    return xenium_nucleus_exterior_coords_list, segment_nucleus_hull_list, xenium_nucleus_max_min_list, segment_nucleus_max_min_list, xenium_nucleus_barcode_list, segment_nucleus_index_list
    
####consider relative resize
def xenium_segment_nucleus_coordinate_transform_relative_resize(xenium_nucleus_exterior_coords_list, segment_nucleus_hull_list, xenium_nucleus_max_min_list, segment_nucleus_max_min_list, xenium_nucleus_barcode_list, segment_nucleus_index_list, output_path, xenium_nucleus_pixel_values_pd, xenium_mip_ome_tif_image, segment_nucleus_pixel_values_pd, segment_label_image_init_affine, crop_image_resize, current_epoch_num, transform_ratio, image_check_path, segment_init_affine_image_width, segment_init_affine_image_height, sample):
    
    #os setting
    keypoints_polygon_check_save_path = output_path+'/keypoints_polygon_check_epoch'+str(current_epoch_num)+'_save/'
    if not os.path.exists(keypoints_polygon_check_save_path):
        os.makedirs(keypoints_polygon_check_save_path)

    keypoints_polygon_patch_save_path = output_path+'/keypoints_polygon_patch_epoch'+str(current_epoch_num)+'_save/'
    if not os.path.exists(keypoints_polygon_patch_save_path):
        os.makedirs(keypoints_polygon_patch_save_path)
    
    #list settings
    xenium_geometry_exterior_coords_transform_list = []
    xenium_nucleus_max_min_transform_list = []
    segment_hull_coords_transform_list = []
    segment_nucleus_max_min_transform_list = []
    
    xenium_geometry_exterior_coords_transform_resize_list = []
    xenium_nucleus_transform_resize_max_min_list = []
    segment_hull_coords_transform_resize_list = []
    segment_nucleus_transform_resize_max_min_list = []
    
    xenium_geometry_exterior_coords_transform_resize_patch_list = []
    segment_hull_coords_transform_resize_patch_list = []
    
    #load rotate file
    rotate_mse_list_pd = pd.read_csv(image_check_path+sample+'_he_rotate_mse_values.csv',index_col=0)
    rotate_full_list = rotate_mse_list_pd.columns.tolist()
    
    rotate_mse_list_np = rotate_mse_list_pd.values.reshape(1,-1)
    row_max_index = np.argmax(rotate_mse_list_np)
    rotate_full_name = rotate_full_list[row_max_index]
    rotate_name_parts = rotate_full_name.split('_')
    rotate_name_init = rotate_name_parts[0]
    rotate_name_last = rotate_name_parts[1]

    #count=0
    for check_cell_index in range(len(segment_nucleus_hull_list)):
        #count=count+1
        #print('count!!!!!!',count)
        ###xenium
        current_xenium_geometry_exterior_coords_list = xenium_nucleus_exterior_coords_list[check_cell_index]
        #print('current_xenium_geometry_exterior_coords_list',current_xenium_geometry_exterior_coords_list)
        current_xenium_nucleus_max_min_list = xenium_nucleus_max_min_list[check_cell_index]
        current_xenium_nucleus_radius = max((current_xenium_nucleus_max_min_list[1]-current_xenium_nucleus_max_min_list[0])/2,(current_xenium_nucleus_max_min_list[3]-current_xenium_nucleus_max_min_list[2])/2)
        current_xenium_nucleus_x_center = (current_xenium_nucleus_max_min_list[1]-current_xenium_nucleus_max_min_list[0])/2+current_xenium_nucleus_max_min_list[0]
        current_xenium_nucleus_y_center = (current_xenium_nucleus_max_min_list[3]-current_xenium_nucleus_max_min_list[2])/2+current_xenium_nucleus_max_min_list[2]
        #print('current_xenium_nucleus_radius',current_xenium_nucleus_radius)
        #print('current_xenium_nucleus_x_center',current_xenium_nucleus_x_center)
        #print('current_xenium_nucleus_y_center',current_xenium_nucleus_y_center)
        
        current_xenium_nucleus_to_x_dis = current_xenium_nucleus_x_center - current_xenium_nucleus_radius
        current_xenium_nucleus_to_y_dis = current_xenium_nucleus_y_center - current_xenium_nucleus_radius
        #print('current_xenium_nucleus_to_x_dis',current_xenium_nucleus_to_x_dis)
        #print('current_xenium_nucleus_to_y_dis',current_xenium_nucleus_to_y_dis)
        
        current_xenium_geometry_exterior_coords_transform_list = []
        for xenium_check_index in range(len(current_xenium_geometry_exterior_coords_list)):
            current_xenium_geometry_exterior_coords_tuple = current_xenium_geometry_exterior_coords_list[xenium_check_index]
            current_xenium_geometry_exterior_coords_transform_tuple = (current_xenium_geometry_exterior_coords_tuple[0]-current_xenium_nucleus_to_x_dis , current_xenium_geometry_exterior_coords_tuple[1]-current_xenium_nucleus_to_y_dis)
            current_xenium_geometry_exterior_coords_transform_list.append(current_xenium_geometry_exterior_coords_transform_tuple)
        #print('current_xenium_geometry_exterior_coords_transform_list',current_xenium_geometry_exterior_coords_transform_list)
        xenium_geometry_exterior_coords_transform_list.append(current_xenium_geometry_exterior_coords_transform_list)
        
        current_xenium_geometry_exterior_coords_max_min_transform_list = []
        for xenium_max_min_index in range(len(current_xenium_nucleus_max_min_list)):
            current_xenium_max_min = current_xenium_nucleus_max_min_list[xenium_max_min_index]
            if xenium_max_min_index <=1:
                current_xenium_max_min_transform = current_xenium_max_min - current_xenium_nucleus_to_x_dis
            elif xenium_max_min_index >1:
                current_xenium_max_min_transform = current_xenium_max_min - current_xenium_nucleus_to_y_dis
            current_xenium_geometry_exterior_coords_max_min_transform_list.append(current_xenium_max_min_transform)
        #print('current_xenium_geometry_exterior_coords_max_min_transform_list',current_xenium_geometry_exterior_coords_max_min_transform_list)
        xenium_nucleus_max_min_transform_list.append(current_xenium_geometry_exterior_coords_max_min_transform_list)
        
        #save xenium_nucleus_polygons
        current_xenium_nucleus_polygons = np.asarray(current_xenium_geometry_exterior_coords_transform_list, np.int32)
        current_xenium_nucleus_mask = np.zeros([int(current_xenium_nucleus_radius*2+1),int(current_xenium_nucleus_radius*2+1)], dtype=np.uint8)
        cv2.fillConvexPoly(current_xenium_nucleus_mask, current_xenium_nucleus_polygons, 255)
        cv2.imwrite(keypoints_polygon_patch_save_path+'cell'+str(check_cell_index)+'_barcode_'+xenium_nucleus_barcode_list[check_cell_index]+'_xenium_nucleus_transform_mask.jpg',current_xenium_nucleus_mask)
        
        #image resize
        current_xenium_nucleus_mask_pillow = Image.fromarray(current_xenium_nucleus_mask)
        current_xenium_nucleus_mask_pillow_resize = current_xenium_nucleus_mask_pillow.resize((crop_image_resize,crop_image_resize))
        current_xenium_nucleus_mask_pillow_resize_np = np.array(current_xenium_nucleus_mask_pillow_resize)
        current_xenium_nucleus_mask_pillow_resize.save(keypoints_polygon_patch_save_path+'cell'+str(check_cell_index)+'_barcode_'+xenium_nucleus_barcode_list[check_cell_index]+'_resize'+str(crop_image_resize)+'_xenium_nucleus_transform_mask.jpg', "JPEG")
        
        #save xenium_nucleus_patches
        current_xenium_nucleus_label = xenium_nucleus_barcode_list[check_cell_index]
        current_xenium_nucleus_pixel_values_pd = xenium_nucleus_pixel_values_pd.loc[current_xenium_nucleus_label]
        #print('current_xenium_nucleus_pixel_values_pd[pixel_center_radius_nucleus]',current_xenium_nucleus_pixel_values_pd['pixel_center_radius_nucleus'])
        #print(type(current_xenium_nucleus_pixel_values_pd['pixel_center_radius_nucleus']))
        current_crop_radius_pixel = current_xenium_nucleus_pixel_values_pd['pixel_center_radius_nucleus']
        current_pixel_x = current_xenium_nucleus_pixel_values_pd['pixel_center_x_nucleus']
        current_pixel_y = current_xenium_nucleus_pixel_values_pd['pixel_center_y_nucleus']
        
        #crop setting
        imagerow_down  = current_pixel_y - current_crop_radius_pixel
        imagerow_up    = current_pixel_y + current_crop_radius_pixel
        imagecol_left  = current_pixel_x - current_crop_radius_pixel
        imagecol_right = current_pixel_x + current_crop_radius_pixel
        
        #xenium_nucleus_patch_crop
        xenium_mip_ome_tif_image_crop_tile = xenium_mip_ome_tif_image.crop((imagecol_left, imagerow_down, imagecol_right, imagerow_up))
        if xenium_mip_ome_tif_image_crop_tile.mode != 'RGB':
            #print('xenium_mip_ome_tif_image_crop_tile mode is',xenium_mip_ome_tif_image_crop_tile.mode)
            xenium_mip_ome_tif_image_crop_tile = xenium_mip_ome_tif_image_crop_tile.convert('RGB')
            #print('xenium_mip_ome_tif_image_crop_tile mode has been converted!')
        xenium_mip_ome_tif_image_crop_tile_resize = xenium_mip_ome_tif_image_crop_tile.resize((crop_image_resize,crop_image_resize))
        xenium_mip_ome_tif_image_crop_tile_resize_np = np.array(xenium_mip_ome_tif_image_crop_tile_resize)
        xenium_mip_ome_tif_image_crop_tile_resize.save(keypoints_polygon_patch_save_path+'cell'+str(check_cell_index)+'_barcode_'+xenium_nucleus_barcode_list[check_cell_index]+'_resize'+str(crop_image_resize)+'_xenium_mip_ome_tif_patch_pillow.jpg', "JPEG")
        
        ###segment
        current_segment_hull_coords_list = segment_nucleus_hull_list[check_cell_index]
        #print('current_segment_hull_coords_list',current_segment_hull_coords_list)
        current_segment_nucleus_max_min_list = segment_nucleus_max_min_list[check_cell_index]
        current_segment_nucleus_radius = max((current_segment_nucleus_max_min_list[1]-current_segment_nucleus_max_min_list[0])/2,(current_segment_nucleus_max_min_list[3]-current_segment_nucleus_max_min_list[2])/2)
        
        current_segment_nucleus_x_center = (current_segment_nucleus_max_min_list[1]-current_segment_nucleus_max_min_list[0])/2+current_segment_nucleus_max_min_list[0]
        current_segment_nucleus_y_center = (current_segment_nucleus_max_min_list[3]-current_segment_nucleus_max_min_list[2])/2+current_segment_nucleus_max_min_list[2]
        #print('current_segment_nucleus_radius',current_segment_nucleus_radius)
        #print('current_segment_nucleus_x_center',current_segment_nucleus_x_center)
        #print('current_segment_nucleus_y_center',current_segment_nucleus_y_center)
        
        current_segment_nucleus_to_x_dis = current_segment_nucleus_x_center - current_segment_nucleus_radius
        current_segment_nucleus_to_y_dis = current_segment_nucleus_y_center - current_segment_nucleus_radius
        #print('current_segment_nucleus_to_x_dis',current_segment_nucleus_to_x_dis)
        #print('current_segment_nucleus_to_y_dis',current_segment_nucleus_to_y_dis)
        
        current_segment_hull_coords_transform_list = []
        for segment_check_index in range(len(current_segment_hull_coords_list)):
            current_segment_hull_coords_tuple = current_segment_hull_coords_list[segment_check_index]
            current_segment_hull_coords_transform_tuple = (current_segment_hull_coords_tuple[0]-current_segment_nucleus_to_x_dis , current_segment_hull_coords_tuple[1]-current_segment_nucleus_to_y_dis)
            current_segment_hull_coords_transform_list.append(current_segment_hull_coords_transform_tuple)
        #print('current_segment_hull_coords_transform_list',current_segment_hull_coords_transform_list)
        segment_hull_coords_transform_list.append(current_segment_hull_coords_transform_list)
        
        current_segment_nucleus_max_min_transform_list = []
        for segment_max_min_index in range(len(current_segment_nucleus_max_min_list)):
            current_segment_max_min = current_segment_nucleus_max_min_list[segment_max_min_index]

            if segment_max_min_index <=1:
                current_segment_max_min_transform = current_segment_max_min - current_segment_nucleus_to_x_dis
            elif segment_max_min_index >1:
                current_segment_max_min_transform = current_segment_max_min - current_segment_nucleus_to_y_dis

            current_segment_nucleus_max_min_transform_list.append(current_segment_max_min_transform)
        #print('current_segment_nucleus_max_min_transform_list',current_segment_nucleus_max_min_transform_list)
        segment_nucleus_max_min_transform_list.append(current_segment_nucleus_max_min_transform_list)

        #save segment_nucleus_polygons
        current_segment_nucleus_polygons = np.asarray(current_segment_hull_coords_transform_list, np.int32)
        current_segment_nucleus_mask = np.zeros([int(current_segment_nucleus_radius*2+1),int(current_segment_nucleus_radius*2+1)], dtype=np.uint8)
        cv2.fillConvexPoly(current_segment_nucleus_mask, current_segment_nucleus_polygons, 255)
        cv2.imwrite(keypoints_polygon_patch_save_path+'cell'+str(check_cell_index)+'_label_index_'+str(segment_nucleus_index_list[check_cell_index])+'_segment_nucleus_transform_mask.jpg',current_segment_nucleus_mask)
        
        #image resize
        current_segment_nucleus_mask_pillow = Image.fromarray(current_segment_nucleus_mask)
        current_segment_nucleus_mask_pillow_resize = current_segment_nucleus_mask_pillow.resize((crop_image_resize,crop_image_resize))
        current_segment_nucleus_mask_pillow_resize_np = np.array(current_segment_nucleus_mask_pillow_resize)
        current_segment_nucleus_mask_pillow_resize.save(keypoints_polygon_patch_save_path+'cell'+str(check_cell_index)+'_label_index_'+str(segment_nucleus_index_list[check_cell_index])+'_resize'+str(crop_image_resize)+'_segment_nucleus_transform_mask.jpg', "JPEG")
        
        #save segment_nucleus_patches
        current_segment_nucleus_label = segment_nucleus_index_list[check_cell_index]
        current_segment_nucleus_pixel_values_pd = segment_nucleus_pixel_values_pd.loc[current_segment_nucleus_label]
        current_crop_radius_pixel = current_segment_nucleus_pixel_values_pd['center_radius_pixel']
        
        #rotate
        if rotate_name_init == 'rotate90':
            if rotate_name_last == 'original':
                current_pixel_x = current_segment_nucleus_pixel_values_pd['y_center_pixel']
                current_pixel_y = segment_init_affine_image_height - current_segment_nucleus_pixel_values_pd['x_center_pixel']
                
            elif rotate_name_last == 'flipLR':
                current_pixel_x = segment_init_affine_image_width - current_segment_nucleus_pixel_values_pd['y_center_pixel']
                current_pixel_y = segment_init_affine_image_height - current_segment_nucleus_pixel_values_pd['x_center_pixel']

            elif rotate_name_last == 'flipTB':
                current_pixel_x = current_segment_nucleus_pixel_values_pd['y_center_pixel']
                current_pixel_y = current_segment_nucleus_pixel_values_pd['x_center_pixel']

            elif rotate_name_last == 'rotate180':
                current_pixel_x = segment_init_affine_image_width - current_segment_nucleus_pixel_values_pd['y_center_pixel']
                current_pixel_y = current_segment_nucleus_pixel_values_pd['x_center_pixel']
        
        elif rotate_name_init == 'rotate0':
            if rotate_name_last == 'original':
                current_pixel_x = current_segment_nucleus_pixel_values_pd['x_center_pixel']
                current_pixel_y = current_segment_nucleus_pixel_values_pd['y_center_pixel']
                
            elif rotate_name_last == 'flipLR':
                current_pixel_x = segment_init_affine_image_width - current_segment_nucleus_pixel_values_pd['x_center_pixel']
                current_pixel_y = current_segment_nucleus_pixel_values_pd['y_center_pixel']
                
            elif rotate_name_last == 'flipTB':
                current_pixel_x = current_segment_nucleus_pixel_values_pd['x_center_pixel']
                current_pixel_y = segment_init_affine_image_height - current_segment_nucleus_pixel_values_pd['y_center_pixel']
                
            elif rotate_name_last == 'rotate180':
                current_pixel_x = segment_init_affine_image_width - current_segment_nucleus_pixel_values_pd['x_center_pixel']
                current_pixel_y = segment_init_affine_image_height - current_segment_nucleus_pixel_values_pd['y_center_pixel']
        
        #crop setting
        imagerow_down  = current_pixel_y - current_crop_radius_pixel
        imagerow_up    = current_pixel_y + current_crop_radius_pixel
        imagecol_left  = current_pixel_x - current_crop_radius_pixel
        imagecol_right = current_pixel_x + current_crop_radius_pixel
        
        #segment_label_patch_crop
        segment_label_image_crop_tile = segment_label_image_init_affine.crop((imagecol_left, imagerow_down, imagecol_right, imagerow_up))
        if segment_label_image_crop_tile.mode != 'RGB':
            #print('segment_label_image_crop_tile mode is',segment_label_image_crop_tile.mode)
            segment_label_image_crop_tile = segment_label_image_crop_tile.convert('RGB')
            #print('segment_label_image_crop_tile mode has been converted!')
        segment_label_image_crop_tile_resize = segment_label_image_crop_tile.resize((crop_image_resize,crop_image_resize))
        segment_label_image_crop_tile_resize_np = np.array(segment_label_image_crop_tile_resize)
        segment_label_image_crop_tile_resize.save(keypoints_polygon_patch_save_path+'cell'+str(check_cell_index)+'_label_index_'+str(segment_nucleus_index_list[check_cell_index])+'_polygon_resize'+str(crop_image_resize)+'_segment_label_image_patch_pillow.jpg', "JPEG")
        
        ###polygon relative resize
        current_segment_nucleus_radius_ratio = current_segment_nucleus_radius * transform_ratio
        #current_xenium_segment_radius_actual_ratio = current_xenium_nucleus_radius / current_segment_nucleus_radius_ratio
        #print('current_xenium_segment_radius_actual_ratio',current_xenium_segment_radius_actual_ratio)

        if current_xenium_nucleus_radius >= current_segment_nucleus_radius_ratio:
            #print('current_xenium_nucleus_radius is larger than current_segment_nucleus_radius_ratio !')
            
            #xenium polygon resize
            xenium_polygon_transform_ratio = current_xenium_nucleus_radius*2/(crop_image_resize-1)
            current_xenium_geometry_exterior_coords_transform_resize_list = []
            for xenium_transform_list_index in range(len(current_xenium_geometry_exterior_coords_transform_list)):
                current_xenium_geometry_exterior_coords_transform_tuple = current_xenium_geometry_exterior_coords_transform_list[xenium_transform_list_index]
                current_xenium_geometry_exterior_coords_transform_resize_tuple = (current_xenium_geometry_exterior_coords_transform_tuple[0]/xenium_polygon_transform_ratio, current_xenium_geometry_exterior_coords_transform_tuple[1]/xenium_polygon_transform_ratio)
                current_xenium_geometry_exterior_coords_transform_resize_list.append(current_xenium_geometry_exterior_coords_transform_resize_tuple)
            #print('current_xenium_geometry_exterior_coords_transform_resize_list',current_xenium_geometry_exterior_coords_transform_resize_list)
            xenium_geometry_exterior_coords_transform_resize_list.append(current_xenium_geometry_exterior_coords_transform_resize_list)
            
            #xenium_transform_resize_max_min_list
            current_xenium_nucleus_transform_resize_max_min_list = [current_xenium_geometry_exterior_coords_max_min_transform_list[0]/xenium_polygon_transform_ratio,current_xenium_geometry_exterior_coords_max_min_transform_list[1]/xenium_polygon_transform_ratio,current_xenium_geometry_exterior_coords_max_min_transform_list[2]/xenium_polygon_transform_ratio,current_xenium_geometry_exterior_coords_max_min_transform_list[3]/xenium_polygon_transform_ratio]
            #print('current_xenium_nucleus_transform_resize_max_min_list',current_xenium_nucleus_transform_resize_max_min_list)
            xenium_nucleus_transform_resize_max_min_list.append(current_xenium_nucleus_transform_resize_max_min_list)
            
            current_xenium_nucleus_polygons_resize = np.asarray(current_xenium_geometry_exterior_coords_transform_resize_list, np.int32)
            current_xenium_nucleus_mask_resize = np.zeros([crop_image_resize,crop_image_resize], dtype=np.uint8)
            cv2.fillConvexPoly(current_xenium_nucleus_mask_resize, current_xenium_nucleus_polygons_resize, 255)
            cv2.imwrite(keypoints_polygon_patch_save_path+'cell'+str(check_cell_index)+'_barcode_'+xenium_nucleus_barcode_list[check_cell_index]+'_polygon_relative_resize'+str(crop_image_resize)+'_xenium_nucleus_transform_mask.jpg',current_xenium_nucleus_mask_resize)
            xenium_geometry_exterior_coords_transform_resize_patch_list.append(current_xenium_nucleus_mask_resize)
            
            #xenium_nucleus_transform_resize_center
            current_xenium_nucleus_transform_resize_center_x = (crop_image_resize-1)/2
            current_xenium_nucleus_transform_resize_center_y = (crop_image_resize-1)/2
            
            #segment polygon resize
            current_segment_hull_coords_transform_resize_init_list = []
            for segment_transform_list_init_index in range(len(current_segment_hull_coords_transform_list)):
                current_segment_hull_coords_transform_init_tuple = current_segment_hull_coords_transform_list[segment_transform_list_init_index]
                current_segment_hull_coords_transform_resize_init_tuple = (current_segment_hull_coords_transform_init_tuple[0]*transform_ratio/xenium_polygon_transform_ratio, current_segment_hull_coords_transform_init_tuple[1]*transform_ratio/xenium_polygon_transform_ratio)
                current_segment_hull_coords_transform_resize_init_list.append(current_segment_hull_coords_transform_resize_init_tuple)
            #print('current_segment_hull_coords_transform_resize_init_list',current_segment_hull_coords_transform_resize_init_list)
            
            #segment_transform_resize_max_min_init_list
            current_segment_nucleus_transform_resize_max_min_init_list = [current_segment_nucleus_max_min_transform_list[0]*transform_ratio/xenium_polygon_transform_ratio,current_segment_nucleus_max_min_transform_list[1]*transform_ratio/xenium_polygon_transform_ratio, current_segment_nucleus_max_min_transform_list[2]*transform_ratio/xenium_polygon_transform_ratio,current_segment_nucleus_max_min_transform_list[3]*transform_ratio/xenium_polygon_transform_ratio]
            #print('current_segment_nucleus_transform_resize_max_min_init_list',current_segment_nucleus_transform_resize_max_min_init_list)
            
            #segment_nucleus_transform_resize_init_center
            current_segment_nucleus_transform_resize_init_center_x = (current_segment_nucleus_transform_resize_max_min_init_list[1] + current_segment_nucleus_transform_resize_max_min_init_list[0])/2
            current_segment_nucleus_transform_resize_init_center_y = (current_segment_nucleus_transform_resize_max_min_init_list[3] + current_segment_nucleus_transform_resize_max_min_init_list[2])/2
                        
            #segment_nucleus_transform_resize_move_dis
            current_segment_nucleus_transform_resize_move_x = current_xenium_nucleus_transform_resize_center_x - current_segment_nucleus_transform_resize_init_center_x
            current_segment_nucleus_transform_resize_move_y = current_xenium_nucleus_transform_resize_center_y - current_segment_nucleus_transform_resize_init_center_y
            
            current_segment_hull_coords_transform_resize_list = []
            for segment_transform_list_index in range(len(current_segment_hull_coords_transform_resize_init_list)):
                current_segment_hull_coords_transform_init_tuple = current_segment_hull_coords_transform_resize_init_list[segment_transform_list_index]
                if current_segment_nucleus_transform_resize_move_x >= 0 and current_segment_nucleus_transform_resize_move_y >= 0 :
                    current_segment_hull_coords_transform_resize_tuple = (current_segment_hull_coords_transform_init_tuple[0]+current_segment_nucleus_transform_resize_move_x, current_segment_hull_coords_transform_init_tuple[1]+current_segment_nucleus_transform_resize_move_y)
                elif current_segment_nucleus_transform_resize_move_x >= 0 and current_segment_nucleus_transform_resize_move_y < 0 :
                    current_segment_hull_coords_transform_resize_tuple = (current_segment_hull_coords_transform_init_tuple[0]+current_segment_nucleus_transform_resize_move_x, current_segment_hull_coords_transform_init_tuple[1]-current_segment_nucleus_transform_resize_move_y)
                elif current_segment_nucleus_transform_resize_move_x < 0 and current_segment_nucleus_transform_resize_move_y >= 0 :
                    current_segment_hull_coords_transform_resize_tuple = (current_segment_hull_coords_transform_init_tuple[0]-current_segment_nucleus_transform_resize_move_x, current_segment_hull_coords_transform_init_tuple[1]+current_segment_nucleus_transform_resize_move_y)
                elif current_segment_nucleus_transform_resize_move_x < 0 and current_segment_nucleus_transform_resize_move_y < 0 :
                    current_segment_hull_coords_transform_resize_tuple = (current_segment_hull_coords_transform_init_tuple[0]-current_segment_nucleus_transform_resize_move_x, current_segment_hull_coords_transform_init_tuple[1]-current_segment_nucleus_transform_resize_move_y)
                    
                current_segment_hull_coords_transform_resize_list.append(current_segment_hull_coords_transform_resize_tuple)
            #print('current_segment_hull_coords_transform_resize_list',current_segment_hull_coords_transform_resize_list)
            segment_hull_coords_transform_resize_list.append(current_segment_hull_coords_transform_resize_list)
            
            #segment_transform_resize_max_min_list
            ##
            current_segment_nucleus_polygons_resize = np.asarray(current_segment_hull_coords_transform_resize_list, np.int32)
            current_segment_nucleus_mask_resize = np.zeros([crop_image_resize,crop_image_resize], dtype=np.uint8)
            cv2.fillConvexPoly(current_segment_nucleus_mask_resize, current_segment_nucleus_polygons_resize, 255)
            cv2.imwrite(keypoints_polygon_patch_save_path+'cell'+str(check_cell_index)+'_label_index_'+str(segment_nucleus_index_list[check_cell_index])+'_polygon_relative_resize'+str(crop_image_resize)+'_segment_nucleus_transform_mask.jpg',current_segment_nucleus_mask_resize)
            segment_hull_coords_transform_resize_patch_list.append(current_segment_nucleus_mask_resize)
            
        else:
            #print('current_xenium_nucleus_radius is smaller than current_segment_nucleus_radius_ratio !')
            
            #segment polygon resize
            segment_polygon_transform_ratio = current_segment_nucleus_radius_ratio*2/(crop_image_resize-1)
            current_segment_hull_coords_transform_resize_list = []
            for segment_transform_list_index in range(len(current_segment_hull_coords_transform_list)):
                current_segment_hull_coords_transform_tuple = current_segment_hull_coords_transform_list[segment_transform_list_index]
                current_segment_hull_coords_transform_resize_tuple = (current_segment_hull_coords_transform_tuple[0]*transform_ratio/segment_polygon_transform_ratio, current_segment_hull_coords_transform_tuple[1]*transform_ratio/segment_polygon_transform_ratio)
                current_segment_hull_coords_transform_resize_list.append(current_segment_hull_coords_transform_resize_tuple)
            #print('current_segment_hull_coords_transform_resize_list',current_segment_hull_coords_transform_resize_list)
            segment_hull_coords_transform_resize_list.append(current_segment_hull_coords_transform_resize_list)
            
            #segment_transform_resize_max_min_list
            current_segment_nucleus_transform_resize_max_min_list = [current_segment_nucleus_max_min_transform_list[0]*transform_ratio/segment_polygon_transform_ratio,current_segment_nucleus_max_min_transform_list[1]*transform_ratio/segment_polygon_transform_ratio,current_segment_nucleus_max_min_transform_list[2]*transform_ratio/segment_polygon_transform_ratio,current_segment_nucleus_max_min_transform_list[3]*transform_ratio/segment_polygon_transform_ratio]
            #print('current_segment_nucleus_transform_resize_max_min_list',current_segment_nucleus_transform_resize_max_min_list)
            segment_nucleus_transform_resize_max_min_list.append(current_segment_nucleus_transform_resize_max_min_list)
            
            current_segment_nucleus_polygons_resize = np.asarray(current_segment_hull_coords_transform_resize_list, np.int32)
            current_segment_nucleus_mask_resize = np.zeros([crop_image_resize,crop_image_resize], dtype=np.uint8)
            cv2.fillConvexPoly(current_segment_nucleus_mask_resize, current_segment_nucleus_polygons_resize, 255)
            cv2.imwrite(keypoints_polygon_patch_save_path+'cell'+str(check_cell_index)+'_label_index_'+str(segment_nucleus_index_list[check_cell_index])+'_polygon_relative_resize'+str(crop_image_resize)+'_segment_nucleus_transform_mask.jpg',current_segment_nucleus_mask_resize)
            segment_hull_coords_transform_resize_patch_list.append(current_segment_nucleus_mask_resize)
            
            #segment_nucleus_transform_resize_center
            current_segment_nucleus_transform_resize_center_x = (crop_image_resize-1)/2
            current_segment_nucleus_transform_resize_center_y = (crop_image_resize-1)/2
            
            #xenium polygon resize
            current_xenium_geometry_exterior_coords_transform_resize_init_list = []
            for xenium_transform_list_init_index in range(len(current_xenium_geometry_exterior_coords_transform_list)):
                current_xenium_geometry_exterior_coords_transform_init_tuple = current_xenium_geometry_exterior_coords_transform_list[xenium_transform_list_init_index]
                current_xenium_geometry_exterior_coords_transform_resize_init_tuple = (current_xenium_geometry_exterior_coords_transform_init_tuple[0]/segment_polygon_transform_ratio, current_xenium_geometry_exterior_coords_transform_init_tuple[1]/segment_polygon_transform_ratio)
                current_xenium_geometry_exterior_coords_transform_resize_init_list.append(current_xenium_geometry_exterior_coords_transform_resize_init_tuple)
            #print('current_xenium_geometry_exterior_coords_transform_resize_init_list',current_xenium_geometry_exterior_coords_transform_resize_init_list)
            
            #xenium_transform_resize_max_min_init_list
            current_xenium_nucleus_transform_resize_max_min_init_list = [current_xenium_geometry_exterior_coords_max_min_transform_list[0]/segment_polygon_transform_ratio,current_xenium_geometry_exterior_coords_max_min_transform_list[1]/segment_polygon_transform_ratio,current_xenium_geometry_exterior_coords_max_min_transform_list[2]/segment_polygon_transform_ratio,current_xenium_geometry_exterior_coords_max_min_transform_list[3]/segment_polygon_transform_ratio]
            #print('current_xenium_nucleus_transform_resize_max_min_init_list',current_xenium_nucleus_transform_resize_max_min_init_list)
            
            #xenium_nucleus_transform_resize_init_center
            current_xenium_nucleus_transform_resize_init_center_x = (current_xenium_nucleus_transform_resize_max_min_init_list[1] + current_xenium_nucleus_transform_resize_max_min_init_list[0])/2
            current_xenium_nucleus_transform_resize_init_center_y = (current_xenium_nucleus_transform_resize_max_min_init_list[3] + current_xenium_nucleus_transform_resize_max_min_init_list[2])/2
                        
            #xenium_nucleus_transform_resize_move_dis
            current_xenium_nucleus_transform_resize_move_x = current_segment_nucleus_transform_resize_center_x - current_xenium_nucleus_transform_resize_init_center_x
            current_xenium_nucleus_transform_resize_move_y = current_segment_nucleus_transform_resize_center_y - current_xenium_nucleus_transform_resize_init_center_y
            
            current_xenium_geometry_exterior_coords_transform_resize_list = []
            for xenium_transform_list_index in range(len(current_xenium_geometry_exterior_coords_transform_resize_init_list)):
                current_xenium_geometry_exterior_coords_transform_init_tuple = current_xenium_geometry_exterior_coords_transform_resize_init_list[xenium_transform_list_index]
                if current_xenium_nucleus_transform_resize_move_x >= 0 and current_xenium_nucleus_transform_resize_move_y >= 0 :                
                    current_xenium_geometry_exterior_coords_transform_resize_tuple = (current_xenium_geometry_exterior_coords_transform_init_tuple[0]+current_xenium_nucleus_transform_resize_move_x, current_xenium_geometry_exterior_coords_transform_init_tuple[1]+current_xenium_nucleus_transform_resize_move_y)
                elif current_xenium_nucleus_transform_resize_move_x >= 0 and current_xenium_nucleus_transform_resize_move_y < 0 :                
                    current_xenium_geometry_exterior_coords_transform_resize_tuple = (current_xenium_geometry_exterior_coords_transform_init_tuple[0]+current_xenium_nucleus_transform_resize_move_x, current_xenium_geometry_exterior_coords_transform_init_tuple[1]-current_xenium_nucleus_transform_resize_move_y)
                elif current_xenium_nucleus_transform_resize_move_x < 0 and current_xenium_nucleus_transform_resize_move_y >= 0 :                
                    current_xenium_geometry_exterior_coords_transform_resize_tuple = (current_xenium_geometry_exterior_coords_transform_init_tuple[0]-current_xenium_nucleus_transform_resize_move_x, current_xenium_geometry_exterior_coords_transform_init_tuple[1]+current_xenium_nucleus_transform_resize_move_y)                
                elif current_xenium_nucleus_transform_resize_move_x < 0 and current_xenium_nucleus_transform_resize_move_y < 0 :                
                    current_xenium_geometry_exterior_coords_transform_resize_tuple = (current_xenium_geometry_exterior_coords_transform_init_tuple[0]-current_xenium_nucleus_transform_resize_move_x, current_xenium_geometry_exterior_coords_transform_init_tuple[1]-current_xenium_nucleus_transform_resize_move_y)
                
                current_xenium_geometry_exterior_coords_transform_resize_list.append(current_xenium_geometry_exterior_coords_transform_resize_tuple)
            #print('current_xenium_geometry_exterior_coords_transform_resize_list',current_xenium_geometry_exterior_coords_transform_resize_list)
            xenium_geometry_exterior_coords_transform_resize_list.append(current_xenium_geometry_exterior_coords_transform_resize_list)
            
            #xenium_transform_resize_max_min_list
            ##
            current_xenium_nucleus_polygons_resize = np.asarray(current_xenium_geometry_exterior_coords_transform_resize_list, np.int32)
            current_xenium_nucleus_mask_resize = np.zeros([crop_image_resize,crop_image_resize], dtype=np.uint8)
            cv2.fillConvexPoly(current_xenium_nucleus_mask_resize, current_xenium_nucleus_polygons_resize, 255)
            cv2.imwrite(keypoints_polygon_patch_save_path+'cell'+str(check_cell_index)+'_barcode_'+xenium_nucleus_barcode_list[check_cell_index]+'_polygon_relative_resize'+str(crop_image_resize)+'_xenium_nucleus_transform_mask.jpg',current_xenium_nucleus_mask_resize)
            xenium_geometry_exterior_coords_transform_resize_patch_list.append(current_xenium_nucleus_mask_resize)
    
    return xenium_geometry_exterior_coords_transform_resize_list, segment_hull_coords_transform_resize_list, xenium_geometry_exterior_coords_transform_resize_patch_list, segment_hull_coords_transform_resize_patch_list
    
def polygon_check_function(xenium_geometry_exterior_coords_transform_resize_list, segment_hull_coords_transform_resize_list, xenium_geometry_exterior_coords_transform_resize_patch_list, segment_hull_coords_transform_resize_patch_list, output_path, value_save_path, learning_name, overlap_threshold_ave, overlap_threshold_min, overlap_type, current_epoch_num):
    
    #os setting
    keypoints_polygon_check_save_path = output_path+'/keypoints_polygon_check_epoch'+str(current_epoch_num)+'_save/'
    if not os.path.exists(keypoints_polygon_check_save_path):
        os.makedirs(keypoints_polygon_check_save_path)

    keypoints_polygon_patch_save_path = output_path+'/keypoints_polygon_patch_epoch'+str(current_epoch_num)+'_save/'
    if not os.path.exists(keypoints_polygon_patch_save_path):
        os.makedirs(keypoints_polygon_patch_save_path)

    #check polygons and patches
    filtered_ranked_top1_barcode_psnr_rgb_five_angles_pd = pd.read_csv(value_save_path+learning_name+'_filtered_ranked_top1_barcode_psnr_rgb_epoch'+str(current_epoch_num)+'_five_angles.csv')
    filtered_ranked_top1_barcode_psnr_rgb_five_angles_index_list = filtered_ranked_top1_barcode_psnr_rgb_five_angles_pd.index.tolist()
    
    xenium_segment_inter_area_ration_list = []
    xenium_segment_inter_area_min_ration_list = []
    
    for check_polygon_patch_index in range(len(segment_hull_coords_transform_resize_list)):
        #print('check_polygon_patch_index',check_polygon_patch_index)
        #check polygons
        current_xenium_nucleus_coords_list = xenium_geometry_exterior_coords_transform_resize_list[check_polygon_patch_index]
        current_segment_nucleus_coords_list = segment_hull_coords_transform_resize_list[check_polygon_patch_index]
        
        current_xenium_nucleus_polygon = Polygon(current_xenium_nucleus_coords_list)
        current_segment_nucleus_polygon = Polygon(current_segment_nucleus_coords_list)
        
        #print('current_xenium_nucleus_polygon valid',current_xenium_nucleus_polygon.is_valid)
        #print('current_segment_nucleus_polygon valid',current_segment_nucleus_polygon.is_valid)
        
        current_xenium_nucleus_polygon_area = current_xenium_nucleus_polygon.area
        current_segment_nucleus_polygon_area = current_segment_nucleus_polygon.area
        
        #print('current_xenium_nucleus_polygon_area',current_xenium_nucleus_polygon_area)
        #print('current_segment_nucleus_polygon_area',current_segment_nucleus_polygon_area)
        
        xenium_segment_area = current_xenium_nucleus_polygon.intersection(current_segment_nucleus_polygon).area
        segment_xenium_area = current_segment_nucleus_polygon.intersection(current_xenium_nucleus_polygon).area
        
        #print('xenium_segment_area',xenium_segment_area)
        #print('segment_xenium_area',segment_xenium_area)
        assert(abs(xenium_segment_area-segment_xenium_area)<1)
        
        current_xenium_nucleus_polygon_inter_ratio = xenium_segment_area/current_xenium_nucleus_polygon_area
        current_segment_nucleus_polygon_inter_ratio = xenium_segment_area/current_segment_nucleus_polygon_area
        current_inter_ratio = (current_xenium_nucleus_polygon_inter_ratio+current_segment_nucleus_polygon_inter_ratio)/2
        
        xenium_segment_inter_area_ration_list.append(current_inter_ratio)
        xenium_segment_inter_area_min_ration_list.append(min(current_xenium_nucleus_polygon_inter_ratio,current_segment_nucleus_polygon_inter_ratio))
    
    if overlap_type == 'overlap_ave':
        #print('overlap_type',overlap_type)
        #ave
        xenium_segment_inter_area_ration_list_np = np.array(xenium_segment_inter_area_ration_list).reshape(-1,1)
        xenium_segment_inter_area_ration_list_pd = pd.DataFrame(xenium_segment_inter_area_ration_list_np,columns=['inter_area_ration'])
        xenium_segment_inter_area_ration_list_pd_sorted = xenium_segment_inter_area_ration_list_pd.sort_values(by='inter_area_ration', ascending=False)
        xenium_segment_inter_area_ration_list_pd_sorted.to_csv(keypoints_polygon_check_save_path+'keypoints_polygon_inter_area_sorted.csv')
        
        #return kept keypoints
        xenium_segment_inter_area_ration_list_pd_sorted_kept = xenium_segment_inter_area_ration_list_pd_sorted.loc[xenium_segment_inter_area_ration_list_pd_sorted['inter_area_ration'] >= overlap_threshold_ave]
        xenium_segment_inter_area_ration_list_pd_sorted_kept.to_csv(keypoints_polygon_check_save_path+'keypoints_polygon_inter_area_sorted_filtered_by_'+str(overlap_threshold_ave)+'.csv')
        xenium_segment_inter_area_ration_list_pd_sorted_kept_index_np = np.array(xenium_segment_inter_area_ration_list_pd_sorted_kept.index.tolist())
        xenium_segment_kept_index_np = xenium_segment_inter_area_ration_list_pd_sorted_kept_index_np    
    elif overlap_type == 'overlap_min':
        #print('overlap_type',overlap_type)
        #min
        xenium_segment_inter_area_min_ration_list_np = np.array(xenium_segment_inter_area_min_ration_list).reshape(-1,1)
        xenium_segment_inter_area_min_ration_list_pd = pd.DataFrame(xenium_segment_inter_area_min_ration_list_np,columns=['inter_area_min_ration'])
        xenium_segment_inter_area_min_ration_list_pd_sorted = xenium_segment_inter_area_min_ration_list_pd.sort_values(by='inter_area_min_ration', ascending=False)
        #xenium_segment_inter_area_min_ration_list_pd_sorted.to_csv(keypoints_polygon_check_save_path+'keypoints_polygon_inter_area_min_sorted.csv')
        
        #return kept keypoints min
        xenium_segment_inter_area_min_ration_list_pd_sorted_kept = xenium_segment_inter_area_min_ration_list_pd_sorted.loc[xenium_segment_inter_area_min_ration_list_pd_sorted['inter_area_min_ration'] >= overlap_threshold_min]
        #xenium_segment_inter_area_min_ration_list_pd_sorted_kept.to_csv(keypoints_polygon_check_save_path+'keypoints_polygon_inter_area_min_sorted_filtered_by_'+str(overlap_threshold_min)+'.csv')
        xenium_segment_inter_area_min_ration_list_pd_sorted_kept_index_np = np.array(xenium_segment_inter_area_min_ration_list_pd_sorted_kept.index.tolist())
        xenium_segment_kept_index_np = xenium_segment_inter_area_min_ration_list_pd_sorted_kept_index_np
    
    return xenium_segment_kept_index_np 

def save_polygon_kept_explorer_keypoints(keypoints_save_path, sample, xenium_segment_explorer_keypoints, current_epoch_num):
    
    ###with open(keypoints_save_path+sample+'_epoch'+str(current_epoch_num)+'_polygon_kept_keypoints.csv', 'w', newline='') as file:
    ###    writer = csv.writer(file)  #lineterminator='\n'
    ###    count_row = 0
    ###    for row_num in range(xenium_segment_explorer_keypoints.shape[0]):
    ###        count_row = count_row + 1
    ###        if count_row < xenium_segment_explorer_keypoints.shape[0]:
    ###            current_row = xenium_segment_explorer_keypoints[row_num,:].tolist()
    ###            writer.writerow(current_row)
    ###        else:
    ###            writer = csv.writer(file,lineterminator='')
    ###            current_row = xenium_segment_explorer_keypoints[row_num,:].tolist()
    ###            writer.writerow(current_row)
                
    #print('Please check the polygon kept keypoints file in the folder of '+keypoints_save_path+' !')
    
    count_row = xenium_segment_explorer_keypoints.shape[0]  #add
    
    print('The keypoints filtered by multi angles, Delaunay triangulation graph and nucleus polygon of epoch'+str(current_epoch_num)+' have been generated!')
    
    return count_row

def save_final_explorer_keypoints(keypoints_save_path, sample, xenium_segment_explorer_keypoints):
    
    with open(keypoints_save_path+sample+'_keypoints.csv', 'w', newline='') as file:
        writer = csv.writer(file)  #lineterminator='\n'
        count_row = 0
        for row_num in range(xenium_segment_explorer_keypoints.shape[0]):
            count_row = count_row + 1
            if count_row < xenium_segment_explorer_keypoints.shape[0]:
                current_row = xenium_segment_explorer_keypoints[row_num,:].tolist()
                writer.writerow(current_row)
            else:
                writer = csv.writer(file,lineterminator='')
                current_row = xenium_segment_explorer_keypoints[row_num,:].tolist()
                writer.writerow(current_row)

    return print('Please check the final keypoints file in the folder of '+keypoints_save_path+' !')
