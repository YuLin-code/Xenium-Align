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

def score_and_rank_cellpose_segment_nucleus_multi_angle(sdata, sample, pixel_size, crop_radius_pixel, center_move_pixel, tif_image_init_affine, segment_label_image_init_affine, xenium_mip_ome_tif_image, learning_name, output_path, transform_ratio, crop_image_resize, segment_nucleus_pixel_values_pd, check_cell_list, save_check_sort_image_num):

    #load os
    score_save_path = output_path+'/score_save/'
    value_save_path = output_path+'/value_save/'
    crop_save_path = output_path+'/crop_Rp'+str(crop_radius_pixel)+'_save/'
    image_save_path = output_path+'/image_save/'
    
    #load settings
    segment_nucleus_pixel_values_np = segment_nucleus_pixel_values_pd.values
    move_pixel_for_he = center_move_pixel / transform_ratio
    crop_radius_pixel_for_he = crop_radius_pixel / transform_ratio
    
    angle_list = ['center','up','down','left','right']

    for angle_num in range(len(angle_list)):
        current_angle_name = angle_list[angle_num]
        
        #score for individual cell
        score_psnr_rgb_list = []
        check_segment_cell_value_list = []
        check_segment_cell_index_list= []
        
        for check_cell_index in range(len(check_cell_list)):
            current_check_cell_index = check_cell_list[check_cell_index]
            current_pixel_y = segment_nucleus_pixel_values_np[current_check_cell_index,0]
            current_pixel_x = segment_nucleus_pixel_values_np[current_check_cell_index,1]
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
            segment_label_image_crop_tile_resize.save(crop_save_path+'index'+str(check_cell_index)+'_Rpsave'+str(round(current_pixel_radius,3))+'_Rpcrop'+str(round(crop_radius_pixel_for_he,3))+'_segment_label_image_crop_pillow_'+current_angle_name+'.jpg', "JPEG")
            
            #tif_image_pillow
            tif_image_crop_tile = tif_image_init_affine.crop((imagecol_left, imagerow_down, imagecol_right, imagerow_up))
            if tif_image_crop_tile.mode != 'RGB':
                #print('tif_image_crop_tile mode is',tif_image_crop_tile.mode)
                tif_image_crop_tile = tif_image_crop_tile.convert('RGB')
                #print('tif_image_crop_tile mode has been converted!')
    
            tif_image_crop_tile_resize = tif_image_crop_tile.resize((crop_image_resize,crop_image_resize))
            tif_image_crop_tile_resize_rgb_np = np.array(tif_image_crop_tile_resize)
            tif_image_crop_tile_resize.save(crop_save_path+'index'+str(check_cell_index)+'_Rpsave'+str(round(current_pixel_radius,3))+'_Rpcrop'+str(round(crop_radius_pixel_for_he,3))+'_HE_crop_pillow'+'_'+current_angle_name+'.jpg', "JPEG")
    
            #set list
            cell_score_search_psnr_rgb_list = []
            cell_all_barcode_list = []
            
            for cell_index in range(sdata.tables.data['table'].X.A.shape[0]):
                #print('cell_index',cell_index)
                cell_circles_init = sdata.shapes['cell_circles']
                cell_circles_coords = cell_circles_init['geometry']
                micron_x = cell_circles_coords.iloc[cell_index].x
                micron_y = cell_circles_coords.iloc[cell_index].y
                micron_radius = cell_circles_init['radius'].iloc[cell_index]
                cell_barcode = cell_circles_init.index[cell_index]
                cell_all_barcode_list.append(cell_barcode)
                
                #nucleus_boundaries
                nucleus_boundaries_init = sdata.shapes['nucleus_boundaries']
                nucleus_boundaries_coords = nucleus_boundaries_init['geometry']
                micron_xmin_nucleus, micron_ymin_nucleus, micron_xmax_nucleus, micron_ymax_nucleus = nucleus_boundaries_coords.iloc[cell_index].bounds
                micron_center_x_nucleus = (micron_xmax_nucleus-micron_xmin_nucleus)/2+micron_xmin_nucleus
                micron_center_y_nucleus = (micron_ymax_nucleus-micron_ymin_nucleus)/2+micron_ymin_nucleus
                micron_center_radius_nucleus = max((micron_xmax_nucleus-micron_xmin_nucleus)/2,(micron_ymax_nucleus-micron_ymin_nucleus)/2)

                #set center and crop value
                if current_angle_name == 'center':
                    pixel_center_radius_nucleus = micron_center_radius_nucleus/pixel_size
                    pixel_center_x_nucleus = micron_center_x_nucleus/pixel_size
                    pixel_center_y_nucleus = micron_center_y_nucleus/pixel_size
                elif current_angle_name == 'up':
                    pixel_center_x_nucleus = micron_center_x_nucleus/pixel_size
                    pixel_center_y_nucleus = micron_center_y_nucleus/pixel_size - center_move_pixel
                    pixel_center_radius_nucleus = micron_center_radius_nucleus/pixel_size
                elif current_angle_name == 'down':
                    pixel_center_x_nucleus = micron_center_x_nucleus/pixel_size
                    pixel_center_y_nucleus = micron_center_y_nucleus/pixel_size + center_move_pixel
                    pixel_center_radius_nucleus = micron_center_radius_nucleus/pixel_size
                elif current_angle_name == 'left':
                    pixel_center_x_nucleus = micron_center_x_nucleus/pixel_size - center_move_pixel
                    pixel_center_y_nucleus = micron_center_y_nucleus/pixel_size
                    pixel_center_radius_nucleus = micron_center_radius_nucleus/pixel_size
                elif current_angle_name == 'right':
                    pixel_center_x_nucleus = micron_center_x_nucleus/pixel_size + center_move_pixel
                    pixel_center_y_nucleus = micron_center_y_nucleus/pixel_size
                    pixel_center_radius_nucleus = micron_center_radius_nucleus/pixel_size
                
                pixel_center_radius_nucleus_used = crop_radius_pixel
                imagerow_down_nucleus  = pixel_center_y_nucleus - pixel_center_radius_nucleus_used
                imagerow_up_nucleus    = pixel_center_y_nucleus + pixel_center_radius_nucleus_used
                imagecol_left_nucleus  = pixel_center_x_nucleus - pixel_center_radius_nucleus_used
                imagecol_right_nucleus = pixel_center_x_nucleus + pixel_center_radius_nucleus_used
                
                crop_nucleus_for_check_image_rgb_tile = xenium_mip_ome_tif_image.crop((imagecol_left_nucleus, imagerow_down_nucleus, imagecol_right_nucleus, imagerow_up_nucleus))
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
            current_sorted_save_path = value_save_path+'check_cell'+str(check_cell_index)+'_'+current_angle_name+'/'
            if not os.path.exists(current_sorted_save_path):
                os.makedirs(current_sorted_save_path)
            
            ###psnr_summary
            cell_score_search_psnr_rgb_list_np = np.array(cell_score_search_psnr_rgb_list).reshape(1,-1)
            score_psnr_rgb_list.append(cell_score_search_psnr_rgb_list_np)
            score_psnr_rgb_list_np = np.array(cell_score_search_psnr_rgb_list_np).reshape(-1,1)
            score_psnr_rgb_list_pd = pd.DataFrame(score_psnr_rgb_list_np,index=cell_all_barcode_list,columns=['score_psnr_rgb'])
            score_psnr_rgb_list_pd_sorted = score_psnr_rgb_list_pd.sort_values(by='score_psnr_rgb', ascending=False)
            score_psnr_rgb_list_pd_sorted_index_list = score_psnr_rgb_list_pd_sorted.index.tolist()
            score_psnr_rgb_list_pd_sorted_save = score_psnr_rgb_list_pd_sorted.head(save_check_sort_image_num)
            score_psnr_rgb_list_pd_sorted_save.to_csv(current_sorted_save_path+'check_cell'+str(check_cell_index)+'_sorted'+str(save_check_sort_image_num)+'_score_psnr_rgb_'+current_angle_name+'.csv')
        
        #psnr_sort_rank
        score_rgb_psnr_list_np = np.concatenate(score_psnr_rgb_list,axis=0)
        score_rgb_psnr_list_np_T = score_rgb_psnr_list_np.T
        score_rgb_psnr_list_pd = pd.DataFrame(score_rgb_psnr_list_np_T,index=cell_all_barcode_list,columns=check_segment_cell_index_list)
        score_rgb_psnr_list_pd.to_csv(score_save_path+'/'+learning_name+'_'+str(len(check_cell_list))+'cell_R'+sample+'_psnr_score_rgb_'+current_angle_name+'.csv')
    
        #save check segment cell values
        check_segment_cell_value_list_concatenate = np.concatenate(check_segment_cell_value_list,axis=0)
        #print('check_segment_cell_value_list_concatenate shape is',check_segment_cell_value_list_concatenate.shape)
        check_segment_cell_value_list_concatenate_pd = pd.DataFrame(check_segment_cell_value_list_concatenate, index = check_segment_cell_index_list, columns = ['pixel_center_x_nucleus','pixel_center_y_nucleus','pixel_center_radius_nucleus'])
        check_segment_cell_value_list_concatenate_pd.to_csv(value_save_path+'filterd_cell'+str(len(check_cell_list))+'_to_check_psnr_segment_nucleus_pixel_values_'+current_angle_name+'.csv')
    
    return print('Current sample psnr checking finished!')

def check_rank_top5_multi_angle(output_path, check_cell_num, sample, learning_name):

    #load os
    score_save_path = output_path+'/score_save/'
    value_save_path = output_path+'/value_save/'
    image_save_path = output_path+'/image_save/'
    
    #set list
    angle_list = ['center','up','down','left','right']
    
    top1_barcode_multi_angle_list = []
    top1_index_multi_angle_list = []
    for angle_num in range(len(angle_list)):
        current_angle_name = angle_list[angle_num]
        
        top5_list = []
        columns_list = []
        first_barcode_list = []
        second_barcode_list = []
        third_barcode_list = []
        fourth_barcode_list = []
        fifth_barcode_list = []
        cell_index_psnr_rgb_list = []
        
        for check_current_cell in range(check_cell_num):
            current_folder = value_save_path+'check_cell'+str(check_current_cell)+'_'+current_angle_name+'/'
            current_top5_psnr_rgb_pd = pd.read_csv(current_folder+'check_cell'+str(check_current_cell)+'_sorted5_score_psnr_rgb_'+current_angle_name+'.csv')
            current_top5_psnr_rgb_np = current_top5_psnr_rgb_pd.values
            top5_list.append(current_top5_psnr_rgb_np)
            columns_list.append('cell'+str(check_current_cell)+'_barcode_psnr_rgb')
            columns_list.append('cell'+str(check_current_cell)+'_score_psnr_rgb')
            first_barcode_list.append(current_top5_psnr_rgb_np[0,0])
            second_barcode_list.append(current_top5_psnr_rgb_np[1,0])
            third_barcode_list.append(current_top5_psnr_rgb_np[2,0])
            fourth_barcode_list.append(current_top5_psnr_rgb_np[3,0])
            fifth_barcode_list.append(current_top5_psnr_rgb_np[4,0])
            cell_index_psnr_rgb_list.append('cell'+str(check_current_cell))
    
        top5_list_np = np.concatenate(top5_list,axis=1)
        top5_list_pd = pd.DataFrame(top5_list_np,columns=columns_list,index=['top1','top2','top3','top4','top5'])
        file_check_name0 = value_save_path+learning_name+'_ranked_top5_psnr_rgb_'+current_angle_name+'.csv'
        if os.path.exists(file_check_name0):
            #print(sample+'_ranked_top5_psnr_rgb_'+current_angle_name+'.csv is exist!')
            pass
        else:
            #print('Start to generate '+sample+'_ranked_top5_psnr_rgb_'+current_angle_name+'.csv!')
            top5_list_pd.to_csv(value_save_path+learning_name+'_ranked_top5_psnr_rgb_'+current_angle_name+'.csv')
        
        first_barcode_list_np = np.array(first_barcode_list).reshape(-1,1)
        second_barcode_list_np = np.array(second_barcode_list).reshape(-1,1)
        third_barcode_list_np = np.array(third_barcode_list).reshape(-1,1)
        fourth_barcode_list_np = np.array(fourth_barcode_list).reshape(-1,1)
        fifth_barcode_list_np = np.array(fifth_barcode_list).reshape(-1,1)
        top5_barcode_psnr_rgb_np = np.hstack((first_barcode_list_np,second_barcode_list_np,third_barcode_list_np,fourth_barcode_list_np,fifth_barcode_list_np))
        top5_barcode_psnr_rgb_pd = pd.DataFrame(top5_barcode_psnr_rgb_np,index =cell_index_psnr_rgb_list,columns=['top1','top2','top3','top4','top5'])
        top1_barcode_multi_angle_list.append(first_barcode_list_np)
        top1_index_multi_angle_list.append('top1_'+current_angle_name)
        
        file_check_name1 = value_save_path+learning_name+'_ranked_top5_barcode_psnr_rgb_'+current_angle_name+'.csv'
        if os.path.exists(file_check_name1):
            #print(sample+'_ranked_top5_barcode_psnr_rgb_'+current_angle_name+'.csv is exist!')
            pass
        else:
            #print('Start to generate '+sample+'_ranked_top5_barcode_psnr_rgb_'+current_angle_name+'.csv!')
            top5_barcode_psnr_rgb_pd.to_csv(value_save_path+learning_name+'_ranked_top5_barcode_psnr_rgb_'+current_angle_name+'.csv')

    #save ranked_top1_barcode_psnr_rgb_five_angles
    top1_barcode_multi_angle_list_np = np.concatenate(top1_barcode_multi_angle_list,axis=1)
    top1_barcode_multi_angle_list_pd = pd.DataFrame(top1_barcode_multi_angle_list_np,index=cell_index_psnr_rgb_list,columns=top1_index_multi_angle_list)
    file_check_name2 = value_save_path+learning_name+'_ranked_top1_barcode_psnr_rgb_five_angles.csv'
    if os.path.exists(file_check_name2):
        #print(sample+'_ranked_top1_barcode_psnr_rgb_five_angles.csv is exist!')
        pass
    else:
        #print('Start to generate '+sample+'_ranked_top1_barcode_psnr_rgb_five_angles.csv!')
        top1_barcode_multi_angle_list_pd.to_csv(value_save_path+learning_name+'_ranked_top1_barcode_psnr_rgb_five_angles.csv')
    
    #save filtered_ranked_top1_barcode_psnr_rgb_five_angles
    filtered_cell_index_list = []
    for check_cell_num in range(top1_barcode_multi_angle_list_np.shape[0]):
        current_check_row = top1_barcode_multi_angle_list_np[check_cell_num,:]
        if len(np.unique(current_check_row)) == 1:
            current_check_index = 'cell'+str(check_cell_num)
            filtered_cell_index_list.append(current_check_index)
    filtered_top1_barcode_multi_angle_list_pd = top1_barcode_multi_angle_list_pd.loc[filtered_cell_index_list]
    file_check_name3 = value_save_path+learning_name+'_filtered_ranked_top1_barcode_psnr_rgb_five_angles.csv'
    if os.path.exists(file_check_name3):
        #print(sample+'_filtered_ranked_top1_barcode_psnr_rgb_five_angles.csv is exist!')
        pass
    else:
        #print('Start to generate '+sample+'_filtered_ranked_top1_barcode_psnr_rgb_five_angles.csv!')
        filtered_top1_barcode_multi_angle_list_pd.to_csv(value_save_path+learning_name+'_filtered_ranked_top1_barcode_psnr_rgb_five_angles.csv')

    return print('Current sample multi angles checking finished!')
    
def pixel_values_filtered_by_multi_angles_for_xenium_explorer(sample, value_save_path, cellpose_save_path, keypoints_save_path, learning_name, check_cell_num, check_cell_list, xenium_nucleus_pixel_values_pd):
    
    #read filterd values
    rank_filtered_top1_index_pd = pd.read_csv(value_save_path+learning_name+'_filtered_ranked_top1_barcode_psnr_rgb_five_angles.csv',index_col=0)
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
    segment_center_pixel_value_pd = pd.read_csv(cellpose_save_path+sample+'_cellpose_segment_H&E_label_location_save.csv',index_col=0)
    segment_center_pixel_value_filterd_pd = segment_center_pixel_value_pd.loc[segment_center_pixel_value_pd['center_radius_pixel']!=-1]
    segment_center_pixel_value_filterd_cell_num_list = list(range(segment_center_pixel_value_filterd_pd.shape[0]))
    segment_center_pixel_value_filterd_pd.index = segment_center_pixel_value_filterd_cell_num_list
    
    #score_cell_list
    segment_center_pixel_value_filterd_pd_scored = segment_center_pixel_value_filterd_pd.loc[check_cell_list]
    segment_center_pixel_value_filterd_pd_scored.to_csv(value_save_path+learning_name+'_filtered_segment_nucleus_center_pixel_values_index_by_cell_list.csv',index=True)
    cell_num_list = list(range(check_cell_num))
    segment_center_pixel_value_filterd_pd_scored.index = cell_num_list
    segment_center_pixel_value_filterd_pd_scored.to_csv(value_save_path+learning_name+'_filtered_segment_nucleus_center_pixel_values_index_by_cell_num.csv',index=True)
    segment_center_pixel_value_filterd_pd_scored_filtered = segment_center_pixel_value_filterd_pd_scored.loc[rank_filtered_top1_index_list_num]
    segment_center_pixel_value_filterd_scored_filtered_np = segment_center_pixel_value_filterd_pd_scored_filtered[['x_center_pixel','y_center_pixel']].values
    
    #check part
    filterd_check_psnr_pixel_values_center_pd = pd.read_csv(value_save_path+'filterd_cell'+str(check_cell_num)+'_to_check_psnr_segment_nucleus_pixel_values_center.csv',index_col=0)
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

    with open(keypoints_save_path+learning_name+'_filtered_by_multi_angle_keypoints.csv', 'w', newline='') as file:
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

    return print('The keypoints filtered by multi angles have been generated!')
    
def pixel_values_filtered_by_multi_angles_and_graph_for_xenium_explorer(sample, graph_save_path, keypoints_save_path, learning_name, graph_source_str, fig_size):

    #os setting
    dataset_root_xenium = graph_save_path+learning_name+'_xenium_explorer/'
    dataset_root_segment = graph_save_path+learning_name+'_segment_explorer/'
    
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
    
    xenium_segment_coords_filtered_by_multi_angles_pd = pd.read_csv(keypoints_save_path+learning_name+'_filtered_by_multi_angle_keypoints.csv')
    
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
            graph_output=None,                                   #graph_output_xenium,
            voronoi_polygon_img_output=None,                     #voronoi_polygon_img_output_xenium,
            graph_img_output=None,                               #graph_img_output_xenium,
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
            graph_output=None,                                  #graph_output_segment,
            voronoi_polygon_img_output=None,                    #voronoi_polygon_img_output_segment,
            graph_img_output=None,                              #graph_img_output_segment,
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
    
    #with open(keypoints_save_path+learning_name+'_filtered_by_multi_angle_and_graph_keypoints.csv', 'w', newline='') as file:
    #    writer = csv.writer(file)  #lineterminator='\n'
    #    count_row = 0
    #    for row_num in range(update_xenium_segment_explorer_np.shape[0]):
    #        count_row = count_row + 1
    #        if count_row < update_xenium_segment_explorer_np.shape[0]:
    #            current_row = update_xenium_segment_explorer_np[row_num,:].tolist()
    #            writer.writerow(current_row)
    #        else:
    #            writer = csv.writer(file,lineterminator='')
    #            current_row = update_xenium_segment_explorer_np[row_num,:].tolist()
    #            writer.writerow(current_row)
    
    filtered_keypoints_num = update_xenium_segment_explorer_np.shape[0]
    
    return filtered_keypoints_num, update_xenium_segment_explorer_np

def pixel_values_filtered_by_multi_angles_and_graph_inverse_for_xenium_explorer(sample, graph_save_path, keypoints_save_path, learning_name, graph_source_str, fig_size):

    #os setting
    dataset_root_xenium = graph_save_path+learning_name+'_inverse_xenium_explorer/'
    dataset_root_segment = graph_save_path+learning_name+'_inverse_segment_explorer/'
    
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
    
    xenium_segment_coords_filtered_by_multi_angles_pd = pd.read_csv(keypoints_save_path+learning_name+'_filtered_by_multi_angle_keypoints.csv')
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
            graph_output=None,                                #graph_output_xenium,
            voronoi_polygon_img_output=None,                  #voronoi_polygon_img_output_xenium,
            graph_img_output=None,                            #graph_img_output_xenium,
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
            graph_output=None,                                   #graph_output_segment,
            voronoi_polygon_img_output=None,                     #voronoi_polygon_img_output_segment,
            graph_img_output=None,                               #graph_img_output_segment,
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
    
    #with open(keypoints_save_path+learning_name+'_filtered_by_multi_angle_and_graph_inverse_keypoints.csv', 'w', newline='') as file:
    #    writer = csv.writer(file)  #lineterminator='\n'
    #    count_row = 0
    #    for row_num in range(update_xenium_segment_explorer_np.shape[0]):
    #        count_row = count_row + 1
    #        if count_row < update_xenium_segment_explorer_np.shape[0]:
    #            current_row = update_xenium_segment_explorer_np[row_num,:].tolist()
    #            writer.writerow(current_row)
    #        else:
    #            writer = csv.writer(file,lineterminator='')
    #            current_row = update_xenium_segment_explorer_np[row_num,:].tolist()
    #            writer.writerow(current_row)

    filtered_keypoints_num = update_xenium_segment_explorer_np.shape[0]
    
    return filtered_keypoints_num, update_xenium_segment_explorer_np

def save_xenium_segment_explorer_keypoints(keypoints_save_path,sample,filtered_keypoints_num,xenium_segment_explorer_np,filtered_keypoints_inverse_num,xenium_segment_explorer_inverse_np,learning_name):
    
    #set xenium_segment_explorer_keypoints
    if filtered_keypoints_num >= filtered_keypoints_inverse_num:
        xenium_segment_explorer_keypoints = xenium_segment_explorer_np
    else:
        xenium_segment_explorer_keypoints = xenium_segment_explorer_inverse_np
    
    print('The keypoints filtered by multi angles and delaunay graph have been generated!')
    
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
    if count_row <= 30:
        os.remove(keypoints_save_path+sample+'_keypoints.csv')
        os.remove(keypoints_save_path+learning_name+'_filtered_by_multi_angle_keypoints.csv')
    else:
        os.remove(keypoints_save_path+learning_name+'_filtered_by_multi_angle_keypoints.csv')
        print('Please check the final keypoints file in the folder of '+keypoints_save_path+' !')
    
    return count_row

def pixel_values_min3_angle_for_xenium_explorer(sample, value_save_path, cellpose_save_path, keypoints_save_path, learning_name, check_cell_num, check_cell_list, xenium_nucleus_pixel_values_pd):
    
    #read filterd values
    rank_filtered_top1_pd = pd.read_csv(value_save_path+learning_name+'_ranked_top1_barcode_psnr_rgb_five_angles.csv',index_col=0)
    rank_filtered_top1_np = rank_filtered_top1_pd.values

    filtered_min3_cell_index_list = []
    rank_filtered_min3_index_list_num = []
    rank_filtered_min3_barcode_list = []
    for cell_index in range(rank_filtered_top1_np.shape[0]):
        current_check_row = rank_filtered_top1_np[cell_index,:]
        #print('current_check_row',current_check_row)
        if len(np.unique(current_check_row)) <= 3:
            for current_check_row_element_index in range(current_check_row.shape[0]):
                current_check_row_element = current_check_row[current_check_row_element_index]
                if np.sum(current_check_row==current_check_row_element) == 4 or np.sum(current_check_row==current_check_row_element) == 5 or np.sum(current_check_row==current_check_row_element) == 3:
                    current_check_index = 'cell'+str(cell_index)
                    filtered_min3_cell_index_list.append(current_check_index)
                    rank_filtered_min3_index_list_num.append(cell_index)
                    rank_filtered_min3_barcode_list.append(current_check_row_element)
                    break

    rank_filtered_top1_min3_pd = rank_filtered_top1_pd.loc[filtered_min3_cell_index_list]
    file_check_name = value_save_path+learning_name+'_ranked_top1_min3_barcode_psnr_rgb_five_angles.csv'
    if os.path.exists(file_check_name):
        pass
    else:
        rank_filtered_top1_min3_pd.to_csv(value_save_path+learning_name+'_ranked_top1_min3_barcode_psnr_rgb_five_angles.csv')
    
    #load segmentation
    segment_center_pixel_value_pd = pd.read_csv(cellpose_save_path+sample+'_cellpose_segment_H&E_label_location_save.csv',index_col=0)
    segment_center_pixel_value_filterd_pd = segment_center_pixel_value_pd.loc[segment_center_pixel_value_pd['center_radius_pixel']!=-1]
    segment_center_pixel_value_filterd_cell_num_list = list(range(segment_center_pixel_value_filterd_pd.shape[0]))
    segment_center_pixel_value_filterd_pd.index = segment_center_pixel_value_filterd_cell_num_list
    
    #score_cell_list
    segment_center_pixel_value_filterd_pd_scored = segment_center_pixel_value_filterd_pd.loc[check_cell_list]
    #segment_center_pixel_value_filterd_pd_scored.to_csv(value_save_path+learning_name+'_filtered_segment_nucleus_center_pixel_values_index_by_cell_list.csv',index=True)
    cell_num_list = list(range(check_cell_num))
    segment_center_pixel_value_filterd_pd_scored.index = cell_num_list
    #segment_center_pixel_value_filterd_pd_scored.to_csv(value_save_path+learning_name+'_filtered_segment_nucleus_center_pixel_values_index_by_cell_num.csv',index=True)
    segment_center_pixel_value_filterd_pd_scored_filtered = segment_center_pixel_value_filterd_pd_scored.loc[rank_filtered_min3_index_list_num]
    segment_center_pixel_value_filterd_scored_filtered_np = segment_center_pixel_value_filterd_pd_scored_filtered[['x_center_pixel','y_center_pixel']].values
    
    #check part
    #ssegment_center_pixel_value_filterd_pd_scored_np = segment_center_pixel_value_filterd_pd_scored[['x_center_pixel','y_center_pixel']].values
    #assert((ssegment_center_pixel_value_filterd_pd_scored_np==segment_center_pixel_value_filterd_scored_filtered_np).all())
    
    #xenium_pixel_nucleus_values
    xenium_nucleus_pixel_values_pd_filtered = xenium_nucleus_pixel_values_pd.loc[rank_filtered_min3_barcode_list]
    xenium_nucleus_pixel_values_pd_filtered_np = xenium_nucleus_pixel_values_pd_filtered[['pixel_center_x_nucleus','pixel_center_y_nucleus']].values
    
    #pixel_value_filtered_for_xenium_explorer
    pixel_value_for_xenium_explorer_filtered_np = np.hstack((xenium_nucleus_pixel_values_pd_filtered_np,segment_center_pixel_value_filterd_scored_filtered_np))
    pixel_value_for_xenium_explorer_filtered_columns_list = ['fixedX','fixedY','alignmentX','alignmentY']
    pixel_value_for_xenium_explorer_filtered_columns_list_np = np.array(pixel_value_for_xenium_explorer_filtered_columns_list).reshape(1,-1)
    pixel_value_for_xenium_explorer_filtered_add_columns_np = np.vstack((pixel_value_for_xenium_explorer_filtered_columns_list_np, pixel_value_for_xenium_explorer_filtered_np))

    with open(keypoints_save_path+learning_name+'_by_multi_angle_min3_keypoints.csv', 'w', newline='') as file:
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

    return print('The keypoints filtered by further multi angles have been generated!')   #print('The keypoints filtered by multi angles min3 have been generated!')

def pixel_values_filtered_by_multi_angles_min3_and_graph_for_xenium_explorer(sample, graph_save_path, keypoints_save_path, learning_name, graph_source_str, fig_size):

    #os setting
    dataset_root_xenium = graph_save_path+learning_name+'_xenium_min3_explorer/'
    dataset_root_segment = graph_save_path+learning_name+'_segment_min3_explorer/'
    
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
    
    xenium_segment_coords_use_center_angle_pd = pd.read_csv(keypoints_save_path+learning_name+'_by_multi_angle_min3_keypoints.csv')
    
    xenium_coords_for_graph_pd = xenium_segment_coords_use_center_angle_pd[['fixedX','fixedY']]
    segment_coords_for_graph_pd = xenium_segment_coords_use_center_angle_pd[['alignmentY','alignmentX']]
    
    cell_index_list = list(range(0,xenium_segment_coords_use_center_angle_pd.shape[0]))
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
            graph_output=None,                                          #graph_output_xenium,
            voronoi_polygon_img_output=None,                                 #voronoi_polygon_img_output_xenium,
            graph_img_output=None,                                              #graph_img_output_xenium,
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
            graph_output=None,                                             #graph_output_segment,
            voronoi_polygon_img_output=None,                               #voronoi_polygon_img_output_segment,
            graph_img_output=None,                                         #graph_img_output_segment,
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
    
    #with open(keypoints_save_path+learning_name+'_by_multi_angles_min3_and_graph_keypoints.csv', 'w', newline='') as file:
    #    writer = csv.writer(file)  #lineterminator='\n'
    #    count_row = 0
    #    for row_num in range(update_xenium_segment_explorer_np.shape[0]):
    #        count_row = count_row + 1
    #        if count_row < update_xenium_segment_explorer_np.shape[0]:
    #            current_row = update_xenium_segment_explorer_np[row_num,:].tolist()
    #            writer.writerow(current_row)
    #        else:
    #            writer = csv.writer(file,lineterminator='')
    #            current_row = update_xenium_segment_explorer_np[row_num,:].tolist()
    #            writer.writerow(current_row)

    filtered_keypoints_num = update_xenium_segment_explorer_np.shape[0]
    
    return filtered_keypoints_num, update_xenium_segment_explorer_np

def pixel_values_filtered_by_multi_angles_min3_and_graph_inverse_for_xenium_explorer(sample, graph_save_path, keypoints_save_path, learning_name, graph_source_str, fig_size):

    #os setting
    dataset_root_xenium = graph_save_path+learning_name+'_inverse_xenium_min3_explorer/'
    dataset_root_segment = graph_save_path+learning_name+'_inverse_segment_min3_explorer/'
    
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
    
    xenium_segment_coords_use_center_angle_pd = pd.read_csv(keypoints_save_path+learning_name+'_by_multi_angle_min3_keypoints.csv')
    xenium_coords_for_graph_pd = xenium_segment_coords_use_center_angle_pd[['fixedX','fixedY']]
    segment_coords_for_graph_pd = xenium_segment_coords_use_center_angle_pd[['alignmentY','alignmentX']]
    
    cell_index_list = list(range(0,xenium_segment_coords_use_center_angle_pd.shape[0]))
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
            graph_output=None,                                        #graph_output_xenium,
            voronoi_polygon_img_output=None,                          #voronoi_polygon_img_output_xenium,
            graph_img_output=None,                                    #graph_img_output_xenium,
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
            graph_output=None,                                         #graph_output_segment,
            voronoi_polygon_img_output=None,                           #voronoi_polygon_img_output_segment,
            graph_img_output=None,                                     #graph_img_output_segment,
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
    
    #with open(keypoints_save_path+learning_name+'_by_multi_angles_min3_and_graph_inverse_keypoints.csv', 'w', newline='') as file:
    #    writer = csv.writer(file)  #lineterminator='\n'
    #    count_row = 0
    #    for row_num in range(update_xenium_segment_explorer_np.shape[0]):
    #        count_row = count_row + 1
    #        if count_row < update_xenium_segment_explorer_np.shape[0]:
    #            current_row = update_xenium_segment_explorer_np[row_num,:].tolist()
    #            writer.writerow(current_row)
    #        else:
    #            writer = csv.writer(file,lineterminator='')
    #            current_row = update_xenium_segment_explorer_np[row_num,:].tolist()
    #            writer.writerow(current_row)

    filtered_keypoints_num = update_xenium_segment_explorer_np.shape[0]
    
    return filtered_keypoints_num, update_xenium_segment_explorer_np

def save_xenium_segment_by_multi_angles_min3_explorer_keypoints(keypoints_save_path,sample,filtered_keypoints_num,xenium_segment_explorer_np,filtered_keypoints_inverse_num,xenium_segment_explorer_inverse_np,learning_name):
    
    #set xenium_segment_explorer_keypoints
    if filtered_keypoints_num >= filtered_keypoints_inverse_num:
        xenium_segment_explorer_keypoints = xenium_segment_explorer_np
    else:
        xenium_segment_explorer_keypoints = xenium_segment_explorer_inverse_np
    
    print('The keypoints filtered by further multi angles and delaunay graph have been generated!')  #print('The keypoints filtered by multi angles min3 and delaunay graph have been generated!')
    
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
    
    #delete initial file
    os.remove(keypoints_save_path+learning_name+'_by_multi_angle_min3_keypoints.csv')
    
    return print('Please check the final keypoints file in the folder of '+keypoints_save_path+' !')
