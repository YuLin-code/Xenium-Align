import pandas as pd
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import tifffile
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
import glob
import random
import cv2

def find_files_with_field(directory, field):
    pattern = os.path.join(directory, '*' + field + '*.jpg')
    return glob.glob(pattern)
 
def find_files_with_field_for_rank(directory, field):
    pattern = os.path.join(directory, '*' + field + '*rgb_xenium.jpg')
    return glob.glob(pattern)

def load_os_ff_sample(sample, data_file_path):

    if sample == '3775' or sample == '5582' or sample == '40440' or sample == '40610' or sample == '40775':
        data_path = data_file_path+'/output-XETG00126_0010200_'+sample+'_20240214_210015/'
        ome_tif_data_path = data_file_path+'/output-XETG00126_0010200_'+sample+'_20240214_210015/morphology.ome.tif'
        focus_ome_tif_data_path = data_file_path+'/output-XETG00126_0010200_'+sample+'_20240214_210015/morphology_focus.ome.tif'
        mip_ome_tif_data_path = data_file_path+'/output-XETG00126_0010200_'+sample+'_20240214_210015/morphology_mip.ome.tif'
        nucleus_boundary_path = data_file_path+'/output-XETG00126_0010200_'+sample+'_20240214_210015/nucleus_boundaries.csv/'
    elif sample == 'f59':
        data_path = data_file_path+'/output-XETG00126_0010207_f59_20240214_210015/'
        ome_tif_data_path = data_file_path+'/output-XETG00126_0010207_f59_20240214_210015/morphology.ome.tif'
        focus_ome_tif_data_path = data_file_path+'/output-XETG00126_0010207_f59_20240214_210015/morphology_focus.ome.tif'
        mip_ome_tif_data_path = data_file_path+'/output-XETG00126_0010207_f59_20240214_210015/morphology_mip.ome.tif'
        nucleus_boundary_path = data_file_path+'/output-XETG00126_0010207_f59_20240214_210015/nucleus_boundaries.csv/'
        sample = 'F59'
    else:
        data_path = data_file_path+'/output-XETG00126_0010207_'+sample+'_20240214_210015/'
        ome_tif_data_path = data_file_path+'/output-XETG00126_0010207_'+sample+'_20240214_210015/morphology.ome.tif'
        focus_ome_tif_data_path = data_file_path+'/output-XETG00126_0010207_'+sample+'_20240214_210015/morphology_focus.ome.tif'
        mip_ome_tif_data_path = data_file_path+'/output-XETG00126_0010207_'+sample+'_20240214_210015/morphology_mip.ome.tif'
        nucleus_boundary_path = data_file_path+'/output-XETG00126_0010207_'+sample+'_20240214_210015/nucleus_boundaries.csv/'
    he_img_path = data_file_path+'/histology_ff/'+sample+'.tif'

    return data_path, mip_ome_tif_data_path, sample, he_img_path

def load_os_ffpe_sample(sample, data_file_path, image_path):
    
    if sample == 'kidney_cancer':
        data_path = data_file_path+'/Xenium_V1_hKidney_cancer_section_outs/'
        mip_ome_tif_data_path = data_path+'morphology_mip.ome.tif'
        ome_tif_img_path = data_file_path+'Xenium_V1_hKidney_cancer_section_he_image.ome.tif'
    elif sample == 'kidney_nondiseased':
        data_path = data_file_path+'/Xenium_V1_hKidney_nondiseased_section_outs/'
        mip_ome_tif_data_path = data_path+'morphology_mip.ome.tif'
        ome_tif_img_path = data_file_path+'Xenium_V1_hKidney_nondiseased_section_he_image.ome.tif'
    
    #read ome.tif file
    ome_tif_img = tifffile.TiffReader(ome_tif_img_path)
    tif_img = ome_tif_img.pages[0].asarray()
    
    he_img_path = image_path+sample+'_HE.tif'
    if not os.path.exists(he_img_path):
        cv2.imwrite(image_path+sample+'_HE.tif',tif_img)
    else:
        print(he_img_path+' is exist!')
    
    return data_path, mip_ome_tif_data_path, sample, he_img_path

def normalization_min_max_grayscale(inputdata):
    _range = np.max(inputdata) - np.min(inputdata)
    return ((inputdata - np.min(inputdata)) / _range)*255

def xenium_nucleus_pixel_values_check(sdata, sample, pixel_size, value_save_path):

    #list for individual cell
    cell_barcode_list = []
    pixel_center_x_nucleus_list = []
    pixel_center_y_nucleus_list = []
    pixel_center_radius_nucleus_list = []

    for cell_index in range(sdata.tables.data['table'].X.A.shape[0]):
        #load current
        cell_circles_init = sdata.shapes['cell_circles']
        cell_circles_coords = cell_circles_init['geometry']
        micron_x = cell_circles_coords.iloc[cell_index].x
        micron_y = cell_circles_coords.iloc[cell_index].y
        micron_radius = cell_circles_init['radius'].iloc[cell_index]
        cell_barcode = cell_circles_init.index[cell_index]
        cell_barcode_list.append(cell_barcode)
    
        #nucleus_boundaries
        nucleus_boundaries_init = sdata.shapes['nucleus_boundaries']
        nucleus_boundaries_coords = nucleus_boundaries_init['geometry']
        micron_xmin_nucleus, micron_ymin_nucleus, micron_xmax_nucleus, micron_ymax_nucleus = nucleus_boundaries_coords.iloc[cell_index].bounds
        micron_center_x_nucleus = (micron_xmax_nucleus-micron_xmin_nucleus)/2+micron_xmin_nucleus
        micron_center_y_nucleus = (micron_ymax_nucleus-micron_ymin_nucleus)/2+micron_ymin_nucleus
        micron_center_radius_nucleus = max((micron_xmax_nucleus-micron_xmin_nucleus)/2,(micron_ymax_nucleus-micron_ymin_nucleus)/2)
    
        pixel_center_x_nucleus = micron_center_x_nucleus/pixel_size
        pixel_center_y_nucleus = micron_center_y_nucleus/pixel_size
        pixel_center_radius_nucleus = micron_center_radius_nucleus/pixel_size
        
        pixel_center_x_nucleus_list.append(pixel_center_x_nucleus)
        pixel_center_y_nucleus_list.append(pixel_center_y_nucleus)
        pixel_center_radius_nucleus_list.append(pixel_center_radius_nucleus)
        
    #save
    pixel_center_x_nucleus_list_np = np.array(pixel_center_x_nucleus_list).reshape(-1,1)
    pixel_center_y_nucleus_list_np = np.array(pixel_center_y_nucleus_list).reshape(-1,1)
    pixel_center_radius_nucleus_list_np = np.array(pixel_center_radius_nucleus_list).reshape(-1,1)
    
    pixel_nucleus_list_np = np.hstack((pixel_center_x_nucleus_list_np,pixel_center_y_nucleus_list_np,pixel_center_radius_nucleus_list_np))
    xenium_nucleus_pixel_values_pd = pd.DataFrame(pixel_nucleus_list_np,columns=['pixel_center_x_nucleus','pixel_center_y_nucleus','pixel_center_radius_nucleus'],index=cell_barcode_list)
    xenium_nucleus_pixel_values_pd.to_csv(value_save_path+sample+'_xenium_nucleus_pixel_values.csv')

    return xenium_nucleus_pixel_values_pd

def initialize_tif_segment_xenium_image(he_img_path, segment_save_path, segment_method, sample, mip_ome_tif_data_path, channel_cellpose, flow_threshold, min_size, prob_thresh, image_check_path):
    
    #load rotate file
    rotate_mse_list_pd = pd.read_csv(image_check_path+sample+'_he_rotate_mse_values.csv',index_col=0)
    rotate_full_list = rotate_mse_list_pd.columns.tolist()
    
    rotate_mse_list_np = rotate_mse_list_pd.values.reshape(1,-1)
    row_max_index = np.argmax(rotate_mse_list_np)
    rotate_full_name = rotate_full_list[row_max_index]
    rotate_name_parts = rotate_full_name.split('_')
    rotate_name_init = rotate_name_parts[0]
    rotate_name_last = rotate_name_parts[1]
    
    #load init tif segment_label
    tif_image_pillow = Image.open(he_img_path)
    if segment_method == 'cellpose':
        segment_image_label_mark_scale_file = segment_save_path+'channel_cellpose'+str(channel_cellpose)+'_flow_threshold'+str(flow_threshold)+'_min_size'+str(min_size)+'_image_segmented_cellpose_label_mark_scale.jpg'
    elif segment_method == 'stardist':
        segment_image_label_mark_scale_file = segment_save_path+'prob_thresh'+str(prob_thresh)+'_image_segmented_stardist_label_mark_scale.jpg'
    segment_image_label_mark_scale_pillow = Image.open(segment_image_label_mark_scale_file)
    
    if rotate_name_init == 'rotate90':
        if rotate_name_last == 'original':
            tif_image_init_affine = tif_image_pillow.transpose(Image.ROTATE_90) 
            segment_label_image_init_affine = segment_image_label_mark_scale_pillow.transpose(Image.ROTATE_90)
            
        elif rotate_name_last == 'flipLR':
            rotated_90_image_tif = tif_image_pillow.transpose(Image.ROTATE_90)
            tif_image_init_affine = rotated_90_image_tif.transpose(Image.FLIP_LEFT_RIGHT)
            rotated_90_image_segment = segment_image_label_mark_scale_pillow.transpose(Image.ROTATE_90)
            segment_label_image_init_affine = rotated_90_image_segment.transpose(Image.FLIP_LEFT_RIGHT)
            
        elif rotate_name_last == 'flipTB':
            rotated_90_image_tif = tif_image_pillow.transpose(Image.ROTATE_90)
            tif_image_init_affine = rotated_90_image_tif.transpose(Image.FLIP_TOP_BOTTOM)
            rotated_90_image_segment = segment_image_label_mark_scale_pillow.transpose(Image.ROTATE_90)
            segment_label_image_init_affine = rotated_90_image_segment.transpose(Image.FLIP_TOP_BOTTOM)
            
        elif rotate_name_last == 'rotate180':
            tif_image_init_affine = tif_image_pillow.transpose(Image.ROTATE_270) 
            segment_label_image_init_affine = segment_image_label_mark_scale_pillow.transpose(Image.ROTATE_270)
    
    elif rotate_name_init == 'rotate0':
        if rotate_name_last == 'original':
            tif_image_init_affine = tif_image_pillow
            segment_label_image_init_affine = segment_image_label_mark_scale_pillow
            
        elif rotate_name_last == 'flipLR':
            tif_image_init_affine = tif_image_pillow.transpose(Image.FLIP_LEFT_RIGHT)
            segment_label_image_init_affine = segment_image_label_mark_scale_pillow.transpose(Image.FLIP_LEFT_RIGHT)
            
        elif rotate_name_last == 'flipTB':
            tif_image_init_affine = tif_image_pillow.transpose(Image.FLIP_TOP_BOTTOM)
            segment_label_image_init_affine = segment_image_label_mark_scale_pillow.transpose(Image.FLIP_TOP_BOTTOM)
            
        elif rotate_name_last == 'rotate180':
            tif_image_init_affine = tif_image_pillow.transpose(Image.ROTATE_180) 
            segment_label_image_init_affine = segment_image_label_mark_scale_pillow.transpose(Image.ROTATE_180)
    
    segment_image_width, segment_image_height = tif_image_pillow.size
    segment_init_affine_image_width, segment_init_affine_image_height = segment_label_image_init_affine.size

    ###xenium_mip_ome_tif_image
    #xenium_mip_ome_tif_image_file = image_save_path+sample+'_cell_morphology_image.jpg'
    #xenium_mip_ome_tif_image_pillow = Image.open(xenium_mip_ome_tif_image_file)
    #xenium_mip_ome_tif_image = xenium_mip_ome_tif_image_pillow
    mip_ome_tif_r = tifffile.TiffReader(mip_ome_tif_data_path)
    mip_ome_tif_image = mip_ome_tif_r.pages[0].asarray()
    mip_ome_tif_rows, mip_ome_tif_cols = mip_ome_tif_image.shape[:2]
    xenium_image_width = mip_ome_tif_cols
    xenium_image_height = mip_ome_tif_rows
    mip_ome_tif_image_gray = normalization_min_max_grayscale(mip_ome_tif_image)
    mip_ome_tif_image_gray_rgb = np.expand_dims(mip_ome_tif_image_gray,2).repeat(3,axis=2)
    mip_ome_tif_image_gray_rgb_uint8 = np.uint8(mip_ome_tif_image_gray_rgb)
    mip_ome_tif_image_gray_rgb_uint8_pillow = Image.fromarray(mip_ome_tif_image_gray_rgb_uint8)
    if mip_ome_tif_image_gray_rgb_uint8_pillow.mode != 'RGB':
        #print('mip_ome_tif_image_gray_rgb_uint8_pillow mode is',mip_ome_tif_image_gray_rgb_uint8_pillow.mode)
        mip_ome_tif_image_gray_rgb_uint8_pillow = mip_ome_tif_image_gray_rgb_uint8_pillow.convert('RGB')
    xenium_mip_ome_tif_image = mip_ome_tif_image_gray_rgb_uint8_pillow
    
    return tif_image_init_affine, segment_label_image_init_affine, xenium_mip_ome_tif_image, segment_init_affine_image_width, segment_init_affine_image_height, segment_image_width, segment_image_height, xenium_image_width, xenium_image_height

def load_segment_xenium_pixel_value(segment_save_path, segment_method, value_save_path, sample, sdata, pixel_size, segment_image_width, segment_image_height, xenium_image_width, xenium_image_height, remove_segment_edge, crop_radius_pixel, center_move_pixel, image_check_path):
    
    #load rotate file
    rotate_mse_list_pd = pd.read_csv(image_check_path+sample+'_he_rotate_mse_values.csv',index_col=0)
    rotate_full_list = rotate_mse_list_pd.columns.tolist()
    
    rotate_mse_list_np = rotate_mse_list_pd.values.reshape(1,-1)
    row_max_index = np.argmax(rotate_mse_list_np)
    rotate_full_name = rotate_full_list[row_max_index]
    rotate_name_parts = rotate_full_name.split('_')
    rotate_name_init = rotate_name_parts[0]
    rotate_name_last = rotate_name_parts[1]

    #load segmentation
    #if segment_method == 'cellpose':
    #    center_pixel_values_pd = pd.read_csv(segment_save_path+sample+'_cellpose_segment_H&E_label_location_save.csv',index_col=0)
    #elif segment_method == 'stardist':
    #    center_pixel_values_pd = pd.read_csv(segment_save_path+sample+'_stardist_segment_H&E_label_location_save.csv',index_col=0)
    center_pixel_values_pd = pd.read_csv(segment_save_path+sample+'_'+segment_method+'_segment_H&E_label_location_save.csv',index_col=0)
    center_pixel_values_filterd_pd = center_pixel_values_pd.loc[center_pixel_values_pd['center_radius_pixel']!=-1]

    #remove egde region
    segment_image_width_max_keep = segment_image_width - remove_segment_edge
    segment_image_height_max_keep = segment_image_height - remove_segment_edge
    center_pixel_values_filterd_remove_edge_pd = center_pixel_values_filterd_pd.loc[(center_pixel_values_filterd_pd['x_center_pixel'] < segment_image_width_max_keep) & (center_pixel_values_filterd_pd['x_center_pixel'] > remove_segment_edge) & (center_pixel_values_filterd_pd['y_center_pixel'] < segment_image_height_max_keep) & (center_pixel_values_filterd_pd['y_center_pixel'] > remove_segment_edge)]
    #if segment_method == 'cellpose':
    #    center_pixel_values_filterd_remove_edge_pd.to_csv(segment_save_path+sample+'_cellpose_segment_H&E_label_location_filterd_remove_edge_save.csv',index=True)
    #elif segment_method == 'stardist':
    #    center_pixel_values_filterd_remove_edge_pd.to_csv(segment_save_path+sample+'_stardist_segment_H&E_label_location_filterd_remove_edge_save.csv',index=True)
    #center_pixel_values_filterd_remove_edge_pd.to_csv(segment_save_path+sample+'_'+segment_method+'_segment_H&E_label_location_filterd_remove_edge_save.csv',index=True)
        
    #load xenium
    #xenium_nucleus_pixel_values_pd = pd.read_csv(value_save_path+sample+'_xenium_nucleus_pixel_values.csv',index_col=0)
    file_check_name = value_save_path+'/'+sample+'_xenium_nucleus_pixel_values.csv'
    if os.path.exists(file_check_name):
        #print(sample+'_xenium_nucleus_pixel_values.csv is exist!')
        xenium_nucleus_pixel_values_pd = pd.read_csv(file_check_name,index_col=0)
    else:
        xenium_nucleus_pixel_values_pd = xenium_nucleus_pixel_values_check(sdata, sample, pixel_size, value_save_path)
    
    #caculate transform_ratio
    segment_center_radius_pixel_whole_np = center_pixel_values_filterd_remove_edge_pd['center_radius_pixel'].values
    segment_center_radius_pixel_mean = np.mean(segment_center_radius_pixel_whole_np)
    
    xenium_nucleus_radius_pixel_np = xenium_nucleus_pixel_values_pd['pixel_center_radius_nucleus'].values
    xenium_center_radius_pixel_mean = np.mean(xenium_nucleus_radius_pixel_np)

    transform_ratio = xenium_center_radius_pixel_mean/segment_center_radius_pixel_mean
    print('transform_ratio',transform_ratio)
    
    #consider removing crop in edge parts
    keep_edge_length_xenium = crop_radius_pixel + center_move_pixel
    keep_edge_length = keep_edge_length_xenium/transform_ratio
    segment_image_width_consider_crop_max_keep = segment_image_width - keep_edge_length
    segment_image_height_consider_crop_max_keep = segment_image_height - keep_edge_length
    center_pixel_values_filterd_remove_edge_consider_crop_pd = center_pixel_values_filterd_remove_edge_pd.loc[(center_pixel_values_filterd_remove_edge_pd['x_center_pixel'] < segment_image_width_consider_crop_max_keep) & (center_pixel_values_filterd_remove_edge_pd['x_center_pixel'] > keep_edge_length) & (center_pixel_values_filterd_remove_edge_pd['y_center_pixel'] < segment_image_height_consider_crop_max_keep) & (center_pixel_values_filterd_remove_edge_pd['y_center_pixel'] > keep_edge_length)]

    #if segment_method == 'cellpose':
    #    center_pixel_values_filterd_remove_edge_consider_crop_pd.to_csv(segment_save_path+sample+'_cellpose_segment_H&E_label_location_filterd_remove_edge_consider_crop_save.csv',index=True)
    #elif segment_method == 'stardist':
    #    center_pixel_values_filterd_remove_edge_consider_crop_pd.to_csv(segment_save_path+sample+'_stardist_segment_H&E_label_location_filterd_remove_edge_consider_crop_save.csv',index=True)
    #center_pixel_values_filterd_remove_edge_consider_crop_pd.to_csv(segment_save_path+sample+'_'+segment_method+'_segment_H&E_label_location_filterd_remove_edge_consider_crop_save.csv', index=True)
    
    #consider nuclues overall layout
    xenium_nucleus_pixel_values_x_np = xenium_nucleus_pixel_values_pd['pixel_center_x_nucleus'].values.reshape(-1,)
    xenium_nucleus_pixel_values_y_np = xenium_nucleus_pixel_values_pd['pixel_center_y_nucleus'].values.reshape(-1,)

    xenium_nucleus_pixel_values_x_max = np.max(xenium_nucleus_pixel_values_x_np)
    xenium_nucleus_pixel_values_x_min = np.min(xenium_nucleus_pixel_values_x_np)
    xenium_nucleus_pixel_values_y_max = np.max(xenium_nucleus_pixel_values_y_np)
    xenium_nucleus_pixel_values_y_min = np.min(xenium_nucleus_pixel_values_y_np)
    xenium_nucleus_pixel_values_x_max_ratio = xenium_nucleus_pixel_values_x_max/xenium_image_width
    xenium_nucleus_pixel_values_x_min_ratio = xenium_nucleus_pixel_values_x_min/xenium_image_width
    xenium_nucleus_pixel_values_y_max_ratio = xenium_nucleus_pixel_values_y_max/xenium_image_height
    xenium_nucleus_pixel_values_y_min_ratio = xenium_nucleus_pixel_values_y_min/xenium_image_height

    if rotate_name_init == 'rotate90':
        if rotate_name_last == 'original':
            segment_overall_layout_x_max = segment_image_width - xenium_nucleus_pixel_values_y_min_ratio * segment_image_width
            segment_overall_layout_x_min = segment_image_width - xenium_nucleus_pixel_values_y_max_ratio * segment_image_width
            segment_overall_layout_y_max = xenium_nucleus_pixel_values_x_max_ratio * segment_image_height
            segment_overall_layout_y_min = xenium_nucleus_pixel_values_x_min_ratio * segment_image_height
            
        elif rotate_name_last == 'flipLR':
            segment_overall_layout_x_max = segment_image_width - xenium_nucleus_pixel_values_y_min_ratio * segment_image_width
            segment_overall_layout_x_min = segment_image_width - xenium_nucleus_pixel_values_y_max_ratio * segment_image_width
            segment_overall_layout_y_max = segment_image_height - xenium_nucleus_pixel_values_x_min_ratio * segment_image_height
            segment_overall_layout_y_min = segment_image_height - xenium_nucleus_pixel_values_x_max_ratio * segment_image_height
            
        elif rotate_name_last == 'flipTB':
            segment_overall_layout_x_max = xenium_nucleus_pixel_values_y_max_ratio * segment_image_width
            segment_overall_layout_x_min = xenium_nucleus_pixel_values_y_min_ratio * segment_image_width
            segment_overall_layout_y_max = xenium_nucleus_pixel_values_x_max_ratio * segment_image_height
            segment_overall_layout_y_min = xenium_nucleus_pixel_values_x_min_ratio * segment_image_height
            
        elif rotate_name_last == 'rotate180':
            segment_overall_layout_x_max = xenium_nucleus_pixel_values_y_max_ratio * segment_image_width
            segment_overall_layout_x_min = xenium_nucleus_pixel_values_y_min_ratio * segment_image_width
            segment_overall_layout_y_max = segment_image_height - xenium_nucleus_pixel_values_x_min_ratio * segment_image_height
            segment_overall_layout_y_min = segment_image_height - xenium_nucleus_pixel_values_x_max_ratio * segment_image_height
    
    elif rotate_name_init == 'rotate0':
        if rotate_name_last == 'original':
            segment_overall_layout_x_max = xenium_nucleus_pixel_values_x_max_ratio * segment_image_width
            segment_overall_layout_x_min = xenium_nucleus_pixel_values_x_min_ratio * segment_image_width
            segment_overall_layout_y_max = xenium_nucleus_pixel_values_y_max_ratio * segment_image_height
            segment_overall_layout_y_min = xenium_nucleus_pixel_values_y_min_ratio * segment_image_height
            
        elif rotate_name_last == 'flipLR':
            segment_overall_layout_x_max = segment_image_width - xenium_nucleus_pixel_values_x_min_ratio * segment_image_width
            segment_overall_layout_x_min = segment_image_width - xenium_nucleus_pixel_values_x_max_ratio * segment_image_width
            segment_overall_layout_y_max = xenium_nucleus_pixel_values_y_max_ratio * segment_image_height
            segment_overall_layout_y_min = xenium_nucleus_pixel_values_y_min_ratio * segment_image_height
            
        elif rotate_name_last == 'flipTB':
            segment_overall_layout_x_max = xenium_nucleus_pixel_values_x_max_ratio * segment_image_width
            segment_overall_layout_x_min = xenium_nucleus_pixel_values_x_min_ratio * segment_image_width
            segment_overall_layout_y_max = segment_image_height - xenium_nucleus_pixel_values_y_min_ratio * segment_image_height
            segment_overall_layout_y_min = segment_image_height - xenium_nucleus_pixel_values_y_max_ratio * segment_image_height
            
        elif rotate_name_last == 'rotate180':
            segment_overall_layout_x_max = segment_image_width - xenium_nucleus_pixel_values_x_min_ratio * segment_image_width
            segment_overall_layout_x_min = segment_image_width - xenium_nucleus_pixel_values_x_max_ratio * segment_image_width
            segment_overall_layout_y_max = segment_image_height - xenium_nucleus_pixel_values_y_min_ratio * segment_image_height
            segment_overall_layout_y_min = segment_image_height - xenium_nucleus_pixel_values_y_max_ratio * segment_image_height

    center_pixel_values_filterd_remove_edge_consider_crop_layout_pd = center_pixel_values_filterd_remove_edge_consider_crop_pd.loc[(center_pixel_values_filterd_remove_edge_consider_crop_pd['x_center_pixel'] < segment_overall_layout_x_max) & (center_pixel_values_filterd_remove_edge_consider_crop_pd['x_center_pixel'] > segment_overall_layout_x_min) & (center_pixel_values_filterd_remove_edge_consider_crop_pd['y_center_pixel'] < segment_overall_layout_y_max) & (center_pixel_values_filterd_remove_edge_consider_crop_pd['y_center_pixel'] > segment_overall_layout_y_min)]

    #if segment_method == 'cellpose':
    #    center_pixel_values_filterd_remove_edge_consider_crop_layout_pd.to_csv(segment_save_path+sample+'_cellpose_segment_H&E_label_location_filterd_remove_edge_consider_crop_layout_save.csv',index=True)
    #elif segment_method == 'stardist':
    #    center_pixel_values_filterd_remove_edge_consider_crop_layout_pd.to_csv(segment_save_path+sample+'_stardist_segment_H&E_label_location_filterd_remove_edge_consider_crop_layout_save.csv',index=True)
    #center_pixel_values_filterd_remove_edge_consider_crop_layout_pd.to_csv(segment_save_path+sample+'_'+segment_method+'_segment_H&E_label_location_filterd_remove_edge_consider_crop_layout_save.csv',index=True)
    
    #center_pixel_values_filterd_remove_edge_consider_crop_layout_file = segment_save_path+sample+'_'+segment_method+'_segment_H&E_label_final_location_save.csv'
    center_pixel_values_filterd_remove_edge_consider_crop_layout_file = segment_save_path+sample+'_'+segment_method+'_segment_H&E_label_remove_edge'+str(remove_segment_edge)+'_consider_crop'+str(crop_radius_pixel)+'_move'+str(center_move_pixel)+'_final_location_save.csv'
    if os.path.exists(center_pixel_values_filterd_remove_edge_consider_crop_layout_file):
        pass
    else:
        center_pixel_values_filterd_remove_edge_consider_crop_layout_pd.to_csv(center_pixel_values_filterd_remove_edge_consider_crop_layout_file,index=True)
    
    segment_nucleus_pixel_values_pd = center_pixel_values_filterd_remove_edge_consider_crop_layout_pd
    #print('xenium_nucleus_pixel_values_x_max',xenium_nucleus_pixel_values_x_max)
    #print('xenium_nucleus_pixel_values_x_min',xenium_nucleus_pixel_values_x_min)
    #print('xenium_nucleus_pixel_values_y_max',xenium_nucleus_pixel_values_y_max)
    #print('xenium_nucleus_pixel_values_y_min',xenium_nucleus_pixel_values_y_min)
    #print('segment_overall_layout_x_max',segment_overall_layout_x_max)
    #print('segment_overall_layout_x_min',segment_overall_layout_x_min)
    #print('segment_overall_layout_y_max',segment_overall_layout_y_max)
    #print('segment_overall_layout_y_min',segment_overall_layout_y_min)
    
    #segment_nucleus_pixel_values_num
    segment_nucleus_pixel_values_num = segment_nucleus_pixel_values_pd.shape[0]
    
    return transform_ratio, segment_nucleus_pixel_values_pd, xenium_nucleus_pixel_values_pd, segment_nucleus_pixel_values_num

def check_cell_list_init_function(segment_nucleus_pixel_values_num, check_cell_num, value_save_path):
    
    check_cell_list_init = random.sample(range(0, segment_nucleus_pixel_values_num), check_cell_num)
    cell_num_list_init = list(range(segment_nucleus_pixel_values_num))
    check_cell_left_list_init = list(set(cell_num_list_init)-set(check_cell_list_init))

    #save list
    check_cell_list_init_np = np.array(check_cell_list_init).reshape(-1,1)
    check_cell_list_init_pd = pd.DataFrame(check_cell_list_init_np,columns=['cell_list_index_init'])
    check_cell_list_init_pd.to_csv(value_save_path+'check_cell_list_init_epoch0_save.csv')
    
    check_cell_left_list_init_np = np.array(check_cell_left_list_init).reshape(-1,1)
    check_cell_left_list_init_pd = pd.DataFrame(check_cell_left_list_init_np,columns=['cell_list_index_init'])
    check_cell_left_list_init_pd.to_csv(value_save_path+'check_cell_left_list_init_epoch0_save.csv')
    
    return check_cell_list_init, check_cell_left_list_init

def check_cell_list_update_function(check_cell_list, check_cell_list_left, check_cell_num, value_save_path, current_epoch_num):
    
    current_left_num = len(check_cell_list_left)
    current_sample_cell_index_list = random.sample(range(0, current_left_num), check_cell_num)
    current_cell_list_left_np = np.array(check_cell_list_left)
    current_sample_cell_list_np = current_cell_list_left_np[current_sample_cell_index_list]
    current_sample_cell_list = current_sample_cell_list_np.tolist()
    current_cell_list_left_for_next = list(set(check_cell_list_left)-set(current_sample_cell_list))
    
    #save list
    current_sample_cell_list_pd = pd.DataFrame(current_sample_cell_list_np,columns=['cell_list_index'])
    current_sample_cell_list_pd.to_csv(value_save_path+'check_cell_list_epoch'+str(current_epoch_num+1)+'_save.csv')
    
    current_cell_list_left_for_next_np = np.array(current_cell_list_left_for_next).reshape(-1,1)
    current_cell_list_left_for_next_pd = pd.DataFrame(current_cell_list_left_for_next_np,columns=['cell_list_index'])
    current_cell_list_left_for_next_pd.to_csv(value_save_path+'check_cell_left_list_epoch'+str(current_epoch_num+1)+'_save.csv')

    return current_sample_cell_list, current_cell_list_left_for_next

def load_xenium_search_region_for_each_segment_nucleus(segment_nucleus_pixel_values_pd, xenium_nucleus_pixel_values_pd, tif_image_init_affine, segment_label_image_init_affine, xenium_mip_ome_tif_image, check_cell_list_init, check_cell_left_list_init, mip_ome_extract_ratio, segment_init_affine_image_width, segment_init_affine_image_height, xenium_image_width, xenium_image_height, transform_ratio, mip_ome_extract_min, output_path, current_epoch_num, image_check_path, sample):
    
    #os settings
    crop_xenium_save_path = output_path+'/crop_epoch'+str(current_epoch_num)+'_xenium_save/'
    if not os.path.exists(crop_xenium_save_path):
        os.makedirs(crop_xenium_save_path)
    crop_he_save_path = output_path+'/crop_epoch'+str(current_epoch_num)+'_he_save/'
    if not os.path.exists(crop_he_save_path):
        os.makedirs(crop_he_save_path)

    crop_segment_save_path = output_path+'/crop_epoch'+str(current_epoch_num)+'_segment_save/'
    if not os.path.exists(crop_segment_save_path):
        os.makedirs(crop_segment_save_path)
    
    value_save_path = output_path+'/value_save/'
    
    #load settings
    segment_nucleus_pixel_values_np = segment_nucleus_pixel_values_pd.values
    xenium_image_base_length = min(xenium_image_width, xenium_image_height)
    xenium_image_extract_crop_radius = xenium_image_base_length * mip_ome_extract_ratio
    #print('xenium_image_extract_crop_radius',xenium_image_extract_crop_radius)
    segment_image_extract_crop_radius = xenium_image_extract_crop_radius / transform_ratio
    
    #set list
    check_segment_cell_value_list = []
    check_segment_cell_index_list= []
    
    cell_num_in_sample_list = len(check_cell_list_init)
    cell_num_in_total = segment_nucleus_pixel_values_pd.shape[0]
    #check_cell_list_tmp = check_cell_list_init
    #check_cell_list_current = check_cell_list_tmp
    check_cell_list_current = check_cell_list_init
    #cell_num_list = list(range(cell_num_in_total))
    #check_cell_left_list_current = list(set(cell_num_list)-set(check_cell_list_current))
    check_cell_left_list_current = check_cell_left_list_init
    check_cell_total_list_current = check_cell_list_init + check_cell_left_list_init
    
    #print('cell_num_in_sample_list',cell_num_in_sample_list)
    #print('cell_num_in_total',cell_num_in_total)
    #print('check_cell_list_current',check_cell_list_current)
    #print('cell_num_list',cell_num_list)
    #print('check_cell_left_list_current',check_cell_left_list_current)

    #load rotate file
    rotate_mse_list_pd = pd.read_csv(image_check_path+sample+'_he_rotate_mse_values.csv',index_col=0)
    rotate_full_list = rotate_mse_list_pd.columns.tolist()
    
    rotate_mse_list_np = rotate_mse_list_pd.values.reshape(1,-1)
    row_max_index = np.argmax(rotate_mse_list_np)
    rotate_full_name = rotate_full_list[row_max_index]
    rotate_name_parts = rotate_full_name.split('_')
    rotate_name_init = rotate_name_parts[0]
    rotate_name_last = rotate_name_parts[1]

    for check_cell_index in range(cell_num_in_sample_list):
        current_check_cell_index = check_cell_list_current[check_cell_index]
        
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
        
        current_segment_pixel_y_ratio = current_pixel_y/segment_init_affine_image_height
        current_segment_pixel_x_ratio = current_pixel_x/segment_init_affine_image_width
        
        current_xenium_pixel_y = xenium_image_height * current_segment_pixel_y_ratio
        current_xenium_pixel_x = xenium_image_width * current_segment_pixel_x_ratio
        
        #xenium crop setting
        current_xenium_pixel_y_min = current_xenium_pixel_y - xenium_image_extract_crop_radius
        current_xenium_pixel_y_max = current_xenium_pixel_y + xenium_image_extract_crop_radius
        current_xenium_pixel_x_min = current_xenium_pixel_x - xenium_image_extract_crop_radius
        current_xenium_pixel_x_max = current_xenium_pixel_x + xenium_image_extract_crop_radius
        
        current_xenium_nucleus_pixel_values_pd = xenium_nucleus_pixel_values_pd.loc[(xenium_nucleus_pixel_values_pd['pixel_center_x_nucleus'] < current_xenium_pixel_x_max) & (xenium_nucleus_pixel_values_pd['pixel_center_x_nucleus'] > current_xenium_pixel_x_min) & (xenium_nucleus_pixel_values_pd['pixel_center_y_nucleus'] < current_xenium_pixel_y_max) & (xenium_nucleus_pixel_values_pd['pixel_center_y_nucleus'] > current_xenium_pixel_y_min)]
        
        #resample as the extracted region without nucleus
        if current_xenium_nucleus_pixel_values_pd.shape[0] < mip_ome_extract_min:
            print('There is one region without cell!')
            print('check_cell_index',check_cell_index)
            print('current_check_cell_index',current_check_cell_index)
            for resample_time in range(len(check_cell_left_list_current)):
                print('There is one time resampling!')
                resample_cell_index_current = random.choice(check_cell_left_list_current)
                print('resample_cell_index_current',resample_cell_index_current)
                
                if rotate_name_init == 'rotate90':
                    if rotate_name_last == 'original':
                        current_resample_pixel_y = segment_init_affine_image_height - segment_nucleus_pixel_values_np[resample_cell_index_current,0]
                        current_resample_pixel_x = segment_nucleus_pixel_values_np[resample_cell_index_current,1]
                        
                    elif rotate_name_last == 'flipLR':
                        current_resample_pixel_y = segment_init_affine_image_height - segment_nucleus_pixel_values_np[resample_cell_index_current,0]
                        current_resample_pixel_x = segment_init_affine_image_width - segment_nucleus_pixel_values_np[resample_cell_index_current,1]
                        
                    elif rotate_name_last == 'flipTB':
                        current_resample_pixel_y = segment_nucleus_pixel_values_np[resample_cell_index_current,0]
                        current_resample_pixel_x = segment_nucleus_pixel_values_np[resample_cell_index_current,1]
                        
                    elif rotate_name_last == 'rotate180':
                        current_resample_pixel_y = segment_nucleus_pixel_values_np[resample_cell_index_current,0]
                        current_resample_pixel_x = segment_init_affine_image_width - segment_nucleus_pixel_values_np[resample_cell_index_current,1]
                
                elif rotate_name_init == 'rotate0':
                    if rotate_name_last == 'original':
                        current_resample_pixel_y = segment_nucleus_pixel_values_np[resample_cell_index_current,1]
                        current_resample_pixel_x = segment_nucleus_pixel_values_np[resample_cell_index_current,0]
                        
                    elif rotate_name_last == 'flipLR':
                        current_resample_pixel_y = segment_nucleus_pixel_values_np[resample_cell_index_current,1]
                        current_resample_pixel_x = segment_init_affine_image_width - segment_nucleus_pixel_values_np[resample_cell_index_current,0]
                        
                    elif rotate_name_last == 'flipTB':
                        current_resample_pixel_y = segment_init_affine_image_height - segment_nucleus_pixel_values_np[resample_cell_index_current,1]
                        current_resample_pixel_x = segment_nucleus_pixel_values_np[resample_cell_index_current,0]
                        
                    elif rotate_name_last == 'rotate180':
                        current_resample_pixel_y = segment_init_affine_image_height - segment_nucleus_pixel_values_np[resample_cell_index_current,1]
                        current_resample_pixel_x = segment_init_affine_image_width - segment_nucleus_pixel_values_np[resample_cell_index_current,0]
                
                current_resample_segment_pixel_y_ratio = current_resample_pixel_y/segment_init_affine_image_height
                current_resample_segment_pixel_x_ratio = current_resample_pixel_x/segment_init_affine_image_width

                current_resample_xenium_pixel_y = xenium_image_height * current_resample_segment_pixel_y_ratio
                current_resample_xenium_pixel_x = xenium_image_width * current_resample_segment_pixel_x_ratio
                
                #xenium crop setting
                current_resample_xenium_pixel_y_min = current_resample_xenium_pixel_y - xenium_image_extract_crop_radius
                current_resample_xenium_pixel_y_max = current_resample_xenium_pixel_y + xenium_image_extract_crop_radius
                current_resample_xenium_pixel_x_min = current_resample_xenium_pixel_x - xenium_image_extract_crop_radius
                current_resample_xenium_pixel_x_max = current_resample_xenium_pixel_x + xenium_image_extract_crop_radius
                
                current_resample_xenium_nucleus_pixel_values_pd = xenium_nucleus_pixel_values_pd.loc[(xenium_nucleus_pixel_values_pd['pixel_center_x_nucleus'] < current_resample_xenium_pixel_x_max) & (xenium_nucleus_pixel_values_pd['pixel_center_x_nucleus'] > current_resample_xenium_pixel_x_min) & (xenium_nucleus_pixel_values_pd['pixel_center_y_nucleus'] < current_resample_xenium_pixel_y_max) & (xenium_nucleus_pixel_values_pd['pixel_center_y_nucleus'] > current_resample_xenium_pixel_y_min)]
                
                if current_resample_xenium_nucleus_pixel_values_pd.shape[0] > mip_ome_extract_min:
                    current_xenium_nucleus_pixel_values_pd = current_resample_xenium_nucleus_pixel_values_pd
                    check_cell_list_current[check_cell_index] = resample_cell_index_current
                    check_cell_left_list_current = list(set(check_cell_total_list_current)-set(check_cell_list_current))
                    current_check_cell_index = resample_cell_index_current
                    current_xenium_pixel_y_min = current_resample_xenium_pixel_y_min
                    current_xenium_pixel_y_max = current_resample_xenium_pixel_y_max
                    current_xenium_pixel_x_min = current_resample_xenium_pixel_x_min
                    current_xenium_pixel_x_max = current_resample_xenium_pixel_x_max
                    break
                
        current_xenium_nucleus_pixel_values_pd.to_csv(crop_xenium_save_path+'check_cell'+str(check_cell_index)+'_index'+str(current_check_cell_index)+'_search_range_extract_ratio'+str(round(mip_ome_extract_ratio,3))+'_xenium_nucleus_pixel_values.csv')
        crop_nucleus_for_extract_image_rgb_tile = xenium_mip_ome_tif_image.crop((current_xenium_pixel_x_min, current_xenium_pixel_y_min, current_xenium_pixel_x_max, current_xenium_pixel_y_max))
        if crop_nucleus_for_extract_image_rgb_tile.mode != 'RGB':
            #print('crop_nucleus_for_extract_image_rgb_tile mode is',crop_nucleus_for_extract_image_rgb_tile.mode)
            crop_nucleus_for_extract_image_rgb_tile = crop_nucleus_for_extract_image_rgb_tile.convert('RGB')
            #print('crop_nucleus_for_extract_image_rgb_tile mode has been converted!')
        
        crop_nucleus_for_extract_image_rgb_tile.save(crop_xenium_save_path+'check_cell'+str(check_cell_index)+'_index'+str(current_check_cell_index)+'_Rp'+str(round(xenium_image_extract_crop_radius,3))+'_search_range_xenium_nucleus_crop_patch.jpg', "JPEG")
        
        #segment crop setting
        imagerow_down  = current_pixel_y - segment_image_extract_crop_radius
        imagerow_up    = current_pixel_y + segment_image_extract_crop_radius
        imagecol_left  = current_pixel_x - segment_image_extract_crop_radius
        imagecol_right = current_pixel_x + segment_image_extract_crop_radius
        
        #segment_label_image_init_affine_crop
        segment_label_image_crop_for_extract_tile = segment_label_image_init_affine.crop((imagecol_left, imagerow_down, imagecol_right, imagerow_up))
        if segment_label_image_crop_for_extract_tile.mode != 'RGB':
            #print('segment_label_image_crop_for_extract_tile mode is',segment_label_image_crop_for_extract_tile.mode)
            segment_label_image_crop_for_extract_tile = segment_label_image_crop_for_extract_tile.convert('RGB')
            #print('segment_label_image_crop_for_extract_tile mode has been converted!')
        
        segment_label_image_crop_for_extract_tile.save(crop_he_save_path+'check_cell'+str(check_cell_index)+'_index'+str(current_check_cell_index)+'_Rp'+str(round(xenium_image_extract_crop_radius,3))+'_search_range_segment_for_compare_nucleus_crop_patch.jpg', "JPEG")
        
        #tif_image_pillow
        tif_image_crop_for_extract_tile = tif_image_init_affine.crop((imagecol_left, imagerow_down, imagecol_right, imagerow_up))
        if tif_image_crop_for_extract_tile.mode != 'RGB':
            #print('tif_image_crop_for_extract_tile mode is',tif_image_crop_for_extract_tile.mode)
            tif_image_crop_for_extract_tile = tif_image_crop_for_extract_tile.convert('RGB')
            #print('tif_image_crop_for_extract_tile mode has been converted!')

        tif_image_crop_for_extract_tile.save(crop_he_save_path+'check_cell'+str(check_cell_index)+'_index'+str(current_check_cell_index)+'_Rp'+str(round(xenium_image_extract_crop_radius,3))+'_search_range_he_for_compare_nucleus_crop_patch.jpg', "JPEG")
        
    #save check segment cell values
    check_segment_cell_value_list_concatenate = np.concatenate(check_segment_cell_value_list,axis=0)
    #print('check_segment_cell_value_list_concatenate shape is',check_segment_cell_value_list_concatenate.shape)
    check_segment_cell_value_list_concatenate_pd = pd.DataFrame(check_segment_cell_value_list_concatenate, index = check_segment_cell_index_list, columns = ['pixel_center_x_nucleus','pixel_center_y_nucleus','pixel_center_radius_nucleus'])
    check_segment_cell_value_list_concatenate_pd.to_csv(value_save_path+'cell'+str(len(check_cell_list_current))+'_in_total_epoch'+str(current_epoch_num)+'_segment_nucleus_pixel_values.csv')
    
    print('Loading xenium search regions finished!')
    
    return check_cell_list_current, check_cell_left_list_current