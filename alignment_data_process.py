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

def find_files_with_field(directory, field):
    pattern = os.path.join(directory, '*' + field + '*.jpg')
    return glob.glob(pattern)
 
def find_files_with_field_for_rank(directory, field):
    pattern = os.path.join(directory, '*' + field + '*rgb_xenium.jpg')
    return glob.glob(pattern)

def load_os(sample, data_file_path):

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
    he_img_path = data_file_path+'/histology/'+sample+'.tif'

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

def initialize_tif_segment_xenium_image(he_img_path, cellpose_save_path, sample, mip_ome_tif_data_path, channel_cellpose, flow_threshold, min_size):

    #tif_image_init_affine
    tif_image_pillow = Image.open(he_img_path)
    rotated_270_image_tif = tif_image_pillow.transpose(Image.ROTATE_270)
    flipped_lr_image_tif = rotated_270_image_tif.transpose(Image.FLIP_LEFT_RIGHT)
    tif_image_init_affine = flipped_lr_image_tif
    #tif_image_init_affine.save(cellpose_save_path+'tif_image_init_affine_pillow.jpg', "JPEG")
    
    #segment_label_image_init_affine
    #segment_image_label_mark_scale_file = cellpose_save_path+'channel_cellpose'+str(channel_cellpose)+'_flow_threshold'+str(flow_threshold)+'_min_size'+str(min_size)+'_image_segmented_cellpose_label_mark_scale_pillow.jpg'
    segment_image_label_mark_scale_file = cellpose_save_path+'channel_cellpose'+str(channel_cellpose)+'_flow_threshold'+str(flow_threshold)+'_min_size'+str(min_size)+'_image_segmented_cellpose_label_mark_scale.jpg'
    segment_image_label_mark_scale_pillow = Image.open(segment_image_label_mark_scale_file)
    rotated_270_image_segment = segment_image_label_mark_scale_pillow.transpose(Image.ROTATE_270)
    flipped_lr_image_segment = rotated_270_image_segment.transpose(Image.FLIP_LEFT_RIGHT)
    segment_label_image_init_affine = flipped_lr_image_segment
    #segment_label_image_init_affine.save(cellpose_save_path+'segment_label_image_init_affine_pillow.jpg', "JPEG")

    #xenium_mip_ome_tif_image
    #xenium_mip_ome_tif_image_file = image_save_path+sample+'_cell_morphology_image.jpg'
    #xenium_mip_ome_tif_image_pillow = Image.open(xenium_mip_ome_tif_image_file)
    #xenium_mip_ome_tif_image = xenium_mip_ome_tif_image_pillow
    mip_ome_tif_r = tifffile.TiffReader(mip_ome_tif_data_path)
    mip_ome_tif_image = mip_ome_tif_r.pages[0].asarray()
    mip_ome_tif_rows, mip_ome_tif_cols = mip_ome_tif_image.shape[:2]
    #print('mip_ome_tif_image for train shape is', mip_ome_tif_image.shape)
    mip_ome_tif_image_gray = normalization_min_max_grayscale(mip_ome_tif_image)
    mip_ome_tif_image_gray_rgb = np.expand_dims(mip_ome_tif_image_gray,2).repeat(3,axis=2)
    mip_ome_tif_image_gray_rgb_uint8 = np.uint8(mip_ome_tif_image_gray_rgb)
    mip_ome_tif_image_gray_rgb_uint8_pillow = Image.fromarray(mip_ome_tif_image_gray_rgb_uint8)
    if mip_ome_tif_image_gray_rgb_uint8_pillow.mode != 'RGB':
        #print('mip_ome_tif_image_gray_rgb_uint8_pillow mode is',mip_ome_tif_image_gray_rgb_uint8_pillow.mode)
        mip_ome_tif_image_gray_rgb_uint8_pillow = mip_ome_tif_image_gray_rgb_uint8_pillow.convert('RGB')
    xenium_mip_ome_tif_image = mip_ome_tif_image_gray_rgb_uint8_pillow
    
    return tif_image_init_affine, segment_label_image_init_affine, xenium_mip_ome_tif_image

def load_segment_xenium_pixel_value(cellpose_save_path, value_save_path, sample, sdata, pixel_size):
    
    #load segmentation
    center_pixel_values_pd = pd.read_csv(cellpose_save_path+sample+'_cellpose_segment_H&E_label_location_save.csv',index_col=0)
    center_pixel_values_filterd_pd = center_pixel_values_pd.loc[center_pixel_values_pd['center_radius_pixel']!=-1]
    segment_nucleus_pixel_values_pd = center_pixel_values_filterd_pd
    
    #load xenium
    #xenium_nucleus_pixel_values_pd = pd.read_csv(value_save_path+sample+'_xenium_nucleus_pixel_values.csv',index_col=0)
    file_check_name = value_save_path+'/'+sample+'_xenium_nucleus_pixel_values.csv'
    if os.path.exists(file_check_name):
        #print(sample+'_xenium_nucleus_pixel_values.csv is exist!')
        xenium_nucleus_pixel_values_pd = pd.read_csv(file_check_name,index_col=0)
    else:
        xenium_nucleus_pixel_values_pd = xenium_nucleus_pixel_values_check(sdata, sample, pixel_size, value_save_path)
    
    #caculate transform_ratio
    segment_center_radius_pixel_whole_np = segment_nucleus_pixel_values_pd['center_radius_pixel'].values
    segment_center_radius_pixel_mean = np.mean(segment_center_radius_pixel_whole_np)
    
    xenium_nucleus_radius_pixel_np = xenium_nucleus_pixel_values_pd['pixel_center_radius_nucleus'].values
    xenium_center_radius_pixel_mean = np.mean(xenium_nucleus_radius_pixel_np)

    transform_ratio = xenium_center_radius_pixel_mean/segment_center_radius_pixel_mean

    return transform_ratio, segment_nucleus_pixel_values_pd, xenium_nucleus_pixel_values_pd