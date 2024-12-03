import warnings
import time
warnings.filterwarnings("ignore")
import numpy as np
import cv2
import pandas as pd
import os
import squidpy as sq
import argparse
import torch
import random
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from cellpose import models
from alignment_data_process import * 

def parse_args():
    parser = argparse.ArgumentParser(description='cellpose image segmentation')

    #parameter settings
    parser.add_argument('-sample', type=str, nargs='+', default=['f59'], help='which sample to be applied.')
    parser.add_argument('-channel_cellpose', type=int, nargs='+', default=[1], help='the channel_cellpose value.')
    parser.add_argument('-min_size', type=int, nargs='+', default=[15], help='the min_size value.')
    parser.add_argument('-flow_threshold', type=float, nargs='+', default=[0.8], help='the flow_threshold value.')
    parser.add_argument('-data_file_path', type=str, nargs='+', default=['./Dataset/'], help='the xenium data file path.')
    parser.add_argument('-use_gpu', type=bool, nargs='+', default=[False], help='use gpu or not.')
    
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

def cellpose_he(img, min_size=15, flow_threshold=0.4, channel_cellpose=0, gpu=False):
    model = models.Cellpose(model_type="nuclei", gpu=gpu)
    res, _, _, _ = model.eval(
        img,
        channels=[channel_cellpose, 0],
        diameter=None,
        min_size=min_size,
        invert=True,
        flow_threshold=flow_threshold,
    )
    return res

if __name__ == "__main__":
    
    #random setting
    seed_random(300)

    #load settings
    args = parse_args()
    sample = args.sample[0]
    channel_cellpose = args.channel_cellpose[0]
    min_size = args.min_size[0]
    flow_threshold = args.flow_threshold[0]
    data_file_path = args.data_file_path[0]
    use_gpu = args.use_gpu[0]

    #load os
    data_path, mip_ome_tif_data_path, sample, he_img_path = load_os(sample, data_file_path)

    #set os
    cellpose_save_path = './'+sample+'_cellpose_channel_cellpose'+str(channel_cellpose)+'_flow_threshold'+str(flow_threshold)+'_min_size'+str(min_size)+'/'
    if not os.path.exists(cellpose_save_path):
        os.makedirs(cellpose_save_path)
    
    #time_computing
    start_time = time.time()
    print("Cellpose Segmentation and Label Location File Generation. Start Time: %s seconds" %(start_time))
    
    #load H&E stained tissue image
    tif_image = sq.im.ImageContainer(he_img_path)
    
    #check tif image
    tif_image_np = np.array(tif_image['image'])[:, :, 0, 0:3]

    #cellpose segmentation
    #sq.im.segment(img=tif_image, layer="image", channel=None, method=cellpose_he, min_size=min_size, flow_threshold=flow_threshold, channel_cellpose=channel_cellpose)
    sq.im.segment(img=tif_image, layer="image", channel=None, method=cellpose_he, min_size=min_size, flow_threshold=flow_threshold, channel_cellpose=channel_cellpose, gpu=use_gpu)

    #save segmentation cell num
    num_np = np.array([len(np.unique(tif_image['segmented_custom']))]).reshape(1,1)
    num_pd = pd.DataFrame(num_np,index=['cell_segment'],columns=['cell_num'])
    num_pd.to_csv(cellpose_save_path+'channel_cellpose'+str(channel_cellpose)+'_flow_threshold'+str(flow_threshold)+'_min_size'+str(min_size)+'_cellpose_segmentation_cell_num.csv')
    
    #obtain segmentation
    tif_image_segmented_cellpose_np = np.array(tif_image['segmented_custom'])
    tif_image_segmented_cellpose_np_squeeze = np.squeeze(tif_image_segmented_cellpose_np)
    tif_image_segmented_cellpose_np_squeeze_scale = normalization_min_max_grayscale(tif_image_segmented_cellpose_np_squeeze)

    #save segmentation label
    tif_image_segmented_cellpose_np_squeeze_pd = pd.DataFrame(tif_image_segmented_cellpose_np_squeeze)
    tif_image_segmented_cellpose_np_squeeze_scale_pd = pd.DataFrame(tif_image_segmented_cellpose_np_squeeze_scale)
    
    labels_mark = np.where(tif_image_segmented_cellpose_np_squeeze != 0, 1, tif_image_segmented_cellpose_np_squeeze)
    labels_scale = normalization_min_max_grayscale(labels_mark)
    #print('labels_mark==labels_scale',(labels_mark==labels_scale).all())

    labels_mark_pd = pd.DataFrame(labels_mark)
    labels_scale_pd = pd.DataFrame(labels_scale)
    cv2.imwrite(cellpose_save_path+'channel_cellpose'+str(channel_cellpose)+'_flow_threshold'+str(flow_threshold)+'_min_size'+str(min_size)+'_image_segmented_cellpose_label_mark_scale.jpg',labels_scale)

    labels_scale_pillow = Image.fromarray(labels_scale)
    if labels_scale_pillow.mode != 'RGB':
        #print('labels_scale_pillow mode is',labels_scale_pillow.mode)
        labels_scale_pillow = labels_scale_pillow.convert('RGB')
        #print('labels_scale_pillow mode has been converted!')
    labels_scale_pillow.save(cellpose_save_path+'channel_cellpose'+str(channel_cellpose)+'_flow_threshold'+str(flow_threshold)+'_min_size'+str(min_size)+'_image_segmented_cellpose_label_mark_scale_pillow.jpg', "JPEG")
    
    #save cell locations
    label_init_test = np.zeros((labels_mark.shape[0],labels_mark.shape[1]))
    x_center_pixel_list = []
    y_center_pixel_list = []
    center_radius_pixel_list = []
    for label_num in range(len(np.unique(tif_image_segmented_cellpose_np_squeeze))-1):
        label_num_current = label_num + 1
        labels_index_current = np.where(tif_image_segmented_cellpose_np_squeeze==label_num_current)
        #print('label_num_current',label_num_current)
        assert(labels_index_current[0].shape[0]==labels_index_current[1].shape[0])
        if labels_index_current[0].shape[0]==0:
            #print('label_num_current without label is ',label_num_current)
            x_center_pixel_list.append(-1)
            y_center_pixel_list.append(-1)
            center_radius_pixel_list.append(-1)
        else:
            for mark_num in range(labels_index_current[0].shape[0]):
                label_init_test[labels_index_current[0][mark_num], labels_index_current[1][mark_num]] = 1
            xmin_pixel = np.min(labels_index_current[1])
            xmax_pixel = np.max(labels_index_current[1])
            ymin_pixel = np.min(labels_index_current[0])
            ymax_pixel = np.max(labels_index_current[0])
            
            x_center_pixel = (xmax_pixel-xmin_pixel)/2+xmin_pixel
            y_center_pixel = (ymax_pixel-ymin_pixel)/2+ymin_pixel
            center_radius_pixel = max((xmax_pixel-xmin_pixel)/2,(ymax_pixel-ymin_pixel)/2)
            x_center_pixel_list.append(x_center_pixel)
            y_center_pixel_list.append(y_center_pixel)
            center_radius_pixel_list.append(center_radius_pixel)
    #print('check label ok?',(label_init_test==labels_mark).all())
    
    x_center_pixel_list_np = np.array(x_center_pixel_list).reshape(-1,1)
    y_center_pixel_list_np = np.array(y_center_pixel_list).reshape(-1,1)
    center_radius_pixel_list_np = np.array(center_radius_pixel_list).reshape(-1,1)
    center_pixel_value_np = np.hstack((x_center_pixel_list_np,y_center_pixel_list_np,center_radius_pixel_list_np))
    center_pixel_value_pd = pd.DataFrame(center_pixel_value_np,columns=['x_center_pixel','y_center_pixel','center_radius_pixel'])
    center_pixel_value_pd.to_csv(cellpose_save_path+sample+'_cellpose_segment_H&E_label_location_save.csv')
    
    cell_index_np = np.arange(0, x_center_pixel_list_np.shape[0]).reshape(-1,1)
    center_pixel_value_for_graph_np = np.hstack((cell_index_np,y_center_pixel_list_np,x_center_pixel_list_np))
    center_pixel_value_for_graph_pd = pd.DataFrame(center_pixel_value_for_graph_np,columns=['CELL_ID','X','Y'])
    center_pixel_value_for_graph_pd.to_csv(cellpose_save_path+sample+'_cellpose_segment_H&E_label_location_for_graph_save.csv',index=False)
    
    print('The nucleus segmentation on sample '+sample+' has been completed!')
    
    #time_computing
    end_time = time.time()
    print("Cellpose Segmentation and Label Location File Generation. End Time: %s seconds" %(end_time))
    print("Cellpose Segmentation and Label Location File Generation. Done. Total Running Time: %s seconds" %(end_time - start_time))    