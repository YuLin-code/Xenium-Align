import warnings
import time
warnings.filterwarnings("ignore")
import numpy as np
import cv2
import pandas as pd
#tf.config.set_visible_devices([tf.config.device.CPU_DEVICE_NAME], 'GPU')
import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#import tensorflow as tf
from csbdeep.utils import normalize
from stardist.models import StarDist2D
import squidpy as sq
import argparse
import torch
import random
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from alignment_data_process import * 

def parse_args():
    parser = argparse.ArgumentParser(description='stardist image segmentation')

    # original
    parser.add_argument('-sample', type=str, nargs='+', default=['f59'], help='which sample to be applied.')
    parser.add_argument('-prob_thresh', type=float, nargs='+', default=[0.0], help='the prob_thresh value.')
    parser.add_argument('-data_file_path', type=str, nargs='+', default=['../Dataset/'], help='the xenium data file path.')
    parser.add_argument('-preservation_method', type=str, nargs='+', default=['ff'], help='the preservation method used for tissue section. eg: ff/ffpe')

    args = parser.parse_args()
    return args

def seed_random_torch(seed):
    random.seed(seed)                       
    torch.manual_seed(seed)                 
    np.random.seed(seed)                    
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)            
    torch.cuda.manual_seed_all(seed)        
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.deterministic = True

def normalization_min_max_grayscale(inputdata):
    _range = np.max(inputdata) - np.min(inputdata)
    return ((inputdata - np.min(inputdata)) / _range)*255

def stardist_2D_versatile_he(img, nms_thresh=None, prob_thresh=None):
    # axis_norm = (0,1)   # normalize channels independently
    axis_norm = (0, 1, 2)  # normalize channels jointly
    # Make sure to normalize the input image beforehand or supply a normalizer to the prediction function.
    # this is the default normalizer noted in StarDist examples.
    img = normalize(img, 1, 99.8, axis=axis_norm)
    model = StarDist2D.from_pretrained("2D_versatile_he")
    labels, _ = model.predict_instances(
        img, nms_thresh=nms_thresh, prob_thresh=prob_thresh
    )
    return labels

if __name__ == "__main__":

    #load original setting
    args = parse_args()
    sample = args.sample[0]
    prob_thresh = args.prob_thresh[0]
    data_file_path = args.data_file_path[0]
    preservation_method = args.preservation_method[0]
    
    #print('sample',sample)
    #print('prob_thresh',prob_thresh)
    #print('data_file_path',data_file_path)
    #print('preservation_method',preservation_method)
    
    #set os
    stardist_save_path = './'+sample+'_stardist_prob_thresh'+str(prob_thresh)+'/'
    if not os.path.exists(stardist_save_path):
        os.makedirs(stardist_save_path)
    
    if preservation_method == 'ff':
        #load os
        data_path, mip_ome_tif_data_path, sample, he_img_path = load_os_ff_sample(sample, data_file_path)
    elif preservation_method == 'ffpe':
        if sample == 'kidney_cancer':
            data_file_path = data_file_path+'kidney_cancerprcc_data/'
        elif sample == 'kidney_nondiseased':
            data_file_path = data_file_path+'kidney_nondiseased_data/'
        #load os
        data_path, mip_ome_tif_data_path, sample, he_img_path = load_os_ffpe_sample(sample, data_file_path, stardist_save_path)

    #time_computing
    start_time = time.time()
    print("StarDist Segmentation and Label Location File Generation. Start Time: %s seconds" %(start_time))
    
    #load H&E stained tissue image
    tif_image = sq.im.ImageContainer(he_img_path)
    
    #delete H&E
    if preservation_method == 'ffpe':
        os.remove(he_img_path)
    
    #check tif image
    tif_image_np = np.array(tif_image['image'])[:, :, 0, 0:3]
    #print('tif_image_np original shape is',np.array(tif_image['image']).shape)
    #print('tif_image_np shape is',tif_image_np.shape)
    #print('tif_image_np max is',np.max(tif_image_np))
    #print('tif_image_np min is',np.min(tif_image_np))

    #parameter settings
    if prob_thresh==0:
        prob_thresh_used = None
    else:
        prob_thresh_used = prob_thresh

    StarDist2D.from_pretrained("2D_versatile_he")
    sq.im.segment(
        img=tif_image,
        layer="image",
        channel=None,
        method=stardist_2D_versatile_he,
        layer_added="segmented_stardist_default",
        prob_thresh=prob_thresh_used,
        nms_thresh=None,
    )

    #time_computing
    seg_end_time = time.time()
    print("StarDist Segmentation. End Time: %s seconds" %(seg_end_time))

    #save segmentation cell num
    num_np = np.array([len(np.unique(tif_image['segmented_stardist_default']))]).reshape(1,1)
    num_pd = pd.DataFrame(num_np,index=['cell_segment'],columns=['cell_num'])
    num_pd.to_csv(stardist_save_path+'prob_thresh'+str(prob_thresh)+'_segmentation_cell_num.csv')
    
    #obtain segmentation
    tif_image_segmented_stardist_np = np.array(tif_image['segmented_stardist_default'])
    tif_image_segmented_stardist_np_squeeze = np.squeeze(tif_image_segmented_stardist_np)
    tif_image_segmented_stardist_np_squeeze_scale = normalization_min_max_grayscale(tif_image_segmented_stardist_np_squeeze)

    #save segmentation label
    tif_image_segmented_stardist_squeeze_pd = pd.DataFrame(tif_image_segmented_stardist_np_squeeze)
    tif_image_segmented_stardist_squeeze_scale_pd = pd.DataFrame(tif_image_segmented_stardist_np_squeeze_scale)
    tif_image_segmented_stardist_squeeze_pd.to_csv(stardist_save_path+'prob_thresh'+str(prob_thresh)+'_tif_image_segmented_stardist.csv')
    #tif_image_segmented_stardist_squeeze_scale_pd.to_csv(stardist_save_path+'prob_thresh'+str(prob_thresh)+'_tif_image_segmented_stardist_scale.csv')
    
    labels_mark = np.where(tif_image_segmented_stardist_np_squeeze != 0, 1, tif_image_segmented_stardist_np_squeeze)
    labels_scale = normalization_min_max_grayscale(labels_mark)
    #print('labels_mark==labels_scale',(labels_mark==labels_scale).all())

    labels_mark_pd = pd.DataFrame(labels_mark)
    labels_scale_pd = pd.DataFrame(labels_scale)
    #labels_mark_pd.to_csv(stardist_save_path+'prob_thresh'+str(prob_thresh)+'_labels_mark.csv')
    #labels_scale_pd.to_csv(stardist_save_path+'prob_thresh'+str(prob_thresh)+'_labels_scale.csv')
    cv2.imwrite(stardist_save_path+'prob_thresh'+str(prob_thresh)+'_image_segmented_stardist_label_mark_scale.jpg',labels_scale)

    labels_scale_pillow = Image.fromarray(labels_scale)
    if labels_scale_pillow.mode != 'RGB':
        #print('labels_scale_pillow mode is',labels_scale_pillow.mode)
        labels_scale_pillow = labels_scale_pillow.convert('RGB')
        #print('labels_scale_pillow mode has been converted!')
    labels_scale_pillow.save(stardist_save_path+'prob_thresh'+str(prob_thresh)+'_image_segmented_stardist_label_mark_scale_pillow.jpg', "JPEG")

    #save cell locations
    label_init_test = np.zeros((labels_mark.shape[0],labels_mark.shape[1]))
    x_center_pixel_list = []
    y_center_pixel_list = []
    center_radius_pixel_list = []
    for label_num in range(len(np.unique(tif_image_segmented_stardist_np_squeeze))-1):
        label_num_current = label_num + 1
        labels_index_current = np.where(tif_image_segmented_stardist_np_squeeze==label_num_current)
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
    print('check label ok?',(label_init_test==labels_mark).all())
    
    x_center_pixel_list_np = np.array(x_center_pixel_list).reshape(-1,1)
    y_center_pixel_list_np = np.array(y_center_pixel_list).reshape(-1,1)
    center_radius_pixel_list_np = np.array(center_radius_pixel_list).reshape(-1,1)
    center_pixel_value_np = np.hstack((x_center_pixel_list_np,y_center_pixel_list_np,center_radius_pixel_list_np))
    center_pixel_value_pd = pd.DataFrame(center_pixel_value_np,columns=['x_center_pixel','y_center_pixel','center_radius_pixel'])
    center_pixel_value_pd.to_csv(stardist_save_path+sample+'_stardist_segment_H&E_label_location_save.csv')
    
    cell_index_np = np.arange(0, x_center_pixel_list_np.shape[0]).reshape(-1,1)
    center_pixel_value_for_graph_np = np.hstack((cell_index_np,y_center_pixel_list_np,x_center_pixel_list_np))
    center_pixel_value_for_graph_pd = pd.DataFrame(center_pixel_value_for_graph_np,columns=['CELL_ID','X','Y'])
    #center_pixel_value_for_graph_pd.to_csv(stardist_save_path+sample+'_stardist_segment_H&E_label_location_for_graph_save.csv',index=False)
    
    print('The nucleus segmentation on sample '+sample+' has been completed!')
    
    #time_computing
    end_time = time.time()
    print("StarDist Segmentation and Label Location File Generation. End Time: %s seconds" %(end_time))
    print("StarDist Segmentation and Label Location File Generation. Done. Total Running Time: %s seconds" %(end_time - start_time))

    run_time_second = end_time - start_time
    run_time_minute = run_time_second/60
    run_time_hour = run_time_minute/60
    rum_time_np = np.array([run_time_second,run_time_minute,run_time_hour]).reshape(1,-1)
    rum_time_pd = pd.DataFrame(rum_time_np,columns=['run_time_second','run_time_minute','run_time_hour'],index=['rum_time'])
    rum_time_pd.to_csv(stardist_save_path+'stardist_segmentaion_and_file_generation_run_time_save.csv',index=True)
    
    seg_run_time_second = seg_end_time - start_time
    seg_run_time_minute = seg_run_time_second/60
    seg_run_time_hour = seg_run_time_minute/60
    seg_rum_time_np = np.array([seg_run_time_second,seg_run_time_minute,seg_run_time_hour]).reshape(1,-1)
    seg_rum_time_pd = pd.DataFrame(seg_rum_time_np,columns=['seg_run_time_second','seg_run_time_minute','seg_run_time_hour'],index=['seg_rum_time'])
    seg_rum_time_pd.to_csv(stardist_save_path+'stardist_segmentaion_run_time_save.csv',index=True)
    