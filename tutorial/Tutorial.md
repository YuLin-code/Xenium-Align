# Tutorial of Xenium-Align

## Defining keypoints to align H&E images and Xenium DAPI-stained images automatically

Xenium-Align is a keypoints identification method that can generate the accurate keypoints file to automatically conduct image alignment between imported H&E image and DAPI-stained image for Xenium Explorer software.

### System and OS Requirements: 

The developed tool can run on both Linux and Windows. It has been tested on a computing server with 2.2 GHz, 144 cores CPU, 503 GB RAM and one NVIDIA TU102 [TITAN RTX] GPU under an ubuntu 18.04 operating system.

### Install Xenium-Align from Github:

```bash
git clone https://github.com/YuLin-code/Xenium-Align.git
cd Xenium-Align
```

### Python Dependencies: 

Xenium-Align depends on the Python scientific stack and python virutal environment with conda (<https://anaconda.org/>) is recommended.

```shell
conda create -n Xenium_Align python=3.9
conda activate Xenium_Align
pip install -r requirements.txt
```

## Examples:

### 1. Data Preprocess Check

For the imported H&E image in Xenium Explorer, we automatically rotate the image direction to keep consistent of the image layout between DAPI-stained image with black-and-white color and H&E image-stained image. 

- **sample** defines the input sample name.
- **preservation_method** defines the preservation method of the input Xenium sample. It includes ff and ffpe methods.
- **data_file_path** defines the specific path address of Xenium datasets. The Xenium data files and H&E-stained images of each sample are placed inside. If the path address or file name is modified, please go to the 'load_os_ff_sample' and 'load_os_ffpe_sample' function in alignment_data_process.py script and make the corresponding modification.

```bash
cd Xenium_Align
python data_preprocess_check.py -sample f59 -preservation_method ff -data_file_path ../Dataset/
```

### 2. Cellpose Image Segmentation

Next, the Cellpose image segmentation is applied and the recommended hyper-parameter settings are shown as follows:

- **channel_cellpose** defines the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
- **min_size** defines the minimum number of pixels per mask.
- **flow_threshold** defines all cells with errors below threshold are kept.
- **use_gpu** defines whether to use GPU to run model or not.

```bash
python cellpose_image_segmentation.py -sample f59 -preservation_method ff -data_file_path ../Dataset/ -channel_cellpose 1 -min_size 15 -flow_threshold 0.8
```

### 3. Xenium-Align for Keypoints Generation

Scoring and ranking functions of multi-angles enhanced image assessment are applied to generate the initially paired keypoints of H&E image and DAPI-stained image. Delaunay triangulation graph matching and nucleus polygon matching are also used to filter out the outliers from two aspects of topology consistency between two graphs and overlap degree between two cell nuclei, respectively. Here we use the following recommended hyper-parameter settings on sample f59 to demo purposes.

- **crop_radius_pixel** defines the pixel value of radius size to crop nucleus as each patch.
- **center_move_pixel** defines the pixel value to move each cropped patch of the other four angles.
- **check_cell_num** defines the number of randomly sampled cells in H&E image that are used to identify the matched keypoints.
- **mip_ome_extract_ratio** defines the ratio of the minimum value in width and height of DAPI-stained image to set the search radius.
- **mip_ome_extract_min** defines the minimum number of cells in the search region that would be kept.
- **segment_method** defines the nucleus segmentation model used for H&E image.
- **overlap_type** defines the overlap type used in nucleus polygon matching.
- **overlap_threshold_ave** defines the threshold value of average overlap in nucleus polygon matching.
- **overlap_threshold_min** defines the threshold value of minimum overlap in nucleus polygon matching.
- **keypoints_min_num** defines the minimum number of keypoints to output.
- **epoch_num** defines the maximum number of epochs to implement Xenium-Align for generating keypoints.
- **crop_image_resize** defines the resize value of the cropped image for multi-angles image assessment.
- **graph_source_str** defines the mode to build Delaunay triangulation graph.

```bash
python xenium_alignment_for_keypoints.py -sample f59 -preservation_method ff -data_file_path ../Dataset/ -crop_radius_pixel 400 -center_move_pixel 300 -check_cell_num 100 -mip_ome_extract_ratio 0.125 -mip_ome_extract_min 50 -segment_method cellpose -overlap_type overlap_ave -overlap_threshold_ave 0.9 -keypoints_min_num 15 -epoch_num 30
```

## Document Description:

In the respective file paths, we have the following files.

- ***_HE_image.jpg**:    The checked H&E image after automatic rotation.

- ***_Xenium_DAPI_image.jpg**:    The DAPI-stained image are extracted from the original Xenium data in OME-TIFF format.

- ***_keypoints.csv**:    The final keypoints file that is used to be imported to Xenium Explorer.