# Tutorial of Xenium-Alignment

## A software tool to generate keypoints for the alignment between H&E image and DAPI-stained image in Xenium Explorer

Xenium-Alignment is a software tool that can generate keypoints csv file for the desktop application of Xenium Explorer to align H&E image and DAPI-stained image and visualize RNA transcript localization in tissues with subcellular resolution.

### System and OS Requirements: 

The developed tool can run on both Linux and Windows. It has been tested on a computing server with 2.2 GHz, 144 cores CPU, 503 GB RAM and one NVIDIA TU102 [TITAN RTX] GPU under an ubuntu 18.04 operating system.

### Install Xenium-Alignment from Github:

```bash
git clone https://github.com/YuLin-code/Xenium-Alignment.git
cd Xenium-Alignment
```

### Python Dependencies: 

Xenium-Alignment depends on the Python scientific stack and python virutal environment with conda (<https://anaconda.org/>) is recommended.

```shell
conda create -n Xenium_Alignment python=3.9
conda activate Xenium_Alignment
pip install -r requirements.txt
```

## Examples:

### 1. Data Preprocess Check

Before implementing tool, the initialized H&E-stained image and cell morphology image should be checked. If the image layout between the cell morphology image with black-and-white color and the H&E-stained image is not consistent, the following commands can not be used and the developed tool needs to be adjusted.

- **sample** defines the input sample name.
- **data_file_path** defines the specific path address of Xenium datasets. The Xenium data files and H&E-stained images of each sample are placed inside. If the path address or file name is modified, please go to the 'load_os' function in alignment_data_process.py script and make the corresponding modification.

```bash
python data_preprocess_check.py -sample f59 -data_file_path ./Dataset/
```

### 2. Cellpose Image Segmentation

Next, the Cellpose image segmentation is applied and the recommended hyper-parameter settings are shown as follows:

- **channel_cellpose** defines the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
- **min_size** defines the minimum number of pixels per mask.
- **flow_threshold** defines all cells with errors below threshold are kept.
- **use_gpu** defines whether to use GPU to run model or not.

```bash
python cellpose_image_segmentation.py -sample f59 -channel_cellpose 1 -min_size 15 -flow_threshold 0.8 -data_file_path ./Dataset/
```

### 3. Score Method for Keypoints Generation

Scoring and ranking functions are applied to generate the paired keypoints of H&E image and DAPI-stained image. Delaunay graph matching is also used to filter out the outliers of unmatched keypoints for more accurate results. It takes about one or two days to generate the keypoints file. Here we use the following recommended hyper-parameter settings to demo purposes. If the keypoints result is poor, we recommend to adjust the hyper-parameters of 'crop_radius_pixel' and 'center_move_pixel'.

- **crop_radius_pixel** defines the pixel value to crop nucleus as each patch.
- **center_move_pixel** defines the pixel value to move each cropped patch of the other four angles.
- **check_cell_num** defines the number of randomly sampled cells that are used to score and rank for keypoints.
- **crop_image_resize** defines the resize value of the cropped image for scoring.
- **graph_source_str** defines the mode to build delaunay graph.

```bash
python score_method_to_alignment_for_keypoints.py -sample f59 -crop_radius_pixel 400 -center_move_pixel 300 -check_cell_num 100 -channel_cellpose 1 -min_size 15 -flow_threshold 0.8 -data_file_path ./Dataset/
```

## Document Description:

In the respective file paths, we have the following files.

- ***_initialied_HE_image.jpg**:    The checked H&E image after initial rotation.

- ***_cell_morphology_image.jpg**:    The cell morphology images are extracted from the DAPI image in OME-TIFF format. 

- ***_keypoints.csv**:    The final keypoints file that is used to be imported to Xenium Explorer.