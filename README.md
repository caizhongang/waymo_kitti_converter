# waymo_kitti_converter

This repository provides tools for:
- [x] Converting Waymo Open Dataset(WOD)-format data to KITTI-format data
- [x] Converting KITTI-format prediction results to WOD-format results
- [x] Visualization for both formats, allowing prediction vs ground truth comparison

The tools convert the following data types:
- [x] Point clouds
- [x] Images
- [x] Bounding box labels
- [x] Calibration
- [x] Self driving car's poses <span style="color:red">[New!]</span>

The tools has some additional features:
- [x] Config file (coming soon)
- [x] Multiprocessing
- [x] Progress bar

## Setup

####Step 1. Create and use a new environment (taking anaconda for example)

Note: Python versions 3.6/3.7 are supported.
```
conda create -n waymo-kitti python=3.7
conda activate waymo-kitti
```

####Step 2. Install TensorFlow

Note: TensorFlow version 1.1.5/2.0.0/2.1.0 are supported.

The following command install the newest version 2.1.0. 
For this repository, the cpu version is sufficient.
```
pip3 install tensorflow 
```

####Step 3. Install Waymo Open Dataset precompiled packages.

Note: the following command assumes TensorFlow version 2.1.0. 
You may modify the command according to your TensorFlow version. 
See [official guide](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md) 
for additional details.

```
pip3 install --upgrade pip
pip3 install waymo-open-dataset-tf-2-1-0==1.2.0 --user
```

####Step 4. Install other packages

```
pip3 install numpy opencv-python matplotlib tqdm
```
The following optional package is needed for the 3D visualization tools in tools/:
```
pip3 install open3d
```
 
## Convert WOD-format data to KITTI-format data

```
python converter.py
```

## Convert KITTI-format results to WOD-format results
It is assumed that the KITTI-format results has the following format:


Suppose your codebase generates prediction in the KITTI-format,
this tool converts these results to Waymo-format,
and wraps it into a single .bin.

```
python prediction_kitti_to_waymo.py
```

## Others

Additional tools can be found in tools/
- Visualization tool for KITTI-format predictions against WOD-format ground truths 


## References

- [waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset)
- [Waymo_Kitti_Adapter](https://github.com/Yao-Shao/Waymo_Kitti_Adapter)