# lidar  
## The workflow 
This folder contains notebooks and scripts uniquely used to process lidar data. At the moment it contains a few notebooks and scripts to process/analyse mobile lidar data from the QA4Mobile project. Ongoing updates are underway to process street view lidar data that are collected for this project.  

- [0_Generate_tile_bbox_qa4mobile.ipynb](0_Generate_tile_bbox_qa4mobile.ipynb): Generate bounding boxe for each lidar point cloud as a geojson file for data checking and query purpose. Note that you will need your AWS account and MFA token.      

- [1_Query_clip_point_cloud.ipynb](1_Query_clip_point_cloud.ipynb): Query lidar file using tile bounding boxes file and clip point cloud to a building boundary.     

- [2_Filter_points_project_to_facades.ipynb](2_Filter_points_project_to_facades.ipynb): Filter point cloud, reproject point cloud as an image to a given viewpoint.   


## Other files
In addtion there is a python script [point_cloud_processings.py](point_cloud_processings.py) which contains functions used in the notebooks.

**Note:** For the subsequent image analysis the relevant notebooks within the [GSV](./GSV) folder will be used.
