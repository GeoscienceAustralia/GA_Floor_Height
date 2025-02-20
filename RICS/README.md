# RICS (Rapid Image Collection System)  
## The workflow 
This folder contains some notebooks that were used for FFH estimation using the RICS images:  

- Converts RICS database file (.sdb) into geojson files: [0_Convert_RICS_database_to_geojson.ipynb](0_Convert_RICS_database_to_geojson.ipynb)  

- Query and download RICS images from AWS S3 buckets: [1_RICS_query_download.ipynb](1_RICS_query_download.ipynb). Note that this notebook requires your s3 account information and MFA token.      

- Split your building points into training and validation sets: [2_Split_training_points_RICS.ipynb](2_Split_training_points_RICS.ipynb)  

- Split corresponding RICS images and labels into training and validation sets: [3_Split_RICS_training_samples.ipynb](3_Split_RICS_training_samples.ipynb)  

- As the image segmentation or object detection model fine-tuning and inferencing are basically very similar to GSV processing workflow except for the input and output files. Therefore no notebooks are duplicated here. For these steps refer to the [GSV folder](./GSV).

- Estimate FFH using inferencing results from OneFormer: [4_FFH_estimation_RICS_simplified.ipynb](4_FFH_estimation_RICS_simplified.ipynb);  


## Other files
In addtion there are a few other notebooks that may be intermediate/unused or just for experimental purpose. The other notebooks that are worthwhile to note at the moment are:

- [Back_calculate_RICS_coordinates.ipynb](Back_calculate_RICS_coordinates.ipynb): This notebook estimated the original coordinates for RICS trajectory points due to significant error in the original data.  
- [Estimate_RICS_focal_length.ipynb](Estimate_RICS_focal_length.ipynb): Notebook to estimate focal length from street view images with typical residential front door visible.
- [Sensitivity_analysis_perspective_projection.ipynb](Sensitivity_analysis_perspective_projection.ipynb): Notebook to assess sensitivity of errors to various parameters.
