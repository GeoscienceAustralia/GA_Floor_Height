# Google Street View (GSV)  
## The workflow 
This folder contains notebooks and scripts used for the workflow to estimate FFH using GSV images. To implement the workflow, there are multiple steps (notebooks):  

- Sample from ground-truth building points to extract a subset of representive buildings which will be used for model training and assessment later: [0_Sample_training_data.ipynb](0_Sample_training_data.ipynb)  

- Query and download GSV panorama images and corresponding depth maps for the sampled building points: [1_Download_GSV_pano_depth_Wagga.ipynb](1_Download_GSV_pano_depth_Wagga.ipynb)    

- Clip GSV panorama images and depth maps to the facade of each building of interest: [2_Clip_GSV_panoramas_streetlevel.ipynb](2_Clip_GSV_panoramas_streetlevel.ipynb)  

- Creat building features training/validation labels from GSV streetview training images using LabelMe tools. Note that you wil need to install LabelMe. Some of the tips can be found in [Run_LabelMe_cmds.ipynb](Run_LabelMe_cmds.ipynb)  

- Convert annotated Labels to required format. If you're using image segmentation model (OneFormer) convert to json files using [Run_LabelMe_cmds.ipynb](Run_LabelMe_cmds.ipynb); if using object detection model (YOLO) convert them using [3_Convert_LabelMe_YOLO.ipynb](3_Convert_LabelMe_YOLO.ipynb)  

- Split your sampled building points into training and validation sets: [4_Split_training_points.ipynb](4_Split_training_points.ipynb)  

- Split corresponding GSV images and labels into training and validation sets: [5_Split_GSV_training_samples.ipynb](5_Split_GSV_training_samples.ipynb)  

- Fine-tune a deep learning model using the training and validation datasets. If you use OneFormer image segmentation, use [6_Fine_tune_OneFormer_GSV.ipynb](6_Fine_tune_OneFormer_GSV.ipynb); if using object detection, use [6_Finetune_Yolo_GSV.ipynb](6_Finetune_Yolo_GSV.ipynb)  

- Inference fine-tuned deep learning model. If you use OneFormer image segmentation, use [7_Inference_evaluate_pretrained_OneFormer.ipynb](7_Inference_evaluate_pretrained_OneFormer.ipynb); if using object detection, use [7_Inference_finetuned_YOLO.ipynb](7_Inference_finetuned_YOLO.ipynb)  

- Estimate FFH using inferencing results. For OneFormer use [8_FFH_estimation_GSV_OneFormer.ipynb](8_FFH_estimation_GSV_OneFormer.ipynb); for YOLO use [8_FFH_estimation_GSV_YOLO.ipynb](8_FFH_estimation_GSV_YOLO.ipynb)  

- Estimate gap-filling depths (distance between building and camera) due to the lack of reliability of GSV depth maps: [7.5_Estimate_GSV_gapfilling_depth.ipynb](7.5_Estimate_GSV_gapfilling_depth.ipynb)  

## Other files
In addtion there are a few other notebooks that may be intermediate/unused or just for experimental purpose. The other two notebooks/scripts that are worthwhile to note at the moment are:

- [Groundtruth_FFH_checking.ipynb](Groundtruth_FFH_checking.ipynb): an interactive notebook to estimate roughly the FFH using high-resolution building height data (AGO) and the GSV street view images.  
- [geometry.py] (geometry.py): Scripts containinig the functions used for geometry restoration.

**Note:** This folder requires installation of the streetlevel package to access Google Street View data: https://streetlevel.readthedocs.io/en/master/
