# First Floor Height estimation using regression method   
## The workflow 
This folder contains notebooks used to inspect building attributes and estimate FFH using regression method. There are multiple notebooks  with their names indicating study areas, e.g. Launceston, Wagga.  

## Launceston
- [launceston_0_nexis_join.ipynb](launceston_0_nexis_join.ipynb): Join NEXIS building points dataset, ground-truth building points dataset and building footprint geometries.      

- [launceston_1_inspection_and_prep.ipynb](launceston_1_inspection_and_prep.ipynb): Inspect distribution of attributes including FFH in groundtruth building points data, match with building footprint geometries and prepare DEM virtual raster.     

- [launceston_2_floor_height_correlations.ipynb](launceston_2_floor_height_correlations.ipynb): Explore correlation between groundtruth floor height values and DEM and other building attributes.    

- [launceston_3_floor_height_correlations_dem_deriv.ipynb](launceston_3_floor_height_correlations_dem_deriv.ipynb): Explore correlation between groundtruth floor height values and DEM and other building attributes, with DEM derivatives included, e.g. slope, roughness etc.   

- [launceston_4_regression_nexis.ipynb](launceston_4_regression_nexis.ipynb): Train a RF regression model using groundtruth FFH values and multiple attributes at Launceston; assess the model at Launceston. Assess the trained model to Wagga to test transferability.    

- [launceston_5_regression_nexis_building_height.ipynb](launceston_5_regression_nexis_building_height.ipynb): Similar implementation to previous notebook but with building height measures (extracted from AGO elevation datasets) included in the regression.   

### Wagga

- [wagga_0_inspection.ipynb](wagga_0_inspection.ipynb): Inspect and compare FFH values from local council , GA combined final groundtruth and NEXIS datasets.  
- 
- [wagga_1_dem.ipynb](wagga_1_dem.ipynb): Prepare DEM virtual raster.  
- 
- [wagga_2_floor_height_correlations.ipynb](wagga_2_floor_height_correlations.ipynb): Train a RF regression model using groundtruth FFH values and multiple attributes; assess the model performance.  
