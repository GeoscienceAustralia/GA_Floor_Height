# GA-floor-height

## What this repo does
- This repo contains code for the GA first floor height estimation project. 
  - It uses conda to manage python environment specified in the environment.yml file

**Note:** This repo uses the streetview repository code to access Google Street View data: https://github.com/robolyst/streetview

## Setting up the environment to run the code
- Clone the repository to your local machine
````
git clone https://github.com/frontiersi/GA-floor-height.git
cd GA-floor-height
````
- make sure submodules are properly initialised
````
git submodule update --init --recursive
````
- Create the conda environment for your project  
````
cd GA-floor-height
conda env create -f environment.yml
````
- You should now be up and running with a new repo and conda environment


## Updating this repo
- Members of the "FrontierSI staff" Github team have permissions to update this repository. 
