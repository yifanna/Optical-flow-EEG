# Optical-flow-EEG
Multilayer and multi-dimensional optical flow vector feature decoding for non-invasive brain electrical imaging of language

## The script runs in a strict sequence to ensure the smooth flow of data and avoid errors:
1. Data_extractions.py —— Reading and loading of the original EEG data (Step 1)
2. Data_processing.py —— Data preprocessing, slicing, filtering, and feature alignment (Step 2)
3. eeg_methods_AF_01.py —— Generation of phase-amplitude topography (RBEAM method, corresponding to Section 3.1 of the paper, Step 3)
4. EEG_OFAMM_02.py —— EEG optical flow dynamics analysis (OFAMM method, Step 4)
5. RF-Hunhe_03.py —— Optical flow features + Random Forest direction four-classification (Step 5)
   
## Environment Installation (Copy and paste into the terminal, install all dependencies in one step)
All script dependencies have been organized. No separate installation is required. Copy the following command and run it directly:

### Core basic dependencies (common to all scripts) 
pip install mne numpy matplotlib scipy
### Terrain map generation relies on (specific to eeg_methods_AF_01.py) 
pip install pykrige scikit-learn joblib
### Optical Flow Analysis Dependency (EEG_OFAMM_02.py Specific) 
pip install opencv-python tqdm pandas
### Classification and Visualization Dependencies (RF-Hunhe_03.py Specific) 
pip install seaborn
Note: pickle, gc, and os are built-in libraries of Python and do not require additional installation.
