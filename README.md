# Optical-flow-EEG
   Multilayer and multi-dimensional optical flow vector feature decoding for non-invasive brain electrical imaging of language
# Supports batch processing of multiple subjects, multiple conditions, and multiple categories. By modifying the corresponding parameters, batch operation can be achieved.
# All output files will be automatically saved to the designated path, eliminating the need for manual folder creation.

## Execution Sequence (Must be strictly followed; otherwise, an error will occur)
   1. python Data_extractions.py (Read the original BDF data)
   2. python Data_processing.py (Preprocess the data and align the formats)
   3. python eeg_methods_AF_01.py (Generate amplitude/phase topographic maps)
   4. python EEG_OFAMM_02.py (Optical flow analysis, generate flow_results.mat)
   5. python RF-Hunhe_03.py (Random forest classification, output the final results)
   
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
  ** Note:** pickle, gc, and os are built-in libraries of Python and do not require additional installation.

## Detailed Descriptions of Each Script (Functions + Usage Methods + Input/Output) 

### 1. Data_extractions.py
   Developed based on MNE-Python, it reads the original BDF files of the Inner Speech Dataset, supports single-subject and multi-subject, single-block and multi-block data reading and loading, and provides standardized data for subsequent preprocessing.
#### Core Functions (Can be directly called):
   - extract_subject_from_bdf: Read the original BDF data of a specified block for a single subject
   - extract_data_from_subject: Extract all 3 blocks of preprocessed data (EEG/EXG/baseline) for a single subject
   - extract_block_data_from_subject: Extract the preprocessed data of a single block for a single subject
   - load_events: Load the event labels for a specified block of a subject
   - extract_data_multisubject: Batch extract data of multiple subjects and concatenate them
   - extract_report/extract_tfr: Load the analysis report and time-frequency analysis data (optional, not necessary)
**Input:** Original BDF files of the Inner Speech Dataset
**Output:** Preprocessed EEG (Shape: Trials × Channels × Time)

### 2. Data_processing.py
   Preprocess the three-dimensional EEG data output by Data_extractions.py, including slicing, filtering, baseline correction, and label alignment, to ensure the data format is uniform and compatible with the subsequent terrain map generation script.The shape of the input data X must be (sessions, channels, time points); Y is the label matrix, used for condition/category filtering, and the parameters are case-insensitive.
**Input:** The three-dimensional EEG array output by Data_extractions.py
**Output:** Standardized preprocessed data (adapted for eeg_methods_AF_01.py)

### 3. eeg_methods_AF_01.py (corresponding to Section 3.1 of the paper)
   Inner Speech EEG Phase-Amplitude Terrain Map Generation Tool, using the RBEAM method, automatically completing "condition/category filtering → filtering → Hilbert transformation → Kriging interpolation", and finally outputting rectangular scalp phase maps and amplitude maps.
   Usage: Directly modify the core parameters in the script (no need to modify other code):
         root_dir = "G:/InnerSpeech2021/"  # Dataset root path (modify according to your own environment)
         save_dir = "G:/InnerSpeech2021-RBEAM/"  # Terrain map save path (modify according to your own environment)
         N_S_list = [4]  # Trial numbers to be processed (can be modified to multiple, such as [4,5,6])
         datatype = "eeg"  # Data type (fixed as eeg, no need to modify)
         Conditions_list = ["Pron", "Inner", "Vis"]  # Experimental conditions (fixed, no need to modify)
         Classes_list = ["Up", "Down", "Right", "Left"]  # Classification categories (fixed, no need to modify)
**Input:** Standardized EEG data output by Data_processing.py 
**Output:**
      - Amplitude/Folder: Amplitude topographic map PNG file (for use with EEG_OFAMM_02.py) 
      - Phase/Folder: Phase topographic map PNG file (optional, not necessary) 
      
### 4. EEG_OFAMM_02.py
   EEG optical flow analysis tool (OFAMM method), focusing on three main tasks: generating the optical flow field → identifying source points/drainage points → tracking pixel trajectories, automatically saving visual images and CSV data. The core output is flow_results.mat (for use by classification scripts).
  Usage: Ensure the script can read the Amplitude folder generated by eeg_methods_AF_01.py (the path does not need to be manually modified, default linkage is in place), and run directly.
**Input: **The PNG sequence of amplitude topographic maps within the Amplitude folder generated by eeg_methods_AF_01.py. 
**Output:**
- Visualized images: Flow vector field diagram, Source/Sink point identification diagram, Trajectory tracking diagram 
- CSV file: Velocity components of optical flow (u/v), source/destination coordinates, trajectory coordinates 
- flow_results.mat: Core data of optical flow (must be output, for use by RF-Hunhe_03.py)
- 
### 5. RF-Hunhe_03.py
   Core Function: Extract optical flow features from the "flow_results.mat" file output by EEG_OFAMM_02.py, train a random forest model, and achieve classification and recognition of the "Up, Down, Right, Left" four directions. Automatically output the model, visualization charts, and a complete set of evaluation indicators.
   Usage: Modify the data root path in the script (pointing to the output directory of EEG_OFAMM_02.py), and run directly. No other manual operations are required.
**Input: **
  "flow_results.mat" output by EEG_OFAMM_02.py 
**Output:**
- Model files: Random Forest model (.pkl), Data standardizer (.pkl) 
- Feature file: High-dimensional feature matrix (.mat) 
- Visual charts: Feature heat map, confusion matrix chart, comparison chart of recall rate/precision rate for each category 
- CSV indicators: Classification indicators (accuracy, recall rate, F1 value), confusion matrix (numerical version + percentage version), experimental summary information

