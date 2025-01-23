## 1. Download the Dataset
Please visit the official website to download the TUH EEG Corpus dataset (v1.5.2). You can download the dataset from the following link:

[Download TUH EEG Corpus v1.5.2](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/)

```
E:\dataSet\TUSZ\v1.5.2
│
├── _DOCS                 
├── edf                   
```

## 2. Data Preparation
Before running the model training, you need to prepare the data. To do this, run the `build_data.py` script located in the `data_preparation` folder. This script will process the data and prepare it for model training.

```bash
python build_data.py --base_dir <path_to_raw_dataset> --save_data_dir <path_to_save_processed_data> --tuh_eeg_szr_ver v1.5.2
```


After running this script, the processed data will be saved as .pkl files. The files will be automatically grouped into folders based on seizure types. The resulting structure will look as follows:
```
D:\dataSet
│
├── ABSZ                  
├── CPSZ                 
├── FNSZ                 
├── GNSZ                  
├── SPSZ                  
├── TCSZ                 
├── TNSZ             
```

## Generate Normal EEG Data
To obtain a dataset of normal EEG recordings, perform the following additional steps:

Modify the _DOCS folder's seizures_v36r.xlsx file. Update the start and end times in the spreadsheet to specify the time intervals for normal EEG data.

Save the updated seizures_v36r.xlsx file.

Re-run the build_data.py




## 3. Run the Training Code

To use the processed data, update the tusz-cnn-gnn-train.py script to include the paths to the appropriate folders. 

```bash
python tusz-cnn-gnn-train.py
```


More code will be provided after the paper is published


## Citation 🖊️

If you find our work useful, please consider citing our paper:

```
@article{comprehensive_seizure_detection,
  title = {Comprehensive Seizure Detection via Synchronized Inter- and Intra-Channel EEG Feature Fusion},
  journal = {The Visual Computer},
}

```
