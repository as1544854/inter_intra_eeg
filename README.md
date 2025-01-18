## 1. Download the Dataset
Please visit the official website to download the TUH EEG Corpus dataset (v1.5.2). You can download the dataset from the following link:

[Download TUH EEG Corpus v1.5.2](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/)

## 2. Data Preparation
Before running the model training, you need to prepare the data. To do this, run the `build_data.py` script located in the `data_preparation` folder. This script will process the data and prepare it for model training.

```bash
python build_data.py --base_dir <path_to_raw_dataset> --save_data_dir <path_to_save_processed_data> --tuh_eeg_szr_ver v1.5.2
```
## 3. Run the Training Code
```bash
python tusz-cnn-gnn-train.py
```


More code will be provided after the paper is published
