# BD4H 2020Fall NIH-Chest-X-rays-Classification Project
This is the repository for the course project of Georgia Tech CSE6250 Big Data Health. 

# Project Overview

This project aims to build an automated tool for diagnosing thoracic disease using X-ray images. An end-to-end deep convolutional neural network architecture was developed to detect and classify thoracic diseases in the lung region facilitated with transfer learning techniques. The performance of the model has been compared with state-of-art in the literature with NIH Chest X-ray Dataset.

# Installation

This program can be run on a computer either with or without GPU. Only one package in the environment need to be revised to corresponding setup, which is `tensforflow` for cpu only and `tensorflow-gpu` when equipped with GPU. 

### AWS

If using AWS EC2 instance, the Deep Learning AMI Ubuntu 18.04 `p2.xlarge` instance is preferred.  It has one Tesla K80 GPU that allows for the training. Additional elastic storage (e.g. 500 GB) is needed to be mounted to the instance to store the image data and results.  

### Python Environment
The python environment to run the program can be installed from `environment.yml` using `Conda`. 
```
conda env create -f environment.yml
```

### Download Dataset
To download the dataset from Kaggle, run the `kaggle_data_download.sh` script with Kaggle username and key.
```
./kaggle_data_download.sh
```

### Train the Model
The main entrance for the program is the `distill_train.py`. To run the program, simply use
```
python distill_train.py
```

# References

- Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. [ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases](docs/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf).

- Kaggle NIH X-ray Dataset [https://www.kaggle.com/nih-chest-xrays/data](https://www.kaggle.com/nih-chest-xrays/data).

- NIH-Chest-X-rays-Classification [https://github.com/paloukari/NIH-Chest-X-rays-Classification](https://github.com/paloukari/NIH-Chest-X-rays-Classification)
