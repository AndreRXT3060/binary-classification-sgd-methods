# Stochastic Gradient Methods for Binary Image Classification

## 📌 Overview

This project implements and compares several optimization methods for **binary image classification**, focusing on **stochastic gradient-based techniques**.

The main goal is to study the behavior and performance of different optimization strategies under a **time-constrained setting**, highlighting their convergence properties and practical efficiency.

---

## ⚙️ Implemented Methods

### 🔹 Deterministic Methods
- **Steepest Descent (Gradient Descent)**
- **Armijo Line Search (Adaptive Step Size)**

### 🔹 Stochastic Methods
- **Stochastic Gradient Descent (SGD) with Minibatch**
- **SGD with Stochastic Armijo-like Rule**
- **SGD with Momentum**

---

##  Key Features

- ⏱️ **Time-based stopping criterion** for fair comparison  
- 📊 **Loss tracking over time**  
- ⚡ **Parallel computation with Numba**  
- 🔬 **Comparison between deterministic and stochastic optimization**  
- 📈 **Visualization of convergence behavior**  

---

## 📊 Results

Each method is evaluated based on:

- Objective function value over time  
- Classification accuracy on a test set  

Typical output:

- 📉 Semilog plot of objective vs time  
- ✅ Final classification accuracy  

---

## 📦 Dataset

This project uses a **preprocessed version of the GISETTE dataset** for binary image classification.

### 📥 Download

The datasets are not included due to size and preprocessing constraints.

You can download them here:

**[https://drive.google.com/drive/folders/1vIoB4GF0yrY08-uGdlWi2-wOeEn_jv1W?usp=sharing]**

---

### 📁 Setup

After downloading, place the datasets in:
datasets/GISETTE.mat
datasets/MNIST_8_9.mat

### ⚙️ Notes on preprocessing

The datasets used in this project:

- contain **flattened (vectorized) images**  
- are **normalized**  
- may include additional preprocessing steps

### Install the dependencies
Set up and activate the virtual environement then from command prompt enter
pip install -r requirements.txt

### To run
To run a demo on one of the 2 included datasets 
python main.py

---

### ⚠️ Disclaimer

These datasets may differ from the original datasets available from public sources.  
Results may not be directly comparable with the standard dataset.

## 🧪 Technical Details

- Optimization implemented using **NumPy**
- Performance improvements via **Numba JIT compilation**
- Custom implementation of:
  - Logistic regression loss
  - Gradient computation
  - Line search methods

## 🎯 Motivation

This project was developed to:

- deepen understanding of **optimization algorithms in machine learning**  
- explore **trade-offs between deterministic and stochastic methods**  
- build a **fully custom training pipeline from scratch**



