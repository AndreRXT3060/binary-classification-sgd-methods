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

## 🧠 Key Features

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

The dataset is not included due to size and preprocessing constraints.

You can download it here:

**[INSERT GOOGLE DRIVE LINK HERE]**

---

### 📁 Setup

After downloading, place the dataset in:
datasets/GISETTE.mat

### ⚙️ Notes on preprocessing

The dataset used in this project:

- contains **flattened (vectorized) images**  
- is **normalized**  
- may include additional preprocessing steps

---

### ⚠️ Disclaimer

This dataset may differ from the original GISETTE dataset available from public sources.  
Results may not be directly comparable with the standard dataset.


