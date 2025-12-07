# Customer Churn Prediction

**Course Project – CSE 598 (Statistical Learning, Special Topics)**  
**Arizona State University – ASU ID: 1237308568**

---

## 1. Project Overview

This repository contains a customer churn prediction project focused on telecom customers.  
The goal is to build a statistical learning pipeline that predicts whether a customer is likely to **churn** (leave the service) based on their demographics, service usage, and subscription details.

The work is implemented in Python using Jupyter Notebooks and standard machine learning libraries such as **pandas**, **scikit‑learn**, and **RandomForestClassifier** from `sklearn.ensemble`. The main workflow is captured in `pipeline_final.ipynb`, where the full data preprocessing and modeling pipeline is defined. :contentReference[oaicite:0]{index=0}

This repository is part of my **college course project** for:

> **CSE 598 – Statistical Learning (Special Topics)**  
> Arizona State University (ASU ID: 1237308568)

---

## 2. Dataset

The project uses the **Telco Customer Churn** dataset loaded from the Hugging Face datasets hub via:

```python
from datasets import load_dataset

dataset = load_dataset("aai510-group1/telco-customer-churn")
train_data = dataset["train"].to_pandas()
validation_data = dataset["validation"].to_pandas()
test_data = dataset["test"].to_pandas()
