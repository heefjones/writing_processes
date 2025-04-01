# Linking Writing Processes to Writing Quality
Linking Writing Processes to Writing Quality Kaggle Competition: 
This [Kaggle competition](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality) tasked me to use keystroke log data to predict overall writing quality.

## Data
The dataset contained keystroke log data from 2471 unique writers. The logs contained keystrokes and mouse clicks taken during the composition of an essay. Each essay was scored on a scale of 0 to 6.
- **Rows:** ~8.4 million 
- **Columns:** 12 (including writer ID and score/label)  

### Null Values
There were no null values in the data.

## Modeling
- **Model:** XGBoost Regressor  
- **Hyperparameter Tuning:** Bayesian Optimization with 110 iterations  
- **Results:**
    - Final RMSE of 0.7014 on a 20% unseen test set.

## Files
- 📊 analysis.ipynb – EDA, feature engineering, model iteration, and final submission.
- 🛠️ helper.py – Custom functions for data processing and model training.
- 📈 submission.csv – Final predictions on the dummy test data.

## Repository Structure
```
/writing_processes
├── analysis.ipynb
├── helper.py
├── /submission.csv
└── README.md
```
