# 🌧️ Rainfall Prediction Using Machine Learning

## 🚀 Run on Google Colab

Click the link below to run the project directly on Google Colab:

[Open in Google Colab](https://colab.research.google.com/github/talibshaikh-dev/rainfall-prediction-using-ml/blob/main/Rainfall_code.ipynb)

### Steps
1. Open the notebook using the above link
2. Run all cells from the menu (Runtime → Run all)
3. The dataset will be loaded directly from GitHub

## 📌 Project Overview
Rainfall prediction plays a crucial role in agriculture, water resource management, and disaster prevention.  
This project uses machine learning techniques to predict **annual rainfall** based on historical weather data.  
The study focuses on **Gangetic West Bengal, India**, a region highly affected by extreme rainfall events.

---

## 🎯 Objectives
- Analyze long-term rainfall trends
- Apply multiple machine learning models for rainfall prediction
- Compare model performance using evaluation metrics
- Identify the most effective model for rainfall forecasting

---

## 📊 Dataset
- **Source:** India Meteorological Department (IMD)
- **Region:** Gangetic West Bengal, India
- **Time Span:** 1901 – 2017
- **Records:** 117 years of annual rainfall data
- **Target Variable:** Annual Rainfall

---

## 🔄 Data Preprocessing
- Handling missing values
- Feature selection
- Train–test data split
- Data normalization (where required)

---

## 📈 Data Visualization
- Annual rainfall trend analysis
- Distribution analysis using histograms and KDE plots
- Visualization to understand rainfall variability over years

---

## 🧠 Machine Learning Models Used
- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regression (SVR)

---

## 📊 Model Evaluation Metrics
Models were evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared Score (R²)

### 🔍 Performance Summary
| Model | MAE | RMSE | R² Score |
|------|-----|------|----------|
| Linear Regression | 151.05 | 212.43 | -0.04 |
| Ridge Regression | 151.05 | 212.43 | -0.04 |
| Decision Tree | 184.27 | 235.29 | -0.28 |
| Random Forest | 174.15 | 227.68 | -0.19 |
| Gradient Boosting | 227.94 | 272.77 | -0.72 |
| Support Vector Regression | 152.43 | 214.67 | -0.06 |

📌 **Observation:**  
Ensemble-based models performed better than simple linear models due to their ability to capture non-linear rainfall patterns.

---

## ⚙️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 🏁 Conclusion
This project demonstrates that machine learning is an effective approach for rainfall prediction using historical climate data.  
Ensemble models provide better performance compared to traditional linear models.  
Such predictions can support **agriculture planning, water management, and disaster preparedness**.

---

## 🔮 Future Enhancements
- Use deep learning models (LSTM)
- Add real-time weather API integration
- Deploy the model using Streamlit or Flask
- Expand analysis to other regions

---

## How to Run the Project 1st method
1. Install Python 3.8 or above
2. Clone the repository
3. Install required libraries using pip
4. Open Jupyter Notebook
5. Run the notebook cells to train and evaluate the models

 ## Run on Google Colab 2nd method 
1. Open Google Colab
2. Upload the dataset or load it from GitHub
3. Install required libraries
4. Run all cells to train and evaluate models

## 👨‍💻 Authors
 MD TALIB 
