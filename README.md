# EV_Charging_prediction
AICTE internship cycle 1
# ⚡ EV Charging Demand Prediction

A machine learning-based project to forecast electric vehicle (EV) adoption across U.S. counties using historical data. This predictive system assists in estimating charging infrastructure requirements and supports sustainable transportation planning.

---

## 🧠 Problem Statement

Electric vehicle adoption is accelerating across the U.S., creating a growing demand for charging infrastructure. However, without accurate forecasting, infrastructure planning can fall short. This project aims to predict monthly EV adoption trends to help policymakers, urban developers, and energy providers make informed decisions.

---

## 🚧 Project Improvements Over Time

This project was developed and improved progressively over **4+ weeks**, with multiple iterations and upgrades. Key improvements include:

1. **Initial Development:**
   - Identified problem statement and gathered U.S. county-level EV population data.
   - Built a Random Forest model using basic temporal features (Year, Month).

2. **Data Cleaning & Preprocessing:**
   - Handled missing values, incorrect date formats, and outliers.
   - Converted numerical columns with commas and unified data types.

3. **Feature Engineering:**
   - Extracted `Year`, `Month` from the `Date` field for time-series prediction.
   - Performed exploratory analysis and visualizations (EV trend lines, outlier analysis).

4. **Model Evaluation & Optimization:**
   - Trained a Random Forest Regressor with performance tuning.
   - Achieved strong performance metrics: **R² = 0.9988**, **MAE = 0.01**, **RMSE = 0.06**.

5. **Deployment with Streamlit App:**
   - Built `app.py` for interactive prediction and visualization.
   - Included model loading, user interface, graph display, and evaluation metrics.

6. **Presentation & Reporting:**
   - Created a clean, visually consistent **PPT presentation** and professional documentation.
   - Wrote a detailed README with proper formatting, file structure, and usage guide.

---

## 🔧 Tools & Technologies Used

- **Programming Language:**  
  Python was used as the primary language for data processing, model training, and deployment.

- **Development Environment:**  
  The project was developed using **Visual Studio Code (VS Code)** for writing, running, and debugging Python scripts.

- **Data Handling & Analysis Libraries:**  
  - `pandas` for loading and manipulating structured datasets  
  - `numpy` for numerical operations and array transformations

- **Visualization Libraries:**  
  - `matplotlib` and `seaborn` for generating line charts, distribution plots, and trend visualizations

- **Machine Learning & Modeling:**  
  - `scikit-learn` for training and evaluating the Random Forest Regressor  
  - `train_test_split`, `RandomForestRegressor`, and model evaluation metrics like `MAE`, `RMSE`, and `R²`

- **Model Serialization:**  
  - `joblib` for saving and loading the trained model in `.pkl` format

- **Deployment:**  
  - **Streamlit** was used to build an interactive web application (`app.py`) that allows users to view forecasts and model performance.

- **File Formats:**  
  The project used `.csv` for data, `.py` for Python scripts, `.pkl` for the saved model, `.ipynb` for notebook-based development, and `.pptx` for presentation.

---

## 📈 Model Performance

- **Mean Absolute Error (MAE):** 0.01  
- **Root Mean Squared Error (RMSE):** 0.06  
- **R² Score:** 0.9988  

These values demonstrate strong predictive accuracy and minimal variance from real-world EV adoption trends.

---

## 🔮 Future Scope

- Integrate more features such as population density, fuel prices, and public incentives.
- Explore time-series models (e.g., ARIMA, LSTM) for sequential forecasting.
- Add map-based visualizations for county-level forecasting in the app.
- Host the app on **Streamlit Cloud** or **Heroku** for public access.
- Enable CSV upload feature in the app for user-customized forecasts.

---

## ✅ Key Learnings

- Hands-on experience in real-world dataset cleaning and preprocessing.
- Solidified understanding of regression modeling and evaluation techniques.
- Model deployment using Streamlit for interactive analytics.
- Gained insight into structuring end-to-end ML projects professionally.

---

## 🖼️ Output Preview

- Line chart showing Actual vs Predicted EV adoption over time.
- Performance metrics and prediction results shown in real-time via the app interface.

---

## 📬 Contact

**Developer:** [Gundu Abhinav]  
**Email:** [gundu.abhinav.2005@gmail.com/ abhinavsharmaa200510@gmail.com]  
GitHub:[github.com/yourusername](https://github.com/Abhinavsharma10-hash/EV_Charging_prediction.git / Gundu-Abhinav-Sharma)

---


