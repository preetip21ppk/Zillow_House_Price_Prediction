# 🏡 Zillow House Price Prediction  
Predicting Single-Family Home Sale Prices Using Zillow’s Public Zestimate Dataset

This project builds an end-to-end machine learning workflow to predict house prices by minimizing **logerror**, the difference between Zillow’s Zestimate and actual sale prices. The pipeline includes **data cleaning, feature engineering, model training, tuning, and evaluation**, with ensemble models demonstrating the strongest performance.

---

## 🎯 Objective  
Accurately estimate home values and reduce logerror by leveraging:

- Property attributes (beds, baths, area, tax value, quality)
- Geolocation features (latitude, longitude, region)
- Engineered numerical and categorical variables
- Cross-validated model comparison with detailed error analysis

---

## 🧠 Key Results (Impact)

- **Ensemble models (Gradient Boosting / XGBoost) outperformed linear models**
- Achieved **~15% reduction in logerror** compared to baseline
- **~12% lower prediction variance** vs. linear models  
- Example performance:  
  - **Linear Regression RMSE ≈ 0.085**  
  - **GBM/XGBoost RMSE ≈ 0.0847**  
- Geospatial features significantly improved accuracy in high-data regions (e.g., **Los Angeles County — FIPS 6037**)

---

## 📦 Data Sources

### **train_2016.csv**
- `parcelid`
- `transactiondate`
- `logerror` (target)

### **properties_2016.csv**
Contains ~60 features, including:
- Bedrooms, bathrooms
- Square footage (living area, lot size)
- Construction/quality metrics
- Land use codes
- Latitude & longitude
- Region, county, ZIP codes
- Tax amount & tax value

---

## 🛠️ Tech Stack  
- Python  
- NumPy, Pandas  
- Scikit-learn  
- XGBoost, Gradient Boosting  
- Matplotlib, Seaborn  
- SciPy  
- Jupyter Notebook  

---

## 🔁 Pipeline

### **1️⃣ Exploratory Data Analysis (EDA)**
- Distribution inspection  
- Correlation heatmaps  
- Scatter plots against logerror  
- Categorical distributions (city, land use, quality)  
- Identified skewness, nonlinearity, and multicollinearity

---

### **2️⃣ Feature Engineering**
- Duplicate removal  
- Missing value imputation:  
  - Mean/median/mode  
  - KNN Imputer  
  - MICE (Iterative)  
- Standardization and scaling  
- Categorical encoding:  
  - One-hot  
  - Label encoding  
  - Target encoding for high-cardinality  
- Interaction & time-based features  
- Outlier detection:  
  - IQR  
  - Z-score  
  - Isolation Forest  
- VIF checks for multicollinearity  
- Dropped low-variance, redundant fields

---

### **3️⃣ Modeling**
Trained and compared:

- Linear Regression  
- Ridge / Lasso / ElasticNet  
- Decision Tree  
- Random Forest  
- AdaBoost  
- Gradient Boosting  
- **XGBoost (best result)**

**Metrics:**  
- RMSE  
- R²  
- Cross-validation  
- Hyperparameter tuning  
  - e.g., `RandomForest(n_estimators=500, max_depth=6)`

---

### **4️⃣ Evaluation & Interpretation**
- Residual plots  
- Error distribution analysis  
- Variance comparison  
- Feature importance ranking  
- Geospatial signal analysis  

---

## 🔍 Top Predictors (Random Forest Importance)
Most influential features included:

- `taxamount`  
- `finishedsquarefeet12`  
- Latitude  
- Longitude  
- `lotsizesquarefeet`  
- Room counts  
- Zoning & land use variables  

These insights validated engineering choices and helped remove low-signal features.

---

## 📈 Cumulative Feature Importance
- **~80% predictive power** captured by top **7 features**  
- **~90%** captured by top **10** features  
- Diminishing returns after ~15  
- Trained optimized model using top-K features for stability + reduced runtime

---

## 📊 Representative Findings
- Ensemble models consistently achieved **lowest RMSE**  
- High-density counties (e.g., LA) provide more reliable predictions  
- Geospatial attributes played a critical role  
- Linear models struggled with nonlinear interactions and produced higher variance  

