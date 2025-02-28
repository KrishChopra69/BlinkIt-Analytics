# BlinkIt Sales Analytics 🚀

## 📌 Overview
This project analyzes BlinkIt’s historical sales data to derive actionable insights and predict sales. It leverages **Power BI for interactive visualizations** and **machine learning models for sales forecasting**, helping optimize inventory, stock management, pricing, and store performance.

---

## 🎯 Key Objectives
- **Identify Sales Trends** across **Outlet Tiers (1,2,3)**, **Sizes (Small, Medium, High)**, and **Item Categories**.
- **Predict Sales** using a **Random Forest Model**.
- **Develop a Power BI Dashboard** for dynamic data visualization.
- **Optimize Inventory & Store Strategy** based on insights.

---

## 📂 Dataset
**Total Records:** 8,523 rows  
**Columns:** 12 Features, including:
- **Item Details:** Fat Content, Type, Weight, Visibility
- **Outlet Info:** Location Type, Size, Establishment Year, Type
- **Sales Metrics:** Historical Sales, Rating

---

## 🛠️ Tools & Technologies
- **Python** (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
- **Power BI** (Interactive Dashboard for Exploratory Data Analysis)
- **VSCode** (Development Environment)

---

## 📊 Data Cleaning & Feature Engineering
✅ **Handled Missing Values:**
- **Item Weight & Visibility** imputed with median values.
- Outlier capping using **Interquartile Range (IQR) Method**.

✅ **Engineered New Features:**
- **Avg_Sales_per_Outlet** (Performance indicator per outlet)
- **Years_Since_Establishment** (2025 - Establishment Year)
- **Visibility_per_Weight** (Item Visibility / Item Weight)
- **Sales_per_Outlet_Type** (Average sales per outlet type)
- **Interaction Features:** (Visibility x Item Type, Weight x Item Type)

---

## 📈 Power BI Dashboard
**6 Interactive Charts:**

✔ **Sales by Fat Content**
✔ **Outlet Location & Fat Content Impact on Sales**
✔ **Item Type & Fat Content Impact on Sales**
✔ **Outlet Size Performance Comparison**
✔ **Outlet Location Performance Comparison**
✔ **Sales by Outlet Establishment Year**

**7 Filters:**

✔ **Outlet Location Type**
✔ **Outlet Size**
✔ **Item Type**
✔ **Total Sales**
✔ **Average Sales**
✔ **No. Of Items**
✔ **Average Rating**

---

## 🤖 Machine Learning Model - Sales Prediction
**Final Model:** **Random Forest Regressor** (Optimized with GridSearchCV)
- **R² Score:** 0.6142 (Explains 61.42% of sales variance)
- **MAE:** 26.16
- **MSE:** 1528.41

✅ **Cross-Validation:** Mean R² = 0.6069 (Good generalization capability)

🔹 **How Sales Prediction Helps Inventory & Stock Management:**
- **Prevents Overstocking & Understocking**: Helps in better demand planning.
- **Optimizes Warehouse Allocation**: Ensures efficient product distribution.
- **Supports Seasonal Trend Analysis**: Allows proactive inventory adjustments.

---

## 📌 Key Business Insights
📊 **Store Performance Analysis:**
- **OUT049 & OUT027 outperform others**, while **OUT010 & OUT019 underperform**.
- **Medium-sized outlets lag behind in expected sales (~$4,000 difference).**

📊 **Sales Trends:**
- **Tier 3 Outlets show the highest predicted growth**.
- **Fruits & Vegetables + Snack Foods are the highest revenue-generating categories**.
- **Shift towards Low Fat products (~$750K vs. Regular Fat at ~$400K).**

📊 **Optimization Strategies:**
- **Reallocate inventory to high-performing stores.**
- **Increase promotion of underperforming item categories.**
- **Leverage pricing & discounts based on demand prediction.**

---

## 🚀 Future Enhancements
🔹 **Incorporate Seasonal Trends & Promotions** for better forecasting.  
🔹 **Develop a Recommendation System** for product bundling.  
🔹 **Improve Store-Level Analysis** by integrating external data sources.  

---

## 📢 How to Use
1️⃣ Clone the repo: `git clone https://github.com/KrishChopra69/BlinkIt-Analytics.git`
2️⃣ Install dependencies: `pip install -r requirements.txt`
3️⃣ Run the Python scripts in **VSCode** for **EDA & Model Training**
4️⃣ Open the **Power BI file** to explore the Dashboard

---

## 📌 Author & Contact
📌 **Author:** Krish Chopra  
🔗 **GitHub Repo:** [BlinkIt-Analytics](https://github.com/KrishChopra69/BlinkIt-Analytics)  
📧 **Email:** er.krishchopra@gmail.com
