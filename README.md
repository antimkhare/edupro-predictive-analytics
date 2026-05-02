# 🎓 EduPro — Predictive Modeling for Course Demand & Revenue Forecasting

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> A complete machine learning pipeline that predicts **course enrollment demand** and **revenue** on the EduPro online learning platform — enabling data-driven decisions on course launches, pricing, and instructor recruitment.

---

## 📌 Problem Statement

EduPro currently lacks:
- Predictive models for course enrollment demand
- Revenue forecasting at course and category level
- Quantitative evidence to support pricing and launch decisions

This project shifts EduPro from **reactive reporting** to **proactive planning**.

---

## 🏗️ Project Structure

```
edupro-predictive-analytics/
│
├── app.py                        ← Streamlit web application (entry point)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py            ← Data merging & feature engineering
│   ├── features.py               ← Feature definitions & constants
│   ├── models.py                 ← ModelPipeline: train, evaluate, predict
│   └── visualizations.py        ← All Plotly chart functions
│
├── data/
│   └── EduPro_Online_Platform.xlsx   ← Dataset (3 sheets: Courses, Teachers, Transactions)
│
├── docs/
│   └── EduPro_Research_Paper.docx   ← Full research paper
│
├── .streamlit/
│   └── config.toml               ← Streamlit theme config
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

| Sheet | Records | Key Fields |
|-------|---------|------------|
| Courses | 60 | CourseID, Category, Type, Level, Price, Duration, Rating |
| Teachers | 60 | TeacherID, Expertise, YearsOfExperience, TeacherRating |
| Transactions | 10,000 | TransactionID, CourseID, TeacherID, Date, Amount |

**Platform Summary:**
- 💰 Total Revenue: $911,323
- 📚 12 Course Categories
- 🎓 3 Difficulty Levels (Beginner, Intermediate, Advanced)

---

## 🤖 Models & Results

### Revenue Prediction (Target: CourseRevenue)

| Model | MAE | RMSE | R² | CV R² |
|-------|-----|------|----|-------|
| **Ridge Regression ★** | $2,575 | $3,529 | **0.9840** | **0.9918** |
| Lasso Regression | $2,761 | $3,644 | 0.9829 | 0.9917 |
| Linear Regression | $2,768 | $3,649 | 0.9828 | 0.9916 |
| Random Forest | $3,764 | $5,460 | 0.9616 | 0.9684 |
| Gradient Boosting | $4,901 | $7,860 | 0.9204 | 0.9693 |

### Enrollment Prediction (Target: EnrollmentCount)

| Model | MAE | RMSE | R² | CV R² |
|-------|-----|------|----|-------|
| **Random Forest ★** | 11.22 | 13.14 | **0.053** | -0.204 |
| Ridge Regression | 10.87 | 13.41 | 0.013 | -0.073 |
| Lasso Regression | 10.83 | 13.54 | -0.005 | -0.046 |
| Linear Regression | 10.92 | 13.51 | -0.001 | -0.087 |
| Gradient Boosting | 12.00 | 13.94 | -0.067 | -0.386 |

> 📝 Low enrollment R² is expected — enrollment is driven by marketing and external signals not present in course metadata.

---

## 🔑 Key Findings

| Finding | Detail |
|---------|--------|
| **Revenue is highly predictable** | R² = 0.984 (Ridge). Price is 97.7% of feature importance. |
| **Enrollment is stochastic** | External signals (marketing, timing) dominate. R² ≈ 0.05. |
| **AI & Business lead revenue** | $202K + $182K = 42% of total platform revenue |
| **Course Rating = top enrollment driver** | 33.5% feature importance |
| **Free courses are an untapped opportunity** | Marketing/ML generate $0 on 1,625 enrollments |

---

## 🗂️ App Pages

| Page | Description |
|------|-------------|
| 📊 Overview | KPIs, problem statement, methodology pipeline |
| 🔍 Exploratory Analysis | Monthly trends, category charts, scatter plots, raw data |
| 🤖 Model Evaluation | 5-model comparison, R², MAE/RMSE charts |
| 📌 Feature Importance | Bar charts + radar comparison (enrollment vs revenue) |
| 🔮 Prediction Tool | Enter course params → instant revenue & enrollment forecast |
| 💡 Insights | Key findings, recommendations, freemium gap analysis |

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/edupro-predictive-analytics.git
cd edupro-predictive-analytics
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🌐 Deployment (Streamlit Cloud)

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo
4. Set **Main file path** to `app.py`
5. Click **Deploy**

---

## 📦 Dependencies

```
streamlit==1.35.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
plotly==5.22.0
openpyxl==3.1.2
```

---

## 🔧 Feature Engineering

| Feature | Type | Description |
|---------|------|-------------|
| CoursePrice | Numeric | Exact price in USD |
| CourseDuration | Numeric | Length in hours |
| CourseRating | Numeric | Avg student rating (1–5) |
| YearsOfExperience | Numeric | Instructor experience |
| TeacherRating | Numeric | Instructor rating (1–5) |
| ExpertiseMatch | Binary | 1 if instructor expertise = course category |
| CourseCategory_enc | Encoded | Label-encoded category (12 classes) |
| CourseType_enc | Encoded | Free=0, Paid=1 |
| CourseLevel_enc | Encoded | Beginner / Intermediate / Advanced |

---

## 📄 Research Paper

Full research paper available in [`docs/EduPro_Research_Paper.docx`](docs/EduPro_Research_Paper.docx)

Covers:
- Literature review
- Dataset description & statistics
- Methodology
- Full model results tables
- Feature importance analysis
- Business recommendations

---

## 👤 Author

**Project submitted to:** Unified Mentor  
**Platform:** EduPro Online Learning  
**Tools:** Python · scikit-learn · Streamlit · Plotly

---

## 📃 License

MIT License — free to use and modify with attribution.
