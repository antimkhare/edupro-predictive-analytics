"""
app.py  ─  EduPro Predictive Analytics Dashboard
=================================================
Run with:
    streamlit run app.py

Requires:
    pip install -r requirements.txt
    Place EduPro_Online_Platform.xlsx inside the /data folder.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from src.data_loader import build_merged, get_monthly_trends, get_category_summary, load_raw_sheets
from src.features import FEATURE_COLS, FEATURE_LABELS, TARGET_ENROLLMENT, TARGET_REVENUE
from src.models import ModelPipeline
from src.visualizations import (
    fig_category_enrollments,
    fig_category_revenue,
    fig_error_comparison,
    fig_importance_bar,
    fig_importance_radar,
    fig_level_enrollments,
    fig_monthly_trend,
    fig_r2_comparison,
    fig_rating_distribution,
    fig_revenue_gap,
    fig_scatter_price_revenue,
    fig_course_type_pie,
)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduPro — Predictive Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.hero-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2.5rem 2rem; border-radius: 16px; margin-bottom: 1.5rem; color: white;
}
.hero-title  { font-size: 2.2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
.hero-sub    { font-size: 1rem; color: #94a3b8; margin-top: 0.5rem; }
.hero-accent { color: #f97316; }
.kpi-card {
    background: white; border-radius: 12px; padding: 1.2rem 1.5rem;
    border-left: 4px solid var(--bc, #f97316);
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}
.kpi-value { font-size: 1.9rem; font-weight: 700; color: #1e293b; line-height: 1; }
.kpi-label { font-size: 0.78rem; color: #64748b; margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 0.5px; }
.section-header {
    font-size: 1.2rem; font-weight: 700; color: #1e293b;
    border-bottom: 2px solid #f97316; padding-bottom: 0.4rem; margin: 1.2rem 0 0.8rem 0;
}
.insight-card {
    background: white; border-radius: 10px; padding: 1rem 1.2rem;
    border-left: 3px solid var(--c, #f97316);
    box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 0.8rem;
}
.insight-title { font-weight: 600; font-size: 0.9rem; color: #1e293b; }
.insight-text  { font-size: 0.82rem; color: #64748b; margin-top: 0.2rem; line-height: 1.5; }
.pred-result {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 1px solid #6ee7b7; border-radius: 12px; padding: 1.5rem;
    text-align: center; margin-top: 1rem;
}
.pred-value { font-size: 2rem; font-weight: 700; color: #065f46; }
.pred-label { font-size: 0.78rem; color: #047857; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA & TRAIN MODELS (cached) ─────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def get_data():
    merged = build_merged()
    _, _, transactions = load_raw_sheets()
    monthly  = get_monthly_trends(transactions)
    cat_df   = get_category_summary(merged)
    return merged, monthly, cat_df

@st.cache_resource(show_spinner="Training models… (this runs once)")
def get_pipeline(merged):
    pipeline = ModelPipeline()
    pipeline.fit(merged)
    return pipeline

merged, monthly, cat_df = get_data()
pipeline = get_pipeline(merged)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 EduPro Analytics")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "📊 Overview",
            "🔍 Exploratory Analysis",
            "🤖 Model Evaluation",
            "📌 Feature Importance",
            "🔮 Prediction Tool",
            "💡 Insights & Recommendations",
        ],
    )
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown(f"- 📚 60 Courses")
    st.markdown(f"- 👨‍🏫 60 Teachers")
    st.markdown(f"- 🔄 10,000 Transactions")
    st.markdown(f"- 💰 ${merged['CourseRevenue'].sum():,.0f} Revenue")
    st.markdown("---")
    st.caption("EduPro Predictive Analytics · Built with Streamlit + scikit-learn")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":

    st.markdown("""
    <div class="hero-card">
      <div class="hero-title">EduPro — <span class="hero-accent">Predictive Modeling</span></div>
      <div class="hero-title" style="font-size:1.3rem;font-weight:400;margin-top:4px">
        Course Demand &amp; Revenue Forecasting
      </div>
      <div class="hero-sub" style="margin-top:1rem;max-width:700px">
        A complete ML pipeline predicting enrollment demand and revenue across 60 courses,
        60 instructors and 10,000 transactions — using Ridge, Lasso, Random Forest
        &amp; Gradient Boosting models.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    kpis = [
        ("60",   "Courses",      "#f97316"),
        (f"${merged['CourseRevenue'].sum()/1000:.0f}K", "Total Revenue", "#3b82f6"),
        ("10,000","Enrollments", "#10b981"),
        (f"{merged['CourseRating'].mean():.2f}/5","Avg Rating","#f59e0b"),
        ("0.984","Best Rev R²",  "#8b5cf6"),
    ]
    cols = st.columns(5)
    for col, (val, lbl, color) in zip(cols, kpis):
        col.markdown(
            f'<div class="kpi-card" style="--bc:{color}">'
            f'<div class="kpi-value">{val}</div>'
            f'<div class="kpi-label">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.markdown('<div class="section-header">Problem Statement</div>', unsafe_allow_html=True)
        st.markdown("""
EduPro currently lacks:
- **Predictive models** for course enrollment demand
- **Revenue forecasting** at course and category level
- **Quantitative evidence** for pricing and course launch decisions

This project introduces ML-powered intelligence to move EduPro from
**reactive reporting** to **proactive planning**.
        """)

        st.markdown('<div class="section-header">Prediction Targets</div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame({
                "Target":      ["Enrollment Count", "Course Revenue ($)", "Category Revenue"],
                "Description": [
                    "Number of enrollments per course",
                    "Total revenue generated per course",
                    "Aggregated revenue by category",
                ],
                "Best Model R²": ["0.053 (Random Forest)", "0.984 (Ridge)", "Derived"],
            }),
            use_container_width=True, hide_index=True,
        )

    with c2:
        st.markdown('<div class="section-header">Pipeline Steps</div>', unsafe_allow_html=True)
        for num, title, desc in [
            ("1", "Data Merging",       "Courses ↔ Transactions ↔ Teachers"),
            ("2", "Feature Engineering","Price bands, rating tiers, experience buckets"),
            ("3", "Preprocessing",      "Label encoding · normalise · 80/20 split"),
            ("4", "Model Training",     "5 models × 2 targets = 10 trained models"),
            ("5", "Evaluation",         "MAE · RMSE · R² · 5-fold CV"),
            ("6", "Insights",           "Feature importance → business recommendations"),
        ]:
            st.markdown(
                f'<div style="display:flex;gap:12px;margin-bottom:10px;align-items:flex-start">'
                f'<div style="background:#f97316;color:white;border-radius:50%;width:28px;height:28px;'
                f'display:flex;align-items:center;justify-content:center;font-weight:700;'
                f'font-size:13px;flex-shrink:0;margin-top:2px">{num}</div>'
                f'<div><strong style="font-size:.88rem">{title}</strong><br>'
                f'<span style="font-size:.8rem;color:#64748b">{desc}</span></div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-header">Revenue by Category</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_category_revenue(cat_df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Exploratory Analysis":
    st.markdown("## Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "📊 Categories", "🔵 Scatter", "📋 Raw Data"])

    with tab1:
        st.plotly_chart(fig_monthly_trend(monthly), use_container_width=True)
        c1, c2 = st.columns(2)
        c1.plotly_chart(fig_course_type_pie(merged),      use_container_width=True)
        c2.plotly_chart(fig_level_enrollments(merged),    use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        c1.plotly_chart(fig_category_revenue(cat_df),     use_container_width=True)
        c2.plotly_chart(fig_category_enrollments(cat_df), use_container_width=True)

        st.markdown("### Full Category Table")
        display = cat_df.copy()
        display["TotalRevenue"]         = display["TotalRevenue"].map("${:,.0f}".format)
        display["AvgPrice"]             = display["AvgPrice"].map("${:,.2f}".format)
        display["AvgRating"]            = display["AvgRating"].map("{:.2f}".format)
        display["RevenuePerEnrollment"] = display["RevenuePerEnrollment"].map("${:,.2f}".format)
        st.dataframe(display.rename(columns={
            "CourseCategory": "Category", "TotalRevenue": "Revenue",
            "TotalEnrollments": "Enrollments", "AvgRating": "Avg Rating",
            "AvgPrice": "Avg Price", "NumCourses": "# Courses",
            "RevenuePerEnrollment": "Rev/Enrollment",
        }), use_container_width=True, hide_index=True)

    with tab3:
        st.plotly_chart(fig_scatter_price_revenue(merged), use_container_width=True)
        st.plotly_chart(fig_rating_distribution(merged),   use_container_width=True)

    with tab4:
        cols_show = [
            "CourseName", "CourseCategory", "CourseType", "CourseLevel",
            "CoursePrice", "CourseDuration", "CourseRating",
            "YearsOfExperience", "TeacherRating",
            "EnrollmentCount", "CourseRevenue", "RevenuePerEnrollment",
        ]
        st.dataframe(merged[cols_show].fillna(0), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Evaluation":
    st.markdown("## Model Evaluation & Comparison")

    target_label = st.radio(
        "Prediction target:",
        ["💰 Course Revenue", "📈 Enrollment Count"],
        horizontal=True,
    )
    target_key = TARGET_REVENUE if "Revenue" in target_label else TARGET_ENROLLMENT
    is_rev     = target_key == TARGET_REVENUE
    res        = pipeline.results[target_key]
    metrics_df = res["metrics"]
    best_name  = res["best_name"]

    st.info(
        f"✅ **Best model:** {best_name}  |  "
        f"R² = {metrics_df.iloc[0]['R2']:.4f}  |  "
        f"CV R² = {metrics_df.iloc[0]['CV_R2']:.4f}"
    )

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("### Performance Table")
        disp = metrics_df.copy()
        if is_rev:
            disp["MAE"]  = disp["MAE"].map("${:,.2f}".format)
            disp["RMSE"] = disp["RMSE"].map("${:,.2f}".format)
        disp = disp.rename(columns={"R2": "R²", "CV_R2": "CV R²"})

        def highlight(row):
            if row["Model"] == best_name:
                return ["background-color:#dcfce7;font-weight:bold"] * len(row)
            return [""] * len(row)

        st.dataframe(disp.style.apply(highlight, axis=1),
                     use_container_width=True, hide_index=True)

        st.markdown("### Metric Definitions")
        st.markdown("""
| Metric | Meaning | Goal |
|--------|---------|------|
| **MAE** | Mean Absolute Error | Lower = better |
| **RMSE** | Root Mean Squared Error (penalises large errors) | Lower = better |
| **R²** | Proportion of variance explained | Closer to 1 |
| **CV R²** | 5-fold cross-validated R² | Closer to 1 |
        """)

    with c2:
        st.plotly_chart(fig_r2_comparison(metrics_df, best_name), use_container_width=True)

    st.plotly_chart(fig_error_comparison(metrics_df), use_container_width=True)

    if is_rev:
        st.success(
            "**Revenue is highly predictable (R² = 0.984).** Ridge Regression achieves the best "
            "balance with the lowest MAE and strong cross-validation stability. Course Price is "
            "the dominant feature — pricing strategy is EduPro's primary revenue lever."
        )
    else:
        st.warning(
            "**Enrollment prediction is challenging (best R² ≈ 0.053).** This is expected — "
            "enrollment depends on external factors (marketing, timing, word-of-mouth) not captured "
            "in course metadata. Behavioural features would significantly improve this model."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📌 Feature Importance":
    st.markdown("## Feature Importance Analysis")

    c1, c2 = st.columns(2)
    c1.plotly_chart(
        fig_importance_bar(
            pipeline.results[TARGET_ENROLLMENT]["importances"],
            "📈 Enrollment — Feature Importances", "Blues",
        ),
        use_container_width=True,
    )
    c2.plotly_chart(
        fig_importance_bar(
            pipeline.results[TARGET_REVENUE]["importances"],
            "💰 Revenue — Feature Importances", "Oranges",
        ),
        use_container_width=True,
    )

    st.plotly_chart(
        fig_importance_radar(
            pipeline.results[TARGET_ENROLLMENT]["importances"],
            pipeline.results[TARGET_REVENUE]["importances"],
        ),
        use_container_width=True,
    )

    st.markdown("### Business Interpretation")
    insights = [
        ("🌟", "Course Rating — #1 Enrollment Driver (33.5%)",
         "High-rated courses attract significantly more students. Investing in content quality "
         "and instructor coaching is the single most impactful enrollment lever.",
         "#3b82f6"),
        ("⏱️", "Course Duration — #2 Enrollment Driver (33.2%)",
         "Longer courses signal depth and are perceived as more valuable. Avoid overly short "
         "courses in competitive categories.",
         "#8b5cf6"),
        ("💵", "Course Price — #1 Revenue Driver (97.7%)",
         "Price is the overwhelming revenue predictor. Strategic premium pricing in AI and "
         "Business categories directly maximises revenue.",
         "#f97316"),
        ("📂", "Category — #3 Enrollment Driver (14.4%)",
         "Category choice affects both audience size and price tolerance. AI and Business "
         "categories command premium pricing and high enrolment.",
         "#10b981"),
    ]
    cols = st.columns(2)
    for i, (icon, title, text, color) in enumerate(insights):
        with cols[i % 2]:
            st.markdown(
                f'<div class="insight-card" style="--c:{color}">'
                f'<div class="insight-title">{icon} {title}</div>'
                f'<div class="insight-text">{text}</div></div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PREDICTION TOOL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediction Tool":
    st.markdown("## Course Revenue & Enrollment Predictor")
    st.markdown(
        "Enter course parameters. Revenue uses **Ridge Regression (R² = 0.984)**; "
        "Enrollment uses **Random Forest**."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        price    = st.number_input("💵 Course Price ($)", 0.0, 600.0, 299.0, 10.0)
        duration = st.number_input("⏱️ Duration (hours)",  1.0, 100.0,  25.0,  1.0)
        rating   = st.slider("⭐ Expected Rating", 1.0, 5.0, 4.0, 0.1)
    with c2:
        level       = st.selectbox("📚 Level",    ["Beginner", "Intermediate", "Advanced"])
        course_type = st.selectbox("🏷️ Type",     ["Paid", "Free"])
        category    = st.selectbox("📂 Category", sorted(merged["CourseCategory"].unique()))
    with c3:
        exp             = st.number_input("👨‍🏫 Teacher Exp (yrs)", 1, 30, 8)
        t_rating        = st.slider("🌟 Teacher Rating", 1.0, 5.0, 4.0, 0.1)
        expertise_match = st.checkbox("✅ Instructor expertise matches category", value=True)

    if st.button("▶  Run Prediction", type="primary", use_container_width=True):
        # Encode inputs the same way training data was encoded
        def safe_encode(le, val):
            try:
                return le.transform([val])[0]
            except ValueError:
                return 0

        le_cat   = LabelEncoder().fit(merged["CourseCategory"])
        le_type  = LabelEncoder().fit(merged["CourseType"])
        le_level = LabelEncoder().fit(merged["CourseLevel"])

        X_input = np.array([[
            price, duration, rating, exp, t_rating,
            int(expertise_match),
            safe_encode(le_cat,   category),
            safe_encode(le_type,  course_type),
            safe_encode(le_level, level),
        ]])

        pred_rev = pipeline.predict_revenue(X_input)
        pred_enr = pipeline.predict_enrollment(X_input)
        pred_rpe = pred_rev / pred_enr if pred_enr > 0 else 0.0

        cols = st.columns(3)
        for col, val, lbl in [
            (cols[0], f"{pred_enr:,}",      "Predicted Enrollments"),
            (cols[1], f"${pred_rev:,.0f}",  "Predicted Revenue"),
            (cols[2], f"${pred_rpe:,.2f}",  "Revenue per Enrollment"),
        ]:
            col.markdown(
                f'<div class="pred-result">'
                f'<div class="pred-value">{val}</div>'
                f'<div class="pred-label">{lbl}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        avg_rev = merged[merged["CourseType"] == course_type]["CourseRevenue"].mean()
        avg_enr = merged[merged["CourseType"] == course_type]["EnrollmentCount"].mean()
        c1, c2 = st.columns(2)
        c1.metric("vs Avg Revenue (same type)", f"${avg_rev:,.0f}", f"${pred_rev - avg_rev:+,.0f}")
        c2.metric("vs Avg Enrollment (same type)", f"{avg_enr:.0f}", f"{pred_enr - avg_enr:+.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Insights & Recommendations":
    st.markdown("## Strategic Insights & Recommendations")

    st.markdown("### 🔑 Key Findings")
    findings = [
        ("💰", "Revenue is Highly Predictable",
         "Ridge Regression achieves R² = 0.984. Over 98% of revenue variance explained. "
         "Revenue forecasting is reliable and production-ready.", "#10b981"),
        ("📊", "Enrollment is Stochastic",
         "Best enrollment R² ≈ 0.053. External factors (marketing, timing, social proof) "
         "not captured in metadata dominate enrollment outcomes.", "#f59e0b"),
        ("🏆", "AI & Business Lead Revenue",
         "Artificial Intelligence ($202K) and Business ($181K) are the top revenue categories — "
         "highest ROI per course launch.", "#3b82f6"),
        ("💵", "Price is #1 Revenue Driver",
         "Course Price accounts for 97.7% of feature importance for revenue. "
         "Pricing strategy is EduPro's primary lever.", "#f97316"),
        ("⭐", "Rating Drives Enrollments",
         "Course Rating (33.5%) and Duration (33.2%) dominate enrollment importance. "
         "Quality and substance attract students regardless of price.", "#8b5cf6"),
        ("🆓", "Free Courses Underutilised",
         "Marketing, ML, and Web Dev courses are free, generating $0 revenue despite strong "
         "enrollments. A freemium upgrade path would unlock significant value.", "#e11d48"),
    ]
    cols = st.columns(2)
    for i, (icon, title, text, color) in enumerate(findings):
        with cols[i % 2]:
            st.markdown(
                f'<div class="insight-card" style="--c:{color};margin-bottom:12px">'
                f'<div class="insight-title" style="font-size:1rem">{icon} {title}</div>'
                f'<div class="insight-text" style="font-size:.85rem;margin-top:4px">{text}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### 📋 Recommendations")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Immediate Actions")
        for item in [
            "Deploy Ridge Regression for all revenue forecasts",
            "Launch new AI / Business / Project Management courses",
            "Set premium price targets: $300–$490",
            "Invest in course quality — rating is the #1 enrollment driver",
            "Match instructors to their expertise category",
        ]:
            st.markdown(f"- {item}")
    with c2:
        st.markdown("#### Strategic Opportunities")
        for item in [
            "Monetise high-enrollment free categories (ML, Marketing, Web Dev)",
            "Build a freemium → premium upgrade funnel",
            "Add behavioural features to improve enrollment R² beyond 0.05",
            "Use category revenue forecasting for annual content roadmap",
            "A/B test pricing tiers: $150, $299, $490 to map price elasticity",
        ]:
            st.markdown(f"- {item}")

    st.markdown("---")
    st.markdown("### 📊 Revenue Opportunity — Freemium Gap Analysis")
    free_potential = merged[merged["CoursePrice"] == 0]["EnrollmentCount"].sum() * 50
    st.metric("Potential Revenue (30 % freemium conversion at $50)", f"${free_potential:,.0f}")
    st.plotly_chart(fig_revenue_gap(merged), use_container_width=True)
