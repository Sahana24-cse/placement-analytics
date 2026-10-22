"""AI-Based Placement Readiness Analytics  -  run with:  streamlit run app.py"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from generate_data import generate

st.set_page_config(page_title="Placement Readiness Analytics", page_icon="🎓", layout="wide")

DATA_FILE = "placement_data.csv"
CAT = ["branch", "college_tier"]
FEATURES = ["branch", "college_tier", "cgpa", "backlogs", "coding_skills", "dsa_score",
            "aptitude_score", "communication_skills", "ml_knowledge", "system_design",
            "internships", "projects_count", "certifications", "hackathons",
            "open_source_contributions", "extracurriculars"]

# ---------------------------------------------------------------- data / models
@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        generate().to_csv(DATA_FILE, index=False)
    return pd.read_csv(DATA_FILE)

@st.cache_resource
def train_models():
    df = load_data()
    X, y = df[FEATURES], df["placed"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT)],
                            remainder=StandardScaler())
    algos = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10,
                                                n_jobs=-1, random_state=42),
    }
    fitted, rows = {}, {}
    for name, clf in algos.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)]).fit(Xtr, ytr)
        pred, proba = pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1]
        rows[name] = dict(accuracy=accuracy_score(yte, pred),
                          precision=precision_score(yte, pred),
                          recall=recall_score(yte, pred), f1=f1_score(yte, pred),
                          roc_auc=roc_auc_score(yte, proba))
        fitted[name] = pipe
    metrics = pd.DataFrame(rows).T
    best = metrics["roc_auc"].idxmax()
    return fitted, metrics, best

def readiness(p):
    academic = min(p["cgpa"] * 10, 100)
    technical = np.mean([p["coding_skills"], p["dsa_score"], p["ml_knowledge"],
                         p["system_design"]]) * 10
    aptitude = p["aptitude_score"]
    comm = p["communication_skills"] * 10
    exp = min((p["internships"] * 40 + p["projects_count"] * 15 + p["certifications"] * 10
               + p["hackathons"] * 10 + p["open_source_contributions"] * 10), 100)
    parts = {"Academic (30%)": academic, "Technical (30%)": technical,
             "Aptitude (20%)": aptitude, "Communication (10%)": comm, "Experience (10%)": exp}
    total = (0.3 * academic + 0.3 * technical + 0.2 * aptitude + 0.1 * comm + 0.1 * exp)
    return round(total), {k: round(v) for k, v in parts.items()}

# ---------------------------------------------------------------- sidebar
st.sidebar.title("🎓 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Student Assessment", "Placement Prediction",
                                  "Readiness Score", "Skill Gap & Recommendations",
                                  "Model Performance", "Analytics Dashboard"])
df = load_data()
models, metrics, best_name = train_models()

def need_profile():
    if "profile" not in st.session_state:
        st.warning("Please fill the **Student Assessment** form first.")
        st.stop()
    return st.session_state["profile"]

# ---------------------------------------------------------------- pages
if page == "Home":
    st.title("AI-Based Placement Readiness Analytics 🚀")
    st.subheader("Welcome to the Placement Analytics Hub")
    st.write("This platform leverages Machine Learning to predict student placement "
             "readiness, identify skill gaps, and provide actionable recommendations.")
    st.markdown("""**Key Features:**
- 📊 **Analytics Dashboard**: View overall trends and insights from historical data.
- 📝 **Student Assessment**: Input your profile to get analyzed.
- 🎯 **Prediction**: Real-time placement prediction using an optimized AI model.
- 📈 **Readiness Score**: A 100-point metric across 5 distinct domains.
- 💡 **Skill Gap & Recommendations**: Personalized actionable advice.""")

elif page == "Student Assessment":
    st.title("Student Assessment Form 📝")
    st.write("Fill out your academic and technical profile.")
    c1, c2, c3 = st.columns(3)
    with c1:
        branch = st.selectbox("Branch", ["CSE", "IT", "ECE", "ME", "EE", "CE", "Chemical"])
        tier = st.selectbox("College Tier", ["Tier-1", "Tier-2", "Tier-3"])
        cgpa = st.number_input("CGPA", 5.0, 10.0, 7.6, 0.1)
        backlogs = st.number_input("Backlogs", 0, 10, 0)
        aptitude = st.number_input("Aptitude Score (0-100)", 0.0, 100.0, 60.0, 1.0)
    with c2:
        coding = st.slider("Coding Skills (0-10)", 0.0, 10.0, 5.8, 0.1)
        dsa = st.slider("DSA Score (0-10)", 0.0, 10.0, 6.6, 0.1)
        comm = st.slider("Communication Skills (0-10)", 0.0, 10.0, 7.0, 0.1)
        ml = st.slider("ML Knowledge (0-10)", 0.0, 10.0, 5.0, 0.1)
        sd = st.slider("System Design (0-10)", 0.0, 10.0, 4.3, 0.1)
    with c3:
        internships = st.number_input("Internships", 0, 10, 2)
        projects = st.number_input("Projects Count", 0, 20, 3)
        certs = st.number_input("Certifications", 0, 20, 2)
        hack = st.number_input("Hackathons Attended", 0, 20, 1)
        oss = st.number_input("Open Source Contributions", 0, 20, 1)
        extra = st.number_input("Extracurriculars", 0, 20, 2)
    if st.button("Save Profile & Analyze"):
        st.session_state["profile"] = dict(
            branch=branch, college_tier=tier, cgpa=cgpa, backlogs=backlogs,
            coding_skills=coding, dsa_score=dsa, aptitude_score=aptitude,
            communication_skills=comm, ml_knowledge=ml, system_design=sd,
            internships=internships, projects_count=projects, certifications=certs,
            hackathons=hack, open_source_contributions=oss, extracurriculars=extra)
        st.success("Profile saved! Proceed to 'Placement Prediction' to see your chances.")

elif page == "Placement Prediction":
    st.title("Placement Prediction 🎯")
    prof = need_profile()
    st.subheader("Review Your Profile:")
    st.dataframe(pd.DataFrame([prof]))
    if st.button("Predict Placement Status"):
        prob = models[best_name].predict_proba(pd.DataFrame([prof])[FEATURES])[0, 1]
        if prob >= 0.7:
            st.success(f"🚀 Highly likely to be placed! Probability: {prob:.1%}")
        elif prob >= 0.5:
            st.warning(f"🙂 Moderate chance of placement. Probability: {prob:.1%}")
        else:
            st.error(f"⚠️ Low chance of placement. Probability: {prob:.1%}")

elif page == "Readiness Score":
    st.title("Placement Readiness Score 📈")
    st.caption("Based on an aggregated weighted average of academic, technical, aptitude, "
               "communication and experience domains.")
    total, parts = readiness(need_profile())
    color = "#2ecc71" if total >= 75 else "#f5a623" if total >= 50 else "#e74c3c"
    st.markdown(f"<h1 style='text-align:center;color:{color};font-size:56px'>{total}/100</h1>"
                "<h4 style='text-align:center'>Overall Readiness Score</h4>",
                unsafe_allow_html=True)
    for col, (k, v) in zip(st.columns(5), parts.items()):
        col.markdown(f"<div style='background:#111;border-radius:8px;padding:14px;"
                     f"text-align:center'><span style='color:#2ecc71;font-size:28px;"
                     f"font-weight:700'>{v}</span><br><span style='color:#ccc;"
                     f"font-size:12px'>{k}</span></div>", unsafe_allow_html=True)

elif page == "Skill Gap & Recommendations":
    st.title("Skill Gap Analysis & Recommendations 💡")
    p = need_profile()
    gaps, recs = [], []
    if p["dsa_score"] < 7:
        gaps.append(f"DSA Score ({p['dsa_score']}/10) is insufficient.")
        recs.append("**Technical:** Solve 2-3 LeetCode/HackerRank problems daily. "
                    "Participate in competitive programming.")
    if p["coding_skills"] < 7:
        gaps.append(f"Coding Skills ({p['coding_skills']}/10) needs improvement.")
    if p["aptitude_score"] < 70:
        gaps.append(f"Aptitude ({p['aptitude_score']}/100) is low.")
        recs.append("**Aptitude:** Practice quantitative and logical reasoning on "
                    "platforms like IndiaBix.")
    if p["communication_skills"] < 6:
        gaps.append(f"Communication ({p['communication_skills']}/10) is low.")
        recs.append("**Communication:** Do mock interviews and group discussions.")
    if p["system_design"] < 5:
        gaps.append(f"System Design ({p['system_design']}/10) is weak.")
        recs.append("**System Design:** Learn basics of scalable systems and LLD/HLD.")
    if p["internships"] == 0:
        gaps.append("No internship experience.")
        recs.append("**Experience:** Apply for internships and build 2-3 solid projects.")
    if p["backlogs"] > 0:
        gaps.append(f"{p['backlogs']} active backlog(s).")
        recs.append("**Academic:** Clear all backlogs before placement season.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Skill Gap Analysis")
        if gaps:
            for g in gaps: st.error(f"⚠️ {g}")
        else:
            st.success("No major skill gaps found. Great profile!")
    with c2:
        st.subheader("Personalized Recommendations")
        for r in recs: st.markdown(f"- {r}")
        if not recs: st.write("Keep up the good work and maintain consistency.")

elif page == "Model Performance":
    st.title("Model Performance Analysis 🤖")
    st.subheader(f"Best Performing Model: {best_name}")
    st.dataframe(metrics.style.background_gradient(cmap="viridis", axis=None).format("{:.6f}"))
    st.caption("Model Comparison metrics")
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics[["accuracy", "f1", "roc_auc"]].plot(kind="bar", ax=ax, colormap="tab20")
    ax.set_title("Algorithm Comparison"); ax.set_ylabel("Score"); ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
    st.pyplot(fig)

elif page == "Analytics Dashboard":
    st.title("Dataset Analytics Dashboard 📊")
    placed = df[df.placed == 1]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Students", len(df))
    k2.metric("Placement Rate", f"{df.placed.mean() * 100:.1f}%")
    k3.metric("Avg CGPA", f"{df.cgpa.mean():.2f}")
    k4.metric("Avg Salary (Placed)", f"{placed.salary_package_lpa.mean():.2f} LPA")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Placement via Branch**")
        b = (df.groupby("branch").placed.mean() * 100).sort_values()
        fig, ax = plt.subplots(); b.plot(kind="barh", color="skyblue", ax=ax)
        ax.set_xlabel("Placement %"); st.pyplot(fig)
    with c2:
        st.markdown("**CGPA vs Placement Probability**")
        bins = pd.cut(df.cgpa, bins=np.arange(5, 10.5, 0.5))
        c = df.groupby(bins, observed=True).placed.mean() * 100
        fig, ax = plt.subplots()
        ax.plot(range(len(c)), c.values, marker="o", color="orange")
        ax.set_xticks(range(len(c))); ax.set_xticklabels([str(i) for i in c.index], rotation=45)
        ax.set_ylabel("Placement %"); ax.set_xlabel("cgpa"); st.pyplot(fig)
    st.markdown("**Salary Distribution by College Tier (Only Placed Students)**")
    fig, ax = plt.subplots(figsize=(10, 5))
    placed.boxplot(column="salary_package_lpa", by="college_tier", ax=ax)
    ax.set_title("Salary Package (LPA) by College Tier"); plt.suptitle("")
    st.pyplot(fig)
