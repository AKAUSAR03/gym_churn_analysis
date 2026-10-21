import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

st.set_page_config(
    page_title="Gym Churn Dashboard",
    page_icon="🏋️",
    layout="wide"
)
# Professional dashboard theme
st.markdown("""
<style>
    /* Main application background */
    .stApp {
        background-color: #F4F7FB;
        color: #172033;
    }

    /* Main content spacing */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    /* Main headings */
    h1, h2, h3 {
        color: #172B4D;
        font-weight: 700;
    }

    /* KPI metric cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E1E8F0;
        padding: 18px 20px;
        border-radius: 14px;
        box-shadow: 0 3px 12px rgba(30, 50, 80, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #000000;
        font-size: 14px;
    }

    div[data-testid="stMetricValue"] {
        color: #000000;
        font-weight: 700;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #14243A;
    }

    section[data-testid="stSidebar"] * {
        color: #F3F6FC;
    }

    /* Buttons */
    .stButton > button,
    .stFormSubmitButton > button {
        background-color: #087F8C;
        color: white;
        border: none;
        border-radius: 9px;
        padding: 0.55rem 1rem;
        font-weight: 600;
        width: 100%;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background-color: #066570;
        color: white;
        border: none;
    }

    /* Containers */
    div[data-testid="stExpander"] {
        background-color: #FFFFFF;
        border: 1px solid #E1E8F0;
        border-radius: 10px;
    }

    /* Dividers */
    hr {
        border-color: #DFE7F0;
    }
    /* Make form labels clearly visible */
.stApp label,
.stApp label p,
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stWidgetLabel"] label {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

/* Keep the main section descriptions readable */
.stApp .stMarkdown p {
    color: #333333;
}
/* Improve text visibility inside input boxes */
.stApp input,
.stApp textarea,
.stApp [data-baseweb="select"] * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Keep the input boxes dark */
.stApp input,
.stApp textarea,
.stApp [data-baseweb="select"] > div {
    background-color: #22212D !important;
}
</style>
""", unsafe_allow_html=True)

DATA_FILE = Path("gym_members_dataset.csv")
MODEL_FILE = Path("churn_pipeline.pkl")

# Dashboard header
st.markdown("""
<div style="
    background: linear-gradient(120deg, #14243A, #087F8C);
    padding: 28px;
    border-radius: 16px;
    margin-bottom: 24px;
">
    <div style="
        color: #9DE5E6;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 2px;
    ">
        GYM INTELLIGENCE
    </div>
    <h1 style="
        color: white;
        margin: 10px 0 8px 0;
        font-size: 32px;
    ">
        Churn Prediction Dashboard
    </h1>
    <p style="
        color: #E3F1F7;
        font-size: 16px;
        margin-bottom: 0;
    ">
        Member analytics • Retention insights • ML-powered predictions
    </p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE)
    data.columns = data.columns.str.strip()
    data["Churn"] = (
        data["Churn"].astype("string").str.strip().str.title()
    )
    return data

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

# Check required files
if not DATA_FILE.exists():
    st.error("Dataset not found. Keep gym_members_dataset.csv beside app.py.")
    st.stop()

if not MODEL_FILE.exists():
    st.error("Model not found. Run python train_model.py first.")
    st.stop()

df = load_data()
saved = load_model()
model = saved["pipeline"]
feature_columns = saved["feature_columns"]

# Dashboard overview
st.header("📊 Membership Overview")
st.caption("A quick summary of member retention and churn")

total_members = len(df)
churned = int((df["Churn"] == "Yes").sum())
retained = int((df["Churn"] == "No").sum())
churn_rate = churned / total_members * 100 if total_members else 0
retention_rate = retained / total_members * 100 if total_members else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Members", total_members)
c2.metric("Retained Members", retained)
c3.metric("Churned Members", churned)
c4.metric("Churn Rate", f"{churn_rate:.1f}%")

st.progress(min(max(retention_rate / 100, 0.0), 1.0))
st.caption(f"Member retention rate: {retention_rate:.1f}%")

st.divider()

# Member input panel
st.header("🔮 Member Churn Assessment")
st.caption(
    "Enter a member's information to estimate their churn risk "
    "and view a suggested retention action."
)

with st.form("member_form"):
    input_values = {}

    cols = st.columns(2)

    for i, feature in enumerate(feature_columns):
        with cols[i % 2]:
            if feature not in df.columns:
                continue

            series = df[feature]

            if pd.api.types.is_numeric_dtype(series):
                median = pd.to_numeric(
                    series, errors="coerce"
                ).median()

                if pd.isna(median):
                    median = 0.0

                input_values[feature] = st.number_input(
                    feature,
                    value=float(median)
                )
            else:
                options = (
                    series.dropna().astype(str).unique().tolist()
                )
                options = sorted(options)

                if options:
                    input_values[feature] = st.selectbox(
                        feature, options
                    )
                else:
                    input_values[feature] = st.text_input(feature)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    input_df = pd.DataFrame(
        [[input_values[c] for c in feature_columns]],
        columns=feature_columns
    )

    try:
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)

        # Convert model labels into a consistent format.
        def is_churn_label(label):
            value = str(label).strip().lower()

            if value in {"yes", "1", "true", "churn", "churned"}:
                return True

            if value in {"no", "0", "false", "retained", "not churned"}:
                return False

            return None

        # Find the probability belonging to each class.
        churn_probability = 0.0
        retention_probability = 0.0

        for label, probability in zip(classes, probabilities):
            churn_label = is_churn_label(label)

            if churn_label is True:
                churn_probability = float(probability) * 100
            elif churn_label is False:
                retention_probability = float(probability) * 100

        prediction_is_churn = is_churn_label(prediction)

        st.subheader("Prediction Result")

        # Temporary diagnostic information to verify the model.
        with st.expander("View prediction diagnostics"):
            st.write("Model classes:", classes)
            st.write("Raw prediction:", repr(prediction))
            st.write("Churn probability:", f"{churn_probability:.1f}%")
            st.write("Retention probability:", f"{retention_probability:.1f}%")

        if prediction_is_churn is True:
            st.error("⚠️ The model predicts that this member may churn.")
            st.write(
                "Suggested action: contact the member, collect feedback, "
                "and consider a suitable retention offer."
            )

        elif prediction_is_churn is False:
            st.success(
                "✅ The model predicts that this member may be retained."
            )
            st.write(
                "Suggested action: maintain engagement and monitor visits."
            )

        else:
            st.warning(
                "The model returned an unrecognized class label. "
                "Check the diagnostics above."
            )

        p1, p2 = st.columns(2)
        p1.metric(
            "Estimated Churn Probability",
            f"{churn_probability:.1f}%"
        )
        p2.metric(
            "Estimated Retention Probability",
            f"{retention_probability:.1f}%"
        )

        st.caption(
            "These are model estimates, not guarantees of future behaviour."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
st.divider()

# Visual analytics
st.header("📈 RETENTION ANALYTICS")

left, right = st.columns(2)

with left:
    st.subheader("Member Churn Distribution")
    churn_counts = df["Churn"].value_counts().reindex(
        ["No", "Yes"], fill_value=0
    )

    fig, ax = plt.subplots()
    ax.pie(
        churn_counts.values,
        labels=["Retained", "Churned"],
        autopct="%1.1f%%",
        startangle=90
    )
    ax.axis("equal")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Average Visits Per Month")

    if "Visits_Per_Month" in df.columns:
        visit_data = df.groupby("Churn")[
            "Visits_Per_Month"
        ].mean().reindex(["No", "Yes"])

        fig, ax = plt.subplots()
        visit_data.plot(kind="bar", ax=ax)
        ax.set_xlabel("Churn (No = Retained, Yes = Churned)")
        ax.set_ylabel("Average Visits Per Month")
        ax.set_title("Member Visit Comparison")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Visits_Per_Month is not available in this dataset.")

# ==========================================
# STAGE 2: ADVANCED ANALYTICS
# ==========================================

st.divider()
st.header("📊 Advanced Membership Analytics")
st.caption(
    "Explore membership patterns, attendance, "
    "and factors used by the churn prediction model."
)

# 1. Membership type analysis
if "Membership_Type" in df.columns:

    st.subheader("🏷️ Churn Rate by Membership Type")

    membership_data = df[
        df["Churn"].isin(["Yes", "No"])
    ].copy()

    membership_summary = (
        membership_data.groupby("Membership_Type")
        .agg(
            Total_Members=("Churn", "size"),
            Churned_Members=(
                "Churn",
                lambda x: (x == "Yes").sum()
            )
        )
        .reset_index()
    )

    membership_summary["Churn_Rate_%"] = (
        membership_summary["Churned_Members"]
        / membership_summary["Total_Members"] * 100
    ).round(2)

    if not membership_summary.empty:
        left, right = st.columns(2)

        with left:
            st.write("**Members by membership type**")
            st.bar_chart(
                membership_summary.set_index(
                    "Membership_Type"
                )["Total_Members"]
            )

        with right:
            st.write("**Churn rate by membership type (%)**")
            st.bar_chart(
                membership_summary.set_index(
                    "Membership_Type"
                )["Churn_Rate_%"]
            )

        st.dataframe(
            membership_summary,
            use_container_width=True,
            hide_index=True
        )
else:
    st.info("Membership_Type is not available in this dataset.")


# 2. Attendance analysis

   # 2. Attendance analysis
attendance = pd.Series(dtype=float)  # defined up front so later sections never fail

if "Visits_Per_Month" in df.columns:

    st.subheader("🏃 Attendance and Member Retention")

    attendance = (
        df[df["Churn"].isin(["Yes", "No"])]
        .groupby("Churn")["Visits_Per_Month"]
        .mean()
        .reindex(["No", "Yes"])
    )

    attendance.index = [
        "Retained" if label == "No" else "Churned"
        for label in attendance.index
    ]

    attendance = attendance.dropna()

    if not attendance.empty:
        fig, ax = plt.subplots(figsize=(8, 4))

        attendance.plot(kind="bar", ax=ax)

        ax.set_title("Average Monthly Visits: Retained vs Churned")
        ax.set_xlabel("Member Status")
        ax.set_ylabel("Average Visits Per Month")
        ax.set_xticklabels(attendance.index, rotation=0)
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        ax.tick_params(axis="both", colors="black")
        ax.title.set_color("black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.grid(axis="y", alpha=0.25)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Compare average monthly visits for retained "
            "and churned members. This is an observed pattern, "
            "not proof that attendance causes churn."
        )


# 3. Random Forest feature importance
st.subheader("🎯 Random Forest Feature Importance")

try:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    encoded_names = preprocessor.get_feature_names_out()
    importance_values = classifier.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": encoded_names,
        "Importance": importance_values
    })

    # Aggregate encoded categories back to original features
    def original_feature(encoded_name):
        name = encoded_name.split("__", 1)[-1]

        for col in sorted(feature_columns, key=len, reverse=True):
            if name == col or name.startswith(col + "_"):
                return col

        return name

    importance_df["Original_Feature"] = (
        importance_df["Feature"].apply(original_feature)
    )

    grouped_importance = (
        importance_df.groupby("Original_Feature")["Importance"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    # The matplotlib chart now lives HERE, after grouped_importance exists
    fig, ax = plt.subplots(figsize=(9, 5))

    grouped_importance.sort_values().plot(kind="barh", ax=ax)

    ax.set_title("Top Features Influencing Churn Predictions")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.tick_params(axis="both", colors="black")
    ax.title.set_color("black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Higher importance means the model relied more on "
        "that feature. It does not establish cause and effect."
    )

except Exception as e:
    st.warning(f"Could not display feature importance: {e}")


# 4. Automatic business insights
st.subheader("💡 Key Business Insights")

if total_members > 0:

    st.write(
        f"**Member base:** The dataset contains "
        f"{total_members} members."
    )

    st.write(
        f"**Retention:** {retention_rate:.1f}% of members "
        f"are labelled as retained in the dataset."
    )

    st.write(
        f"**Churn:** {churn_rate:.1f}% of members "
        f"are labelled as churned in the dataset."
    )

    if (
        "Membership_Type" in df.columns
        and "membership_summary" in locals()
        and not membership_summary.empty
    ):
        highest = membership_summary.loc[
            membership_summary["Churn_Rate_%"].idxmax()
        ]

        st.write(
            f"**Highest observed membership-type churn rate:** "
            f"{highest['Membership_Type']} "
            f"({highest['Churn_Rate_%']:.1f}%)."
        )

    # Fixed: the index was renamed to Retained/Churned above
    if "Retained" in attendance.index and "Churned" in attendance.index:
        retained_visits = attendance["Retained"]
        churned_visits = attendance["Churned"]

        st.write(
            f"**Attendance comparison:** Retained members "
            f"averaged {retained_visits:.1f} visits per month; "
            f"churned members averaged {churned_visits:.1f}."
        )

    st.info(
        "Business suggestion: review membership groups with higher "
        "observed churn and consider contacting members whose visits "
        "have decreased. Validate these patterns before making decisions."
    )


# Dataset explorer
#==============================

#STAGE 3: MODEL EVALUATION

#==============================

st.header("🤖 Machine Learning Model Evaluation")
st.caption(
"Evaluation results are calculated on held-out test data "
"that was not used to train the models."
)

evaluation_data = joblib.load("churn_pipeline.pkl")
rf_results = evaluation_data["evaluation"]
comparison_results = evaluation_data["comparison"]

#1. Random Forest performance metrics

st.subheader("Random Forest Performance")

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric("Accuracy", f"{rf_results['accuracy']:.1%}")
metric2.metric("Precision", f"{rf_results['precision']:.1%}")
metric3.metric("Recall", f"{rf_results['recall']:.1%}")
metric4.metric("F1-Score", f"{rf_results['f1_score']:.1%}")

st.caption(
"Precision, recall and F1-score treat churned members as the "
"positive class. Results depend on the dataset and test split."
)

#2. Confusion matrix

st.subheader("Confusion Matrix")

cm = rf_results["confusion_matrix"]

confusion_df = pd.DataFrame(
cm,
index=["Actual Retained", "Actual Churned"],
columns=["Predicted Retained", "Predicted Churned"]
)

st.dataframe(confusion_df, use_container_width=True)

fig, ax = plt.subplots(figsize=(6, 4))
image = ax.imshow(cm, interpolation="nearest")

ax.set_title("Random Forest Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Retained", "Churned"])
ax.set_yticklabels(["Retained", "Churned"])

for row in range(2):
  for col in range(2):
   ax.text(
col,
row,
str(cm[row][col]),
ha="center",
va="center"
)

fig.colorbar(image, ax=ax)
st.pyplot(fig)
plt.close(fig)

st.markdown(
"""
How to read the confusion matrix

- **Actual retained, predicted retained:** correctly identified retained members.
- **Actual retained, predicted churned:** false alarms.
- **Actual churned, predicted retained:** missed churn cases.
- **Actual churned, predicted churned:** correctly identified churn cases.
"""

)

#3. Compare Random Forest and Logistic Regression

st.subheader("Model Comparison")

comparison_rows = []

for model_name, metrics in comparison_results.items():
 comparison_rows.append({
"Model": model_name,
"Accuracy": metrics["accuracy"],
"Precision": metrics["precision"],
"Recall": metrics["recall"],
"F1-Score": metrics["f1_score"]
})

comparison_df = pd.DataFrame(comparison_rows).set_index("Model")

st.dataframe(
comparison_df.style.format("{:.1%}"),
use_container_width=True
)

st.bar_chart(comparison_df)

#4. Explain the results

best_f1_model = comparison_df["F1-Score"].idxmax()
best_recall_model = comparison_df["Recall"].idxmax()

st.subheader("Evaluation Insights")

st.write(
f"Best F1-score: {best_f1_model}. "
"F1-score balances precision and recall."
)

st.write(
f"Best recall: {best_recall_model}. "
"Higher recall means the model identifies a larger share "
"of actual churn cases."
)

st.info(
"For a gym retention system, recall is important because missed "
"churn cases may mean losing opportunities to retain members. "
"Precision also matters because unnecessary retention offers can cost money."
)
st.divider()
st.header("📋 MEMBER DATA EXPLORER")

with st.expander("View member records"):
    st.dataframe(df, use_container_width=True)

with st.expander("View numerical summary"):
    st.dataframe(df.describe(include="all").transpose(),
                 use_container_width=True)

st.caption(
    "Project: Gym Churn Prediction using Random Forest, "
    "Pandas, Scikit-learn and Streamlit."
)