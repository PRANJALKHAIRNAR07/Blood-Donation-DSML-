
import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Blood Donation Prediction",
    page_icon="🩸",
    layout="wide"
)

st.title("🩸 Blood Donation Prediction System")
st.write(
    "Predict a recorded blood donation outcome using "
    "historical donor information and machine learning."
)

st.info(
    "This model uses historical data from March 2007. "
    "It is an educational prediction tool, not a guarantee "
    "of future donation behaviour."
)

# Load saved model comparison
comparison_path = "models/model_comparison.csv"

if os.path.exists(comparison_path):
    results = pd.read_csv(comparison_path)

    st.subheader("Model Performance Comparison")
    st.dataframe(
        results.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1-score": "{:.2%}"
        }),
        use_container_width=True
    )

    st.bar_chart(
        results.set_index("Algorithm")[
            ["Accuracy", "Precision", "Recall", "F1-score"]
        ]
    )

st.divider()

st.subheader("Enter Donor Information")

col1, col2 = st.columns(2)

with col1:
    recency = st.number_input(
        "Months since last donation",
        min_value=0,
        max_value=100,
        value=2
    )

    frequency = st.number_input(
        "Total previous donations",
        min_value=1,
        max_value=100,
        value=10
    )

with col2:
    monetary = st.number_input(
        "Total blood donated (cc)",
        min_value=250,
        max_value=25000,
        value=2500,
        step=250
    )

    time = st.number_input(
        "Months since first donation",
        min_value=0,
        max_value=100,
        value=30
    )

model_names = {
    "Logistic Regression": "models/logistic_regression.joblib",
    "Decision Tree": "models/decision_tree.joblib",
    "Random Forest": "models/random_forest.joblib"
}

selected_model = st.selectbox(
    "Choose an ML algorithm",
    list(model_names.keys())
)

if st.button("Predict Donation Outcome", type="primary"):
    model_path = model_names[selected_model]

    if not os.path.exists(model_path):
        st.error("Model file not found. Run train_model.py first.")
    else:
        model = joblib.load(model_path)

        input_data = pd.DataFrame([{
            "Recency": recency,
            "Frequency": frequency,
            "Monetary": monetary,
            "Time": time
        }])

        prediction = int(model.predict(input_data)[0])

        st.subheader("Prediction Result")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)
            positive_index = classes.index(1)
            probability = probabilities[positive_index]

            st.metric(
                "Estimated probability of recorded donation",
                f"{probability:.1%}"
            )

        if prediction == 1:
            st.success(
                "Predicted class: 1 — Recorded donation"
            )
        else:
            st.warning(
                "Predicted class: 0 — No recorded donation"
            )

        st.caption(
            "This prediction describes patterns in historical "
            "data and should not be interpreted as a reliable "
            "forecast of a person's current or future behaviour."
        )
