
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction")
st.write(
    "Upload a student dataset and predict examination "
    "performance using Machine Learning."
)

st.sidebar.header("Project Information")
st.sidebar.write("Algorithm: Linear Regression")
st.sidebar.write("Application: Student Performance Prediction")

uploaded_file = st.file_uploader(
    "Upload your student CSV dataset",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.subheader("1. Dataset Preview")
    st.dataframe(data.head(10), use_container_width=True)

    st.write("Dataset shape:", data.shape)

    st.subheader("2. Select Target Column")

    numeric_columns = data.select_dtypes(
        include=["number"]
    ).columns.tolist()

    if not numeric_columns:
        st.error("Your dataset needs a numeric target column.")

    else:

        default_target = (
            "exam_score"
            if "exam_score" in numeric_columns
            else numeric_columns[-1]
        )

        target = st.selectbox(
            "Choose the score column to predict",
            numeric_columns,
            index=numeric_columns.index(default_target)
        )

        feature_options = [
            col for col in data.columns if col != target
        ]

        features = st.multiselect(
            "Choose input features",
            feature_options,
            default=feature_options
        )

        st.subheader("3. Train Machine Learning Model")

        if st.button("Train Model and Predict"):

            if not features:
                st.error("Please select at least one feature.")

            else:

                clean_data = data[
                    features + [target]
                ].copy()

                clean_data[target] = pd.to_numeric(
                    clean_data[target],
                    errors="coerce"
                )

                clean_data = clean_data.dropna(
                    subset=[target]
                )

                X = clean_data[features]
                y = clean_data[target]

                if len(clean_data) < 10:
                    st.error(
                        "Please upload at least 10 valid records."
                    )

                else:

                    numeric_features = X.select_dtypes(
                        include=["number"]
                    ).columns.tolist()

                    categorical_features = X.select_dtypes(
                        exclude=["number"]
                    ).columns.tolist()

                    numeric_pipeline = Pipeline([
                        ("imputer", SimpleImputer(
                            strategy="median"
                        ))
                    ])

                    categorical_pipeline = Pipeline([
                        ("imputer", SimpleImputer(
                            strategy="most_frequent"
                        )),
                        ("encoder", OneHotEncoder(
                            handle_unknown="ignore"
                        ))
                    ])

                    preprocessing = ColumnTransformer([
                        ("numeric", numeric_pipeline,
                         numeric_features),
                        ("categorical", categorical_pipeline,
                         categorical_features)
                    ])

                    model = Pipeline([
                        ("preprocessing", preprocessing),
                        ("regression", LinearRegression())
                    ])

                    X_train, X_test, y_train, y_test = (
                        train_test_split(
                            X,
                            y,
                            test_size=0.2,
                            random_state=42
                        )
                    )

                    model.fit(X_train, y_train)

                    predictions = model.predict(X_test)

                    mae = mean_absolute_error(
                        y_test, predictions
                    )

                    mse = mean_squared_error(
                        y_test, predictions
                    )

                    r2 = r2_score(
                        y_test, predictions
                    )

                    st.success("Model training completed!")

                    st.subheader("4. Model Evaluation")

                    col1, col2, col3 = st.columns(3)

                    col1.metric("MAE", f"{mae:.2f}")
                    col2.metric("MSE", f"{mse:.2f}")
                    col3.metric("R² Score", f"{r2:.3f}")

                    st.subheader("5. Actual vs Predicted Scores")

                    results = pd.DataFrame({
                        "Actual Score": y_test.values,
                        "Predicted Score": predictions
                    })

                    st.dataframe(
                        results,
                        use_container_width=True
                    )

                    fig, ax = plt.subplots()

                    ax.scatter(y_test, predictions)

                    ax.set_xlabel("Actual Scores")
                    ax.set_ylabel("Predicted Scores")

                    ax.set_title(
                        "Student Performance Prediction"
                    )

                    st.pyplot(fig)

                    st.subheader("6. Predict New Student Score")

                    new_student = {}

                    for feature in features:

                        if feature in numeric_features:

                            new_student[feature] = st.number_input(
                                f"Enter {feature}",
                                value=float(
                                    pd.to_numeric(
                                        X[feature],
                                        errors="coerce"
                                    ).median()
                                )
                            )

                        else:

                            choices = (
                                X[feature]
                                .dropna()
                                .astype(str)
                                .unique()
                                .tolist()
                            )

                            new_student[feature] = st.selectbox(
                                f"Select {feature}",
                                choices,
                                key=f"new_{feature}"
                            )

                    if st.button("Predict Student Score"):

                        new_data = pd.DataFrame(
                            [new_student]
                        )

                        prediction = model.predict(new_data)

                        st.success(
                            f"Predicted {target}: "
                            f"{prediction[0]:.2f}"
                        )

else:

    st.info("Please upload a CSV file to get started.")
