import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

st.set_page_config(page_title="Food Infrastructure Tracker", layout="centered")

st.title(" Food Infrastructure Tracker")
st.write("AI Project with Market Access & Agricultural Yield Analysis")

# Upload file option
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Load dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Preview of Uploaded Data", df.head())

    # ----------------- Data Preparation -----------------
    # Keep only numeric columns for ML
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        st.error(" Need at least 2 numeric columns in dataset.")
    else:
        # Features = all except last numeric column
        X = numeric_df.iloc[:, :-1]
        y = numeric_df.iloc[:, -1]   # last numeric column as target

        # ---------- 1) 3D Clustering ----------
        if X.shape[1] >= 3:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            df["Cluster"] = kmeans.fit_predict(X)

            fig1 = plt.figure(figsize=(6, 4))
            ax = fig1.add_subplot(111, projection="3d")
            ax.scatter(X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, 2],
                       c=df["Cluster"], cmap="viridis")
            ax.set_xlabel(X.columns[0])
            ax.set_ylabel(X.columns[1])
            ax.set_zlabel(X.columns[2])
            ax.set_title("3D Clustering Result")
            st.pyplot(fig1)
        else:
            st.warning("ℹ️ Need at least 3 numeric columns for 3D clustering.")

        # ---------- 2) Decision Tree + Confusion Matrix ----------
        labels = ["Low", "Medium", "High"]
        try:
            y_bins = pd.qcut(y, q=len(labels), labels=labels, duplicates="drop")
        except Exception:
            y_bins = pd.cut(y, bins=len(labels), labels=labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_bins, test_size=0.3, random_state=42
        )

        tree = DecisionTreeClassifier(random_state=42)
        tree.fit(X_train, y_train)
        y_pred_tree = tree.predict(X_test)
        acc = accuracy_score(y_test, y_pred_tree)
        st.write("###  Decision Tree Accuracy:", round(acc, 2))

        logistic = LogisticRegression(max_iter=500)
        logistic.fit(X_train, y_train.codes)  # convert categories to numbers
        preds = logistic.predict(X_test)

        cm = confusion_matrix(y_test.codes, preds)

        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax2)
        ax2.set_title("Confusion Matrix")
        st.pyplot(fig2)

        # ---------- 3) Correlation Heatmap ----------
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax3)
        ax3.set_title("Correlation Heatmap (Feature Relationships)")
        st.pyplot(fig3)

else:
    st.info(" Please upload a file to continue.")
