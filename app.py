# ----------------- Data Preparation -----------------
# Drop columns that are IDs or categorical text
numeric_df = df.select_dtypes(include=[np.number]).copy()

# Try to convert numeric-looking strings into numbers
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace("\n", "").str.strip(), errors="ignore")
    except Exception:
        pass

numeric_df = df.select_dtypes(include=[np.number])   # reselect numeric columns

if numeric_df.shape[1] < 2:
    st.error("Dataset must have at least 2 numeric columns.")
else:
    # Use all numeric columns except last as features
    X = numeric_df.iloc[:, :-1]
    y = numeric_df.iloc[:, -1]

