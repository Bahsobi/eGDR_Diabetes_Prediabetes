import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import statsmodels.api as sm

# ---------- Custom Styling (Blue Theme) ----------
st.markdown(
    """
    <style>
        .stApp {
            background-color: #e0f7fa;  /* Light Blue */
        }
        .stSidebar {
            background-color: #b2ebf2;  /* Sidebar Blue */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Header ----------
st.markdown(
    """
    <div style='display: flex; justify-content: center; align-items: center; flex-direction: column;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/8/83/TUMS_Signature_Variation_1_BLUE.png' width='200' style='margin-bottom: 10px;'/>
    </div>
    """,
    unsafe_allow_html=True
)

st.title('🤖🩺 Machine Learning APP for Predicting Retinopathy Risk')
st.info('Predict **Retinopathy** risk based on health data using XGBoost and Logistic Regression.')

# ---------- Load and Preprocess Data ----------
# ---------- Load and Preprocess Data ----------
import pandas as pd

@st.cache_data
def load_data():
    url = "https://github.com/Bahsobi/Diabetic-Retinopathy/raw/refs/heads/main/filtered_data_corrected.xlsx"
    df = pd.read_excel(url)
   
    # Check for NaN or infinite values and remove them
    if df.isnull().values.any():
        st.warning("Warning: Missing values found and dropped.")
        df = df.dropna()  # Drop rows with missing values (or use df.fillna(method='ffill') for forward fill)
        
    if np.any(np.isinf(df.values)):
        st.warning("Warning: Infinite values found and replaced with NaN.")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna()  # Remove rows with NaN values after replacing infinities
        
    return df

df = load_data()
# ---------- Features ----------
target = 'Retinopathy'
categorical_features = ['Hypertension']
numerical_features = ['Age', 'BMI', 'Total_Cholesterol', 'Triglycerides', 'Fasting_Glucose', 'HOMA_IR', 'eGDR']
features = categorical_features + numerical_features

X = df[features]
y = df[target]

# ---------- Preprocessing ----------
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

# ---------- XGBoost Pipeline ----------
model = Pipeline([
    ('prep', preprocessor),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model.fit(X_train, y_train)

# ---------- Feature Importance ----------
xgb_model = model.named_steps['xgb']
encoder = model.named_steps['prep'].named_transformers_['cat']
feature_names = encoder.get_feature_names_out(categorical_features).tolist() + numerical_features
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# ---------- Logistic Regression for Odds Ratio ----------
odds_pipeline = Pipeline([
    ('prep', preprocessor),
    ('logreg', LogisticRegression(max_iter=1000))
])
odds_pipeline.fit(X_train, y_train)
log_model = odds_pipeline.named_steps['logreg']
odds_df = pd.DataFrame({
    'Feature': feature_names,
    'Odds Ratio': np.exp(log_model.coef_[0])
}).sort_values(by='Odds Ratio', ascending=False)

# ---------- Sidebar Input ----------
st.sidebar.header("📝 Input Data")

#part1: Categorical Inputs
hypertension_options = df['Hypertension'].dropna().unique().tolist()

#part2: Numerical Inputs
egdr = st.sidebar.number_input("eGDR (2 - 15)", min_value=2.0, max_value=25.0, value=10.0)
age = st.sidebar.number_input("Age (18 - 80)", min_value=18, max_value=80, value=30)
bmi = st.sidebar.number_input("BMI (14.6 - 82.0)", min_value=14.6, max_value=82.0, value=25.0)
total_cholesterol = st.sidebar.number_input("Total Cholesterol (80 - 400)", min_value=80.0, max_value=400.0, value=200.0)
triglycerides = st.sidebar.number_input("Triglycerides (30 - 600)", min_value=30.0, max_value=600.0, value=150.0)
fasting_glucose = st.sidebar.number_input("Fasting Glucose", min_value=50, max_value=400, value=100)
homa_ir = st.sidebar.number_input("HOMA-IR", min_value=0.0, max_value=20.0, value=5.0)

#part3: Categorical Inputs
hypertension = st.sidebar.selectbox("Hypertension", hypertension_options)

user_input = pd.DataFrame([{
    'Age': age,
    'BMI': bmi,
    'Total_Cholesterol': total_cholesterol,
    'Triglycerides': triglycerides,
    'Fasting_Glucose': fasting_glucose,
    'HOMA_IR': homa_ir,
    'eGDR': egdr,
    'Hypertension': hypertension,
}])

# ---------- Prediction ----------
prediction = model.predict(user_input)[0]
probability = model.predict_proba(user_input)[0][1]
odds_value = probability / (1 - probability)

# ---------- Display Result ----------
if prediction == 1:
    st.error(f"""
        ⚠️ **Prediction: Retinopathy**

        🧮 **Probability:** {probability:.2%}  
        🎲 **Odds:** {odds_value:.2f}
    """)
else:
    st.success(f"""
        ✅ **Prediction: No Retinopathy**

        🧮 **Probability:** {probability:.2%}  
        🎲 **Odds:** {odds_value:.2f}
    """)

# ---------- Show Tables ----------
st.subheader("📊 Odds Ratios for Retinopathy (Logistic Regression)")
st.dataframe(odds_df)

st.subheader("💡 Feature Importances (XGBoost)")
st.dataframe(importance_df)

# ---------- Plot Feature Importances ----------
st.subheader("📈 Bar Chart: Feature Importances")
fig, ax = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
st.pyplot(fig)

# ---------- Summary ----------
with st.expander("📋 Data Summary"):
    st.write(df.describe())

st.subheader("🎯 Retinopathy Distribution")
fig2, ax2 = plt.subplots()
df[target].value_counts().plot.pie(
    autopct='%1.1f%%', labels=['No Retinopathy', 'Retinopathy'], ax=ax2, colors=["#81c784", "#e57373"])
ax2.set_ylabel("")
st.pyplot(fig2)

with st.expander("🔍 Sample Data (First 10 Rows)"):
    st.dataframe(df.head(10)) 
