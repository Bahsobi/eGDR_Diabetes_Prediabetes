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

# ---------- Custom Styling ----------
st.markdown(
    """
    <style>
        .stApp {
            background-color: #E3F2FD;  /* Light Blue */
        }
        .stSidebar {
            background-color: #BBDEFB;  /* Deeper Blue */
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

st.title('🤖 Machine Learning Models APP for Predicting Prediabetes & Diabetes Risk in Women')
st.info('Predict the **Prediabetes & Diabetes** based on health data using XGBoost and Logistic Regression.')





# ---------- Load Data ----------
@st.cache_data
def load_data():
    url = "https://github.com/Bahsobi/eGDR_Diabetes_Prediabetes/raw/refs/heads/main/FEATURE%20FINAL.xlsx"
    return pd.read_excel(url)

df = load_data()


# ---------- Features & Target ----------
target = 'Diabetes_Prediabetes'
features = [col for col in df.columns if col != target]
df = df[features + [target]].dropna()

X = df[features]
y = df[target]

# ---------- Identify Feature Types ----------
categorical_features = []
numerical_features = []

for col in features:
    if df[col].dtype == 'object' or df[col].nunique() < 10:  # Assuming categorical if <10 unique values
        categorical_features.append(col)
    else:
        numerical_features.append(col)

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
importances = xgb_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# ---------- Logistic Regression for Odds Ratio ----------
odds_pipeline = Pipeline([
    ('prep', preprocessor),
    ('logreg', LogisticRegression(max_iter=1000))
])
odds_pipeline.fit(X_train, y_train)
log_model = odds_pipeline.named_steps['logreg']
odds_ratios = np.exp(log_model.coef_[0])

odds_df = pd.DataFrame({
    'Feature': feature_names,
    'Odds Ratio': odds_ratios
}).sort_values(by='Odds Ratio', ascending=False)

# ---------- Sidebar User Input ----------
st.sidebar.header("📝 Input Individual Data")

user_input = {}
for col in features:
    if col in categorical_features:
        unique_values = df[col].unique()
        if len(unique_values) > 10:  # If too many unique values, use number input
            min_val = df[col].min()
            max_val = df[col].max()
            default_val = (min_val + max_val) / 2
            user_input[col] = st.sidebar.number_input(
                f"{col} ({min_val:.1f}-{max_val:.1f})",
                min_value=min_val,
                max_value=max_val,
                value=default_val
            )
        else:
            user_input[col] = st.sidebar.selectbox(col, unique_values)
    else:
        min_val = df[col].min()
        max_val = df[col].max()
        default_val = (min_val + max_val) / 2
        user_input[col] = st.sidebar.number_input(
            f"{col} ({min_val:.1f}-{max_val:.1f})",
            min_value=min_val,
            max_value=max_val,
            value=default_val
        )

# ---------- Prediction ----------
user_df = pd.DataFrame([user_input])

prediction = model.predict(user_df)[0]
probability = model.predict_proba(user_df)[0][1]
odds_value = probability / (1 - probability)

# ---------- Display Result ----------
if prediction == 1:
    st.error(f"""
        ⚠️ **Prediction: Diabetes/Prediabetes**

        🧮 **Probability:** {probability:.2%}  
        🎲 **Odds:** {odds_value:.2f}
    """)
else:
    st.success(f"""
        ✅ **Prediction: No Diabetes/Prediabetes**

        🧮 **Probability:** {probability:.2%}  
        🎲 **Odds:** {odds_value:.2f}
    """)

# ---------- Show Tables ----------
st.subheader("📊 Odds Ratios for Diabetes/Prediabetes (Logistic Regression)")
st.dataframe(odds_df)

st.subheader("💡 Feature Importances (XGBoost)")
st.dataframe(importance_df)

# ---------- Plot Feature Importances ----------
st.subheader("📈 Bar Chart: Feature Importances")
fig, ax = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
ax.set_title("Feature Importance for Diabetes/Prediabetes Prediction")
st.pyplot(fig)

# ---------- Summary ----------
with st.expander("📋 Data Summary"):
    st.write(df.describe())

st.subheader("🎯 Diabetes/Prediabetes Distribution")
fig2, ax2 = plt.subplots()
df['Diabetes_Prediabetes'].value_counts().plot.pie(
    autopct='%1.1f%%', labels=['No Diabetes/Prediabetes', 'Diabetes/Prediabetes'], 
    ax=ax2, colors=["#81c784", "#e57373"])
ax2.set_ylabel("")
ax2.set_title("Diabetes/Prediabetes Distribution")
st.pyplot(fig2)

with st.expander("🔍 Sample Data (First 10 Rows)"):
    st.dataframe(df.head(10))
