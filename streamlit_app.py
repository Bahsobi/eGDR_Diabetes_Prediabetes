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

# ---------- Detect Feature Types ----------
categorical_features = df[features].select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = df[features].select_dtypes(include=['int64', 'float64']).columns.tolist()




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
cat_features = encoder.get_feature_names_out(categorical_features).tolist()
feature_names = cat_features + numerical_features
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

filtered_odds_df = odds_df[~odds_df['Feature'].str.contains("race")]

# ---------- Sidebar User Input ----------
st.sidebar.header("📝 Input Individual Data")

user_input_data = {}
for col in features:
    if df[col].dtype == 'object':
        options = df[col].dropna().unique().tolist()
        user_input_data[col] = st.sidebar.selectbox(col, options)
    else:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        default_val = float(df[col].mean())
        user_input_data[col] = st.sidebar.number_input(f"{col} ({min_val:.2f} - {max_val:.2f})", min_value=min_val, max_value=max_val, value=default_val)

user_input = pd.DataFrame([user_input_data])

# ---------- Prediction ----------
prediction = model.predict(user_input)[0]
probability = model.predict_proba(user_input)[0][1]
odds_value = probability / (1 - probability)

# ---------- Display Result ----------
if prediction == 1:
    st.error(f"""
        ⚠️ **Prediction: At Risk for Prediabetes/Diabetes**

        🧮 **Probability:** {probability:.2%}  
        🎲 **Odds:** {odds_value:.2f}
    """)
else:
    st.success(f"""
        ✅ **Prediction: Not at Risk**

        🧮 **Probability:** {probability:.2%}  
        🎲 **Odds:** {odds_value:.2f}
    """)

# ---------- Show Tables ----------
st.subheader("📊 Odds Ratios (Logistic Regression) (Excluding Race)")
st.dataframe(filtered_odds_df)

st.subheader("💡 Feature Importances (XGBoost)")
st.dataframe(importance_df)

# ---------- Plot Feature Importances ----------
st.subheader("📈 Bar Chart: Feature Importances")
fig, ax = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax, color="#2196f3")
st.pyplot(fig)

# ---------- Quartile Odds Ratio for WWI ----------
if 'WWI' in df.columns:
    st.subheader("📉 Odds Ratios by WWI Quartiles")
    df_wwi = df[['WWI', target]].copy()
    df_wwi['WWI_quartile'] = pd.qcut(df_wwi['WWI'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    X_q = pd.get_dummies(df_wwi['WWI_quartile'], drop_first=True)
    X_q = sm.add_constant(X_q).astype(float)
    y_q = df_wwi[target].astype(float)

    model_q = sm.Logit(y_q, X_q).fit(disp=False)
    ors = np.exp(model_q.params)
    ci = model_q.conf_int()
    ci.columns = ['2.5%', '97.5%']
    ci = np.exp(ci)

    or_df = pd.DataFrame({
        'Quartile': ors.index,
        'Odds Ratio': ors.values,
        'CI Lower': ci['2.5%'],
        'CI Upper': ci['97.5%'],
        'p-value': model_q.pvalues
    }).query("Quartile != 'const'")

    st.dataframe(or_df.set_index('Quartile').style.format("{:.2f}"))

    fig3, ax3 = plt.subplots()
    sns.pointplot(data=or_df, x='Quartile', y='Odds Ratio', join=False, capsize=0.2, errwidth=1.5, color="#0d47a1")
    ax3.axhline(1, linestyle='--', color='gray')
    ax3.set_title("Odds Ratios by WWI Quartiles")
    st.pyplot(fig3)

# ---------- Summary ----------
with st.expander("📋 Data Summary"):
    st.write(df.describe())

st.subheader("🎯 Distribution of Diabetes_Prediabetes")
fig2, ax2 = plt.subplots()
df[target].value_counts().plot.pie(
    autopct='%1.1f%%', labels=['Not at Risk', 'At Risk'], ax=ax2, colors=["#90caf9", "#f44336"])
ax2.set_ylabel("")
st.pyplot(fig2)

with st.expander("🔍 Sample Data (First 10 Rows)"):
    st.dataframe(df.head(10))
