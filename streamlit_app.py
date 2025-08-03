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

st.title('🤖🩺 Machine Learning APP for Predicting Diabetes & Prediabetes Risk')
st.info('Predict the **Diabetes/Prediabetes** status based on health data using NNet and Logistic Regression.')







# ---------- Load Data ----------
@st.cache_data
def load_data():
    url = "https://github.com/Bahsobi/eGDR_Diabetes_Prediabetes/raw/refs/heads/main/FEATURE%20FINAL.xlsx"
    return pd.read_excel(url)

df = load_data()

# ---------- Features & Target ----------
target = 'Diabetes_Prediabetes'
features = [col for col in df.columns if col != target]

categorical_features = ['Race_Ethnicity', 'Marital_Status',
                        'Smoked_100_Cigarettes', 'Ever_Drank_Alcohol']
numerical_features = ['Age', 'BMI', 'Total_Cholesterol', 'Triglycerides', 'eGDR']

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







#########################################################################################################################



# ---------- Sidebar User Input ----------
# ---------- Sidebar User Input ----------
st.sidebar.header("📝 Input Individual Data")




#part1
# Fixed Category Options (Based on Your Divisions)
race_options = [
    "Mexican American", "Other Hispanic", "Non-Hispanic White",
    "Non-Hispanic Black", "Non-Hispanic Asian", "Other Race - Including Multi-Racial"
]

marital_status_options = [
    "Living alone", "Married/living with partner"
]

education_level_options = [
    "Less than 9th grade", "9-11th grade", "High school graduate or equivalent",
    "Some college or AA degree", "College graduate or above"
]

smoke_options = [
    "Never smoker"
]

alcohol_options = [
    "Never drinker", "Former drinker", "Light-to-moderate", "Heavy drinker"
]



#part2
# Numerical Inputs (Fixed Range like previous style)
egdr = st.sidebar.number_input("eGDR (2 - 15)", min_value=2.0, max_value=25.0, value=10.0)

age = st.sidebar.number_input("Age (18 - 80)", min_value=18, max_value=80, value=30)
bmi = st.sidebar.number_input("BMI (14.6 - 82.0)", min_value=14.6, max_value=82.0, value=25.0)
total_cholesterol = st.sidebar.number_input("Total Cholesterol (80 - 400)", min_value=80.0, max_value=400.0, value=200.0)
triglycerides = st.sidebar.number_input("Triglycerides (30 - 600)", min_value=30.0, max_value=600.0, value=150.0)



#part3
# Categorical Inputs with New Divisions
race_ethnicity = st.sidebar.selectbox("Race/Ethnicity", race_options)
marital_status = st.sidebar.selectbox("Marital Status", marital_status_options)
smoked_100_cigarettes = st.sidebar.selectbox("Smoked at least 100 Cigarettes", smoke_options)
ever_drank_alcohol = st.sidebar.selectbox("Alcohol Consumption", alcohol_options)

# Create User Input DataFrame
user_input = pd.DataFrame([{
    'Age': age,
    'BMI': bmi,
    'Total_Cholesterol': total_cholesterol,
    'Triglycerides': triglycerides,
    'eGDR': egdr,
    'Race_Ethnicity': race_ethnicity,
    'Marital_Status': marital_status,
    'Smoked_100_Cigarettes': smoked_100_cigarettes,
    'Ever_Drank_Alcohol': ever_drank_alcohol,
}])



##########################################################################################



# ---------- Prediction ----------
prediction = model.predict(user_input)[0]
probability = model.predict_proba(user_input)[0][1]
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
st.pyplot(fig)

# ---------- Quartile Odds Ratio for eGDR ----------
st.subheader("📉 Odds Ratios for Diabetes/Prediabetes by eGDR Quartiles")
df_egdr = df[['eGDR', target]].copy()
df_egdr['eGDR_quartile'] = pd.qcut(df_egdr['eGDR'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

X_q = pd.get_dummies(df_egdr['eGDR_quartile'], drop_first=True)
X_q = sm.add_constant(X_q).astype(float)
y_q = df_egdr[target].astype(float)

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
sns.pointplot(data=or_df, x='Quartile', y='Odds Ratio', join=False, capsize=0.2, errwidth=1.5)
ax3.axhline(1, linestyle='--', color='gray')
ax3.set_title("Odds Ratios for Diabetes/Prediabetes by eGDR Quartiles")
st.pyplot(fig3)

# ---------- Summary ----------
with st.expander("📋 Data Summary"):
    st.write(df.describe())

st.subheader("🎯 Diabetes/Prediabetes Distribution")
fig2, ax2 = plt.subplots()
df[target].value_counts().plot.pie(
    autopct='%1.1f%%', labels=['No Diabetes/Prediabetes', 'Diabetes/Prediabetes'], ax=ax2, colors=["#81c784", "#e57373"])
ax2.set_ylabel("")
st.pyplot(fig2)

with st.expander("🔍 Sample Data (First 10 Rows)"):
    st.dataframe(df.head(10))
