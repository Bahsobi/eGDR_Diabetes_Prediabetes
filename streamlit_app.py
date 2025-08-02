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





---------- Define Feature Types ----------
Categorical features with their mappings
categorical_features = {
'Race_Ethnicity': {
1: 'Non-Hispanic White',
2: 'Non-Hispanic Black',
3: 'Mexican American',
4: 'Other Hispanic',
5: 'Non-Hispanic Asian',
6: 'Other Race',
7: 'Multi-Racial'
},
'Education_Level': {
1: 'Less than 9th grade',
2: '9-11th grade',
3: 'High school graduate',
4: 'Some college or AA degree',
5: 'College graduate or above'
},
'Marital_Status': {
1: 'Married',
2: 'Not married'
},
'Smoked_100_Cigarettes': {
1: 'Yes',
2: 'No'
},
'Ever_Drank_Alcohol': {
1: 'Yes',
2: 'No'
},
'Hyperlipidemia': {
0: 'No',
1: 'Yes'
}
}

Numerical features
numerical_features = ['Age', 'BMI', 'Total_Cholesterol', 'Triglycerides', 'eGDR']

Target variable
target = 'Diabetes_Prediabetes'

Apply categorical mappings
for col, mapping in categorical_features.items():
df[col] = df[col].map(mapping)

---------- Features & Target ----------
features = list(categorical_features.keys()) + numerical_features
df = df[features + [target]].dropna()

X = df[features]
y = df[target]

---------- Preprocessing ----------
cat_features = list(categorical_features.keys())
num_features = numerical_features

preprocessor = ColumnTransformer([
('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
('num', StandardScaler(), num_features)
])

---------- XGBoost Pipeline ----------
model = Pipeline([
('prep', preprocessor),
('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model.fit(X_train, y_train)

---------- Feature Importance ----------
xgb_model = model.named_steps['xgb']
encoder = model.named_steps['prep'].named_transformers_['cat']
feature_names = encoder.get_feature_names_out(cat_features).tolist() + num_features
importances = xgb_model.feature_importances_
importance_df = pd.DataFrame({
'Feature': feature_names,
'Importance': importances
}).sort_values(by='Importance', ascending=False)

---------- Logistic Regression for Odds Ratio ----------
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

---------- Sidebar User Input ----------
st.sidebar.header("📝 Input Individual Data")

Dynamically create input widgets based on feature types
user_input = {}
for feature in features:
if feature in categorical_features:
options = list(categorical_features[feature].values())
default_value = options[0]
user_input[feature] = st.sidebar.selectbox(feature, options, index=0)
else:
min_val = float(df[feature].min())
max_val = float(df[feature].max())
default_val = float(df[feature].median())
user_input[feature] = st.sidebar.number_input(
f"{feature} ({min_val:.1f}-{max_val:.1f})",
min_value=min_val,
max_value=max_val,
value=default_val
)

---------- Prediction ----------
user_df = pd.DataFrame([user_input])

prediction = model.predict(user_df)[0]
probability = model.predict_proba(user_df)[0][1]
odds_value = probability / (1 - probability)

---------- Display Result ----------
if prediction == 1:
st.error(f"""
⚠️ Prediction: Diabetes/Prediabetes

text
    🧮 **Probability:** {probability:.2%}  
    🎲 **Odds:** {odds_value:.2f}
""")
else:
st.success(f"""
✅ Prediction: No Diabetes/Prediabetes

text
    🧮 **Probability:** {probability:.2%}  
    🎲 **Odds:** {odds_value:.2f}
""")
---------- Show Tables ----------
st.subheader("📊 Odds Ratios for Diabetes/Prediabetes (Logistic Regression)")
st.dataframe(odds_df)

st.subheader("💡 Feature Importances (XGBoost)")
st.dataframe(importance_df)

---------- Plot Feature Importances ----------
st.subheader("📈 Bar Chart: Feature Importances")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), ax=ax)
ax.set_title("Top 10 Most Important Features for Diabetes/Prediabetes Prediction")
st.pyplot(fig)

---------- Target Distribution ----------
st.subheader("🎯 Diabetes/Prediabetes Distribution")
fig2, ax2 = plt.subplots()
df[target].value_counts().plot.pie(
autopct='%1.1f%%',
labels=['No Diabetes/Prediabetes', 'Diabetes/Prediabetes'],
ax=ax2,
colors=["#81c784", "#e57373"])
ax2.set_ylabel("")
ax2.set_title("Distribution of Diabetes/Prediabetes in Dataset")
st.pyplot(fig2)

---------- Summary ----------
with st.expander("📋 Data Summary"):
st.write(df.describe())

with st.expander("🔍 Sample Data (First 10 Rows)"):
st.dataframe(df.head(10))

---------- Correlation Heatmap ----------
st.subheader("🌡️ Correlation Heatmap")
numerical_df = df[numerical_features + [target]]
corr = numerical_df.corr()
fig3, ax3 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax3)
ax3.set_title("Correlation Between Numerical Features and Diabetes/Prediabetes")
st.pyplot(fig3)

---------- Age Distribution by Diabetes Status ----------
st.subheader("📊 Age Distribution by Diabetes/Prediabetes Status")
fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.boxplot(x=target, y='Age', data=df, ax=ax4)
ax4.set_xticklabels(['No Diabetes/Prediabetes', 'Diabetes/Prediabetes'])
ax4.set_title("Age Distribution by Diabetes/Prediabetes Status")
st.pyplot(fig4)

---------- BMI Distribution by Diabetes Status ----------
st.subheader("📊 BMI Distribution by Diabetes/Prediabetes Status")
fig5, ax5 = plt.subplots(figsize=(10, 6))
sns.boxplot(x=target, y='BMI', data=df, ax=ax5)
ax5.set_xticklabels(['No Diabetes/Prediabetes', 'Diabetes/Prediabetes'])
ax5.set_title("BMI Distribution by Diabetes/Prediabetes Status")
st.pyplot(fig5)












