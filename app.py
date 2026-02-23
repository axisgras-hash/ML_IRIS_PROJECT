import streamlit as st
import pandas as pd
import pickle
import time
import numpy as np
st.set_page_config(
    page_title="Flower Species Prediction",
    page_icon="🌸",
    layout="wide"  # This makes the page full width
)

st.title('🪻IRIS Flower Species Classification🪻')
st.image('https://camo.githubusercontent.com/ca4ff60b76c0803d41de1ebf12d6617465e3924fe6845192dd98e9f8503596c2/68747470733a2f2f7777772e656d6265646465642d726f626f746963732e636f6d2f77702d636f6e74656e742f75706c6f6164732f323032322f30312f497269732d446174617365742d436c617373696669636174696f6e2d31303234783336372e706e67')

st.write("""
This Machine Learning web application predicts the species of a flower based on its input features.  
The model is trained on the famous Iris dataset and compares predictions from multiple classification algorithms.

## 🤖 Models Used
- Logistic Regression  
- Support Vector Classifier (SVC)  
- Decision Tree Classifier  
- Random Forest Classifier  
- K-Nearest Neighbors (KNN)  
- Multinomial Naive Bayes  
- Gradient Boosting Classifier  
- AdaBoost Classifier  
- XGBoost Classifier  
- LightGBM Classifier  

Each model outputs the predicted species along with its probability score, allowing users to compare performance across different algorithms.

""")




st.sidebar.title('Select Iris Flower Features')
st.sidebar.image('https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2F1cs36b9rbuc98t7uhb0h.webp')



df = pd.read_csv('iris.csv')
X = df.drop(['target',	'target_names'], axis = 1)


target_names = ['setosa', 'versicolor', 'virginica']
user_input = []
for i in X:
  min_i = X[i].min()
  max_i = X[i].max()
  ans = float(st.sidebar.slider(f'Enter value for {i}: ',min_i,max_i))
  user_input.append(ans)

final_input = [user_input]

all_models_names = ['LogisticRegression','SVC','DecisionTreeClassifier',
              'RandomForestClassifier', 'KNeighborsClassifier', 'MultinomialNB',
              'GradientBoostingClassifier', 'AdaBoostClassifier','XGBClassifier','LGBMClassifier']




# Custom CSS to make sidebar button green
st.markdown("""
<style>
div[data-testid="stSidebar"] button[kind="primary"] {
    background-color: #28a745 !important;
    color: white !important;
    border-radius: 8px;
}
div[data-testid="stSidebar"] button[kind="primary"]:hover {
    background-color: #218838 !important;
}
</style>
""", unsafe_allow_html=True)

all_species = [str(i) for i in target_names]
flower_dict = dict(zip(all_species,np.zeros(len(all_species))))

submit_button = st.sidebar.button("Submit", type="primary")
if submit_button:
    st.markdown("### Predicting *Iris Flower Species*...")
    # Then show spinner
    with st.spinner("Please wait..."):
        time.sleep(2)

    for index,i in enumerate(all_models_names):
      with open(f'{i}_best_model.pkl','rb') as f:
          model = pickle.load(f)
      model_ans = model.predict(final_input)[0]
      final_flower_name = target_names[model_ans]
      flower_dict[final_flower_name] += 1
        
      try:
        pb = round(model.predict_proba(final_input).max()*100,2)
      except:
        pb = 1
    
      st.write(f"""
        ### 🌸 Prediction using **{i}**
        
        - **Predicted Species:** `{final_flower_name}`
        - **Probability:** `{round(pb,2)}%`
        
        ---
        """)

    final_model_prediction = max(flower_dict,key = flower_dict.get)
    
    st.write(f"""
            ---
            ### 🌸 FINAL PREDICTION IS
            
            - **Predicted Species:** `{final_model_prediction}`
            - **Probability:** `{round(100,2)}%`
            
            ---
            """)
    
    st.image(final_model_prediction.lower()+'.png')
    

st.markdown("""
---
<div style="text-align:center; font-size:14px;">
    Made with ❤️ by <b>Ankit Mishra</b> <br>
    🔗 <a href="https://https://www.linkedin.com/in/ankitmishra97/" target="_blank">LinkedIn</a> |
    💻 <a href="https://github.com/axisgras-hash" target="_blank">GitHub</a>
</div>
""", unsafe_allow_html=True)
