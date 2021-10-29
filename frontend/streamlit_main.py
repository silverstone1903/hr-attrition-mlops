import streamlit as st

import requests
import urllib
import json
import os
import numpy as np
import pandas as pd
import urllib.request

columns = ['BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
       'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender',
       'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
       'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked', 'OverTime',
       'StockOptionLevel', 'WorkLifeBalance']

local = True


if local:
    BACKEND_URL = "http://localhost:8000"
else:
    ip = urllib.request.urlopen("http://169.254.169.254/latest/meta-data/public-ipv4").read().decode()
    BACKEND_URL = str("http://") + str(ip) + str(":8000")


MODELS_URL = urllib.parse.urljoin(BACKEND_URL, "models")
TRAIN_URL = urllib.parse.urljoin(BACKEND_URL, "train")
PREDICT_URL = urllib.parse.urljoin(BACKEND_URL, "predict")
FI_URL = urllib.parse.urljoin(BACKEND_URL, "importance")

st.set_page_config(layout="centered",
                   page_icon="💰",
                   page_title="HR Analytics - Employee Attrition Prediction")

page = st.sidebar.radio(label="Pages", options=["Main Page", "Train", "Predict", "Feature Importance"])

if page == "Main Page":
    st.header("fastapi + MLflow + streamlit + AWS = MLOps")
    st.markdown("This is a modified version of [mnist mlops learning](https://github.com/zademn/mnist-mlops-learning) project. Kudos to [zademn](https://github.com/zademn/mnist-mlops-learning)!")
    st.markdown("Changelog:")
    st.markdown("* Used [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset) data with selected features.")
    st.markdown("* Model uses features which were selected on my previous [MLOps post (in Turkish)](https://silverstone1903.github.io/posts/2020/08/github-actions-ile-cml).")
    st.markdown("* Replaced PyTorch DL model to Random Forest Classifer.")
    st.markdown("* DL model parameters were replaced with RF parameters (criteria, maximum features, class weight and number of trees)." )
    st.markdown("* Feature Importance page & endpoint added.")
    st.markdown("* Deployed as IAS to AWS EC2 instance. Runs on t2.micro instance with 2 Network Load Balancers.")
    st.markdown("* Registered domain from [tech](https://get.tech) and added it on Route53 as a hosted zone. (I already used namecheap & name.com from [Github Student Pack](https://education.github.com/pack/offers?sort=popularity&tag=Domains) 😅). ")
    st.markdown("* On [streamlit](http://streamlit.gumustas.tech) you can train a model or get prediction and track experiments on [mlflow](http://mlflow.gumustas.tech).")
    
    
    st.markdown(""" <p style="text-align: center;"> Powered by </p> """, unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.image("https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png",width=100)
    c2.image( "https://databricks.com/wp-content/uploads/2021/06/MLflow-logo-pos-TM-1.png", width=100)
    c3.image( "https://aws1.discourse-cdn.com/business7/uploads/streamlit/original/2X/8/8cb5b6c0e1fe4e4ebfd30b769204c0d30c332fec.png", width=100)
    c4.image( "https://upload.wikimedia.org//wikipedia/commons/thumb/9/93/Amazon_Web_Services_Logo.svg/1200px-Amazon_Web_Services_Logo.svg.png", width=100)
    
    

elif page == "Train":
    st.header("Train")
    st.session_state.model_type = st.selectbox(
        "Model type", options=["RF"])

    model_name = st.text_input(label="Model name", value="Random Forest", max_chars=20)

    if st.session_state.model_type == "RF":
        num_tree = st.slider("N Tree:", value=100, min_value=10, max_value=250)
        criterion = st.selectbox("Criterion", options = ["gini", "entropy"])
        max_feat = st.selectbox("Max. Features", options = ["auto", "sqrt", "log2"])
        cw = st.selectbox("Class Weight", options = ["balanced", None])
        hyperparams = {"n_estimators": num_tree, "criterion": criterion, "max_features": max_feat, "class_weight": cw}

        
    if st.button("Train"):    
        
        to_post = {"model_name": model_name,
                   "hyperparams": hyperparams,
                    "max_features": max_feat,
                    "class_weight": cw}
        
        st.header("Selected Parameters")        
        for key, value in hyperparams.items():
            st.write(key, ' : ', value)

        response = requests.post(url=TRAIN_URL, data=json.dumps(to_post))
        if response.ok:
            res = response.json()["result"]
        else:
            res = "Training task failed"
        st.write(res)
        st.subheader("Track experiments in [MLFlow](http://mlflow.gumustas.tech) ")
        
        
        

elif page == "Predict":
    st.header("Predict")
    st.write("[Dataset Description](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset) ")
    try:
        response = requests.get(MODELS_URL)
        if response.ok:
            model_list = response.json()
            model_name = st.selectbox(
                label="Select your model", options=model_list)
        else:
            st.write("No models found")
    except ConnectionError as e:
        st.write("Couldn't reach backend")
    
    c1, c2, c3 = st.columns(3)
    
    x1 = c1.selectbox("Business Travel:", (0, 1, 2), 
                      help = "0: Non-Travel, 1: Travel Frequently, 2: Travel Rarely")
    
    x2 = c1.slider("Daily Rate:",
                     value=800,
                     min_value=100, max_value = 1500)
    
    x3 = c1.selectbox("Department:", (0, 1, 2), help = 'Research & Development: 0, Human Resources: 1, Sales: 2')
    
    x4 = c1.number_input("Distance From Home:", value=1, min_value=1, max_value = 29, step=1)
    
    x5 = c1.selectbox("Education:", (1,2,3,4,5), help = '1: Below College, 2: College,  3: Bachelor, 4: Master, 5: Doctor')
    
    x6 = c1.selectbox("Education Field:", (0, 1, 2, 3, 4, 5),  help = "Other: 0, Medical: 1, Life Sciences: 2, Marketing: 3, Technical Degree: 4, Human Resources: 5")
    
    x21 = c2.selectbox("Environment Satisfaction:", (1, 2, 3, 4), 
                          help = "1: Low, 2: Medium, 3: High, 4: Very High")
    
    x22 = c2.selectbox("Gender:", (0, 1), help = "Female: 0, Male: 1")
    
    x23 = c2.selectbox("Job Involvement:", (1, 2, 3, 4), 
                      help = "1: Low, 2: Medium, 3: High, 4: Very High")
    
    x24 = c2.selectbox("Job Level:", (1, 2, 3, 4, 5))
    
    x25 = c2.selectbox("Job Role:", (0, 1, 2, 3, 4, 5, 6, 7, 8), 
                      help = "0: Research Director, 1: Manager, 2: Healthcare Representative, 3: Manufacturing Director, ")
    
    x26 = c2.selectbox("Job Satisfaction:", (1, 2, 3, 4), 
                      help = "1: Low, 2: Medium, 3: High, 4: Very High")
    
    x31 = c3.selectbox("Marital Status:", (0, 1, 2), 
                      help = "0: Divorced, 1: Married, 2: Single")
    
    x32 = c3.slider("Monthly Income:", value = 6500, min_value = 1000, max_value = 25000)
    
    x33 = c3.number_input("Number of Companies Worked:",
                     value=3,
                     min_value=0,
                     max_value = 12,
                     step=1)
    
    x34 = c3.selectbox("Over Time:", (0, 1), help = "0: No, 1: Yes")
    
    x35 = c3.selectbox("Stock Option Level:", (0, 1, 2, 3))
    
    x36 = c3.selectbox("Work-Life Balance:", (0, 1, 2, 3))
    
    df = (x1, x2, x3, x4, x5, x6, x21, x22, x23, x24, x25, x26, x31, x32, x33, x34, x35, x36)


    if st.button("Predict"):
    
        try:
            response_predict = requests.post(url=PREDICT_URL,
                                              data=json.dumps({"data": df, "model_name": model_name})
                                              )
            print("Response sent!")
            print(response_predict)
            if response_predict.ok:
                print("Response Code:", response_predict.status_code)
                res = response_predict.json()
                st.markdown(f"**Prediction**: {res['result']}")
                
            else:
                print("Some error occured!")
                st.write("Some error occured")
        except ConnectionError as e:
            st.write("Couldn't reach backend")
elif page == "Feature Importance":
    st.header("Feature Importance")
    try:
        response = requests.get(MODELS_URL)
        if response.ok:
            model_list = response.json()
            model_name = st.selectbox(
                label="Select your model", options=model_list)
        else:
            st.write("No models found")
    except ConnectionError as e:
        st.write("Couldn't reach backend")

    fi = requests.post(FI_URL,  data=json.dumps({"model_name": model_name}))
    fi = pd.DataFrame(fi.json()).reset_index(drop =True )
    st.dataframe(fi, 600, 600)
    
else:
    st.write("Page does not exist")
    

footer = """
<center>
<br>
<br>
<br>
<br>
<p><a href="https://silverstone1903.github.io" target="_blank" rel="noopener">
<img src="https://logoeps.com/wp-content/uploads/2014/05/37318-github-logo-icon-vector-icon-vector-eps.png" 
alt="" width="32" height="32" /></a></p>
</center>
"""

st.markdown(footer, unsafe_allow_html=True)