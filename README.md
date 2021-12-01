# fastapi + MLflow + streamlit + AWS

<p style="text-align:center">
<img src="https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png" width="100" > <img src="https://databricks.com/wp-content/uploads/2021/06/MLflow-logo-pos-TM-1.png" width="100">
<img src="https://aws1.discourse-cdn.com/business7/uploads/streamlit/original/2X/8/8cb5b6c0e1fe4e4ebfd30b769204c0d30c332fec.png" width="100">
<img src="https://upload.wikimedia.org//wikipedia/commons/thumb/9/93/Amazon_Web_Services_Logo.svg/1200px-Amazon_Web_Services_Logo.svg.png" width="100" >
</p>
<br> 
</br>

This is a modified version of [mnist mlops learning](https://github.com/zademn/mnist-mlops-learning) project. Kudos to [zademn](https://github.com/zademn/mnist-mlops-learning)! <br>
<br>
> Due to the monthly cost I shutted down the app. [Screenshots](https://github.com/silverstone1903/hr-attrition-mlops#screenshots) 👇🏻
>  
Streamlit URL: ~~[streamlit.gumustas.tech](http://streamlit.gumustas.tech/)~~ <br>
MLFlow URL: ~~[mlflow.gumustas.tech](http://mlflow.gumustas.tech)~~ <br>

Blog post (in Turkish): [fastapi + MLflow + streamlit + AWS = MLOps](https://silverstone1903.github.io/posts/2021/10/fastapi-mlflow-streamlit-aws/)

---

Setup env.
```bash
pip install -r requirements.txt
```
# Start app
Go in the root dir and run these;

Streamlit
```bash
streamlit run frontend/streamlit_main.py
```

FastAPI 
```
uvicorn backend.main:app
```

MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///db/backend.db
```

## Docker
- Mlflow: http://localhost:5000
- FastApi: http://localhost:8000/docs
- Streamlit: http://localhost:8501/

```bash
docker-compose build
docker-compose up
```

# Architecture
![image](resources/arch.png)

# Screenshots

Due to the monthly cost I shutted down the app. Here are the screenshots: 

<img align="center" src="https://raw.githubusercontent.com/silverstone1903/hr-attrition-mlops/master/assets/mlflow.JPG">
<img align="center" src="https://raw.githubusercontent.com/silverstone1903/hr-attrition-mlops/master/assets/streamlit_1.JPG">
<img align="center" src="https://raw.githubusercontent.com/silverstone1903/hr-attrition-mlops/master/assets/streamlit_2.JPG">
<img align="center" src="https://raw.githubusercontent.com/silverstone1903/hr-attrition-mlops/master/assets/streamlit_3.JPG">
<img align="center" src="https://raw.githubusercontent.com/silverstone1903/hr-attrition-mlops/master/assets/streamlit_4.JPG">
<img align="center" src="https://raw.githubusercontent.com/silverstone1903/hr-attrition-mlops/master/assets/aws_bill.png">