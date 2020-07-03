import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'gender_Male':1,'gender_Other':0, 'age':55, 'hypertension':0, 'heart_disease':0,
                            'avg_glucose_level':113.45, 'bmi':27.9,'smoking_status_never smoked':0,
                            'smoking_status_smokes':0})

print(r.json())