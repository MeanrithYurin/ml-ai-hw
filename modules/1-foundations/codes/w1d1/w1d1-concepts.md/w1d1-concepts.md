# Week 1 Day 1 – Mapping the ML Universe

## Exercise A – ML Problem Types

### Loan default prediction
- Type: Classification  
- Inputs: credit score, income, repayment history  
- Output: default or not default  

---

### Energy consumption forecast
- Type: Regression  
- Inputs: past energy usage, weather, season  
- Output: predicted energy value  

---

### Customer grouping
- Type: Clustering  
- Inputs: purchase frequency, spending, product type  
- Output: customer groups  

---

### Fraud detection
- Type: Classification  
- Inputs: transaction amount, location, time  
- Output: fraud or not fraud  

---

### Product description from image
- Type: Generative  
- Inputs: image, visual features  
- Output: generated text  

---

### Patient severity (1–5)
- Type: Classification  
- Inputs: vital signs, symptoms, history  
- Output: severity level  

---

## Exercise B – ML Lifecycle

**Problem Definition**  
We want to predict which patients might come back within 30 days. This helps doctors take action early and reduce readmissions.

**Data Collection**  
Use hospital records like diagnosis, treatments, and past visits. Data from several years is better.

**EDA & Preprocessing**  
Clean missing values and convert data into usable format. Medical data is usually messy.

**Model Training**  
Start with simple models like logistic regression or decision tree. Easier to understand.

**Evaluation**  
Accuracy is not enough. Recall is important to catch high-risk patients.

**Deployment**  
Show risk score in hospital system when patient is leaving.

**Monitoring**  
Model needs to be checked and updated over time.

---

## Exercise C – AI Types

Chess engine → Rule-based AI  
Spam filter → Classical ML  
GPT-4 → Deep Learning  
Netflix system → Classical ML  
Thermostat → Rule-based AI  
Tumor detection (CNN) → Deep Learning  
Decision tree → Classical ML  

---

## My Note

- yes/no → classification  
- number → regression  
- grouping → clustering  
- create new → generative  