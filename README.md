Titanic Survival Prediction using Logistic Regression in Go
This project implements a logistic regression model from scratch in Go (Golang) to predict passenger survival from the Titanic dataset based on age-related features. The model includes custom feature engineering, regularized gradient descent, and optimal threshold selection.

📂 Project Structure
Language: Go (Golang)

Dataset: titanic (1).csv

Input Feature: Age

Output: Binary survival classification (0 or 1)

✨ Features
🔢 Logistic Regression implementation without external ML libraries

🔍 Custom feature engineering (e.g., age, age², log(age), one-hot age groups)

🧠 Training via gradient descent with L2 regularization

📈 Dynamic threshold tuning for better classification performance

📊 Data statistics and age group survival analysis

🧪 Train/Test split with stratified sampling

🧪 Engineered Features
Feature Name	Description
age	Raw age value
age²	Non-linear age feature
log(age)	Logarithmic transformation to normalize age
isChild	Binary (1 if age ≤ 12)
isTeen	Binary (1 if 13 ≤ age ≤ 17)
isElderly	Binary (1 if age ≥ 60)

🛠️ How to Run
Prepare Dataset

Ensure the file titanic (1).csv is in the same directory as your .go file. It should have:

Age column (column 5)

Survived label (column 2)

Build & Run

bash
Copy
Edit
go run titanic.go
This will:

Load and preprocess the data

Train a logistic regression model

Print evaluation metrics and sample predictions

📊 Sample Output
bash
Copy
Edit
=== Dataset Statistics ===
Total records: 891
Survivors: 342 (38.4%)
Average age: 29.70
...

=== Model Evaluation ===
Training Accuracy: 80.56%
Test Accuracy: 78.14%
Baseline Accuracy (majority class): 61.61%
...

Age 25: 71.3% survival probability - Would survive
📌 Important Functions
createFeatures(age float64) []float64 – Feature engineering

predict(age float64) float64 – Probability prediction

train(...) – Model training with learning rate, epochs, and regularization

evaluateModel(...) – Accuracy computation

findOptimalThreshold(...) – Best classification threshold

📚 References
Titanic Dataset - Kaggle

Course: Programming for AI

National University of Technology (Spring 2025)

👨‍💻 Authors
Umair Ahmed (F23607025)

M. Anas Bhatti (F23607044)

Instructor: Lec. Umar Aftab
Department of Computer Science
BS Artificial Intelligence
