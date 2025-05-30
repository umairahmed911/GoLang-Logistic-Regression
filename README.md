# 🚢 Titanic Survival Prediction in GoLang

> 🎯 A clean implementation of logistic regression from scratch in Go to predict Titanic passenger survival based on age.

---

## 📌 Overview

This project uses a **logistic regression model**, implemented entirely in Go, to predict passenger survival on the Titanic. It focuses on **feature engineering**, **age-based analysis**, **gradient descent optimization**, and **evaluation with threshold tuning**.

---

## 📁 Dataset

- **Source:** Titanic Dataset (`titanic (1).csv`)
- **Features Used:** Age (with polynomial and categorical transformations)
- **Label:** Survived (0 = No, 1 = Yes)

---

## 🧠 Features & Techniques

- ✅ Logistic Regression from scratch
- ✅ Gradient descent with L2 regularization
- ✅ Age normalization, squaring, and logarithmic scaling
- ✅ One-hot encoded age group flags (`Child`, `Teen`, `Elderly`)
- ✅ Train-test split with stratified sampling
- ✅ Optimal threshold search for binary classification

---

## 🛠️ How to Run

> ⚙️ Requirements: Go installed (v1.20+ recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/titanic-logistic-golang.git
cd titanic-logistic-golang

# Run the Go program
go run titanic.go
📌 Make sure the titanic (1).csv file is in the same directory as main.go.

🔍 Engineered Features
Feature	Description
age	Raw age value
age²	Non-linear feature
log(age)	Logarithmic scaling to reduce skew
isChild	Binary: 1 if age ≤ 12
isTeen	Binary: 1 if 13 ≤ age ≤ 17
isElderly	Binary: 1 if age ≥ 60

📊 Example Output
text
Copy
Edit
=== Dataset Statistics ===
Total records: 891
Survivors: 342 (38.4%)
Average age: 29.70
Median age: 28.00

=== Model Evaluation ===
Training Accuracy: 68.56%
Test Accuracy: 64.14%
Baseline Accuracy: 65.61%

=== Sample Predictions ===
Age 5:    91.3% survival probability - Would survive
Age 25:   71.3% survival probability - Would survive
Age 75:   24.7% survival probability - Would not survive
📈 Model Configuration
Hyperparameter	Value
Learning Rate	0.1
Epochs	1000
Regularization	0.01

👥 Authors
👨‍💻 Umair Ahmed — F23607025

👨‍💻 M. Anas Bhatti — F23607044

🧑‍🏫 Instructor: Lec. Umar Aftab

🎓 BS Artificial Intelligence - Spring 2025
🧪 National University of Technology, Pakistan


🌟 Star this Repo
If you found this project helpful or interesting, consider giving it a ⭐ on GitHub!
