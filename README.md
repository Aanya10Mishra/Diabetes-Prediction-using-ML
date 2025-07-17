

# Diabetes Prediction Using Machine Learning
**Overview**
This project implements a machine learning model to predict whether a person has diabetes based on medical attributes. The model uses a Support Vector Machine (SVM) classifier trained on the Pima Indians Diabetes Dataset. The dataset is preprocessed, standardized, and split into training and testing sets to evaluate the model's performance. The project includes data analysis, model training, evaluation, and a predictive system for new inputs.
Features

Data Preprocessing: Loads and analyzes the dataset, standardizes features using StandardScaler.
Model Training: Trains an SVM classifier with a linear kernel.
Model Evaluation: Computes accuracy scores for training and testing data.
Prediction System: Allows input of new data to predict diabetes status (diabetic or non-diabetic).
Dataset: Uses the Pima Indians Diabetes Dataset, which includes features like glucose levels, blood pressure, and BMI.

Requirements
To run this project, you need Python 3.x and the following libraries:

numpy
pandas
scikit-learn

You can install the required libraries using:
pip install -r requirements.txt

Installation

Clone the Repository:
git clone https://github.com/your-username/diabetes-prediction-using-ml.git
cd diabetes-prediction-using-ml


Install Dependencies:
pip install -r requirements.txt


Download the Dataset:

The project uses the diabetes.csv dataset. You can download it from Kaggle or another source.
Place the diabetes.csv file in the project directory.


Run the Script:
python diabetes_prediction_using_ml.py



**Usage**
The script loads the diabetes.csv dataset, preprocesses it, and trains an SVM model.
After training, it evaluates the model’s accuracy on both training and testing data.
A sample input is provided to demonstrate the predictive system:input_data = (2, 197, 70, 45, 543, 30.5, 0.158, 53)

The script outputs whether the person is diabetic or not based on this input.

To make predictions with new data:

Modify the input_data tuple in the script with your own values.
Run the script to see the prediction.

**Dataset**
The dataset (diabetes.csv) contains the following columns:

Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age
Outcome (0 = non-diabetic, 1 = diabetic)

The dataset is not included in the repository due to its size and licensing. Download it from Kaggle.

**Results**
Training Accuracy: Approximately 78-80% (varies with random seed).
Testing Accuracy: Approximately 76-78% (varies with random seed).
The model can predict diabetes status for new inputs with reasonable accuracy.

**Future Improvements**
Experiment with other machine learning models (e.g., Random Forest, XGBoost).
Perform hyperparameter tuning for the SVM classifier.
Add cross-validation to improve model robustness.
Include feature selection to identify the most important predictors.

**Contributing**
Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure your code follows PEP 8 guidelines and includes appropriate documentation.
License
This project is licensed under the MIT License. See the LICENSE file for details.

**Acknowledgments**
The Pima Indians Diabetes Dataset is sourced from Kaggle.
Built using Python, scikit-learn, pandas, and numpy.
