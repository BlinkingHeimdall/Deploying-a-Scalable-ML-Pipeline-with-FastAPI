# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This project uses a RandomForestClassifier trained on the UCI Census Income dataset (Adult Census dataset from 1994).  
The model predicts whether an individual's annual income is:
- **<=50K**, or  
- **>50K**
based on demographic and employment-related features.

The model is trained and saved using Python, scikit-learn, and the data processing utilities provided in `ml/data.py`.  
Categorical features are one-hot encoded, and the target label is binarized.

The following files produced by the pipeline:
- `model/model.pkl`
- `model/encoder.pkl`
The model is deployed using a FastAPI REST API.

## Intended Use

The model is intended for educational and demonstration purposes only, as part of an MLOps course project.
Proper intended uses include:
- Demonstrating ML pipeline construction  
- Showing inference with a deployed FastAPI API  
- Practicing CI/CD, testing, and slice-based model evaluation  
- Illustrating issues with model fairness and model monitoring  


## Training Data

The model was trained on the Adult Census(1994) dataset, containing demographic information such as:
- Age  
- Workclass  
- Education  
- Marital status  
- Occupation  
- Relationship  
- Race  
- Sex  
- Native country  
- Capital gain / loss  
- Hours worked per week  

The dataset was split using an 80/20 stratified train-test split, ensuring balanced label representation.

Categorical columns used:

- workclass  
- education  
- marital-status  
- occupation  
- relationship  
- race  
- sex  
- native-country  

The target (label) is `"salary"`.

## Evaluation Data

The evaluation dataset is the 0% held-out test split created during the stratified train-test split.  
Preprocessing uses the same encoder fitted on the training data.

The evaluation also includes slice-based performance metrics generated for every unique value of each categorical feature.  
These results are saved in `slice_output.txt`.

## Metrics

Global model performance
Using the hold-out test set, the model produced the following scores:
- Precision: ~0.7353
- Recall: ~0.6378
- F1-score: ~0.6831

## Ethical Considerations
- Risk of demographic bias:
  Because this dataset includes attributes such as race, sex, and marital status, the model may learn historical biases present in census income patterns.

- Not appropriate for real-world decisions:
  The predictions cannot be ethically used for employment, housing, lending, or other impactful decisions involving individuals.

- Data quality limitations:
  The census dataset includes missing, noisy, or outdated socioeconomic categories, which can introduce unfair patterns.



## Caveats and Recommendations
- This model is not a production-grade system and should be used only for educational purposes.

- Performance varies significantly across demographic slices; any real deployment would require extensive bias and fairness auditing.

- Additional work such as hyperparameter tuning, feature engineering, and calibration could improve model quality.

- The dataset is from the 1994 Census and may not reflect current population demographics or income patterns.