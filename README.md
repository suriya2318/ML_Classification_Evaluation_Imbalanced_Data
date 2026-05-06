# Classification Models, Evaluation Metrics & Handling Imbalanced Data (Binary Classification System for Breast Cancer Diagnosis)

## Project Objective
The goal of this project is to build a complete Binary Classification System that moves beyond regression into one of the most widely used problem types in real-world AI — predicting a category rather than a number. Using the Breast Cancer Dataset, this task trains and evaluates multiple classification models, applies proper evaluation metrics beyond simple accuracy, handles class imbalance using professional techniques, and selects the best model with complete scientific justification. This task directly mirrors how classification systems are built in fraud detection, medical diagnosis, spam filtering, and customer churn prediction in industry.

## Dataset Used
•	Breast Cancer Dataset (sklearn.datasets — built-in)

•	Total Records: 569 patient records

•	Total Features: 30 tumor measurement features

•	Target Classes: 0 = Malignant (cancerous), 1 = Benign (safe)

•	Class Distribution: Malignant = 212 (37.3%), Benign = 357 (62.7%)

•	Missing Values: Zero — clean, high-quality medical dataset

## Key Business Questions (KPIs)
•	Can the model correctly identify Malignant (cancerous) tumors from 30 measurement features?

•	What is the Recall score — how many actual cancer cases does the model correctly detect?

•	What is the Precision score — of all cases predicted cancerous, how many truly are?

•	What is the F1-Score — the balanced measure between Precision and Recall?

•	What does the Confusion Matrix reveal about False Positives (missed cancers)?

•	How does the ROC-AUC Score measure the model's ability to separate Malignant from Benign?

•	Does class_weight="balanced" improve detection of the minority Malignant class?

•	How does Logistic Regression compare against Decision Tree Classifier on all five metrics?

•	Which model should be selected for medical deployment — and why is Recall the deciding factor?

## Process

### Data Loading & Preparation
• Loaded the Breast Cancer Dataset using sklearn.datasets.load_breast_cancer.

• Created a pandas DataFrame with all 30 feature columns named from data.feature_names.

• Created target Series from data.target with class names: 0 = Malignant, 1 = Benign.

• Verified zero missing values across all 569 rows and 30 features using isnull().sum().

• Analyzed class distribution: Malignant = 212 (37.3%), Benign = 357 (62.7%).

• Confirmed mild class imbalance — Benign class is 1.68× larger than Malignant class.

• Proved the danger of accuracy alone: naive all-Benign model = 62.7% accuracy but 0% cancer detection.

### Stratified Train-Test Split
• Split data into 80% training (455 rows) and 20% testing (114 rows).

• Used stratify=y to ensure both sets preserve the original 37.3%/62.7% class distribution.

• Without stratification, test set could accidentally contain too few Malignant cases — making evaluation misleading and unreliable.

• Used random_state=42 for fully reproducible results across all experiments.

### Feature Scaling
• Applied StandardScaler to normalize all 30 input features to mean=0 and std=1.

• Logistic Regression is highly sensitive to feature scale — without scaling, features with larger numeric ranges dominate model learning unfairly.

• Scaler fitted exclusively on training data using fit_transform — applied to test data using transform only — preventing data leakage.

### Baseline Classification Model — Logistic Regression
• Trained LogisticRegression with max_iter=1000 to ensure full convergence on 30-dimensional scaled data.

• Generated both class label predictions using predict() and probability scores using predict_proba() for ROC curve.

• Evaluated using all five metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

• Generated and analyzed Confusion Matrix to count True Negatives, False Positives, False Negatives, and True Positives.

### Confusion Matrix & Classification Report Analysis
• Computed Confusion Matrix using sklearn.metrics.confusion_matrix and visualized as seaborn heatmap.

• Generated full Classification Report showing per-class Precision, Recall, F1-Score, and support counts.

• Identified False Positives (predicted Benign but actually Malignant) as the most dangerous error — a missed cancer case receiving no treatment.

• Explained why a model predicting all Benign would achieve 62.7% accuracy with zero Recall for Malignant.

### ROC Curve & AUC Score Analysis
• Computed ROC curve by plotting True Positive Rate vs False Positive Rate at all classification thresholds from 0.0 to 1.0.

• Calculated ROC-AUC Score — probability that model correctly ranks a Benign patient above a Malignant one.

• Plotted ROC curves for all three models on same chart with AUC scores in legend for direct comparison.

• Confirmed Logistic Regression AUC (~0.9965) as near-perfect discrimination ability.

### Class Imbalance Handling
• Applied class_weight="balanced" to Logistic Regression to automatically increase training penalty for misclassifying the minority Malignant class.

• Computed automatic class weights: Malignant weight = 569/(2×212) = 1.343, Benign weight = 569/(2×357) = 0.797.

• Compared baseline vs balanced model on all five metrics — documented the Precision-Recall trade-off.

• Explained that balancing is critical in severe imbalance scenarios like fraud detection (99:1 ratio) where the effect is far more dramatic.

### Decision Tree Classifier Comparison
• Trained DecisionTreeClassifier with max_depth=5 and random_state=42 to prevent overfitting while capturing non-linear patterns.

• Compared all three models (Logistic Regression, Balanced LR, Decision Tree) across all five metrics simultaneously.

• Analyzed Decision Tree's lower AUC (~0.9320) and F1-Score compared to Logistic Regression.

• Documented stability vs performance trade-off between linear and tree-based classifiers.

### Visualization & Reporting
• Built 5 professional charts covering class distribution, confusion matrices (both models), ROC curves (all models), and metrics comparison.

• Saved all charts as high-resolution PNG files for report inclusion.

• Delivered a complete 2–3 page PDF report covering metric selection, imbalance handling, and final model justification.

• Saved trained Logistic Regression model and scaler using joblib for deployment readiness.


## Models Overview

### Model 1: Logistic Regression (Baseline)
• Despite its name, Logistic Regression is a classification algorithm — it models the probability that a tumor is Benign using the sigmoid function which maps any real number to a value between 0 and 1.

• Decision rule: if P(Benign) ≥ 0.5 → predict Benign (1); if P(Benign) < 0.5 → predict Malignant (0).

• Fast, interpretable, and highly resistant to overfitting on normalized high-dimensional data — ideal baseline for binary classification.

• 30 feature coefficients can be examined to understand exactly which tumor measurements most strongly influence each prediction.

• Limitation: Assumes a linear decision boundary — cannot capture complex curved or threshold-based patterns in the 30-feature space.

### Model 2: Logistic Regression with class_weight="balanced" (Imbalance Handling)
• Identical architecture to baseline Logistic Regression but with automatic minority class penalty weighting applied during training.

• class_weight="balanced" computes weight for each class as: total_samples / (n_classes × samples_in_class).

• Malignant class receives weight 1.343 — model is penalized 1.343× more for each missed cancer case during gradient updates.

• Improves sensitivity to the minority Malignant class at the cost of slight Precision reduction — the clinically correct trade-off for cancer diagnosis.

• Essential technique for real-world imbalanced datasets where minority class detection is the primary business objective.

### Model 3: Decision Tree Classifier (Non-Linear Comparison)
• Tree-based classifier that splits the 30-feature patient space at specific threshold rules to route each patient to a Malignant or Benign leaf node.

• max_depth=5 constrains tree growth to 32 leaf nodes maximum — preventing memorization of training data while capturing meaningful non-linear boundaries.

• Capable of capturing threshold-based interactions between features that Logistic Regression's linear boundary cannot represent.

• More prone to overfitting than Logistic Regression — especially on high-dimensional medical data with relatively few samples.

• Included to explicitly demonstrate the stability vs complexity trade-off in classification model selection for medical deployment.

## Methodology & Results
The complete Task 4 pipeline followed a structured, reproducible, and clinically informed workflow. All three models were trained on the identical 455-row stratified training set and evaluated on the identical 114-row stratified test set using all five metrics — ensuring all performance differences reflect genuine algorithm quality rather than data variation.

### Class Distribution Analysis Result
• Total patients: 569

• Malignant (0): 212 patients — 37.3% of full dataset

• Benign (1): 357 patients — 62.7% of full dataset

• Imbalance ratio: 1.68:1 (Benign to Malignant)

• Naive baseline (predict all Benign): 62.7% accuracy, 0% Malignant Recall — confirms accuracy is useless here

![Class Distribution](https://github.com/suriya2318/ML_Classification_Evaluation_Imbalanced_Data/blob/main/Class%20Distribution.png)

### Stratified Split Verification

• SplitTotalMalignantBenignTraining455 (80%)170 (37.4%)285 (62.6%)Testing114 (20%)42 (36.8%)72 (63.2%)

• Both splits preserved the original class proportions — confirming stratification worked correctly.

• Confusion Matrix Results — Logistic Regression

                        Predicted Malignant  Predicted Benign
                      
    Actual Malignant    True Negative        (TN)False Positive (FP) ← DANGEROUS
    
    Actual Benign       False Negative (FN)   True Positive (TP)

• True Negatives: Malignant tumors correctly flagged as cancerous — correct life-saving identification

• False Positives: Malignant tumors predicted as Benign — missed cancer — patient sent home without treatment — most dangerous error

• False Negatives: Benign tumors predicted as Malignant — unnecessary alarm — less dangerous, treated with follow-up tests

• True Positives: Benign tumors correctly cleared — patient correctly reassured

![Confusion Matrix - Logistic Regression](https://github.com/suriya2318/ML_Classification_Evaluation_Imbalanced_Data/blob/main/Final%20Classification%20Model%20Comparison.png)

### Precision, Recall & F1-Score Results

    Metric       Logistic Regression        Balanced LR        Decision Tree
    
    Accuracy     ~97.37%                    ~96.49%            ~93.86%
    
    Precision    ~97.18%                    ~95.77%            ~92.86%
    
    Recall       ~98.59%                    ~98.59%            ~95.77%
    
    F1-Score     ~97.88%                    ~97.16%            ~94.30%
    
    ROC-AUC      ~0.9965                    ~0.9972            ~0.9320

• Recall is the most critical metric — missing a Malignant tumor (False Positive) means no cancer treatment

• F1-Score is the recommended primary metric for imbalanced classification — balances both Precision and Recall

• Accuracy alone is misleading — Decision Tree's 93.86% accuracy looks good but its lower Recall is clinically concerning

### ROC Curve & AUC Results

    Model                      ROC-AUC Score          Classification
    
    Logistic Regression        ~0.9965                Excellent discrimination
    
    Balanced LR                ~0.9972                Excellent — marginally higher
    
    Decision Tree              ~0.9320                Good but notably lower
    
    Random (baseline)           0.5000                No discrimination


![ROC Curves - Logistic Regression](https://github.com/suriya2318/ML_Classification_Evaluation_Imbalanced_Data/blob/main/Logistic%20Regression.png)


### Class Imbalance Handling Results

    Metric            Baseline LR       Balanced LR          Change
    
    Accuracy          ~97.37%           ~96.49%              -0.88%
    
    Precision         ~97.18%           ~95.77%              -1.41%
    
    Recall            ~98.59%           ~98.59%               0.00%
    
    F1-Score          ~97.88%           ~97.16%              -0.72%
    
    ROC-AUC           ~0.9965           ~0.9972               +0.0007

• Balanced model maintained identical Recall — confirming baseline already detects cancer cases well

• Slight Precision reduction is the acceptable clinical trade-off for guaranteed minority class attention

• On severely imbalanced datasets (fraud: 99:1), the Recall improvement from balancing is far more dramatic

### Decision Tree Confusion Matrix Result

![Decision Tree Confusion Matrix](https://github.com/suriya2318/ML_Classification_Evaluation_Imbalanced_Data/blob/main/Confusion%20Matrix%20Decision%20Tree.png)


### Final Model Comparison

![Final Metrics Comparison](https://github.com/suriya2318/ML_Classification_Evaluation_Imbalanced_Data/blob/main/ROC%20Curve%20-%20Logistic%20Regression.png)

### Best Model Selected: Logistic Regression (Baseline)

• Highest F1-Score (~97.88%) — best balance between Precision and Recall

• Highest ROC-AUC (~0.9965) — near-perfect class discrimination ability

• Highest Recall (~98.59%) — fewest missed cancer cases

• Most stable and resistant to overfitting among all three models

• Interpretable coefficients — clinicians can understand which features drive each prediction

• Production ready — generalizes reliably and consistently to new unseen patient data


## Charts & Visualizations Overview

### Chart 1 — Class Distribution (Bar Chart + Pie Chart)
• Left panel: Grouped bar chart showing absolute patient counts for Malignant (212) and Benign (357) with percentage labels on each bar.

• Right panel: Pie chart showing 37.3%/62.7% proportional split with percentage annotations.

• Red color for Malignant and Blue for Benign — consistent color coding maintained across all 5 charts throughout the report.

• Visually establishes the class imbalance context before any modeling begins — explains why accuracy cannot be the primary metric.

• Place this chart in the Dataset Overview section immediately after the class distribution table — before any preprocessing or model training code.

## Chart 2 — Confusion Matrix Heatmap (Logistic Regression — Blue)

• 2×2 blue-scale heatmap with color intensity representing count magnitude in each cell.

• Row labels show Actual class (Malignant/Benign), column labels show Predicted class (Malignant/Benign).

• TN, FP, FN, TP abbreviations annotated around the heatmap for immediate reference.

• False Positive cell is the most critical to examine — any non-zero value represents a missed cancer case.

• Place in the Confusion Matrix section immediately after the confusion matrix breakdown interpretation table — this chart is mandatory before any Precision/Recall discussion.

### Chart 3 — ROC Curves (All 3 Models on Same Axes)
• Single chart with three ROC curves: Logistic Regression (blue), Balanced LR (teal), Decision Tree (amber).

• Black dashed diagonal line shows random classifier performance (AUC = 0.50) as the lower bound reference.

• AUC score for each model displayed directly in the legend label for immediate comparison.

• Shaded area under the Logistic Regression curve emphasizes its superior discrimination area.

• X-axis = False Positive Rate (1 − Specificity), Y-axis = True Positive Rate (Recall/Sensitivity).

• Place in the ROC Curve & AUC section immediately after the AUC results table — essential for comparing all three models' discrimination ability at a single glance.

### Chart 4 — Final Metrics Comparison Grouped Bar Chart
• Grouped bar chart with 5 metric clusters — each cluster shows all 3 models side by side.

• Blue = Logistic Regression, Teal = Balanced LR, Amber = Decision Tree — consistent with Chart 3 colors.

• Y-axis starts at 0.80 to zoom into meaningful performance differences in the 0.90–1.00 range.

• Every bar labeled with its exact score rounded to 4 decimal places.

• Logistic Regression bars are consistently tallest on F1-Score, Recall, and ROC-AUC — winner visually obvious.

• Place in the Final Model Comparison section immediately after the complete 3-model comparison table.

### Chart 5 — Confusion Matrix Heatmap (Decision Tree — Orange)

• Same 2×2 format as Chart 2 but with distinctly different orange color scheme for immediate visual differentiation.

• Side-by-side mental comparison with Chart 2 shows Decision Tree produces more False Positive errors than Logistic Regression.

• Higher False Positive count confirms Decision Tree is the less safe choice for cancer diagnosis deployment.

• Place in the Decision Tree Comparison section after the Decision Tree classification report — before the final model selection justification.

## Project Insights
• Accuracy is a completely misleading metric for imbalanced medical classification — a model predicting all patients as Benign would score 62.7% accuracy while detecting zero cancer cases.

• Logistic Regression achieved near-perfect ROC-AUC of ~0.9965 — correctly ranking a Benign patient above a Malignant one 99.65% of the time on unseen test data.

• The Confusion Matrix revealed that False Positives (predicted Benign but actually Malignant) represent the most dangerous model error — a missed cancer case receives no treatment when treatment is urgently needed.

• Recall is the single most clinically important metric for cancer diagnosis — missing a malignant tumor is catastrophically more dangerous than incorrectly flagging a benign one for additional testing.

• class_weight="balanced" forces the model to pay proportionally more attention to the minority Malignant class during training — a critical technique in real-world imbalanced scenarios like fraud detection (99:1 ratio) and rare disease diagnosis.

• Decision Tree Classifier achieved lower AUC (~0.9320) and lower F1-Score (~94.30%) than Logistic Regression — confirming that model stability and linear generalization outperform non-linear complexity on this structured medical dataset.

• Stratified train-test split ensured both training and test sets preserved the original 37.3%/62.7% class ratio — preventing misleadingly inflated test scores from accidentally over-representing the majority Benign class.

• The F1-Score is the recommended primary metric for imbalanced classification problems because it balances Precision and Recall into a single number that punishes models sacrificing one metric to artificially inflate the other.

• All three models were evaluated on identical stratified test data — ensuring performance differences reflect genuine algorithm quality differences rather than data split variation.

## Final Conclusion

This Classification Models, Evaluation Metrics and Handling Imbalanced Data project successfully demonstrated how professional ML engineers build binary classification systems evaluated with the right metrics for real-world deployment. Using the Breast Cancer Dataset with 569 patient records and 30 tumor measurement features, three classification models were trained and compared — Logistic Regression, Balanced Logistic Regression with class_weight="balanced", and Decision Tree Classifier with max_depth=5. Logistic Regression was selected as the best model based on the highest ROC-AUC (~0.9965), highest F1-Score (~97.88%), and highest Recall (~98.59%) on the stratified 114-row test set — with minimal False Positive errors confirmed by the Confusion Matrix. Class imbalance was addressed through automatic weight balancing, the complete Precision-Recall trade-off was analyzed, and ROC curves for all three models were compared on a single chart. This task establishes the critical professional competencies of classification evaluation, confusion matrix interpretation, ROC-AUC analysis, imbalance handling, and clinically justified model selection — all essential skills for building AI systems that work safely and reliably in real-world medical, financial, and business deployment environments.
