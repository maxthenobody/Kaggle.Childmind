# Kaggle Child Mind Institute — Problematic Internet Use (PIU)
https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use

---
## 기술 스택 (Tech Stack)

**Language**: Python
**ML Models**: XGBoost, LightGBM, Catboost

Hyperparameter Tuning: Optuna

Feature Selection: SHAP

Imputation: KNNImputer (Scikit-learn)

Dataframes: pandas, polars

Visualization: matplotlib, seaborn

Cross-validation: StratifiedKFold (Scikit-learn)

OS: Linux (Ubuntu 24.04 LTS)

IDE: VSCode, Jupyter Notebook

---

## 프로젝트 개요

**아동 청소년의 문제성 인터넷 사용 예측 모델 개발**

본 프로젝트에서는 Child Mind Institute Kaggle 경진대회에 참여하여 아동 및 청소년의 인구 통계 신체 데이터, 설문, 액티그래피 측정 기록 데이터 등를 활용하여 문제성 인터넷 사용(PIU)을 예측하는 머신러닝 모델을 개발했습니다.

단독으로 참여하였고, 은메달(상위 1.5%)을 수상하였습니다.

자세한 내용은 아래 영어로 작성된 README를 참조해주세요.

---

## 1. Project Overview

This project addresses the **Child Mind Institute Problematic Internet Use (PIU)** competition on Kaggle, focusing on developing machine learning models to predict problematic internet use in children and adolescents. The solution leverages a diverse dataset comprising demographic, behavioural and actigraphy data.

The primary goal was to build a robust classification model capable of identifying individuals at risk of PIU, contributing to early intervention strategies and improved mental health outcomes for young people.

## 2. Problem Statement

Problematic Internet Use (PIU) is a growing concern, impacting mental health and daily functioning in children and adolescents. Early identification of PIU is crucial for effective intervention. This project aimed to create a predictive model by analyzing complex features from multiple data sources such as actigraphy, demographic and behavioral data to accurately classify individuals with PIU.

## 3. Dataset

The dataset provided by the Child Mind Institute (via Kaggle) includes:

    Demographics - Information about age and sex of participants.
    Internet Use - Number of hours of using computer/internet per day.
    Children's Global Assessment Scale - Numeric scale used by mental health clinicians to rate the general functioning of youths under the age of 18.
    Physical Measures - Collection of blood pressure, heart rate, height, weight and waist, and hip measurements.
    FitnessGram Vitals and Treadmill - Measurements of cardiovascular fitness assessed using the NHANES treadmill protocol.
    FitnessGram Child - Health related physical fitness assessment measuring five different parameters including aerobic capacity, muscular strength, muscular endurance, flexibility, and body composition.
    Bio-electric Impedance Analysis - Measure of key body composition elements, including BMI, fat, muscle, and water content.
    Physical Activity Questionnaire - Information about children's participation in vigorous activities over the last 7 days.
    Sleep Disturbance Scale - Scale to categorize sleep disorders in children.
    Actigraphy - Objective measure of ecological physical activity through a research-grade biotracker.
    Parent-Child Internet Addiction Test - 20-item scale that measures characteristics and behaviors associated with compulsive use of the Internet including compulsivity, escapism, and dependency.

    ### Actigraphy Files and Field Descriptions ###
    During their participation in the HBN study, some participants were given an accelerometer to wear for up to 30 days continually while at home and going about their regular daily lives.

    series_{train|test}.parquet/id={id} - Series to be used as training data, partitioned by id. Each series is a continuous recording of accelerometer data for a single subject spanning many days.
    id - The patient identifier corresponding to the id field in train/test.csv.
    step - An integer timestep for each observation within a series.
    X, Y, Z - Measure of acceleration, in g, experienced by the wrist-worn watch along each standard axis.
    enmo - As calculated and described by the wristpy package, ENMO is the Euclidean Norm Minus One of all accelerometer signals (along each of the x-, y-, and z-axis, measured in g-force) with negative values rounded to zero. Zero values are indicative of periods of no motion. While no standard measure of acceleration exists in this space, this is one of the several commonly computed features.
    anglez - As calculated and described by the wristpy package, Angle-Z is a metric derived from individual accelerometer components and refers to the angle of the arm relative to the horizontal plane.
    non-wear_flag - A flag (0: watch is being worn, 1: the watch is not worn) to help determine periods when the watch has been removed, based on the GGIR definition, which uses the standard deviation and range of the accelerometer data.
    light - Measure of ambient light in lux. See ​​here for details.
    battery_voltage - A measure of the battery voltage in mV.
    time_of_day - Time of day representing the start of a 5s window that the data has been sampled over, with format %H:%M:%S.%9f.
    weekday - The day of the week, coded as an integer with 1 being Monday and 7 being Sunday.
    quarter - The quarter of the year, an integer from 1 to 4.
    relative_date_PCIAT - The number of days (integer) since the PCIAT test was administered (negative days indicate that the actigraphy data has been collected before the test was administered).

    * **Target Variable: Severity Impairment Index (SII)** Multiclass classification identifying the level of Problematic Internet Use (PIU).


## 4. Methodology & Approach

This project followed a comprehensive machine learning pipeline:

1.  **Data Loading & Initial Exploration:**
    * Loaded and understood the various data formats (parquet for actigraphy, CSV for tabular data).
    * Performed initial sanity checks and basic statistical analyses.

2.  **Data Preprocessing & Feature Engineering:**
    * **Actigraphy Data:**
        * Applied standard fMRI preprocessing steps (e.g., motion correction, skull stripping, normalization, spatial smoothing, time-series extraction from ROIs or voxel data).
        * Extracted meaningful features from fMRI (e.g., functional connectivity matrices, regional brain activity measures, specific ROI signals).
        * [If applicable: Describe how anatomical MRI was used, e.g., for segmentation or normalization reference.]
    * **Tabular Data:** Handled missing values, encoded categorical features, and scaled numerical features.
    * **Data Integration:** Developed strategies to combine features extracted from neuroimaging with demographic and behavioral data into a unified dataset for model training.

3.  **Model Selection & Architecture:**
    * Explored various machine learning and deep learning models suitable for classification.
    * **[If you used CNNs for fMRI]:** Implemented a Convolutional Neural Network (CNN) architecture specifically designed for processing time-series fMRI data or extracted functional connectivity matrices.
    * [If you combined models]: Discussed how tabular features were integrated (e.g., concatenated to CNN output, or trained separate models and ensemble).
    * Used `TensorFlow` and `Keras` for building and training the neural network models.

4.  **Training & Validation:**
    * Utilized appropriate cross-validation strategies to ensure model generalization.
    * Monitored training progress using metrics like AUC, accuracy, and loss.
    * Applied techniques to prevent overfitting (e.g., dropout, early stopping, L1/L2 regularization).

5.  **Model Evaluation:**
    * Evaluated the final model performance on an unseen test set using the **Area Under the Receiver Operating Characteristic Curve (AUC)**, the primary competition metric.
    * Analyzed other metrics like accuracy, precision, recall, and F1-score to gain a comprehensive understanding of model strengths and weaknesses.
    * [If applicable: Performed interpretability analyses to understand which features contributed most to predictions, e.g., using SHAP or LIME for tabular data, or saliency maps for CNNs if you managed that for images.]

## 5. Technologies Used

* **Programming Language:** Python
* **Deep Learning Frameworks:** TensorFlow, Keras
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Scientific Computing:** SciPy
* **Machine Learning Libraries:** Scikit-learn
* **Neuroimaging Specific Libraries:** NiBabel, Nilearn (If you used these for fMRI/MRI processing)
* **Data Visualization:** Matplotlib, Seaborn
* **Development Environment:** Jupyter Notebook

## 6. Results & Key Findings

* **Achieved AUC Score:** Our model achieved an AUC score of **[Your Achieved AUC Score, e.g., 0.XX]** on the validation/test set, demonstrating [e.g., 'strong predictive capabilities' or 'competitive performance'].
* **[Your Key Insight 1]:** Briefly describe an important discovery or pattern observed (e.g., "Functional connectivity features extracted from specific brain regions proved to be highly indicative of PIU.").
* **[Your Key Insight 2]:** Another finding (e.g., "The integration of demographic data significantly improved model robustness, suggesting a synergistic effect with neuroimaging features.").
* **Model Performance Visualization:**
    * ![AUC ROC Curve](images/roc_curve.png)
    * ![Training Loss & Accuracy](images/training_plot.png)
    * ![Confusion Matrix](images/confusion_matrix.png)
    * *(Self-note: Replace `images/` with the actual path to your saved plots and ensure these plots are clear and professional.)*

## 7. Conclusion & Future Work

This project successfully developed a predictive model for Problematic Internet Use, demonstrating proficiency in handling complex, multi-modal neuroimaging data and applying deep learning techniques for classification. The insights gained highlight the potential of data-driven approaches in mental health diagnostics.

**Future Enhancements could include:**
* Exploring more advanced deep learning architectures for multimodal data fusion.
* Integrating external datasets for transfer learning or data augmentation.
* Conducting more in-depth feature importance analysis to identify critical biomarkers of PIU.
* Investigating explainable AI (XAI) techniques to make model predictions more interpretable for clinical applications.

## 8. How to Run This Project

To replicate the analysis and model training:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/maxthenobody/Kaggle.Childmind.git](https://github.com/maxthenobody/Kaggle.Childmind.git)
    cd Kaggle.Childmind
    ```
2.  **Download the dataset:**
    * You will need to join the [Child Mind Institute Problematic Internet Use competition](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use) on Kaggle.
    * Download the `archive.zip` (or specific files if you only used a subset) and place it in the project root directory.
    * *(Self-note: Provide more specific instructions if the data needs unpacking or specific folder structures.)*
3.  **Create a virtual environment and install dependencies:**
    ```bash
    conda create -n childmind python=3.9 # or venv
    conda activate childmind
    pip install -r requirements.txt
    ```
    *(Self-note: Ensure you have a `requirements.txt` file listing all libraries used, e.g., `tensorflow`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `nibabel`, `nilearn`)*
4.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook "Kaggle Childmind Project.ipynb"
    ```
    Follow the steps in the notebook to execute the data processing, model training, and evaluation.

## 9. Repository Structure











