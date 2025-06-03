# Kaggle Child Mind Institute — Problematic Internet Use (PIU)
https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use

---
## 기술 스택 (Tech Stack)

Language: Python

ML Models: XGBoost, LightGBM, Catboost

Hyperparameter Tuning: Optuna

Feature Selection: SHAP

Dataframes: pandas, polars

Visualization: matplotlib, seaborn

Cross-validation: StratifiedKFold (Scikit-learn)

OS: Linux (Ubuntu 24.04 LTS)

IDE: VSCode

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
















