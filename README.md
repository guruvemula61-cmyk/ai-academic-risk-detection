# AI-Based Early Detection of Students at Academic Risk

## Overview

This project is an AI-powered early-warning and intervention system designed to identify students who may be moving toward academic risk before poor performance becomes a serious problem.

The system analyzes multiple academic signals including attendance, assessment performance, assignment completion, engagement, and concept mastery to identify emerging risk patterns.

Instead of simply showing marks, the system follows a complete workflow:

**Detect → Explain → Intervene → Track**

## Problem

Academic difficulties often develop gradually. Declining assessment scores, irregular attendance, incomplete assignments, reduced engagement, and learning gaps can appear weeks before a student experiences serious academic difficulty.

In large classrooms, continuously identifying these patterns manually is difficult.

This project aims to provide teachers with an intelligent early-warning mechanism that helps them identify emerging risk and take timely action.

## Solution

The system uses a machine-learning model to generate an academic risk score for each student based on multiple learning signals.

It then provides:

- Class-level risk monitoring
- Student-level risk analysis
- Risk-factor explanations
- Learning trajectory visualization
- Historical Time Travel analysis
- Personalized intervention recommendations
- Parent communication support
- Progress monitoring

## Key Features

### 1. Academic Risk Detection

The system analyzes student learning signals and generates an estimated academic risk score.

### 2. Explainable Risk Analysis

Teachers can see the major factors associated with a student's current risk level instead of receiving only a numerical prediction.

### 3. Time Travel

The system reconstructs the student's academic journey across multiple weeks, helping teachers identify when warning signals started increasing.

### 4. Personalized Intervention

The system generates targeted support recommendations based on the student's weakest academic signals.

### 5. Parent Communication

Supportive parent communication can be generated in English or Telugu.

### 6. Class Risk Dashboard

Teachers can view the overall classroom risk distribution and identify students who may require additional support.

## Technology Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Plotly

## Machine Learning

The prototype uses Logistic Regression to demonstrate explainable academic-risk classification.

The model considers:

- Attendance
- Quiz performance
- Assignment completion
- Engagement
- Concept mastery

The current prototype uses synthetic classroom data for demonstration.

## Workflow

```text
Student Learning Signals
          ↓
   Data Processing
          ↓
   Risk Prediction
          ↓
   Risk Explanation
          ↓
 Intervention Recommendation
          ↓
   Progress Tracking
