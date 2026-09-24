# AI-Based Early Detection of Students at Academic Risk

An explainable AI-powered early-warning and intervention system that helps educators identify students showing emerging academic risk, understand the likely contributing factors, recommend targeted support, and track changes over time.

## 🚀 The Problem

Academic difficulties are often identified only after a student has already experienced repeated low scores, missed assignments, poor attendance, or declining engagement.

Traditional monitoring systems usually answer:

> "Which students are performing poorly?"

But educators need a more useful question:

> "Which students are starting to show warning signals, why are those signals appearing, and what support can be provided early?"

This project addresses that gap through an AI-assisted early-warning workflow.

---

## 💡 Our Approach

The system follows a four-stage workflow:

### DETECT → EXPLAIN → INTERVENE → TRACK

**1. Detect**

Analyze multiple learning signals such as:

- Attendance
- Assessment performance
- Assignment completion
- Learning engagement
- Concept mastery
- Performance trends

**2. Explain**

Instead of showing only a risk score, the system identifies the likely pattern behind the risk.

Examples:

- Academic Difficulty
- Disengagement
- Sudden Decline
- Chronic Struggler

**3. Intervene**

Generate a targeted support recommendation based on the identified pattern.

Examples:

- Targeted topic revision
- Teacher/mentor check-in
- Priority academic review
- Individualized long-term support

**4. Track**

Monitor the student's trajectory over time and observe whether risk signals are improving or worsening.

---

# 🌟 Key Features

## 1. Explainable Academic Risk Detection

The system combines multiple student learning signals and generates an academic risk score.

Risk levels are presented as:

- 🟢 Low Risk
- 🟡 Moderate Risk
- 🔴 High Risk

The goal is not to label students, but to provide an early signal for educator attention.

---

## 2. Root-Cause Classification

A major feature of the system is identifying the likely pattern behind academic risk.

### Academic Difficulty
The student's academic performance and concept mastery show persistent weakness.

### Disengagement
Attendance, assignments, or learning engagement show deterioration.

### Sudden Decline
The student previously showed stable performance but recently experienced a sharp decline.

### Chronic Struggler
The student demonstrates consistently weak performance over an extended period.

This converts a simple risk score into an actionable explanation.

---

## 3. Trend Velocity

A student's current score alone may not tell the complete story.

The system also analyzes the direction and rate of change in the student's risk trajectory.

For example:

> "Risk is increasing rapidly."

This helps educators identify students whose situation may require attention before they cross a critical risk level.

---

## 4. Explainable Risk Card

For every selected student, the system presents:

- Current risk score
- Risk level
- Root cause
- Trend direction
- Signal changes
- Supporting evidence
- Recommended action

This makes the AI output easier for educators to understand and act upon.

---

## 5. Early Risk Forecast

The system examines recent risk trends and estimates when a student may approach a higher-risk threshold if the current trajectory continues.

This is intended as an early-warning signal, not a guaranteed prediction.

---

## 6. Time Travel Analysis

### "When did the risk actually start?"

The Time Travel feature replays historical weekly learning data and visualizes how the student's risk changed over time.

Educators can identify:

- The week risk began increasing
- Sudden changes
- Persistent deterioration
- Periods of recovery
- Week-over-week changes

This shifts the focus from:

> "The student is at risk."

to:

> "When did the warning signals begin?"

---

## 7. What-If Simulator

Educators can simulate changes in learning signals such as:

- Attendance
- Assessment performance
- Assignment completion
- Engagement
- Concept mastery

The system then compares the current risk with the simulated scenario.

Example:

> What happens to the estimated risk if attendance improves?

This helps demonstrate how different learning signals can influence the overall risk assessment.

**Note:** The simulator is a scenario-analysis tool and does not guarantee future outcomes.

---

## 8. Personalized Intervention Engine

The system maps detected risk patterns to suggested support actions.

| Risk Pattern | Suggested Support |
|---|---|
| Academic Difficulty | Targeted topic revision |
| Disengagement | Teacher / mentor check-in |
| Sudden Decline | Priority academic review |
| Chronic Struggler | Individualized long-term support |

The objective is to move from detection to meaningful action.

---

## 9. Intervention Center

The Intervention Center provides educators with a consolidated view of:

- Detected root cause
- Current risk
- Initial risk
- Risk trajectory
- Recommended support plan
- Parent communication draft

This creates a practical workflow for educator intervention.

---

## 10. Parent Communication Support

The system can generate a supportive communication message for parents in:

- English
- Telugu

The communication focuses on providing support rather than publicly labeling or ranking students.

---

# 🧠 AI / Machine Learning

The prototype uses machine learning to estimate academic risk from multiple learning signals.

### Current Model

**Logistic Regression**

The model provides a simple and interpretable baseline for the prototype.

### Input Signals

- Attendance
- Assessment score
- Assignment completion
- Engagement
- Concept mastery

### Additional Analytics

The system also uses:

- Trend analysis
- Risk trajectory analysis
- Rule-based root-cause classification
- Scenario simulation
- Threshold forecasting

The architecture can later be extended with more advanced models and real institutional data.

---

# 📊 Demonstration Data

This prototype uses **synthetic classroom data**.

The synthetic dataset contains multiple student learning patterns designed to demonstrate different academic-risk scenarios.

This allows the system to be demonstrated without exposing real student information.

For real-world deployment, the system would require:

- Institution-approved data
- Appropriate privacy controls
- Validated indicators
- Local model evaluation
- Human oversight

---

# 🎯 Why This Approach Matters

Most academic dashboards focus heavily on historical performance.

This project focuses on **early signals and actionable intervention**.

Instead of only asking:

**"Who is performing poorly?"**

the system attempts to answer:

**"Who is showing emerging warning signals?"**

**"What pattern is associated with the warning?"**

**"What support could be considered?"**

**"Is the student's trajectory changing over time?"**

This creates an early-warning workflow rather than a simple performance dashboard.

---

# 🏗️ System Workflow

```text
Student Learning Signals
        │
        ▼
┌───────────────────────────┐
│ Data Processing           │
│ Attendance                │
│ Assessments               │
│ Assignments               │
│ Engagement                │
│ Concept Mastery           │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│ AI Risk Assessment        │
│ Logistic Regression       │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│ Explainability Layer      │
│ Risk + Trends + Signals   │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│ Root-Cause Classification │
│ Difficulty / Disengage    │
│ Sudden Decline / Chronic  │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│ Intervention Engine       │
│ Personalized Support      │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│ Tracking & Monitoring     │
│ Risk Trajectory           │
│ Time Travel               │
│ What-If Analysis          │
└───────────────────────────┘
