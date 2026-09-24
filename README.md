# AI-Based Early Detection of Students at Academic Risk

An explainable AI-powered early-warning and intervention system that helps educators identify students showing emerging academic risk, understand the likely contributing factors, recommend targeted support, and track changes over time.

---

## 1. The Problem

Academic difficulties are often identified only after a student has already experienced repeated low scores, missed assignments, poor attendance, or declining engagement.

Traditional monitoring systems usually answer:

> "Which students are performing poorly?"

But educators need a more useful question:

> "Which students are starting to show warning signals, why are those signals appearing, and what support can be provided early?"

This project addresses that gap through an AI-assisted early-warning workflow.

---

## 2. Our Approach

The system follows a four-stage workflow:

**DETECT → EXPLAIN → INTERVENE → TRACK**

### Detect

The system analyzes multiple learning signals such as:

- Attendance
- Assessment performance
- Assignment completion
- Learning engagement
- Concept mastery
- Performance trends

### Explain

Instead of showing only a risk score, the system identifies the likely pattern behind the risk.

The prototype identifies four major patterns:

- Academic Difficulty
- Disengagement
- Sudden Decline
- Chronic Struggler

### Intervene

The system generates a targeted support recommendation based on the identified pattern.

Examples include:

- Targeted topic revision
- Teacher or mentor check-in
- Priority academic review
- Individualized long-term support

### Track

The student's trajectory is monitored over time to observe whether risk signals are improving or worsening.

---

## 3. Key Features

### 3.1 Explainable Academic Risk Detection

The system combines multiple student learning signals and generates an academic risk score.

Risk levels are presented as:

- Low Risk
- Moderate Risk
- High Risk

The objective is not to label students, but to provide an early signal for educator attention.

---

### 3.2 Root-Cause Classification

A major feature of the system is identifying the likely pattern behind academic risk.

#### Academic Difficulty

The student's academic performance and concept mastery show persistent weakness.

#### Disengagement

Attendance, assignments, or learning engagement show deterioration.

#### Sudden Decline

The student previously showed stable performance but recently experienced a sharp decline.

#### Chronic Struggler

The student demonstrates consistently weak performance over an extended period.

This converts a simple risk score into an actionable explanation.

---

### 3.3 Trend Velocity

A student's current risk score alone may not tell the complete story.

The system analyzes the direction and rate of change in the student's risk trajectory.

For example:

> "Risk is increasing rapidly."

This helps educators identify students whose situation may require attention before they cross a critical risk level.

---

### 3.4 Explainable Risk Card

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

### 3.5 Early Risk Forecast

The system examines recent risk trends and estimates when a student may approach a higher-risk threshold if the current trajectory continues.

This is intended as an early-warning signal, not a guaranteed prediction.

---

### 3.6 Time Travel Analysis

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

### 3.7 What-If Simulator

Educators can simulate changes in learning signals such as:

- Attendance
- Assessment performance
- Assignment completion
- Engagement
- Concept mastery

The system then compares the current risk with the simulated scenario.

Example:

> "What happens to the estimated risk if attendance improves?"

This helps demonstrate how different learning signals can influence the overall risk assessment.

The simulator is a scenario-analysis tool and does not guarantee future outcomes.

---

### 3.8 Personalized Intervention Engine

The system maps detected risk patterns to suggested support actions.

| Risk Pattern | Suggested Support |
|---|---|
| Academic Difficulty | Targeted topic revision |
| Disengagement | Teacher or mentor check-in |
| Sudden Decline | Priority academic review |
| Chronic Struggler | Individualized long-term support |

The objective is to move from detection to meaningful action.

---

### 3.9 Intervention Center

The Intervention Center provides educators with a consolidated view of:

- Detected root cause
- Current risk
- Initial risk
- Risk trajectory
- Recommended support plan
- Parent communication draft

This creates a practical workflow for educator intervention.

---

### 3.10 Parent Communication Support

The system can generate a supportive communication message for parents in:

- English
- Telugu

The communication focuses on providing support rather than publicly labeling or ranking students.

---

## 4. AI and Machine Learning

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

## 5. Demonstration Data

This prototype uses synthetic classroom data.

The synthetic dataset contains multiple student learning patterns designed to demonstrate different academic-risk scenarios.

This allows the system to be demonstrated without exposing real student information.

For real-world deployment, the system would require:

- Institution-approved data
- Appropriate privacy controls
- Validated indicators
- Local model evaluation
- Human oversight

---

## 6. Why This Approach Matters

Most academic dashboards focus heavily on historical performance.

This project focuses on early signals and actionable intervention.

Instead of only asking:

> "Who is performing poorly?"

the system attempts to answer:

> "Who is showing emerging warning signals?"

> "What pattern is associated with the warning?"

> "What support could be considered?"

> "Is the student's trajectory changing over time?"

This creates an early-warning workflow rather than a simple performance dashboard.

---

## 7. System Workflow

```text
Student Learning Signals
        |
        v
+---------------------------+
| Data Processing           |
| Attendance                |
| Assessments               |
| Assignments               |
| Engagement                |
| Concept Mastery           |
+-------------+-------------+
              |
              v
+---------------------------+
| AI Risk Assessment        |
| Logistic Regression       |
+-------------+-------------+
              |
              v
+---------------------------+
| Explainability Layer      |
| Risk + Trends + Signals   |
+-------------+-------------+
              |
              v
+---------------------------+
| Root-Cause Classification |
| Difficulty / Disengage    |
| Sudden Decline / Chronic  |
+-------------+-------------+
              |
              v
+---------------------------+
| Intervention Engine       |
| Personalized Support      |
+-------------+-------------+
              |
              v
+---------------------------+
| Tracking and Monitoring   |
| Risk Trajectory           |
| Time Travel               |
| What-If Analysis          |
+---------------------------+
