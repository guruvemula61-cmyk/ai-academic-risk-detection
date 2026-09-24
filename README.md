# AI-Based Early Detection of Students at Academic Risk

An explainable early-warning and intervention system designed to help teachers identify emerging academic risk, understand the likely contributing factors, provide targeted support, and track student progress over time.

## 1. Problem Statement

Students may begin showing signs of academic difficulty before the problem becomes clearly visible through final examination results.

Changes in:

- Attendance
- Quiz performance
- Assignment completion
- Learning engagement
- Academic trends

can provide useful early signals.

The challenge is to transform these signals into an understandable early-warning system that helps teachers take timely, targeted action.

## 2. Proposed Solution

**AI-Based Early Detection of Students at Academic Risk** provides a teacher-focused workflow:

**DETECT → EXPLAIN → INTERVENE → TRACK**

The system analyzes synthetic weekly student records and produces an interpretable risk score.

Instead of showing only a prediction, the application explains:

- What changed
- Which factors contributed to the risk
- When the risk started increasing
- What support can be provided
- Whether the student's risk trajectory is improving

The system is designed as a decision-support prototype with human oversight.

---

# 3. Five-Screen Application Workflow

## Screen 1 — Login

A simple demo entry screen with two options:

- Enter as Teacher
- Enter as Student

No real authentication, signup, password, OTP, or account management is implemented.

The application is designed for demonstration purposes.

---

## Screen 2 — Teacher Dashboard

The Teacher Dashboard provides a class-level overview.

### Key information

- Total students
- Students needing support
- Recovering students
- Average attendance

### Class Risk Heatmap

The heatmap displays student risk across eight weeks.

Each cell represents the student's risk level for a specific week.

Risk categories:

| Risk Score | Status |
|---|---|
| Below 35 | On Track |
| 35–54 | Watch |
| 55–69 | Monitor Closely |
| 70 and above | Needs Support |

This allows teachers to identify patterns instead of looking only at the latest score.

### Needs Support List

The dashboard highlights students requiring attention with:

- Student name
- Risk score
- Main contributing reason
- Risk trend
- Profile access

---

# 4. Screen 3 — Student Profile

The Student Profile provides an individual-level explanation of academic risk.

## Risk Gauge

The current risk score is displayed on a 0–100 scale.

The application uses the student-friendly status:

**Needs Support**

instead of labeling a student as failing.

## Top Risk Factors

The application displays the main contributing factors, such as:

- Attendance dropped
- Quiz trend
- Assignment gap

These factors help explain why the current risk score changed.

## Plain-Language Explanation

The system converts numerical signals into a simple explanation.

Example:

> Attendance has declined during the recent weeks and quiz performance is also showing a downward trend. The recent assignment gap is contributing additional risk.

The purpose is to make the output understandable to teachers without requiring them to interpret raw model values.

---

# 5. Trend Analysis

The Student Profile includes eight-week trend analysis for:

- Attendance
- Quiz performance
- Academic risk

The charts help identify changes over time rather than relying only on a single observation.

This is important because academic risk is often better understood as a trajectory.

---

# 6. Time Travel Analysis

The Time Travel feature allows the teacher to move through the student's academic history from:

**Week 1 → Week 8**

As the selected week changes, the application updates the student's:

- Risk score
- Risk status
- Attendance
- Quiz performance
- Assignment status
- Historical context

The interface also identifies the point at which the student was first flagged in the demonstration dataset.

Example:

**Week 4 — Flagged**

This creates a clear visual story of how academic risk emerged.

---

# 7. Screen 4 — Intervention

Once a student is identified as needing support, the teacher can generate a support plan.

## Personalized Support Plan

The prototype generates a structured intervention containing:

### 1. Revision Plan

Suggested academic actions based on the student's observed risk factors.

### 2. Parent Communication

A Telugu parent communication message is generated to help the teacher communicate the concern clearly and respectfully.

The message focuses on:

- The observed academic change
- The student's current situation
- Suggested support
- A collaborative approach

### 3. Teacher Checklist

The teacher can track actions such as:

- Talk to student
- Call parent
- Assign revision plan
- Mark action as completed

The goal is to connect detection with an actionable support workflow.

---

# 8. Parent Voice Message Prototype

The Intervention screen includes a voice-message interface for Telugu parent communication.

For the current prototype, the displayed transcript represents the intended communication content.

If text-to-speech is not connected, the interface clearly identifies the feature as a prototype rather than presenting simulated audio as a real communication.

---

# 9. Screen 5 — Track

The Track screen follows the student's risk trajectory after support.

The prototype can demonstrate a simulated recovery scenario where a student's risk changes over time.

Example:

**Risk: 72 → 54**

Status:

**Recovering**

This recovery scenario is based on synthetic demonstration data.

It does not claim that the intervention itself caused the improvement.

The purpose is to demonstrate how the system could support continuous monitoring.

---

# 10. Risk Detection Engine

The current prototype uses an interpretable weighted rule-based scoring system rather than a complex machine-learning training pipeline.

The risk score considers four major components:

| Signal | Weight |
|---|---:|
| Attendance trend | 35% |
| Quiz score trend | 30% |
| Assignment missing streak | 20% |
| Overall attendance level | 15% |

The resulting score is normalized to a 0–100 risk scale.

Higher scores indicate stronger signals that the student may require attention.

The system is intended for early-warning support, not final academic judgment.

---

# 11. Risk Categories

The prototype uses the following score ranges:

| Score Range | Status |
|---|---|
| < 35 | On Track |
| 35–54 | Watch |
| 55–69 | Monitor Closely |
| ≥ 70 | Needs Support |

These thresholds are part of the demonstration prototype and should be validated and adapted before use with real institutional data.

---

# 12. Synthetic Dataset

The application uses synthetic student records for demonstration.

The prototype contains:

- 60 students
- 8 weeks of academic data

Each weekly record includes:

- Student ID
- Week number
- Attendance percentage
- Quiz score
- Assignment submission status
- LMS login activity

## Student Personas

The synthetic dataset contains different learning patterns:

- 15 steady toppers
- 22 stable middle-performing students
- 8 gradual decliners
- 5 sudden droppers
- 5 chronic absentee patterns
- 3 recovering students
- 2 fluctuating students

These personas are intentionally created to demonstrate different academic trajectories.

The recovering-student pattern is particularly useful for demonstrating the Track workflow.

---

# 13. Explainability

A central design principle of the application is:

**Do not show only a risk score. Explain the signals behind it.**

The application provides:

- Risk score
- Risk status
- Contributing factors
- Trend information
- Historical timeline
- Plain-language explanation
- Suggested support actions

This helps teachers understand why a student was flagged and what information should be considered before taking action.

---

# 14. Application Architecture

The prototype follows this workflow:

```text
Student Weekly Data
        |
        v
Risk Signal Analysis
        |
        v
Weighted Risk Score
        |
        v
Risk Classification
        |
        +------------------+
        |                  |
        v                  v
Explain Risk         Track Trend
        |
        v
Generate Support Plan
        |
        v
Teacher Actions
        |
        v
Monitor Recovery
