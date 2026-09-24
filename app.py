import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI-Based Early Detection of Students at Academic Risk",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background: #f7f9fc;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: #0f172a;
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

/* Headers */

.hero-title {
    font-size: 34px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 4px;
}

.hero-subtitle {
    color: #64748b;
    font-size: 15px;
    margin-bottom: 24px;
}

.section-title {
    font-size: 23px;
    font-weight: 800;
    color: #0f172a;
    margin-top: 15px;
    margin-bottom: 12px;
}

/* Cards */

.metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 20px;
    min-height: 130px;
    box-shadow: 0 4px 14px rgba(15,23,42,0.04);
}

.metric-label {
    font-size: 13px;
    color: #64748b;
    font-weight: 600;
}

.metric-value {
    font-size: 31px;
    font-weight: 800;
    color: #0f172a;
    margin-top: 7px;
}

.metric-small {
    font-size: 12px;
    color: #64748b;
    margin-top: 5px;
}

/* Status */

.status-danger {
    background: #fef2f2;
    color: #b91c1c;
    border: 1px solid #fecaca;
    padding: 7px 12px;
    border-radius: 999px;
    font-weight: 700;
    display: inline-block;
}

.status-warning {
    background: #fffbeb;
    color: #b45309;
    border: 1px solid #fde68a;
    padding: 7px 12px;
    border-radius: 999px;
    font-weight: 700;
    display: inline-block;
}

.status-success {
    background: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
    padding: 7px 12px;
    border-radius: 999px;
    font-weight: 700;
    display: inline-block;
}

.status-neutral {
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    padding: 7px 12px;
    border-radius: 999px;
    font-weight: 700;
    display: inline-block;
}

/* Info cards */

.info-box {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
}

.info-title {
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 7px;
}

.info-text {
    color: #475569;
    font-size: 14px;
    line-height: 1.6;
}

/* Login */

.login-wrapper {
    max-width: 760px;
    margin: 60px auto 0 auto;
    text-align: center;
}

.login-title {
    font-size: 38px;
    font-weight: 800;
    color: #0f172a;
}

.login-subtitle {
    font-size: 16px;
    color: #64748b;
    margin-top: 10px;
    margin-bottom: 30px;
}

.role-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
    min-height: 180px;
    box-shadow: 0 5px 18px rgba(15,23,42,0.05);
}

/* Tables */

.dataframe {
    border-radius: 12px;
}

/* Timeline */

.timeline-card {
    background: white;
    border-left: 5px solid #2563eb;
    padding: 15px 18px;
    margin-bottom: 10px;
    border-radius: 10px;
}

/* Footer */

.small-note {
    font-size: 12px;
    color: #64748b;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = None

if "selected_student" not in st.session_state:
    st.session_state.selected_student = "STU-001"

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "selected_subject" not in st.session_state:
    st.session_state.selected_subject = "Mathematics"

if "parent_message_sent" not in st.session_state:
    st.session_state.parent_message_sent = False


# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================

@st.cache_data
def create_student_data():

    students = [
        ("STU-001", "Aarav Kumar", "VIII-A", "Gradual Decliner"),
        ("STU-002", "Ananya Reddy", "VIII-A", "Steady"),
        ("STU-003", "Rahul Verma", "VIII-A", "Sudden Drop"),
        ("STU-004", "Saanvi Rao", "VIII-A", "Steady"),
        ("STU-005", "Arjun Singh", "VIII-A", "Chronic Absentee"),
        ("STU-006", "Diya Sharma", "VIII-A", "Recovering"),
        ("STU-007", "Vihaan Patel", "VIII-A", "Steady"),
        ("STU-008", "Myra Das", "VIII-A", "Gradual Decliner"),
        ("STU-009", "Aditya Kumar", "VIII-A", "Steady"),
        ("STU-010", "Ishita Rao", "VIII-A", "Sudden Drop"),
        ("STU-011", "Rohan Gupta", "VIII-A", "Steady"),
        ("STU-012", "Aadhya Singh", "VIII-A", "Chronic Absentee"),
        ("STU-013", "Kabir Reddy", "VIII-A", "Gradual Decliner"),
        ("STU-014", "Meera Patel", "VIII-A", "Steady"),
        ("STU-015", "Ayaan Khan", "VIII-A", "Recovering"),
        ("STU-016", "Sara Verma", "VIII-A", "Steady"),
        ("STU-017", "Nikhil Rao", "VIII-A", "Sudden Drop"),
        ("STU-018", "Anika Sharma", "VIII-A", "Steady"),
        ("STU-019", "Ishaan Kumar", "VIII-A", "Gradual Decliner"),
        ("STU-020", "Kiara Das", "VIII-A", "Steady"),
        ("STU-021", "Dev Reddy", "VIII-A", "Chronic Absentee"),
        ("STU-022", "Navya Singh", "VIII-A", "Steady"),
        ("STU-023", "Aryan Patel", "VIII-A", "Gradual Decliner"),
        ("STU-024", "Tara Rao", "VIII-A", "Steady"),
        ("STU-025", "Reyansh Kumar", "VIII-A", "Sudden Drop"),
        ("STU-026", "Ira Verma", "VIII-A", "Steady"),
        ("STU-027", "Karthik Sharma", "VIII-A", "Gradual Decliner"),
        ("STU-028", "Riya Reddy", "VIII-A", "Steady"),
        ("STU-029", "Manav Gupta", "VIII-A", "Recovering"),
        ("STU-030", "Avni Singh", "VIII-A", "Steady"),
        ("STU-031", "Siddharth Rao", "VIII-A", "Gradual Decliner"),
        ("STU-032", "Aanya Patel", "VIII-A", "Steady"),
        ("STU-033", "Ritvik Kumar", "VIII-A", "Sudden Drop"),
        ("STU-034", "Shreya Sharma", "VIII-A", "Steady"),
        ("STU-035", "Yash Verma", "VIII-A", "Chronic Absentee"),
        ("STU-036", "Pihu Reddy", "VIII-A", "Steady"),
        ("STU-037", "Atharv Singh", "VIII-A", "Gradual Decliner"),
        ("STU-038", "Nandini Rao", "VIII-A", "Steady"),
        ("STU-039", "Krishna Patel", "VIII-A", "Sudden Drop"),
        ("STU-040", "Mahi Kumar", "VIII-A", "Steady"),
        ("STU-041", "Varun Das", "VIII-A", "Gradual Decliner"),
        ("STU-042", "Aarohi Sharma", "VIII-A", "Steady"),
        ("STU-043", "Harsh Reddy", "VIII-A", "Chronic Absentee"),
        ("STU-044", "Ishani Verma", "VIII-A", "Steady"),
        ("STU-045", "Rudra Singh", "VIII-A", "Gradual Decliner"),
        ("STU-046", "Tanvi Rao", "VIII-A", "Steady"),
        ("STU-047", "Advik Patel", "VIII-A", "Sudden Drop"),
        ("STU-048", "Riya Kumar", "VIII-A", "Steady"),
        ("STU-049", "Om Sharma", "VIII-A", "Gradual Decliner"),
        ("STU-050", "Kavya Reddy", "VIII-A", "Steady"),
        ("STU-051", "Dhruv Singh", "VIII-A", "Chronic Absentee"),
        ("STU-052", "Prisha Rao", "VIII-A", "Steady"),
        ("STU-053", "Atharv Kumar", "VIII-A", "Recovering"),
        ("STU-054", "Nitya Patel", "VIII-A", "Steady"),
        ("STU-055", "Samar Verma", "VIII-A", "Gradual Decliner"),
        ("STU-056", "Aditi Sharma", "VIII-A", "Steady"),
        ("STU-057", "Vivaan Reddy", "VIII-A", "Sudden Drop"),
        ("STU-058", "Manya Singh", "VIII-A", "Steady"),
        ("STU-059", "Arnav Rao", "VIII-A", "Gradual Decliner"),
        ("STU-060", "Siya Patel", "VIII-A", "Steady"),
    ]

    rng = np.random.default_rng(42)

    rows = []

    for sid, name, cls, persona in students:

        if persona == "Steady":
            base_att = rng.uniform(84, 96)
            base_quiz = rng.uniform(76, 92)
            att_slope = rng.uniform(-0.3, 0.3)
            quiz_slope = rng.uniform(-0.2, 0.3)

        elif persona == "Gradual Decliner":
            base_att = rng.uniform(82, 92)
            base_quiz = rng.uniform(78, 88)
            att_slope = rng.uniform(-2.0, -0.8)
            quiz_slope = rng.uniform(-2.0, -1.0)

        elif persona == "Sudden Drop":
            base_att = rng.uniform(85, 94)
            base_quiz = rng.uniform(80, 90)
            att_slope = 0
            quiz_slope = 0

        elif persona == "Chronic Absentee":
            base_att = rng.uniform(55, 69)
            base_quiz = rng.uniform(65, 78)
            att_slope = rng.uniform(-0.7, 0.1)
            quiz_slope = rng.uniform(-0.7, 0.2)

        elif persona == "Recovering":
            base_att = rng.uniform(68, 76)
            base_quiz = rng.uniform(58, 70)
            att_slope = rng.uniform(1.0, 2.0)
            quiz_slope = rng.uniform(1.0, 2.2)

        for week in range(1, 9):

            if persona == "Sudden Drop" and week >= 6:
                attendance = base_att - rng.uniform(8, 17)
                quiz = base_quiz - rng.uniform(20, 32)

            else:
                attendance = (
                    base_att
                    + att_slope * (week - 1)
                    + rng.normal(0, 1.5)
                )

                quiz = (
                    base_quiz
                    + quiz_slope * (week - 1)
                    + rng.normal(0, 2)
                )

            attendance = float(np.clip(attendance, 45, 99))
            quiz = float(np.clip(quiz, 30, 99))

            missing = 0

            if persona == "Chronic Absentee":
                missing = int(rng.integers(1, 4))

            elif persona == "Sudden Drop" and week >= 6:
                missing = int(rng.integers(1, 4))

            elif persona == "Gradual Decliner" and week >= 5:
                missing = int(rng.integers(1, 3))

            elif persona == "Recovering":
                missing = int(rng.integers(0, 2))

            else:
                missing = int(rng.integers(0, 2))

            rows.append({
                "Student ID": sid,
                "Student": name,
                "Class": cls,
                "Persona": persona,
                "Week": week,
                "Attendance": round(attendance, 1),
                "Quiz Score": round(quiz, 1),
                "Missing Assignments": missing
            })

    return pd.DataFrame(rows)


df = create_student_data()


# ============================================================
# SUBJECT DATA
# ============================================================

subjects = [
    "Mathematics",
    "Science",
    "English",
    "Social Studies",
    "Computer Science"
]


@st.cache_data
def create_subject_data():

    rng = np.random.default_rng(21)

    records = []

    for sid in df["Student ID"].unique():

        student_rows = df[df["Student ID"] == sid]

        latest = student_rows.iloc[-1]

        overall = latest["Quiz Score"]

        for subject in subjects:

            adjustment = {
                "Mathematics": rng.uniform(-12, 5),
                "Science": rng.uniform(-7, 6),
                "English": rng.uniform(-4, 8),
                "Social Studies": rng.uniform(-5, 8),
                "Computer Science": rng.uniform(-2, 10)
            }[subject]

            marks = float(np.clip(overall + adjustment, 35, 98))

            if subject == "Mathematics" and latest["Persona"] in [
                "Gradual Decliner",
                "Sudden Drop"
            ]:
                marks -= 7

            attendance = float(np.clip(
                latest["Attendance"] + rng.uniform(-7, 5),
                45,
                99
            ))

            records.append({
                "Student ID": sid,
                "Subject": subject,
                "Marks": round(marks, 1),
                "Attendance": round(attendance, 1)
            })

    return pd.DataFrame(records)


subject_df = create_subject_data()


# ============================================================
# DAILY ATTENDANCE
# ============================================================

@st.cache_data
def create_daily_attendance():

    rng = np.random.default_rng(55)

    rows = []

    start_date = datetime(2026, 8, 3)

    for sid in df["Student ID"].unique():

        latest_att = df[df["Student ID"] == sid].iloc[-1]["Attendance"]

        for day in range(30):

            date = start_date + timedelta(days=day)

            if date.weekday() >= 5:
                continue

            probability = latest_att / 100

            present = rng.random() < probability

            status = "Present" if present else "Absent"

            if present and rng.random() < 0.04:
                status = "Late"

            rows.append({
                "Student ID": sid,
                "Date": date,
                "Status": status
            })

    return pd.DataFrame(rows)


daily_df = create_daily_attendance()


# ============================================================
# RISK ENGINE
# ============================================================

def calculate_risk(student_id):

    history = df[df["Student ID"] == student_id].sort_values("Week")

    attendance = history["Attendance"].values
    quiz = history["Quiz Score"].values

    attendance_slope = np.polyfit(history["Week"], attendance, 1)[0]
    quiz_slope = np.polyfit(history["Week"], quiz, 1)[0]

    latest_att = attendance[-1]
    latest_quiz = quiz[-1]

    attendance_factor = np.clip(
        (-attendance_slope * 8) + ((85 - latest_att) * 0.65),
        0,
        100
    )

    quiz_factor = np.clip(
        (-quiz_slope * 8) + ((70 - latest_quiz) * 1.4),
        0,
        100
    )

    missing = int(history.tail(3)["Missing Assignments"].sum())

    assignment_factor = np.clip(
        missing / 6 * 100,
        0,
        100
    )

    attendance_level_factor = np.clip(
        (85 - latest_att) * 1.7,
        0,
        100
    )

    risk = (
        attendance_factor * 0.35
        + quiz_factor * 0.30
        + assignment_factor * 0.20
        + attendance_level_factor * 0.15
    )

    risk = round(float(np.clip(risk, 0, 100)), 1)

    if risk >= 70:
        status = "Needs Support"
    elif risk >= 55:
        status = "Monitor Closely"
    elif risk >= 35:
        status = "Watch"
    else:
        status = "On Track"

    return {
        "risk": risk,
        "status": status,
        "attendance_slope": attendance_slope,
        "quiz_slope": quiz_slope,
        "latest_att": latest_att,
        "latest_quiz": latest_quiz,
        "missing": missing,
        "attendance_factor": attendance_factor,
        "quiz_factor": quiz_factor,
        "assignment_factor": assignment_factor,
        "attendance_level_factor": attendance_level_factor
    }


# ============================================================
# RISK TABLE
# ============================================================

@st.cache_data
def create_risk_table():

    records = []

    for sid in df["Student ID"].unique():

        student = df[df["Student ID"] == sid].iloc[-1]
        risk = calculate_risk(sid)

        records.append({
            "Student ID": sid,
            "Student": student["Student"],
            "Class": student["Class"],
            "Risk Score": risk["risk"],
            "Status": risk["status"],
            "Attendance": risk["latest_att"],
            "Latest Quiz": risk["latest_quiz"],
            "Missing Assignments": risk["missing"],
            "Persona": student["Persona"]
        })

    return pd.DataFrame(records)


risk_df = create_risk_table()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def status_html(status):

    if status == "Needs Support":
        return '<span class="status-danger">Needs Support</span>'

    if status == "Monitor Closely":
        return '<span class="status-warning">Monitor Closely</span>'

    if status == "Watch":
        return '<span class="status-neutral">Watch</span>'

    return '<span class="status-success">On Track</span>'


def get_student(student_id):

    row = risk_df[risk_df["Student ID"] == student_id].iloc[0]
    return row


def get_reasons(student_id):

    risk = calculate_risk(student_id)

    reasons = []

    if risk["latest_att"] < 75:
        reasons.append(
            f"Attendance is currently {risk['latest_att']:.0f}%."
        )

    if risk["attendance_slope"] < -0.7:
        reasons.append(
            "Attendance has been declining across recent weeks."
        )

    if risk["latest_quiz"] < 60:
        reasons.append(
            f"Recent assessment performance is {risk['latest_quiz']:.0f}%."
        )

    if risk["quiz_slope"] < -0.7:
        reasons.append(
            "Assessment performance is trending downward."
        )

    if risk["missing"] >= 3:
        reasons.append(
            f"{risk['missing']} assignments were missed in the recent period."
        )

    if not reasons:
        reasons.append(
            "Current indicators are relatively stable."
        )

    return reasons


def weak_subject(student_id):

    data = subject_df[
        subject_df["Student ID"] == student_id
    ].sort_values("Marks")

    return data.iloc[0]["Subject"]


def student_name(student_id):

    return risk_df[
        risk_df["Student ID"] == student_id
    ].iloc[0]["Student"]


# ============================================================
# LOGIN SCREEN
# ============================================================

def login_screen():

    st.markdown(
        '<div class="login-wrapper">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="login-title">Academic Support Platform</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="login-subtitle">'
        'AI-Based Early Detection of Students at Academic Risk'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="info-box">'
        '<div class="info-title">Demo Login</div>'
        '<div class="info-text">'
        'Select a role to explore the role-based academic support workflow. '
        'This prototype uses synthetic student data.'
        '</div></div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            '<div class="role-card">'
            '<h3>Teacher</h3>'
            '<p>Monitor students, identify risk and plan interventions.</p>'
            '</div>',
            unsafe_allow_html=True
        )

        if st.button(
            "Teacher Demo Login",
            use_container_width=True,
            key="teacher_login"
        ):
            st.session_state.logged_in = True
            st.session_state.role = "Teacher"
            st.session_state.page = "Dashboard"
            st.rerun()

    with col2:
        st.markdown(
            '<div class="role-card">'
            '<h3>Student</h3>'
            '<p>View performance, weak subjects and learning support.</p>'
            '</div>',
            unsafe_allow_html=True
        )

        if st.button(
            "Student Demo Login",
            use_container_width=True,
            key="student_login"
        ):
            st.session_state.logged_in = True
            st.session_state.role = "Student"
            st.session_state.page = "My Dashboard"
            st.rerun()

    with col3:
        st.markdown(
            '<div class="role-card">'
            '<h3>Parent</h3>'
            '<p>Track academic progress, attendance and teacher updates.</p>'
            '</div>',
            unsafe_allow_html=True
        )

        if st.button(
            "Parent Demo Login",
            use_container_width=True,
            key="parent_login"
        ):
            st.session_state.logged_in = True
            st.session_state.role = "Parent"
            st.session_state.page = "Child Dashboard"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div style="text-align:center;margin-top:35px;">'
        '<span class="small-note">'
        'Prototype | Synthetic Data | Human Oversight Required'
        '</span></div>',
        unsafe_allow_html=True
    )


# ============================================================
# TEACHER DASHBOARD
# ============================================================

def teacher_dashboard():

    st.markdown(
        '<div class="hero-title">Teacher Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'Monitor academic performance, attendance and emerging support needs.'
        '</div>',
        unsafe_allow_html=True
    )

    total_students = len(risk_df)
    needs_support = len(
        risk_df[risk_df["Status"] == "Needs Support"]
    )

    recovering = len(
        risk_df[
            risk_df["Persona"] == "Recovering"
        ]
    )

    avg_marks = risk_df["Latest Quiz"].mean()
    avg_att = risk_df["Attendance"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)

    metrics = [
        ("Total Students", total_students, "Class VIII-A"),
        ("Needs Support", needs_support, "Risk threshold reached"),
        ("Recovering", recovering, "Positive recent trend"),
        ("Average Marks", f"{avg_marks:.1f}%", "Latest assessment"),
        ("Attendance", f"{avg_att:.1f}%", "Class average")
    ]

    for col, item in zip(
        [c1, c2, c3, c4, c5],
        metrics
    ):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{item[0]}</div>
                    <div class="metric-value">{item[1]}</div>
                    <div class="metric-small">{item[2]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="section-title">Academic Risk Overview</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1.5, 1])

    with left:

        counts = risk_df["Status"].value_counts()

        fig = go.Figure(
            data=[
                go.Bar(
                    x=counts.index,
                    y=counts.values
                )
            ]
        )

        fig.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=20, b=20),
            xaxis_title="Risk Status",
            yaxis_title="Students",
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.markdown(
            '<div class="info-box">'
            '<div class="info-title">Early Warning Signals</div>'
            '<div class="info-text">'
            'The system combines attendance trend, assessment trend, '
            'missing assignments and current attendance level.'
            '</div></div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="info-box">'
            '<div class="info-title">Important</div>'
            '<div class="info-text">'
            'Risk scores are support signals, not final judgments. '
            'Teachers should review the underlying evidence before taking action.'
            '</div></div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">Students Requiring Attention</div>',
        unsafe_allow_html=True
    )

    top_students = risk_df.sort_values(
        "Risk Score",
        ascending=False
    ).head(7)

    for _, row in top_students.iterrows():

        col1, col2, col3, col4 = st.columns(
            [2.2, 1, 1, 1]
        )

        with col1:
            st.write(
                f"**{row['Student']}**  \n"
                f"{row['Student ID']} | {row['Class']}"
            )

        with col2:
            st.metric(
                "Risk",
                f"{row['Risk Score']:.0f}"
            )

        with col3:
            st.write(
                status_html(row["Status"]),
                unsafe_allow_html=True
            )

        with col4:

            if st.button(
                "View",
                key=f"view_{row['Student ID']}"
            ):
                st.session_state.selected_student = row["Student ID"]
                st.session_state.page = "Student Profile"
                st.rerun()


# ============================================================
# STUDENT PROFILE
# ============================================================

def student_profile():

    sid = st.session_state.selected_student

    student = get_student(sid)
    risk = calculate_risk(sid)

    st.markdown(
        '<div class="hero-title">Student Profile</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'Evidence-based student overview with explainable risk signals.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Student</div>
                <div class="metric-value" style="font-size:23px;">
                    {student['Student']}
                </div>
                <div class="metric-small">
                    {student['Student ID']} | {student['Class']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Risk Score</div>
                <div class="metric-value">
                    {risk['risk']:.0f}
                </div>
                <div class="metric-small">0–100 support signal</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Attendance</div>
                <div class="metric-value">
                    {risk['latest_att']:.0f}%
                </div>
                <div class="metric-small">Latest period</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Latest Assessment</div>
                <div class="metric-value">
                    {risk['latest_quiz']:.0f}%
                </div>
                <div class="metric-small">Latest assessment</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        f"<br>{status_html(risk['status'])}",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Why is this student receiving this signal?</div>',
        unsafe_allow_html=True
    )

    reasons = get_reasons(sid)

    for reason in reasons:
        st.markdown(
            f"""
            <div class="info-box">
                <div class="info-text">{reason}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">Risk Factor Breakdown</div>',
        unsafe_allow_html=True
    )

    factor_df = pd.DataFrame({
        "Factor": [
            "Attendance Trend",
            "Assessment Trend",
            "Missing Assignments",
            "Attendance Level"
        ],
        "Contribution": [
            risk["attendance_factor"] * 0.35,
            risk["quiz_factor"] * 0.30,
            risk["assignment_factor"] * 0.20,
            risk["attendance_level_factor"] * 0.15
        ]
    })

    fig = go.Figure(
        go.Bar(
            x=factor_df["Contribution"],
            y=factor_df["Factor"],
            orientation="h"
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Contribution to risk signal",
        yaxis_title=""
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">Subject Performance</div>',
        unsafe_allow_html=True
    )

    sdata = subject_df[
        subject_df["Student ID"] == sid
    ].copy()

    st.dataframe(
        sdata[
            ["Subject", "Marks", "Attendance"]
        ].rename(
            columns={
                "Marks": "Marks %",
                "Attendance": "Attendance %"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">Performance Trend</div>',
        unsafe_allow_html=True
    )

    history = df[
        df["Student ID"] == sid
    ].sort_values("Week")

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=history["Week"],
            y=history["Quiz Score"],
            mode="lines+markers",
            name="Assessment"
        )
    )

    fig2.add_trace(
        go.Scatter(
            x=history["Week"],
            y=history["Attendance"],
            mode="lines+markers",
            name="Attendance"
        )
    )

    fig2.update_layout(
        height=380,
        xaxis_title="Week",
        yaxis_title="Percentage",
        yaxis=dict(range=[0, 100])
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">Time Travel</div>',
        unsafe_allow_html=True
    )

    selected_week = st.slider(
        "Replay student history up to week",
        1,
        8,
        8,
        key=f"time_{sid}"
    )

    snapshot = history[
        history["Week"] <= selected_week
    ].iloc[-1]

    a, b, c = st.columns(3)

    a.metric(
        "Attendance",
        f"{snapshot['Attendance']:.0f}%"
    )

    b.metric(
        "Assessment",
        f"{snapshot['Quiz Score']:.0f}%"
    )

    c.metric(
        "Missing Assignments",
        int(snapshot["Missing Assignments"])
    )

    if selected_week < 8:
        st.info(
            "This replay shows an earlier point in the student's academic history."
        )
    else:
        st.success(
            "Current period selected."
        )


# ============================================================
# INTERVENTION CENTER
# ============================================================

def intervention_page():

    sid = st.session_state.selected_student

    student = get_student(sid)
    risk = calculate_risk(sid)

    weak = weak_subject(sid)

    st.markdown(
        '<div class="hero-title">Intervention Center</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'Convert detected risk signals into targeted academic support.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="info-box">
            <div class="info-title">
                Recommended focus: {weak}
            </div>
            <div class="info-text">
                The student's current indicators suggest that additional
                support may be useful in this subject.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">1. Explain Again</div>',
        unsafe_allow_html=True
    )

    selected_subject = st.selectbox(
        "Select subject",
        subjects,
        index=subjects.index(weak)
    )

    explanations = {
        "Mathematics":
            "Quadratic equations can be understood as equations containing x². "
            "Start by identifying a, b and c, then select a suitable solving method.",
        "Science":
            "Break the concept into definition, process, example and application. "
            "Use a simple real-world example before attempting questions.",
        "English":
            "Read the sentence carefully, identify the key grammar rule and "
            "then apply it to a short example.",
        "Social Studies":
            "Convert the topic into timeline, causes, events and outcomes. "
            "This makes long answers easier to remember.",
        "Computer Science":
            "Start with the basic concept, trace one small example and then "
            "practice a similar problem independently."
    }

    st.markdown(
        f"""
        <div class="info-box">
            <div class="info-title">
                {selected_subject} — Simple Explanation
            </div>
            <div class="info-text">
                {explanations[selected_subject]}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">2. Practice</div>',
        unsafe_allow_html=True
    )

    practice_questions = {
        "Mathematics":
            "Solve x² - 5x + 6 = 0.",
        "Science":
            "Explain the main stages of the process studied in this chapter.",
        "English":
            "Identify the grammar rule used in the following sentence.",
        "Social Studies":
            "Write two causes and two effects of the selected historical event.",
        "Computer Science":
            "Explain the concept using a simple example."
    }

    st.info(
        practice_questions[selected_subject]
    )

    st.button(
        "Check Practice Answer",
        key="practice_check"
    )

    st.markdown(
        '<div class="section-title">3. Personalized Support Plan</div>',
        unsafe_allow_html=True
    )

    plan_col1, plan_col2 = st.columns(2)

    with plan_col1:

        st.markdown(
            """
            <div class="info-box">
                <div class="info-title">Daily Revision</div>
                <div class="info-text">
                    25 minutes focused practice on the identified weak area.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-box">
                <div class="info-title">Practice Set</div>
                <div class="info-text">
                    Complete 5 targeted questions and review incorrect answers.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with plan_col2:

        st.markdown(
            """
            <div class="info-box">
                <div class="info-title">Special Class</div>
                <div class="info-text">
                    Attend a focused online support session for the identified topic.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-box">
                <div class="info-title">Teacher Review</div>
                <div class="info-text">
                    Review progress after the next assessment cycle.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">4. Parent Communication</div>',
        unsafe_allow_html=True
    )

    parent_message = (
        f"Hello Parent, {student['Student']} may benefit from additional "
        f"support in {weak}. Current attendance is {risk['latest_att']:.0f}% "
        f"and the latest assessment score is {risk['latest_quiz']:.0f}%. "
        f"We recommend regular revision and the available special support class."
    )

    st.text_area(
        "Suggested parent update",
        value=parent_message,
        height=130
    )

    if st.button(
        "Send Parent Update",
        type="primary",
        use_container_width=True
    ):
        st.session_state.parent_message_sent = True

    if st.session_state.parent_message_sent:
        st.success(
            "Parent update sent in demo mode."
        )

    st.markdown(
        '<div class="section-title">5. Recommended Special Classes</div>',
        unsafe_allow_html=True
    )

    special_classes = pd.DataFrame({
        "Subject": [
            weak,
            "Science",
            "English"
        ],
        "Focus": [
            "Concept Revision",
            "Chapter Support",
            "Practice Session"
        ],
        "Mode": [
            "Online",
            "Online",
            "Online"
        ],
        "Status": [
            "Recommended",
            "Available",
            "Available"
        ]
    })

    st.dataframe(
        special_classes,
        use_container_width=True,
        hide_index=True
    )

    if st.button(
        "Join Recommended Special Class",
        use_container_width=True
    ):
        st.success(
            f"Demo classroom opened for {weak}."
        )


# ============================================================
# TRACK PAGE
# ============================================================

def track_page():

    sid = st.session_state.selected_student

    student = get_student(sid)

    st.markdown(
        '<div class="hero-title">Progress Tracking</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'Track academic signals before and after support activities.'
        '</div>',
        unsafe_allow_html=True
    )

    history = df[
        df["Student ID"] == sid
    ].sort_values("Week")

    risks = []

    for week in range(1, 9):

        h = history[
            history["Week"] <= week
        ]

        if len(h) < 2:
            risk_value = 25
        else:

            att_slope = np.polyfit(
                h["Week"],
                h["Attendance"],
                1
            )[0]

            quiz_slope = np.polyfit(
                h["Week"],
                h["Quiz Score"],
                1
            )[0]

            last_att = h.iloc[-1]["Attendance"]
            last_quiz = h.iloc[-1]["Quiz Score"]

            risk_value = (
                np.clip(
                    (-att_slope * 8) + ((85 - last_att) * 0.65),
                    0,
                    100
                ) * 0.35
                +
                np.clip(
                    (-quiz_slope * 8) + ((70 - last_quiz) * 1.4),
                    0,
                    100
                ) * 0.30
            )

        risks.append(float(np.clip(risk_value, 0, 100)))

    trend_df = pd.DataFrame({
        "Week": list(range(1, 9)),
        "Risk Signal": risks
    })

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=trend_df["Week"],
            y=trend_df["Risk Signal"],
            mode="lines+markers",
            name="Risk Signal"
        )
    )

    fig.update_layout(
        height=400,
        yaxis=dict(range=[0, 100]),
        xaxis_title="Week",
        yaxis_title="Risk Signal"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">Academic Recovery Indicators</div>',
        unsafe_allow_html=True
    )

    latest = history.iloc[-1]
    earlier = history.iloc[0]

    latest_subjects = subject_df[
        subject_df["Student ID"] == sid
    ]

    weak = latest_subjects.sort_values(
        "Marks"
    ).iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Attendance",
        f"{latest['Attendance']:.0f}%",
        f"{latest['Attendance'] - earlier['Attendance']:+.0f} pts"
    )

    c2.metric(
        "Assessment",
        f"{latest['Quiz Score']:.0f}%",
        f"{latest['Quiz Score'] - earlier['Quiz Score']:+.0f} pts"
    )

    c3.metric(
        "Weak Subject",
        weak["Subject"]
    )

    c4.metric(
        "Current Status",
        calculate_risk(sid)["status"]
    )

    st.markdown(
        '<div class="section-title">Support Timeline</div>',
        unsafe_allow_html=True
    )

    timeline = [
        ("Week 1", "Baseline academic data recorded."),
        ("Week 3", "Attendance and assessment trends monitored."),
        ("Week 5", "Emerging support signal identified."),
        ("Week 6", "Targeted intervention recommended."),
        ("Week 7", "Parent communication and special class recommended."),
        ("Week 8", "Progress reviewed.")
    ]

    for week, text in timeline:

        st.markdown(
            f"""
            <div class="timeline-card">
                <b>{week}</b><br>
                {text}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.warning(
        "This prototype uses synthetic data. The progress visualization demonstrates "
        "how a school could track indicators over time; it does not establish that "
        "an intervention caused a particular outcome."
    )


# ============================================================
# STUDENT DASHBOARD
# ============================================================

def student_dashboard():

    sid = "STU-001"

    student = get_student(sid)
    risk = calculate_risk(sid)

    st.markdown(
        '<div class="hero-title">My Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="hero-subtitle">
            Welcome, {student['Student']}. Here is your academic support overview.
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Overall Marks",
        f"{risk['latest_quiz']:.0f}%"
    )

    c2.metric(
        "Attendance",
        f"{risk['latest_att']:.0f}%"
    )

    c3.metric(
        "Academic Status",
        risk["status"]
    )

    c4.metric(
        "Weak Subject",
        weak_subject(sid)
    )

    st.markdown(
        '<div class="section-title">My Subjects</div>',
        unsafe_allow_html=True
    )

    data = subject_df[
        subject_df["Student ID"] == sid
    ]

    st.dataframe(
        data[
            ["Subject", "Marks", "Attendance"]
        ].rename(
            columns={
                "Marks": "Marks %",
                "Attendance": "Attendance %"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">Learn Again</div>',
        unsafe_allow_html=True
    )

    weak = weak_subject(sid)

    st.info(
        f"Your current support focus is {weak}. "
        "Use the learning support section to review concepts and practice."
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "Explain Again",
            use_container_width=True
        ):
            st.success(
                f"Learning support opened for {weak}."
            )

    with col2:
        if st.button(
            "Practice Questions",
            use_container_width=True
        ):
            st.success(
                f"Practice set opened for {weak}."
            )

    st.markdown(
        '<div class="section-title">Special Classes</div>',
        unsafe_allow_html=True
    )

    classes = pd.DataFrame({
        "Subject": [weak, "Science"],
        "Mode": ["Online", "Online"],
        "Focus": ["Concept Revision", "Chapter Support"],
        "Availability": ["Recommended", "Available"]
    })

    st.dataframe(
        classes,
        use_container_width=True,
        hide_index=True
    )

    if st.button(
        "Open Recommended Class",
        use_container_width=True
    ):
        st.success(
            "Demo classroom opened."
        )


# ============================================================
# PARENT DASHBOARD
# ============================================================

def parent_dashboard():

    sid = "STU-001"

    student = get_student(sid)
    risk = calculate_risk(sid)

    st.markdown(
        '<div class="hero-title">Child Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="hero-subtitle">
            Academic progress overview for {student['Student']}.
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Overall Marks",
        f"{risk['latest_quiz']:.0f}%"
    )

    c2.metric(
        "Attendance",
        f"{risk['latest_att']:.0f}%"
    )

    c3.metric(
        "Status",
        risk["status"]
    )

    c4.metric(
        "Focus Subject",
        weak_subject(sid)
    )

    st.markdown(
        '<div class="section-title">Subject-wise Performance</div>',
        unsafe_allow_html=True
    )

    data = subject_df[
        subject_df["Student ID"] == sid
    ]

    st.dataframe(
        data[
            ["Subject", "Marks", "Attendance"]
        ].rename(
            columns={
                "Marks": "Marks %",
                "Attendance": "Attendance %"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">Attendance</div>',
        unsafe_allow_html=True
    )

    daily = daily_df[
        daily_df["Student ID"] == sid
    ].copy()

    attendance_summary = (
        daily["Status"]
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Days")
    )

    st.dataframe(
        attendance_summary,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">Teacher Update</div>',
        unsafe_allow_html=True
    )

    weak = weak_subject(sid)

    st.markdown(
        f"""
        <div class="info-box">
            <div class="info-title">Suggested Support</div>
            <div class="info-text">
                The teacher has identified {weak} as a current focus area.
                Regular revision, practice and participation in the recommended
                support class may help the student address the identified learning need.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Recommended Special Class</div>',
        unsafe_allow_html=True
    )

    st.info(
        f"{weak} | Online | Concept Revision | Recommended"
    )

    if st.button(
        "View Special Class",
        use_container_width=True
    ):
        st.success(
            "Demo special class details opened."
        )

    st.markdown(
        '<div class="section-title">Progress</div>',
        unsafe_allow_html=True
    )

    history = df[
        df["Student ID"] == sid
    ].sort_values("Week")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history["Week"],
            y=history["Quiz Score"],
            mode="lines+markers",
            name="Marks"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=history["Week"],
            y=history["Attendance"],
            mode="lines+markers",
            name="Attendance"
        )
    )

    fig.update_layout(
        height=350,
        yaxis=dict(range=[0, 100]),
        xaxis_title="Week",
        yaxis_title="Percentage"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

def sidebar():

    with st.sidebar:

        st.markdown(
            """
            <div style="font-size:21px;font-weight:800;">
                Academic Support
            </div>
            <div style="font-size:12px;color:#94a3b8;margin-top:5px;">
                Early Detection and Intervention
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        st.write(
            f"**Role:** {st.session_state.role}"
        )

        if st.session_state.role == "Teacher":

            pages = [
                "Dashboard",
                "Student Profile",
                "Intervention",
                "Track"
            ]

        elif st.session_state.role == "Student":

            pages = [
                "My Dashboard"
            ]

        else:

            pages = [
                "Child Dashboard"
            ]

        selected = st.radio(
            "Navigation",
            pages,
            index=pages.index(
                st.session_state.page
            ) if st.session_state.page in pages else 0
        )

        st.session_state.page = selected

        st.divider()

        if st.button(
            "Logout",
            use_container_width=True
        ):
            st.session_state.logged_in = False
            st.session_state.role = None
            st.session_state.page = "Dashboard"
            st.rerun()

        st.markdown(
            """
            <div style="margin-top:20px;font-size:11px;color:#94a3b8;">
                Synthetic demonstration data<br>
                Human review required for real deployment
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# APP ROUTER
# ============================================================

if not st.session_state.logged_in:

    login_screen()

else:

    sidebar()

    if st.session_state.role == "Teacher":

        if st.session_state.page == "Dashboard":
            teacher_dashboard()

        elif st.session_state.page == "Student Profile":
            student_profile()

        elif st.session_state.page == "Intervention":
            intervention_page()

        elif st.session_state.page == "Track":
            track_page()

    elif st.session_state.role == "Student":

        student_dashboard()

    elif st.session_state.role == "Parent":

        parent_dashboard()
