import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-Based Early Detection of Students at Academic Risk",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    html, body, [class*="css"] {
        font-family: Inter, sans-serif;
    }

    .stApp {
        background: #f5f7fb;
        color: #172033;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .topbar {
        background: #101828;
        color: white;
        padding: 18px 24px;
        border-radius: 14px;
        margin-bottom: 18px;
    }

    .brand {
        font-size: 21px;
        font-weight: 800;
    }

    .top-subtitle {
        color: #98a2b3;
        font-size: 12px;
        margin-top: 4px;
    }

    .card {
        background: white;
        border: 1px solid #e4e7ec;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.04);
        margin-bottom: 14px;
    }

    .metric-label {
        color: #667085;
        font-size: 13px;
        font-weight: 600;
    }

    .metric-value {
        font-size: 31px;
        font-weight: 800;
        margin-top: 4px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 800;
        margin: 25px 0 12px;
    }

    .reason {
        color: #475467;
        font-size: 13px;
        line-height: 1.4;
    }

    .small-note {
        color: #667085;
        font-size: 12px;
    }

    .badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
    }

    .badge-red {
        background: #fee4e2;
        color: #b42318;
    }

    .badge-orange {
        background: #fff0d6;
        color: #b54708;
    }

    .badge-yellow {
        background: #fff7cc;
        color: #8a6116;
    }

    .badge-green {
        background: #dcfae6;
        color: #067647;
    }

    .factor {
        margin: 15px 0;
    }

    .factor-top {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        font-weight: 700;
    }

    .bar {
        height: 9px;
        background: #eaecf0;
        border-radius: 99px;
        margin-top: 7px;
        overflow: hidden;
    }

    .fill-red {
        height: 100%;
        background: #d92d20;
    }

    .fill-orange {
        height: 100%;
        background: #f79009;
    }

    .fill-blue {
        height: 100%;
        background: #3b82f6;
    }

    .login-wrap {
        max-width: 760px;
        margin: 9vh auto;
        text-align: center;
    }

    .login-title {
        font-size: 43px;
        line-height: 1.1;
        font-weight: 800;
        color: #101828;
        margin-top: 15px;
    }

    .login-sub {
        color: #667085;
        font-size: 16px;
        line-height: 1.6;
        margin: 15px auto 30px;
        max-width: 650px;
    }

    .heatmap {
        border-collapse: separate;
        border-spacing: 4px;
        width: 100%;
    }

    .heatmap th {
        font-size: 11px;
        color: #667085;
        padding: 5px;
    }

    .heatmap td {
        height: 28px;
        border-radius: 5px;
        text-align: center;
        font-size: 9px;
        color: #344054;
    }

    .h-green {
        background: #b7ebc6;
    }

    .h-yellow {
        background: #ffe58f;
    }

    .h-orange {
        background: #ffbf69;
    }

    .h-red {
        background: #f98b82;
        color: white !important;
    }

    .info-strip {
        background: #eef4ff;
        border: 1px solid #c7d7fe;
        color: #344054;
        border-radius: 10px;
        padding: 12px 15px;
        font-size: 13px;
        margin-bottom: 15px;
    }

    .demo-strip {
        background: #f9fafb;
        border: 1px solid #eaecf0;
        color: #667085;
        border-radius: 9px;
        padding: 10px 13px;
        font-size: 12px;
        margin-top: 10px;
    }

    div.stButton > button {
        border-radius: 9px;
        font-weight: 700;
        min-height: 44px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SYNTHETIC CLASSROOM DATA
# ============================================================

@st.cache_data
def build_data():
    rng = np.random.default_rng(2026)

    personas = (
        ["Steady"] * 15
        + ["Stable Middle"] * 22
        + ["Gradual Decliner"] * 8
        + ["Sudden Dropper"] * 5
        + ["Chronic Absentee"] * 5
        + ["Recovering"] * 3
        + ["Fluctuating"] * 2
    )

    records = []

    for i, persona in enumerate(personas, 1):

        student = f"Student {i:02d}"

        for week in range(1, 9):

            if persona == "Steady":

                attendance = 92 + rng.normal(0, 2)
                quiz = 84 + rng.normal(0, 3)
                assignment = 95 + rng.normal(0, 2)
                lms = 90 + rng.normal(0, 3)

            elif persona == "Stable Middle":

                attendance = 80 + rng.normal(0, 4)
                quiz = 69 + rng.normal(0, 5)
                assignment = 80 + rng.normal(0, 5)
                lms = 73 + rng.normal(0, 5)

            elif persona == "Gradual Decliner":

                decline = max(0, week - 2)

                attendance = (
                    88
                    - decline * 7
                    + rng.normal(0, 2)
                )

                quiz = (
                    78
                    - decline * 6
                    + rng.normal(0, 3)
                )

                assignment = (
                    90
                    - decline * 7
                    + rng.normal(0, 3)
                )

                lms = (
                    84
                    - decline * 6
                    + rng.normal(0, 3)
                )

            elif persona == "Sudden Dropper":

                if week <= 4:

                    attendance = 92 + rng.normal(0, 2)
                    quiz = 82 + rng.normal(0, 3)
                    assignment = 94 + rng.normal(0, 2)
                    lms = 88 + rng.normal(0, 3)

                else:

                    decline = week - 4

                    attendance = (
                        92
                        - decline * 21
                        + rng.normal(0, 2)
                    )

                    quiz = (
                        82
                        - decline * 9
                        + rng.normal(0, 3)
                    )

                    assignment = (
                        94
                        - decline * 18
                        + rng.normal(0, 2)
                    )

                    lms = (
                        88
                        - decline * 15
                        + rng.normal(0, 3)
                    )

            elif persona == "Chronic Absentee":

                attendance = 47 + rng.normal(0, 4)
                quiz = 54 + rng.normal(0, 5)
                assignment = 50 + rng.normal(0, 5)
                lms = 44 + rng.normal(0, 5)

            elif persona == "Recovering":

                if week <= 3:

                    attendance = (
                        48
                        + week * 2
                        + rng.normal(0, 2)
                    )

                    quiz = (
                        48
                        + week * 2
                        + rng.normal(0, 3)
                    )

                    assignment = (
                        46
                        + week * 3
                        + rng.normal(0, 3)
                    )

                    lms = (
                        45
                        + week * 2
                        + rng.normal(0, 3)
                    )

                else:

                    recovery = week - 3

                    attendance = (
                        58
                        + recovery * 7
                        + rng.normal(0, 2)
                    )

                    quiz = (
                        54
                        + recovery * 7
                        + rng.normal(0, 3)
                    )

                    assignment = (
                        55
                        + recovery * 7
                        + rng.normal(0, 3)
                    )

                    lms = (
                        52
                        + recovery * 7
                        + rng.normal(0, 3)
                    )

            else:

                swing = [
                    0,
                    5,
                    -4,
                    7,
                    -3,
                    4,
                    -2,
                    3,
                ][week - 1]

                attendance = (
                    71
                    + swing
                    + rng.normal(0, 3)
                )

                quiz = (
                    62
                    + swing
                    + rng.normal(0, 4)
                )

                assignment = (
                    70
                    + swing
                    + rng.normal(0, 4)
                )

                lms = (
                    65
                    + swing
                    + rng.normal(0, 4)
                )

            assignment = float(
                np.clip(
                    assignment,
                    0,
                    100
                )
            )

            records.append(
                {
                    "Student": student,
                    "Persona": persona,
                    "Week": week,
                    "Attendance": float(
                        np.clip(
                            attendance,
                            15,
                            99
                        )
                    ),
                    "Quiz": float(
                        np.clip(
                            quiz,
                            10,
                            99
                        )
                    ),
                    "Assignment": assignment,
                    "LMS": float(
                        np.clip(
                            lms,
                            5,
                            100
                        )
                    ),
                    "Submitted": assignment >= 55,
                }
            )

    return pd.DataFrame(records)


df = build_data()


# ============================================================
# RISK ENGINE
# ============================================================

def risk_score(history):

    h = history.sort_values("Week").copy()

    last_attendance = float(
        h.iloc[-1]["Attendance"]
    )

    last_quiz = float(
        h.iloc[-1]["Quiz"]
    )

    recent_attendance = h.tail(3)

    recent_quiz = h.tail(4)

    attendance_slope = np.polyfit(
        recent_attendance["Week"],
        recent_attendance["Attendance"],
        1
    )[0]

    quiz_slope = np.polyfit(
        recent_quiz["Week"],
        recent_quiz["Quiz"],
        1
    )[0]

    missing_streak = 0

    for submitted in h["Submitted"].iloc[::-1]:

        if not submitted:
            missing_streak += 1
        else:
            break

    attendance_level = np.clip(
        (75 - last_attendance) / 75 * 100,
        0,
        100
    )

    attendance_trend = np.clip(
        (-attendance_slope) * 8
        + attendance_level * 0.35,
        0,
        100
    )

    quiz_trend = np.clip(
        (-quiz_slope) * 8
        + max(
            0,
            70 - last_quiz
        ) * 1.25,
        0,
        100
    )

    assignment_gap = np.clip(
        missing_streak / 3 * 100,
        0,
        100
    )

    overall_attendance = np.clip(
        (70 - last_attendance) / 70 * 100,
        0,
        100
    )

    score = (
        attendance_trend * 0.35
        + quiz_trend * 0.30
        + assignment_gap * 0.20
        + overall_attendance * 0.15
    )

    return float(
        np.clip(
            score,
            0,
            100
        )
    )


def risk_label(score):

    if score >= 70:
        return "Needs Support", "red"

    if score >= 55:
        return "Monitor Closely", "orange"

    if score >= 35:
        return "Watch", "yellow"

    return "On Track", "green"


def primary_reason(history):

    h = history.sort_values("Week")

    attendance_slope = np.polyfit(
        h["Week"].tail(3),
        h["Attendance"].tail(3),
        1
    )[0]

    quiz_slope = np.polyfit(
        h["Week"].tail(4),
        h["Quiz"].tail(4),
        1
    )[0]

    missing = int(
        (~h["Submitted"].tail(4)).sum()
    )

    if attendance_slope < -7:

        return (
            "Attendance has dropped sharply "
            "in recent weeks."
        )

    if quiz_slope < -4:

        return (
            "Quiz performance has been declining "
            "across recent weeks."
        )

    if missing >= 2:

        return (
            "Recent assignment submissions show "
            "a completion gap."
        )

    if float(h.iloc[-1]["Attendance"]) < 55:

        return (
            "Attendance is consistently low "
            "across the observed period."
        )

    return (
        "Multiple learning signals are below "
        "the class monitoring range."
    )


def risk_components(history):

    h = history.sort_values("Week")

    attendance_change = (
        float(
            h.iloc[-1]["Attendance"]
        )
        -
        float(
            h.iloc[0]["Attendance"]
        )
    )

    quiz_start = max(
        0,
        len(h) - 4
    )

    quiz_change = (
        float(
            h.iloc[-1]["Quiz"]
        )
        -
        float(
            h.iloc[quiz_start]["Quiz"]
        )
    )

    missing_recent = int(
        (~h["Submitted"].tail(4)).sum()
    )

    attendance_impact = max(
        0,
        -attendance_change
    )

    quiz_impact = max(
        0,
        -quiz_change
    )

    assignment_impact = (
        missing_recent * 20
    )

    values = [
        (
            "Attendance trend",
            attendance_impact,
            (
                f"Attendance changed "
                f"{h.iloc[0]['Attendance']:.0f}% "
                f"→ "
                f"{h.iloc[-1]['Attendance']:.0f}%"
            ),
        ),
        (
            "Quiz trend",
            quiz_impact,
            (
                f"Recent quiz performance changed "
                f"{h.iloc[quiz_start]['Quiz']:.0f} "
                f"→ "
                f"{h.iloc[-1]['Quiz']:.0f}"
            ),
        ),
        (
            "Assignment gap",
            assignment_impact,
            (
                f"{missing_recent} recent assignment(s) "
                f"were not submitted"
            ),
        ),
    ]

    total = sum(
        item[1]
        for item in values
    )

    if total == 0:
        total = 1

    return [
        (
            name,
            value / total * 100,
            explanation
        )
        for name, value, explanation
        in sorted(
            values,
            key=lambda x: x[1],
            reverse=True
        )
    ]


# ============================================================
# STUDENT SUMMARY TABLE
# ============================================================

student_rows = []

for student in df["Student"].unique():

    history = df[
        df["Student"] == student
    ].sort_values("Week")

    score = risk_score(history)

    label, tone = risk_label(
        score
    )

    student_rows.append(
        {
            "Student": student,
            "Risk": score,
            "Level": label,
            "Tone": tone,
            "Reason": primary_reason(
                history
            ),
            "Persona": history.iloc[0]["Persona"],
        }
    )


risk_df = (
    pd.DataFrame(student_rows)
    .sort_values(
        "Risk",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# SESSION STATE
# ============================================================

if "screen" not in st.session_state:
    st.session_state.screen = "Login"

if "selected_student" not in st.session_state:
    st.session_state.selected_student = (
        risk_df.iloc[0]["Student"]
    )

if "support_generated" not in st.session_state:
    st.session_state.support_generated = False

if "teacher_tasks" not in st.session_state:

    st.session_state.teacher_tasks = {
        "student": False,
        "parent": False,
        "revision": False,
        "followup": False,
    }


def navigate(screen):

    st.session_state.screen = screen


# ============================================================
# SCREEN 1 — LOGIN
# ============================================================

if st.session_state.screen == "Login":

    st.markdown(
        '<div class="login-wrap">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand">'
        'AI-Based Early Detection of Students at Academic Risk'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="login-title">'
        'See the struggle early.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="login-sub">'
        'An explainable early-warning and intervention system '
        'that helps teachers identify emerging academic risk, '
        'understand likely drivers, provide targeted support, '
        'and track the student trajectory.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="demo-strip">'
        'Demo environment: synthetic classroom data. '
        'No real student records are used.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    login_left, login_right = st.columns(2)

    with login_left:

        if st.button(
            "Enter as Teacher (Demo)",
            use_container_width=True,
            key="teacher_login"
        ):

            navigate("Dashboard")
            st.rerun()

    with login_right:

        if st.button(
            "Enter as Student (Demo)",
            use_container_width=True,
            key="student_login"
        ):

            navigate("Dashboard")
            st.rerun()

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# TOP BAR
# ============================================================

st.markdown(
    """
    <div class="topbar">
        <div class="brand">
            AI-Based Early Detection of Students at Academic Risk
        </div>
        <div class="top-subtitle">
            Detect early · Explain clearly · Intervene · Track
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# NAVIGATION
# ============================================================

navigation_columns = st.columns(5)

navigation_items = [
    ("Dashboard", "Dashboard"),
    ("Student Profile", "Student Profile"),
    ("Intervention", "Intervention"),
    ("Track", "Track"),
    ("Login", "Login"),
]

for index, (label, target) in enumerate(
    navigation_items
):

    with navigation_columns[index]:

        if st.button(
            label,
            use_container_width=True,
            key=f"navigation_{label}"
        ):

            navigate(target)
            st.rerun()


# ============================================================
# SCREEN 2 — TEACHER DASHBOARD
# ============================================================

if st.session_state.screen == "Dashboard":

    st.markdown(
        "## Teacher Dashboard"
    )

    selector_one, selector_two, selector_three = (
        st.columns(3)
    )

    with selector_one:

        st.selectbox(
            "Class",
            ["Class 8-B"],
            key="class_selector"
        )

    with selector_two:

        st.selectbox(
            "Week",
            ["Week 8"],
            key="week_selector"
        )

    with selector_three:

        st.selectbox(
            "Monitoring Mode",
            ["Early-warning"],
            key="monitoring_mode"
        )

    st.write("")

    total_students = 60

    needs_support = int(
        (
            risk_df["Risk"] >= 70
        ).sum()
    )

    recovering = int(
        (
            risk_df["Persona"]
            == "Recovering"
        ).sum()
    )

    average_attendance = float(
        df[
            df["Week"] == 8
        ]["Attendance"].mean()
    )

    metric_one, metric_two, metric_three, metric_four = (
        st.columns(4)
    )

    with metric_one:

        st.markdown(
            f"""
            <div class="card">
                <div class="metric-label">
                    Students
                </div>
                <div class="metric-value">
                    {total_students}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric_two:

        st.markdown(
            f"""
            <div class="card">
                <div class="metric-label">
                    Needs Support
                </div>
                <div class="metric-value">
                    {needs_support}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric_three:

        st.markdown(
            f"""
            <div class="card">
                <div class="metric-label">
                    Recovering
                </div>
                <div class="metric-value">
                    {recovering}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric_four:

        st.markdown(
            f"""
            <div class="card">
                <div class="metric-label">
                    Avg Attendance
                </div>
                <div class="metric-value">
                    {average_attendance:.0f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">'
        'Class Risk Heatmap'
        '</div>',
        unsafe_allow_html=True
    )

    heatmap_html = """
    <table class="heatmap">
        <tr>
            <th style="text-align:left">
                Student
            </th>
            <th>W1</th>
            <th>W2</th>
            <th>W3</th>
            <th>W4</th>
            <th>W5</th>
            <th>W6</th>
            <th>W7</th>
            <th>W8</th>
        </tr>
    """

    for student in df["Student"].unique():

        student_data = df[
            df["Student"] == student
        ].sort_values("Week")

        heatmap_html += (
            "<tr>"
            f"<td style='text-align:left;font-size:9px'>"
            f"{student}"
            f"</td>"
        )

        for week in range(1, 9):

            partial_history = student_data[
                student_data["Week"] <= week
            ]

            score = risk_score(
                partial_history
            )

            if score < 35:
                css_class = "h-green"

            elif score < 55:
                css_class = "h-yellow"

            elif score < 70:
                css_class = "h-orange"

            else:
                css_class = "h-red"

            heatmap_html += (
                f"<td class='{css_class}' "
                f"title='Risk {score:.0f}/100'>"
                f"{score:.0f}"
                f"</td>"
            )

        heatmap_html += "</tr>"

    heatmap_html += "</table>"

    st.markdown(
        heatmap_html,
        unsafe_allow_html=True
    )

    st.caption(
        "Synthetic classroom data. Each cell represents "
        "the rule-based risk score using information "
        "available up to that week."
    )

    st.markdown(
        '<div class="section-title">'
        'Needs Support'
        '</div>',
        unsafe_allow_html=True
    )

    for _, row in risk_df.head(5).iterrows():

        col_student, col_score, col_reason, col_button = (
            st.columns(
                [1.2, 0.8, 3.6, 1]
            )
        )

        with col_student:

            st.write(
                f"**{row['Student']}**"
            )

        with col_score:

            st.write(
                f"**{row['Risk']:.0f}/100**"
            )

        with col_reason:

            st.markdown(
                f'<div class="reason">'
                f'{row["Reason"]}'
                f'</div>',
                unsafe_allow_html=True
            )

        with col_button:

            if st.button(
                "View",
                key=f"dashboard_view_{row['Student']}"
            ):

                st.session_state.selected_student = (
                    row["Student"]
                )

                navigate(
                    "Student Profile"
                )

                st.rerun()


# ============================================================
# SCREEN 3 — STUDENT PROFILE
# ============================================================

elif st.session_state.screen == "Student Profile":

    st.markdown(
        "## Student Profile"
    )

    students = risk_df[
        "Student"
    ].tolist()

    if (
        st.session_state.selected_student
        in students
    ):

        selected_index = students.index(
            st.session_state.selected_student
        )

    else:

        selected_index = 0

    selected = st.selectbox(
        "Student",
        students,
        index=selected_index,
        key="profile_student"
    )

    st.session_state.selected_student = selected

    history = df[
        df["Student"] == selected
    ].sort_values("Week")

    current_score = risk_score(
        history
    )

    current_label, current_tone = risk_label(
        current_score
    )

    left, right = st.columns(
        [0.95, 1.35]
    )

    # --------------------------------------------------------
    # LEFT SIDE
    # --------------------------------------------------------

    with left:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            f"### {selected}"
        )

        st.markdown(
            f'<span class="badge badge-{current_tone}">'
            f'{current_label}'
            f'</span>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="
                font-size:64px;
                font-weight:800;
                margin-top:12px
            ">
                {current_score:.0f}
                <span style="
                    font-size:20px;
                    color:#667085
                ">
                    /100
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            "Explainable rule-based academic risk score"
        )

        st.markdown(
            "### Top Risk Factors"
        )

        factors = risk_components(
            history
        )

        for index, (
            name,
            impact,
            explanation
        ) in enumerate(
            factors[:3]
        ):

            if index == 0:
                fill_class = "fill-red"

            elif index == 1:
                fill_class = "fill-orange"

            else:
                fill_class = "fill-blue"

            st.markdown(
                f"""
                <div class="factor">
                    <div class="factor-top">
                        <span>{name}</span>
                        <span>{impact:.0f}%</span>
                    </div>

                    <div class="bar">
                        <div class="{fill_class}"
                             style="
                                width:
                                {min(impact,100):.0f}%
                             ">
                        </div>
                    </div>

                    <div class="small-note">
                        {explanation}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        if st.button(
            "Explain this risk",
            use_container_width=True,
            key="explain_risk"
        ):

            st.info(
                f"{selected} is currently classified as "
                f"'{current_label}'. The explanation is based "
                f"on observed attendance, quiz, and assignment "
                f"signals. Current attendance is "
                f"{history.iloc[-1]['Attendance']:.0f}% "
                f"and current quiz performance is "
                f"{history.iloc[-1]['Quiz']:.0f}%. "
                f"This is a decision-support signal and "
                f"should be reviewed by a teacher."
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # RIGHT SIDE
    # --------------------------------------------------------

    with right:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            "### Attendance and Quiz Trend"
        )

        trend_chart = go.Figure()

        trend_chart.add_trace(
            go.Scatter(
                x=history["Week"],
                y=history["Attendance"],
                mode="lines+markers",
                name="Attendance",
            )
        )

        trend_chart.add_trace(
            go.Scatter(
                x=history["Week"],
                y=history["Quiz"],
                mode="lines+markers",
                name="Quiz",
            )
        )

        trend_chart.update_layout(
            height=300,
            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10
            ),
            yaxis=dict(
                range=[0, 100],
                title="Percent"
            ),
            xaxis=dict(
                title="Week",
                dtick=1
            ),
            legend=dict(
                orientation="h"
            ),
        )

        st.plotly_chart(
            trend_chart,
            use_container_width=True
        )

        st.markdown(
            "### Time Travel"
        )

        selected_week = st.slider(
            "Replay week",
            min_value=1,
            max_value=8,
            value=4,
            step=1,
            key="profile_time_travel"
        )

        snapshot_history = history[
            history["Week"] <= selected_week
        ]

        snapshot = history[
            history["Week"] == selected_week
        ].iloc[0]

        snapshot_score = risk_score(
            snapshot_history
        )

        week_col, risk_col, attendance_col = (
            st.columns(3)
        )

        with week_col:

            st.metric(
                "Week",
                selected_week
            )

        with risk_col:

            st.metric(
                "Risk",
                f"{snapshot_score:.0f}/100"
            )

        with attendance_col:

            st.metric(
                "Attendance",
                f"{snapshot['Attendance']:.0f}%"
            )

        if selected_week == 4:

            st.warning(
                "Flagged for teacher review in Week 4 "
                "in this synthetic demonstration."
            )

        elif selected_week == 8:

            st.info(
                "Week 8 snapshot. Review the complete "
                "trajectory before deciding the next "
                "support step."
            )

        else:

            st.caption(
                "Move the slider to replay the "
                "student's trajectory."
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


# ============================================================
# SCREEN 4 — INTERVENTION
# ============================================================

elif st.session_state.screen == "Intervention":

    st.markdown(
        "## Intervention"
    )

    students = risk_df[
        "Student"
    ].tolist()

    if (
        st.session_state.selected_student
        in students
    ):

        selected_index = students.index(
            st.session_state.selected_student
        )

    else:

        selected_index = 0

    selected = st.selectbox(
        "Student",
        students,
        index=selected_index,
        key="intervention_student"
    )

    st.session_state.selected_student = selected

    history = df[
        df["Student"] == selected
    ].sort_values("Week")

    score = risk_score(
        history
    )

    label, tone = risk_label(
        score
    )

    reason = primary_reason(
        history
    )

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        f"### {selected}"
    )

    st.markdown(
        f'<span class="badge badge-{tone}">'
        f'{label}'
        f'</span>',
        unsafe_allow_html=True
    )

    st.write("")

    st.write(
        f"**Observed pattern:** {reason}"
    )

    st.markdown(
        '<div class="info-strip">'
        'Support recommendations are generated from the '
        'observed synthetic signals. They are intended to '
        'assist teacher review, not replace it.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "Generate Support Plan",
        use_container_width=True,
        key="generate_support"
    ):

        st.session_state.support_generated = True

    if st.session_state.support_generated:

        plan_left, plan_right = (
            st.columns(2)
        )

        # ----------------------------------------------------
        # REVISION PLAN
        # ----------------------------------------------------

        with plan_left:

            st.markdown(
                "### Revision Plan"
            )

            st.write(
                "**Day 1–2:** Review the weakest "
                "recent learning area."
            )

            st.write(
                "**Day 3:** Complete a short "
                "targeted practice set."
            )

            st.write(
                "**Day 4:** Teacher check-in "
                "and feedback."
            )

            st.write(
                "**Day 5:** Reassess the relevant "
                "skill and update support."
            )

        # ----------------------------------------------------
        # TELUGU PARENT COMMUNICATION
        # ----------------------------------------------------

        with plan_right:

            st.markdown(
                "### Parent Communication"
            )

            st.info(
                "నమస్కారం గారు, మీ పిల్లవారి చదువులో "
                "కొన్ని సహాయం అవసరమైన సంకేతాలు కనిపిస్తున్నాయి. "
                "కలిసి కొంత అదనపు సహాయం అందిస్తే చదువులో "
                "మెరుగుదల రావడానికి అవకాశం ఉంటుంది. "
                "ఉపాధ్యాయుడితో కలిసి తదుపరి చర్యలను "
                "చర్చిద్దాం."
            )

            st.button(
                "Play Telugu Voice Message",
                disabled=True,
                use_container_width=True,
                key="voice_demo"
            )

            st.caption(
                "Prototype transcript. Voice playback is "
                "not connected to an external voice service "
                "in this version."
            )

        # ----------------------------------------------------
        # TEACHER CHECKLIST
        # ----------------------------------------------------

        st.markdown(
            "### Teacher Checklist"
        )

        task_student = st.checkbox(
            "Talk to the student",
            value=st.session_state.teacher_tasks[
                "student"
            ],
            key="task_student"
        )

        task_parent = st.checkbox(
            "Contact parent or guardian",
            value=st.session_state.teacher_tasks[
                "parent"
            ],
            key="task_parent"
        )

        task_revision = st.checkbox(
            "Assign targeted revision",
            value=st.session_state.teacher_tasks[
                "revision"
            ],
            key="task_revision"
        )

        task_followup = st.checkbox(
            "Schedule follow-up review",
            value=st.session_state.teacher_tasks[
                "followup"
            ],
            key="task_followup"
        )

        st.session_state.teacher_tasks = {
            "student": task_student,
            "parent": task_parent,
            "revision": task_revision,
            "followup": task_followup,
        }

        completed_tasks = sum(
            st.session_state.teacher_tasks.values()
        )

        st.progress(
            completed_tasks / 4
        )

        st.caption(
            f"{completed_tasks} of 4 support actions completed."
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# SCREEN 5 — TRACK
# ============================================================

elif st.session_state.screen == "Track":

    st.markdown(
        "## Track"
    )

    recovering_students = risk_df[
        risk_df["Persona"] == "Recovering"
    ]["Student"].tolist()

    if recovering_students:

        selected = st.selectbox(
            "Recovering Student",
            recovering_students,
            key="track_student"
        )

    else:

        selected = st.selectbox(
            "Student",
            risk_df["Student"].tolist(),
            key="track_student_fallback"
        )

    history = df[
        df["Student"] == selected
    ].sort_values("Week")

    weekly_risks = []

    for week in range(1, 9):

        weekly_history = history[
            history["Week"] <= week
        ]

        weekly_risks.append(
            risk_score(
                weekly_history
            )
        )

    week_three_risk = weekly_risks[2]
    week_eight_risk = weekly_risks[7]

    trajectory_change = (
        week_eight_risk
        - week_three_risk
    )

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    track_one, track_two, track_three = (
        st.columns(3)
    )

    with track_one:

        st.metric(
            "Risk at Week 3",
            f"{week_three_risk:.0f}/100"
        )

    with track_two:

        st.metric(
            "Risk at Week 8",
            f"{week_eight_risk:.0f}/100"
        )

    with track_three:

        st.metric(
            "Trajectory Change",
            f"{trajectory_change:+.0f} pts"
        )

    track_chart = go.Figure()

    track_chart.add_trace(
        go.Scatter(
            x=list(range(1, 9)),
            y=weekly_risks,
            mode="lines+markers",
            name="Risk trajectory",
        )
    )

    track_chart.add_hline(
        y=70,
        line_dash="dash",
        annotation_text="Needs Support"
    )

    track_chart.add_hline(
        y=55,
        line_dash="dot",
        annotation_text="Monitor Closely"
    )

    track_chart.update_layout(
        height=400,
        yaxis=dict(
            range=[0, 100],
            title="Risk Score"
        ),
        xaxis=dict(
            title="Week",
            dtick=1
        ),
        margin=dict(
            l=10,
            r=10,
            t=25,
            b=10
        )
    )

    st.plotly_chart(
        track_chart,
        use_container_width=True
    )

    if trajectory_change < 0:

        st.success(
            "Recovering trajectory shown in this "
            "synthetic demonstration."
        )

    else:

        st.info(
            "This trajectory is shown for demonstration "
            "and requires teacher interpretation."
        )

    st.markdown(
        '<div class="demo-strip">'
        'Important: this is synthetic data. A lower risk '
        'score after support does not by itself prove '
        'that the support intervention caused the change.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )
