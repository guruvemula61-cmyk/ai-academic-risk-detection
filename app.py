import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import plotly.express as px

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Academic Risk Detection",
    page_icon="🎓",
    layout="wide"
)

# =========================================================
# STYLE
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #64748b;
    margin-bottom: 25px;
}

.risk-card {
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    background: #ffffff;
    margin: 10px 0;
}

.insight-card {
    padding: 18px;
    border-radius: 14px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
}

.small-text {
    color: #64748b;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SYNTHETIC STUDENT DATA
# =========================================================

@st.cache_data
def create_student_data():

    rng = np.random.default_rng(42)

    students = [
        "Aarav", "Ananya", "Arjun", "Bhavya", "Charan",
        "Deepika", "Esha", "Harsha", "Ishaan", "Kavya",
        "Manoj", "Meera", "Nikhil", "Pooja", "Rahul",
        "Sanjana", "Sathwik", "Sneha", "Tejas", "Varsha"
    ]

    data = []

    for index, student in enumerate(students):

        # Different synthetic learning patterns
        pattern = index % 5

        for week in range(1, 9):

            if pattern == 0:
                # Stable learner
                attendance_base = 94
                quiz_base = 88
                assignment_base = 94
                engagement_base = 90
                mastery_base = 88

            elif pattern == 1:
                # Academic difficulty
                decline = (week - 1) * 1.6
                attendance_base = 91
                quiz_base = 84 - decline * 2
                assignment_base = 90 - decline * 0.8
                engagement_base = 87
                mastery_base = 84 - decline * 2.3

            elif pattern == 2:
                # Disengagement
                decline = (week - 1) * 1.8
                attendance_base = 94 - decline * 1.5
                quiz_base = 86
                assignment_base = 91
                engagement_base = 90 - decline * 2.4
                mastery_base = 86

            elif pattern == 3:
                # Sudden decline
                if week < 6:
                    attendance_base = 94
                    quiz_base = 90
                    assignment_base = 94
                    engagement_base = 92
                    mastery_base = 89
                else:
                    drop = (week - 5) * 7
                    attendance_base = 94 - drop
                    quiz_base = 90 - drop
                    assignment_base = 94 - drop * 0.8
                    engagement_base = 92 - drop
                    mastery_base = 89 - drop * 0.7

            else:
                # Chronic struggler
                attendance_base = 76
                quiz_base = 55
                assignment_base = 61
                engagement_base = 58
                mastery_base = 52

            attendance = np.clip(
                attendance_base + rng.normal(0, 2.5),
                35, 100
            )

            quiz_score = np.clip(
                quiz_base + rng.normal(0, 4),
                20, 100
            )

            assignment_completion = np.clip(
                assignment_base + rng.normal(0, 3),
                20, 100
            )

            engagement = np.clip(
                engagement_base + rng.normal(0, 4),
                15, 100
            )

            concept_mastery = np.clip(
                mastery_base + rng.normal(0, 4),
                15, 100
            )

            data.append([
                student,
                week,
                round(attendance, 1),
                round(quiz_score, 1),
                round(assignment_completion, 1),
                round(engagement, 1),
                round(concept_mastery, 1)
            ])

    return pd.DataFrame(
        data,
        columns=[
            "Student",
            "Week",
            "Attendance",
            "Quiz Score",
            "Assignment Completion",
            "Engagement",
            "Concept Mastery"
        ]
    )


df = create_student_data()


# =========================================================
# MACHINE LEARNING MODEL
# =========================================================

FEATURES = [
    "Attendance",
    "Quiz Score",
    "Assignment Completion",
    "Engagement",
    "Concept Mastery"
]


@st.cache_resource
def train_model():

    rng = np.random.default_rng(10)

    X = rng.uniform(
        25,
        100,
        size=(2000, 5)
    )

    risk_signal = (
        7
        - 0.035 * X[:, 0]
        - 0.045 * X[:, 1]
        - 0.025 * X[:, 2]
        - 0.020 * X[:, 3]
        - 0.040 * X[:, 4]
        + rng.normal(0, 0.45, 2000)
    )

    probability = 1 / (
        1 + np.exp(-risk_signal)
    )

    y = (
        probability > 0.5
    ).astype(int)

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(X, y)

    return model


model = train_model()

df["Risk Score"] = (
    model.predict_proba(
        df[FEATURES]
    )[:, 1] * 100
)


# =========================================================
# RISK LEVEL
# =========================================================

def risk_level(score):

    if score >= 70:
        return "High Risk"

    if score >= 40:
        return "Moderate Risk"

    return "Low Risk"


df["Risk Level"] = df[
    "Risk Score"
].apply(risk_level)


# =========================================================
# LATEST STUDENT RECORD
# =========================================================

latest = (
    df.sort_values("Week")
    .groupby("Student")
    .tail(1)
    .copy()
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def calculate_risk(values):

    values = np.array(
        values
    ).reshape(1, -1)

    return float(
        model.predict_proba(values)[0][1] * 100
    )


def linear_slope(values):

    if len(values) < 2:
        return 0.0

    x = np.arange(len(values))

    return float(
        np.polyfit(x, values, 1)[0]
    )


def get_trend_information(history):

    recent = history.tail(3)

    risk_values = recent[
        "Risk Score"
    ].values

    slope = linear_slope(
        risk_values
    )

    if slope > 0.5:
        direction = "Increasing"

    elif slope < -0.5:
        direction = "Improving"

    else:
        direction = "Stable"

    return slope, direction


def forecast_threshold(
    current_risk,
    slope,
    threshold=70
):

    if current_risk >= threshold:
        return 0

    if slope <= 0:
        return None

    weeks = (
        threshold - current_risk
    ) / slope

    return max(
        1,
        int(np.ceil(weeks))
    )


def root_cause_classification(history):

    latest_row = history.iloc[-1]

    recent = history.tail(3)

    attendance_slope = linear_slope(
        recent["Attendance"].values
    )

    quiz_slope = linear_slope(
        recent["Quiz Score"].values
    )

    engagement_slope = linear_slope(
        recent["Engagement"].values
    )

    mastery_slope = linear_slope(
        recent["Concept Mastery"].values
    )

    avg_quiz = history[
        "Quiz Score"
    ].mean()

    avg_mastery = history[
        "Concept Mastery"
    ].mean()

    avg_engagement = history[
        "Engagement"
    ].mean()

    current_quiz = latest_row[
        "Quiz Score"
    ]

    current_mastery = latest_row[
        "Concept Mastery"
    ]

    current_attendance = latest_row[
        "Attendance"
    ]

    current_engagement = latest_row[
        "Engagement"
    ]

    # -----------------------------------------------------
    # SUDDEN DECLINE
    # -----------------------------------------------------

    if len(history) >= 5:

        previous = history.iloc[-5]

        recent_drop = (
            previous["Quiz Score"]
            - current_quiz
        )

        if recent_drop >= 15:

            return (
                "Sudden Decline",
                "Recent performance changed sharply "
                "after a relatively stable period."
            )

    # -----------------------------------------------------
    # CHRONIC STRUGGLER
    # -----------------------------------------------------

    if (
        avg_quiz < 65
        and avg_mastery < 65
        and avg_engagement < 70
    ):

        return (
            "Chronic Struggler",
            "Performance has remained consistently low "
            "across multiple learning signals."
        )

    # -----------------------------------------------------
    # DISENGAGEMENT
    # -----------------------------------------------------

    if (
        engagement_slope < -1
        or attendance_slope < -1
    ):

        if (
            current_quiz >= 70
            and current_mastery >= 70
        ):

            return (
                "Disengagement",
                "Attendance or engagement is declining "
                "while academic performance remains comparatively stronger."
            )

    # -----------------------------------------------------
    # ACADEMIC DIFFICULTY
    # -----------------------------------------------------

    if (
        mastery_slope < -1
        or quiz_slope < -1
        or current_mastery < 65
        or current_quiz < 65
    ):

        return (
            "Academic Difficulty",
            "Assessment performance or concept mastery "
            "shows signs of learning difficulty."
        )

    # -----------------------------------------------------
    # GENERAL
    # -----------------------------------------------------

    return (
        "Emerging Risk",
        "Multiple learning signals suggest that "
        "additional monitoring may be useful."
    )


def get_intervention(cause):

    interventions = {

        "Academic Difficulty": [
            "Identify the weakest concepts or topics.",
            "Provide targeted revision material.",
            "Use guided practice questions.",
            "Conduct a short mastery reassessment."
        ],

        "Disengagement": [
            "Schedule a short mentor or teacher check-in.",
            "Review attendance and classroom participation.",
            "Introduce interactive learning activities.",
            "Monitor engagement during the next cycle."
        ],

        "Sudden Decline": [
            "Prioritize teacher review.",
            "Review recent changes in learning behaviour.",
            "Use supportive parent communication.",
            "Monitor the student closely over the next learning cycle."
        ],

        "Chronic Struggler": [
            "Create an individualized learning plan.",
            "Break learning goals into smaller milestones.",
            "Provide regular guided practice.",
            "Review progress weekly."
        ],

        "Emerging Risk": [
            "Continue close monitoring.",
            "Identify the earliest changing learning signal.",
            "Provide targeted academic support.",
            "Reassess after the next learning cycle."
        ]
    }

    return interventions.get(
        cause,
        interventions["Emerging Risk"]
    )


def get_factor_changes(history):

    if len(history) < 3:
        return []

    first = history.iloc[0]
    current = history.iloc[-1]

    changes = []

    for feature in FEATURES:

        change = (
            current[feature]
            - first[feature]
        )

        changes.append(
            (feature, change)
        )

    return sorted(
        changes,
        key=lambda x: x[1]
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    'AI-Based Early Detection of Students at Academic Risk'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Detect early • Identify why • Forecast risk • Recommend action • Track outcomes'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎓 Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Student Analysis",
        "Time Travel",
        "What-If Simulator",
        "Intervention Center"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Prototype uses synthetic classroom data "
    "for demonstration."
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    total_students = len(latest)

    high_risk = len(
        latest[
            latest["Risk Level"] == "High Risk"
        ]
    )

    moderate_risk = len(
        latest[
            latest["Risk Level"] == "Moderate Risk"
        ]
    )

    average_risk = latest[
        "Risk Score"
    ].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Students Monitored",
        total_students
    )

    col2.metric(
        "🔴 High Risk",
        high_risk
    )

    col3.metric(
        "🟡 Moderate Risk",
        moderate_risk
    )

    col4.metric(
        "Average Risk",
        f"{average_risk:.1f}%"
    )

    # -----------------------------------------------------
    # SMART ALERTS
    # -----------------------------------------------------

    st.markdown("## 🚨 Smart Early-Warning Center")

    alerts = []

    for student in latest["Student"]:

        history = df[
            df["Student"] == student
        ].sort_values("Week")

        current = history.iloc[-1]

        slope, direction = get_trend_information(
            history
        )

        if current["Risk Score"] >= 70:

            alerts.append(
                f"🔴 **{student}** — High estimated risk "
                f"({current['Risk Score']:.1f}%)."
            )

        elif slope > 3:

            alerts.append(
                f"⚠️ **{student}** — Risk is increasing rapidly "
                f"({slope:+.1f} points/week)."
            )

    if alerts:

        for alert in alerts[:8]:
            st.warning(alert)

    else:

        st.success(
            "No immediate high-priority warning signals detected."
        )

    # -----------------------------------------------------
    # RISK CHART
    # -----------------------------------------------------

    st.markdown("## 📊 Class Risk Overview")

    chart_data = latest.sort_values(
        "Risk Score",
        ascending=True
    )

    fig = px.bar(
        chart_data,
        x="Risk Score",
        y="Student",
        orientation="h",
        text=chart_data[
            "Risk Score"
        ].round(0),
        labels={
            "Risk Score": "Academic Risk Score (%)",
            "Student": "Student"
        }
    )

    fig.update_layout(
        height=650,
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------------------------------
    # ROOT CAUSE DISTRIBUTION
    # -----------------------------------------------------

    st.markdown("## 🧠 Root-Cause Distribution")

    causes = []

    for student in latest["Student"]:

        history = df[
            df["Student"] == student
        ].sort_values("Week")

        cause, _ = root_cause_classification(
            history
        )

        causes.append({
            "Student": student,
            "Root Cause": cause
        })

    cause_df = pd.DataFrame(causes)

    cause_counts = (
        cause_df[
            "Root Cause"
        ]
        .value_counts()
        .reset_index()
    )

    cause_counts.columns = [
        "Root Cause",
        "Students"
    ]

    fig2 = px.pie(
        cause_counts,
        names="Root Cause",
        values="Students",
        hole=0.45
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# =========================================================
# STUDENT ANALYSIS
# =========================================================

elif page == "Student Analysis":

    st.title("🧠 Explainable Student Risk Analysis")

    selected_student = st.selectbox(
        "Select Student",
        sorted(
            latest["Student"].unique()
        )
    )

    history = df[
        df["Student"] == selected_student
    ].sort_values("Week")

    current = history.iloc[-1]

    current_risk = current[
        "Risk Score"
    ]

    cause, cause_explanation = (
        root_cause_classification(
            history
        )
    )

    slope, direction = (
        get_trend_information(
            history
        )
    )

    forecast = forecast_threshold(
        current_risk,
        slope
    )

    # -----------------------------------------------------
    # TOP METRICS
    # -----------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Current Risk",
        f"{current_risk:.1f}%"
    )

    c2.metric(
        "Risk Level",
        risk_level(current_risk)
    )

    c3.metric(
        "Root Cause",
        cause
    )

    c4.metric(
        "Trend",
        direction
    )

    # -----------------------------------------------------
    # EXPLAINABLE RISK CARD
    # -----------------------------------------------------

    st.markdown("## 🔎 Explainable Risk Card")

    st.markdown(
        f"""
        <div class="risk-card">

        <h3>⚠️ {risk_level(current_risk)}</h3>

        <p>
        <b>Primary pattern:</b> {cause}
        </p>

        <p>
        <b>Why?</b> {cause_explanation}
        </p>

        <p>
        <b>Trend velocity:</b>
        {slope:+.2f} risk points/week
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # EARLY FORECAST
    # -----------------------------------------------------

    st.markdown("## 🔮 Early Risk Forecast")

    if forecast == 0:

        st.warning(
            "The student is already above the high-risk threshold. "
            "Teacher review is recommended."
        )

    elif forecast is not None:

        st.warning(
            f"If the current trend continues, the model estimates "
            f"that the student could reach the 70% risk threshold "
            f"in approximately {forecast} week(s)."
        )

    else:

        st.success(
            "The current trend does not indicate movement toward "
            "the high-risk threshold."
        )

    # -----------------------------------------------------
    # SIGNAL CHANGES
    # -----------------------------------------------------

    st.markdown("## 📉 Learning Signal Changes")

    changes = get_factor_changes(
        history
    )

    for feature, change in changes:

        if change < -5:

            st.warning(
                f"**{feature}** decreased by "
                f"{abs(change):.1f} points across the observed period."
            )

        elif change > 5:

            st.success(
                f"**{feature}** improved by "
                f"{change:.1f} points."
            )

    # -----------------------------------------------------
    # TRAJECTORY
    # -----------------------------------------------------

    st.markdown("## 📈 Learning Trajectory")

    fig = px.line(
        history,
        x="Week",
        y="Risk Score",
        markers=True,
        labels={
            "Risk Score":
            "Academic Risk Score (%)"
        }
    )

    fig.add_hline(
        y=70,
        line_dash="dash",
        annotation_text="High Risk"
    )

    fig.add_hline(
        y=40,
        line_dash="dot",
        annotation_text="Moderate Risk"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------------------------------
    # ACTION
    # -----------------------------------------------------

    st.markdown("## 🎯 Recommended Action")

    actions = get_intervention(
        cause
    )

    for i, action in enumerate(
        actions,
        1
    ):

        st.write(
            f"**{i}.** {action}"
        )


# =========================================================
# TIME TRAVEL
# =========================================================

elif page == "Time Travel":

    st.title("⏪ Time Travel")

    st.write(
        "Replay the student's academic journey to identify "
        "when warning signals began changing."
    )

    selected_student = st.selectbox(
        "Choose Student",
        sorted(
            latest["Student"].unique()
        ),
        key="time_student"
    )

    history = df[
        df["Student"] == selected_student
    ].sort_values("Week")

    fig = px.line(
        history,
        x="Week",
        y="Risk Score",
        markers=True,
        labels={
            "Risk Score":
            "Academic Risk Score (%)"
        }
    )

    fig.add_hline(
        y=70,
        line_dash="dash",
        annotation_text="High Risk"
    )

    fig.add_hline(
        y=40,
        line_dash="dot",
        annotation_text="Moderate Risk"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    selected_week = st.slider(
        "Replay Week",
        1,
        8,
        4
    )

    snapshot = history[
        history["Week"] == selected_week
    ].iloc[0]

    st.markdown(
        f"## Week {selected_week} Snapshot"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Attendance",
        f"{snapshot['Attendance']:.0f}%"
    )

    c2.metric(
        "Quiz",
        f"{snapshot['Quiz Score']:.0f}%"
    )

    c3.metric(
        "Assignments",
        f"{snapshot['Assignment Completion']:.0f}%"
    )

    c4.metric(
        "Engagement",
        f"{snapshot['Engagement']:.0f}%"
    )

    c5.metric(
        "Mastery",
        f"{snapshot['Concept Mastery']:.0f}%"
    )

    if selected_week > 1:

        previous = history[
            history["Week"] < selected_week
        ].iloc[-1]

        risk_change = (
            snapshot["Risk Score"]
            - previous["Risk Score"]
        )

        if risk_change > 3:

            st.warning(
                f"⚠️ Risk increased by "
                f"{risk_change:.1f} points from the previous week."
            )

        elif risk_change < -3:

            st.success(
                f"Risk decreased by "
                f"{abs(risk_change):.1f} points from the previous week."
            )

        else:

            st.info(
                "Risk remained relatively stable compared "
                "with the previous week."
            )


# =========================================================
# WHAT-IF SIMULATOR
# =========================================================

elif page == "What-If Simulator":

    st.title("🔮 What-If Intervention Simulator")

    st.write(
        "Simulate hypothetical improvements in learning signals "
        "and observe how the model's estimated risk score changes."
    )

    st.info(
        "Simulation only — this is not a guaranteed prediction "
        "of future performance."
    )

    selected_student = st.selectbox(
        "Select Student",
        sorted(
            latest["Student"].unique()
        ),
        key="what_student"
    )

    current = latest[
        latest["Student"] == selected_student
    ].iloc[0]

    current_values = [
        current[f]
        for f in FEATURES
    ]

    current_risk = calculate_risk(
        current_values
    )

    cause, _ = root_cause_classification(
        df[
            df["Student"] == selected_student
        ].sort_values("Week")
    )

    st.markdown(
        f"### Primary Root Cause: **{cause}**"
    )

    st.markdown("## 🧪 Simulate Support")

    attendance_change = st.slider(
        "Attendance improvement",
        0,
        15,
        0
    )

    quiz_change = st.slider(
        "Assessment improvement",
        0,
        20,
        0
    )

    assignment_change = st.slider(
        "Assignment completion improvement",
        0,
        20,
        0
    )

    engagement_change = st.slider(
        "Engagement improvement",
        0,
        20,
        0
    )

    mastery_change = st.slider(
        "Concept mastery improvement",
        0,
        20,
        0
    )

    simulated_values = [

        min(
            100,
            current["Attendance"]
            + attendance_change
        ),

        min(
            100,
            current["Quiz Score"]
            + quiz_change
        ),

        min(
            100,
            current["Assignment Completion"]
            + assignment_change
        ),

        min(
            100,
            current["Engagement"]
            + engagement_change
        ),

        min(
            100,
            current["Concept Mastery"]
            + mastery_change
        )
    ]

    simulated_risk = calculate_risk(
        simulated_values
    )

    difference = (
        current_risk
        - simulated_risk
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Current Risk",
        f"{current_risk:.1f}%"
    )

    c2.metric(
        "Simulated Risk",
        f"{simulated_risk:.1f}%"
    )

    c3.metric(
        "Estimated Change",
        f"{difference:+.1f} pts"
    )

    comparison = pd.DataFrame({
        "State": [
            "Current",
            "Simulated"
        ],
        "Risk": [
            current_risk,
            simulated_risk
        ]
    })

    fig = px.bar(
        comparison,
        x="State",
        y="Risk",
        text="Risk",
        range_y=[0, 100],
        title="Current vs Simulated Risk"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if difference > 0:

        st.success(
            f"The simulated scenario reduces the model's "
            f"estimated risk by {difference:.1f} points."
        )

    else:

        st.info(
            "Try adjusting the intervention variables "
            "to explore another scenario."
        )


# =========================================================
# INTERVENTION CENTER
# =========================================================

else:

    st.title("🎯 Intervention Center")

    selected_student = st.selectbox(
        "Select Student",
        sorted(
            latest["Student"].unique()
        ),
        key="intervention_student"
    )

    history = df[
        df["Student"] == selected_student
    ].sort_values("Week")

    current = history.iloc[-1]

    current_risk = current[
        "Risk Score"
    ]

    cause, explanation = (
        root_cause_classification(
            history
        )
    )

    # -----------------------------------------------------
    # ROOT CAUSE
    # -----------------------------------------------------

    st.markdown("## 🧠 Detected Root Cause")

    st.info(
        f"**{cause}** — {explanation}"
    )

    # -----------------------------------------------------
    # INTERVENTION PLAN
    # -----------------------------------------------------

    st.markdown("## 🎯 Personalized Support Plan")

    actions = get_intervention(
        cause
    )

    for i, action in enumerate(
        actions,
        1
    ):

        st.write(
            f"**{i}.** {action}"
        )

    # -----------------------------------------------------
    # TRACKER
    # -----------------------------------------------------

    st.markdown("## 📈 Intervention Impact Tracker")

    initial_risk = history.iloc[0][
        "Risk Score"
    ]

    latest_risk = history.iloc[-1][
        "Risk Score"
    ]

    change = (
        latest_risk
        - initial_risk
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Initial Risk",
        f"{initial_risk:.1f}%"
    )

    c2.metric(
        "Current Risk",
        f"{latest_risk:.1f}%"
    )

    c3.metric(
        "Trajectory Change",
        f"{change:+.1f} pts"
    )

    if change < -2:

        st.success(
            "Estimated risk has decreased across the observed period."
        )

    elif change > 2:

        st.warning(
            "Estimated risk has increased across the observed period. "
            "Closer review may be appropriate."
        )

    else:

        st.info(
            "Estimated risk has remained relatively stable."
        )

    # -----------------------------------------------------
    # PARENT COMMUNICATION
    # -----------------------------------------------------

    st.markdown("## 👨‍👩‍👧 Supportive Parent Communication")

    language = st.radio(
        "Choose Language",
        [
            "English",
            "Telugu"
        ],
        horizontal=True
    )

    if language == "English":

        message = (
            f"Hello, this is a supportive academic update "
            f"regarding {selected_student}. Recent learning "
            "signals suggest that some additional academic "
            "support may be helpful. We recommend focused "
            "support and regular follow-up during the coming "
            "learning cycle. Our goal is to identify learning "
            "gaps early and support the student's progress."
        )

    else:

        message = (
            f"నమస్కారం. {selected_student} విద్యాభ్యాసానికి "
            "సంబంధించిన తాజా సమాచారం ఇది. కొన్ని అంశాల్లో "
            "అదనపు సహాయం అవసరమయ్యే సూచనలు కనిపిస్తున్నాయి. "
            "వచ్చే అభ్యాస దశలో లక్ష్యిత సహాయం మరియు క్రమమైన "
            "పర్యవేక్షణ చేయడం మంచిది. విద్యార్థి అభ్యాసంలో "
            "ఉన్న ఖాళీలను ముందుగానే గుర్తించి సహాయం "
            "అందించడం మా లక్ష్యం."
        )

    st.text_area(
        "Message",
        message,
        height=170
    )
