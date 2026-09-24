import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import plotly.express as px

st.set_page_config(
    page_title="AI Academic Risk Detection",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------------------------------
# PAGE STYLE
# ---------------------------------------------------------
st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}
.subtitle {
    font-size: 17px;
    color: #64748b;
    margin-bottom: 25px;
}
.card {
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    background: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DEMO DATA
# ---------------------------------------------------------
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

        declining_student = index in [2, 7, 14, 18]

        for week in range(1, 9):

            decline = (week - 1) * (
                1.8 if declining_student else 0.15
            )

            attendance = np.clip(
                94 - decline + rng.normal(0, 3),
                50,
                100
            )

            quiz_score = np.clip(
                90 - decline * 2 + rng.normal(0, 5),
                25,
                100
            )

            assignment_completion = np.clip(
                96 - decline * 1.5 + rng.normal(0, 4),
                30,
                100
            )

            engagement = np.clip(
                91 - decline * 1.8 + rng.normal(0, 5),
                20,
                100
            )

            concept_mastery = np.clip(
                88 - decline * 2 + rng.normal(0, 5),
                20,
                100
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

# ---------------------------------------------------------
# AI MODEL
# ---------------------------------------------------------
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

    X = rng.uniform(25, 100, size=(1500, 5))

    risk_signal = (
        7
        - 0.035 * X[:, 0]
        - 0.045 * X[:, 1]
        - 0.025 * X[:, 2]
        - 0.020 * X[:, 3]
        - 0.040 * X[:, 4]
        + rng.normal(0, 0.45, 1500)
    )

    probability = 1 / (1 + np.exp(-risk_signal))

    y = (probability > 0.5).astype(int)

    model = LogisticRegression(max_iter=1000)

    model.fit(X, y)

    return model


model = train_model()

df["Risk Score"] = (
    model.predict_proba(df[FEATURES])[:, 1] * 100
)


def risk_level(score):

    if score >= 70:
        return "High Risk"

    elif score >= 40:
        return "Moderate Risk"

    return "Low Risk"


df["Risk Level"] = df["Risk Score"].apply(risk_level)

latest = (
    df.sort_values("Week")
    .groupby("Student")
    .tail(1)
    .copy()
)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">AI-Based Early Detection of Students at Academic Risk</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Detect early. Explain the signals. Recommend targeted support. Track the change.'
    '</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Student Analysis",
        "Time Travel",
        "Intervention Center"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "This prototype uses synthetic classroom data "
    "for demonstration purposes."
)

# =========================================================
# DASHBOARD
# =========================================================
if page == "Dashboard":

    total_students = len(latest)

    high_risk = len(
        latest[latest["Risk Level"] == "High Risk"]
    )

    moderate_risk = len(
        latest[latest["Risk Level"] == "Moderate Risk"]
    )

    average_risk = latest["Risk Score"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Students Monitored",
        total_students
    )

    col2.metric(
        "High Risk",
        high_risk
    )

    col3.metric(
        "Moderate Risk",
        moderate_risk
    )

    col4.metric(
        "Average Risk",
        f"{average_risk:.1f}%"
    )

    st.markdown("## Class Risk Overview")

    chart_data = latest.sort_values(
        "Risk Score",
        ascending=True
    )

    fig = px.bar(
        chart_data,
        x="Risk Score",
        y="Student",
        orientation="h",
        text=chart_data["Risk Score"].round(0),
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

    st.markdown("## Students Requiring Support")

    support_students = latest[
        latest["Risk Level"] != "Low Risk"
    ].sort_values(
        "Risk Score",
        ascending=False
    )

    table = support_students[
        [
            "Student",
            "Risk Score",
            "Risk Level",
            "Attendance",
            "Quiz Score",
            "Assignment Completion",
            "Engagement",
            "Concept Mastery"
        ]
    ].copy()

    table["Risk Score"] = table[
        "Risk Score"
    ].round(1)

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# STUDENT ANALYSIS
# =========================================================
elif page == "Student Analysis":

    st.title("Student Risk Analysis")

    selected_student = st.selectbox(
        "Select Student",
        sorted(latest["Student"].unique())
    )

    current = latest[
        latest["Student"] == selected_student
    ].iloc[0]

    history = df[
        df["Student"] == selected_student
    ].sort_values("Week")

    st.markdown(
        f"## {selected_student}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Current Risk",
        f"{current['Risk Score']:.1f}%"
    )

    col2.metric(
        "Risk Level",
        current["Risk Level"]
    )

    col3.metric(
        "Concept Mastery",
        f"{current['Concept Mastery']:.1f}%"
    )

    st.markdown("## Why is the student at risk?")

    factors = {
        "Attendance": 100 - current["Attendance"],
        "Assessment Performance": 100 - current["Quiz Score"],
        "Assignment Completion": 100 - current["Assignment Completion"],
        "Engagement": 100 - current["Engagement"],
        "Concept Mastery": 100 - current["Concept Mastery"]
    }

    top_factors = sorted(
        factors.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    for number, (factor, value) in enumerate(
        top_factors,
        start=1
    ):

        actual_value = 100 - value

        st.write(
            f"**{number}. {factor}** — "
            f"Current signal: {actual_value:.1f}%"
        )

    st.markdown("## Learning Trajectory")

    trend = history[
        ["Week", "Risk Score"]
    ].copy()

    fig = px.line(
        trend,
        x="Week",
        y="Risk Score",
        markers=True,
        labels={
            "Risk Score": "Risk Score (%)"
        }
    )

    fig.update_layout(
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("## Recommended Intervention")

    weakest_factor = min(
        FEATURES,
        key=lambda x: current[x]
    )

    recommendations = {

        "Attendance":
            "Review missed lessons and create a short catch-up plan.",

        "Quiz Score":
            "Provide targeted practice on the student's weakest assessment topics.",

        "Assignment Completion":
            "Break pending assignments into smaller deadlines and monitor completion.",

        "Engagement":
            "Use interactive activities and regular teacher check-ins.",

        "Concept Mastery":
            "Provide focused concept revision followed by a short mastery assessment."
    }

    st.info(
        recommendations[weakest_factor]
    )

# =========================================================
# TIME TRAVEL
# =========================================================
elif page == "Time Travel":

    st.title("Time Travel")

    st.write(
        "Replay the student's academic journey and identify "
        "when early warning signals started appearing."
    )

    selected_student = st.selectbox(
        "Choose Student",
        sorted(latest["Student"].unique()),
        key="time_travel_student"
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
            "Risk Score": "Academic Risk Score (%)"
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

    fig.update_layout(
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    selected_week = st.slider(
        "Replay Week",
        min_value=1,
        max_value=8,
        value=4
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
        "Concept Mastery",
        f"{snapshot['Concept Mastery']:.0f}%"
    )

    st.info(
        f"At Week {selected_week}, the estimated academic risk "
        f"score was {snapshot['Risk Score']:.1f}%. "
        "This historical view helps identify when additional "
        "support could have been considered."
    )

# =========================================================
# INTERVENTION CENTER
# =========================================================
else:

    st.title("Intervention Center")

    selected_student = st.selectbox(
        "Select Student",
        sorted(latest["Student"].unique()),
        key="intervention_student"
    )

    current = latest[
        latest["Student"] == selected_student
    ].iloc[0]

    st.markdown(
        f"## Personalized Support Plan for {selected_student}"
    )

    weakest_factor = min(
        FEATURES,
        key=lambda x: current[x]
    )

    plans = {

        "Attendance": [
            "Review missed lessons.",
            "Identify possible attendance barriers.",
            "Create a short catch-up schedule.",
            "Monitor attendance during the next learning cycle."
        ],

        "Quiz Score": [
            "Identify the lowest-performing topics.",
            "Provide targeted practice.",
            "Use worked examples and guided revision.",
            "Conduct a short reassessment."
        ],

        "Assignment Completion": [
            "Identify incomplete assignments.",
            "Break tasks into smaller deadlines.",
            "Provide teacher follow-up.",
            "Monitor completion during the next week."
        ],

        "Engagement": [
            "Introduce interactive classroom activities.",
            "Schedule a short teacher check-in.",
            "Encourage active participation.",
            "Monitor engagement in the next cycle."
        ],

        "Concept Mastery": [
            "Identify weak concepts.",
            "Provide focused concept revision.",
            "Give practice questions.",
            "Conduct a short mastery check."
        ]
    }

    for number, action in enumerate(
        plans[weakest_factor],
        start=1
    ):

        st.write(
            f"**{number}.** {action}"
        )

    st.markdown("## Parent Communication")

    language = st.radio(
        "Choose Language",
        ["English", "Telugu"],
        horizontal=True
    )

    if language == "English":

        message = (
            f"Hello, this is a supportive academic update "
            f"regarding {selected_student}. Recent learning "
            "signals suggest that some additional academic "
            "support may be helpful. We recommend focused "
            "revision and regular follow-up during the coming "
            "week. Our goal is to identify learning gaps early "
            "and support the student's progress."
        )

    else:

        message = (
            f"నమస్కారం. {selected_student} విద్యాభ్యాసానికి "
            "సంబంధించిన తాజా సమాచారం ఇది. కొన్ని అంశాల్లో "
            "అదనపు సహాయం అవసరమయ్యే సూచనలు కనిపిస్తున్నాయి. "
            "వచ్చే వారంలో లక్ష్యిత పునశ్చరణ మరియు క్రమమైన "
            "పర్యవేక్షణ చేయడం మంచిది. విద్యార్థి అభ్యాసంలో "
            "ఉన్న ఖాళీలను ముందుగానే గుర్తించి సహాయం "
            "అందించడం మా లక్ష్యం."
        )

    st.text_area(
        "Message",
        message,
        height=170
    )

    st.success(
        "Support plan generated. Continue monitoring "
        "future learning signals to evaluate progress."
    )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")

st.caption(
    "Prototype for hackathon demonstration. "
    "The application uses synthetic classroom data and "
    "should not be used for high-stakes decisions about real students "
    "without appropriate validation and human oversight."
)
