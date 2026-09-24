import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="AI-Based Early Detection of At-Risk Students",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GLOBAL CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #f5f3ff; }
.block-container { padding-top: 1.2rem; max-width: 1450px; }
section[data-testid="stSidebar"] {
  background: linear-gradient(200deg,#2a0e6e 0%,#1e0a4d 55%,#4c0a63 100%);
}
section[data-testid="stSidebar"] * { color: #e6ddff !important; }

.hero-title { font-size: 32px; font-weight: 800;
  background: linear-gradient(90deg,#7c3aed,#db2777);
  -webkit-background-clip: text; background-clip: text; color: transparent; margin-bottom:4px; }
.hero-subtitle { color: #6b6285; font-size: 15px; margin-bottom: 22px; }
.section-title { font-size: 21px; font-weight: 800; color: #191035; margin: 22px 0 12px; }

.metric-card { background: white; border: 1px solid #e7e1fb; border-radius: 16px; padding: 18px;
  min-height: 120px; box-shadow: 0 6px 18px rgba(109,40,217,.07); }
.metric-label { font-size: 13px; color: #6b6285; font-weight: 600; }
.metric-value { font-size: 28px; font-weight: 800; margin-top: 6px;
  background: linear-gradient(90deg,#7c3aed,#db2777); -webkit-background-clip: text;
  background-clip: text; color: transparent; }
.metric-small { font-size: 12px; color: #6b6285; margin-top: 5px; }

.blue-card { background: white; border: 1px solid #c9d9ff; border-radius: 14px; padding: 16px; }
.blue-card .metric-label { color: #2955d6; }
.blue-card .metric-value { color: #1e3a8a; -webkit-text-fill-color: initial; background: none; }

.status-danger { background: #ffe4ec; color: #e11d48; border: 1px solid #ffc2d6;
  padding: 6px 12px; border-radius: 999px; font-weight: 700; display: inline-block; }
.status-warning { background: #fff2dc; color: #ea8c00; border: 1px solid #ffdfa3;
  padding: 6px 12px; border-radius: 999px; font-weight: 700; display: inline-block; }
.status-watch { background: #efe8ff; color: #6d28d9; border: 1px solid #d9c8ff;
  padding: 6px 12px; border-radius: 999px; font-weight: 700; display: inline-block; }
.status-success { background: #dcfaf3; color: #0d9488; border: 1px solid #a8f0e0;
  padding: 6px 12px; border-radius: 999px; font-weight: 700; display: inline-block; }

.info-box { background: white; border: 1px solid #e7e1fb; border-radius: 14px; padding: 16px; margin-bottom: 10px; }
.info-title { font-weight: 800; color: #191035; margin-bottom: 6px; }
.info-text { color: #6b6285; font-size: 14px; line-height: 1.6; }
.ai-box { border-left: 4px solid #6d28d9; }

.login-wrapper { max-width: 820px; margin: 40px auto 0; text-align: center; }
.login-title { font-size: 36px; font-weight: 800; color: #fff; }
.login-subtitle { font-size: 15px; color: #cabdea; margin-top: 10px; margin-bottom: 26px; }
.role-card { background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.14);
  border-radius: 18px; padding: 24px; text-align: center; min-height: 170px; }
.role-card h3 { color: #fff; margin: 4px 0 6px; }
.role-card p { color: #c7bbe3; font-size: 13px; margin-bottom: 14px; }
.stat-strip { display:flex; justify-content:center; gap:32px; margin: 20px 0 8px; flex-wrap:wrap; }
.stat-strip b { display:block; font-size:22px; color:#fff; }
.stat-strip span { font-size:12px; color:#b6a5da; }
.timeline-card { background: white; border-left: 5px solid #6d28d9; padding: 13px 16px;
  margin-bottom: 9px; border-radius: 10px; }
.small-note { font-size: 12px; color: #6b6285; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
defaults = {"logged_in": False, "role": None, "selected_student": "STU-001",
            "page": "Dashboard", "selected_subject": "Mathematics",
            "parent_message_sent": False, "dir_filter": "All"}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

SUBJECTS = ["Mathematics", "Science", "English", "Social Studies", "Computer Science"]
FIRST_NAMES = ["Aarav","Ananya","Rahul","Saanvi","Arjun","Diya","Vihaan","Myra","Aditya","Ishita",
    "Rohan","Aadhya","Kabir","Meera","Ayaan","Sara","Nikhil","Anika","Ishaan","Kiara",
    "Dev","Navya","Aryan","Tara","Reyansh","Ira","Karthik","Riya","Manav","Avni",
    "Siddharth","Aanya","Ritvik","Shreya","Yash","Pihu","Atharv","Nandini","Krishna","Mahi",
    "Varun","Aarohi","Harsh","Ishani","Rudra","Tanvi","Advik","Om","Kavya","Dhruv"]
LAST_NAMES = ["Kumar","Reddy","Verma","Rao","Singh","Sharma","Patel","Das","Gupta","Khan"]
FATHER_FIRST = ["Suresh","Ramesh","Mahesh","Rajesh","Prakash","Vijay","Anil","Sanjay","Vinod","Ashok"]
PATTERN = ["Steady","Steady","Gradual Decliner","Steady","Chronic Absentee",
           "Steady","Sudden Drop","Steady","Recovering","Steady"]

PERSONA_RANGES = {
    "Steady":            {"att": (86, 94), "quiz": (78, 90), "slope": (-0.2, 0.2)},
    "Gradual Decliner":  {"att": (82, 90), "quiz": (78, 86), "slope": (-1.8, -1.0)},
    "Sudden Drop":       {"att": (86, 93), "quiz": (80, 88), "slope": (0, 0)},
    "Chronic Absentee":  {"att": (56, 68), "quiz": (64, 76), "slope": (-0.5, 0.1)},
    "Recovering":        {"att": (68, 75), "quiz": (58, 68), "slope": (1.0, 2.0)},
}

# ============================================================
# SYNTHETIC DATA
# ============================================================
@st.cache_data
def build_students():
    rng = np.random.default_rng(42)
    students = []
    for i, first in enumerate(FIRST_NAMES):
        last = LAST_NAMES[i % len(LAST_NAMES)]
        name = f"{first} {last}"
        persona = PATTERN[i % len(PATTERN)]
        pr = PERSONA_RANGES[persona]
        base_att = rng.uniform(*pr["att"])
        base_quiz = rng.uniform(*pr["quiz"])
        att_slope = rng.uniform(*pr["slope"])
        quiz_slope = rng.uniform(*pr["slope"])

        weeks = []
        for w in range(1, 9):
            if persona == "Sudden Drop" and w >= 6:
                att = base_att - rng.uniform(8, 17)
                quiz = base_quiz - rng.uniform(20, 32)
                missing = int(rng.integers(1, 4))
            else:
                att = base_att + att_slope * (w - 1) + rng.normal(0, 1.5)
                quiz = base_quiz + quiz_slope * (w - 1) + rng.normal(0, 2)
                if persona == "Chronic Absentee":
                    missing = int(rng.integers(1, 4))
                elif persona == "Gradual Decliner" and w >= 5:
                    missing = int(rng.integers(1, 3))
                else:
                    missing = int(rng.integers(0, 2))
            weeks.append({
                "Week": w,
                "Attendance": round(float(np.clip(att, 45, 99)), 1),
                "Quiz Score": round(float(np.clip(quiz, 30, 99)), 1),
                "Missing Assignments": missing
            })

        subj = {}
        for s in SUBJECTS:
            adj = rng.uniform(-8, 8)
            if s == "Mathematics" and persona in ["Gradual Decliner", "Sudden Drop"]:
                adj -= 7
            subj[s] = {
                "Marks": round(float(np.clip(weeks[-1]["Quiz Score"] + adj, 35, 98)), 1),
                "Attendance": round(float(np.clip(weeks[-1]["Attendance"] + rng.uniform(-6, 4), 45, 99)), 1)
            }

        exam_att = round(float(np.clip(weeks[-1]["Attendance"] + rng.uniform(-6, 4), 40, 99)), 1)
        special_att = round(float(np.clip(weeks[-1]["Attendance"] + rng.uniform(-10, 6), 35, 99)), 1)

        daily_log = []
        d = datetime(2026, 8, 3)
        while len(daily_log) < 20:
            if d.weekday() < 5:
                p = weeks[-1]["Attendance"] / 100
                status = "Present" if rng.random() < p else "Absent"
                if status == "Present" and rng.random() < 0.08:
                    status = "Late"
                daily_log.append({"Date": d, "Status": status})
            d += timedelta(days=1)

        students.append({
            "Student ID": f"STU-{i+1:03d}", "Student": name, "Class": "VIII-A", "Persona": persona,
            "Father": f"{FATHER_FIRST[i % len(FATHER_FIRST)]} {last}",
            "Phone": f"9{700000000 + i*137931 % 100000000:09d}"[:10],
            "Weeks": weeks, "Subjects": subj,
            "ExamAttendance": exam_att, "SpecialClassAttendance": special_att,
            "DailyLog": daily_log
        })
    return students

students = build_students()
by_id = {s["Student ID"]: s for s in students}

# ============================================================
# RISK ENGINE
# ============================================================
def calc_risk(student):
    weeks = pd.DataFrame(student["Weeks"])
    att_slope = np.polyfit(weeks["Week"], weeks["Attendance"], 1)[0]
    quiz_slope = np.polyfit(weeks["Week"], weeks["Quiz Score"], 1)[0]
    last_att = weeks["Attendance"].iloc[-1]
    last_quiz = weeks["Quiz Score"].iloc[-1]
    att_factor = np.clip((-att_slope * 8) + ((85 - last_att) * 0.65), 0, 100)
    quiz_factor = np.clip((-quiz_slope * 8) + ((70 - last_quiz) * 1.4), 0, 100)
    missing = int(weeks.tail(3)["Missing Assignments"].sum())
    assign_factor = np.clip(missing / 6 * 100, 0, 100)
    att_level_factor = np.clip((85 - last_att) * 1.7, 0, 100)
    risk = round(float(np.clip(
        att_factor*0.35 + quiz_factor*0.30 + assign_factor*0.20 + att_level_factor*0.15, 0, 100)), 1)
    status = ("Needs Support" if risk >= 70 else "Monitor Closely" if risk >= 55
              else "Watch" if risk >= 35 else "On Track")
    return dict(risk=risk, status=status, att_slope=att_slope, quiz_slope=quiz_slope,
                last_att=last_att, last_quiz=last_quiz, missing=missing,
                att_factor=att_factor, quiz_factor=quiz_factor,
                assign_factor=assign_factor, att_level_factor=att_level_factor)

@st.cache_data
def risk_table(_students):
    rows = []
    for s in _students:
        r = calc_risk(s)
        rows.append({**{k: s[k] for k in ["Student ID", "Student", "Class", "Persona"]},
                     "Risk Score": r["risk"], "Status": r["status"],
                     "Attendance": r["last_att"], "Latest Quiz": r["last_quiz"],
                     "Missing Assignments": r["missing"], "Father": s["Father"], "Phone": s["Phone"]})
    return pd.DataFrame(rows)

risk_df = risk_table(students)

def status_html(status):
    cls = {"Needs Support": "status-danger", "Monitor Closely": "status-warning",
           "Watch": "status-watch"}.get(status, "status-success")
    return f'<span class="{cls}">{status}</span>'

def weak_subject(student):
    return min(student["Subjects"].items(), key=lambda kv: kv[1]["Marks"])[0]

def get_reasons(r):
    reasons = []
    if r["last_att"] < 75:
        reasons.append(f"Attendance is currently {r['last_att']:.0f}%.")
    if r["att_slope"] < -0.7:
        reasons.append("Attendance has been declining across recent weeks.")
    if r["last_quiz"] < 60:
        reasons.append(f"Recent assessment performance is {r['last_quiz']:.0f}%.")
    if r["quiz_slope"] < -0.7:
        reasons.append("Assessment performance is trending downward.")
    if r["missing"] >= 3:
        reasons.append(f"{r['missing']} assignments were missed in the recent period.")
    if not reasons:
        reasons.append("Current indicators are relatively stable.")
    return reasons

def ai_trend_text(r):
    if r["att_slope"] < -0.5 or r["quiz_slope"] < -0.5:
        return ("Based on the last 8 weeks, this student's risk signal is trending upward — "
                "early intervention now is likely to be more effective than waiting for the next report card.")
    if r["att_slope"] > 0.5 or r["quiz_slope"] > 0.5:
        return "Based on the last 8 weeks, this student's indicators are improving — current support appears to be working."
    return "Based on the last 8 weeks, this student's indicators are relatively stable."

# ============================================================
# LOGIN
# ============================================================
def login_screen():
    needs_support = len(risk_df[risk_df["Status"] != "On Track"])
    st.markdown('<div class="login-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">AI-Based Early Detection<br>of At-Risk Students</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">An explainable early-detection engine that reads attendance, '
                'assessments and assignment patterns to flag students who need support — weeks before report cards do.</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-strip">
        <div><b>{len(students)}</b><span>Students monitored</span></div>
        <div><b>{len(SUBJECTS)}</b><span>Subjects tracked</span></div>
        <div><b>{needs_support}</b><span>Signals flagged this term</span></div>
        <div><b>3</b><span>Role-based views</span></div>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="role-card"><h3>👩‍🏫 Teacher</h3><p>Monitor students, review risk '
                     'signals and plan interventions.</p></div>', unsafe_allow_html=True)
        if st.button("Continue as Teacher", use_container_width=True, key="teacher_login"):
            st.session_state.update(logged_in=True, role="Teacher", page="Dashboard"); st.rerun()
    with col2:
        st.markdown('<div class="role-card"><h3>🎓 Student</h3><p>View marks, attendance and '
                     'learning support by subject.</p></div>', unsafe_allow_html=True)
        if st.button("Continue as Student", use_container_width=True, key="student_login"):
            st.session_state.update(logged_in=True, role="Student", page="My Dashboard"); st.rerun()
    with col3:
        st.markdown('<div class="role-card"><h3>👪 Parent</h3><p>Track academic progress, daily '
                     'attendance and teacher updates.</p></div>', unsafe_allow_html=True)
        if st.button("Continue as Parent", use_container_width=True, key="parent_login"):
            st.session_state.update(logged_in=True, role="Parent", page="Child Dashboard"); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# METRIC CARD HELPER
# ============================================================
def metric(col, label, value, note="", blue=False):
    cls = "blue-card" if blue else "metric-card"
    col.markdown(f'<div class="{cls}"><div class="metric-label">{label}</div>'
                  f'<div class="metric-value">{value}</div>'
                  f'<div class="metric-small">{note}</div></div>', unsafe_allow_html=True)

# ============================================================
# TEACHER DASHBOARD
# ============================================================
def teacher_dashboard():
    st.markdown('<div class="hero-title">Teacher Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Monitor academic performance, attendance and emerging support needs.</div>',
                unsafe_allow_html=True)

    needs_support = len(risk_df[risk_df["Status"] == "Needs Support"])
    recovering = len(risk_df[risk_df["Persona"] == "Recovering"])
    avg_marks = risk_df["Latest Quiz"].mean()
    avg_att = risk_df["Attendance"].mean()
    cols = st.columns(5)
    for c, (label, val, note) in zip(cols, [
        ("Total Students", len(students), "Class VIII-A"),
        ("Needs Support", needs_support, "Risk threshold reached"),
        ("Recovering", recovering, "Positive recent trend"),
        ("Average Marks", f"{avg_marks:.1f}%", "Latest assessment"),
        ("Attendance", f"{avg_att:.1f}%", "Class average")]):
        metric(c, label, val, note)

    st.markdown('<div class="section-title">Academic Risk Overview</div>', unsafe_allow_html=True)
    left, right = st.columns([1.5, 1])
    with left:
        counts = risk_df["Status"].value_counts()
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color="#7c3aed"))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown('<div class="info-box"><div class="info-title">Early Warning Signals</div>'
                     '<div class="info-text">Combines attendance trend, assessment trend, missing '
                     'assignments and current attendance level.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box"><div class="info-title">Important</div><div class="info-text">'
                     'Risk scores are support signals, not final judgments. Review evidence before acting.'
                     '</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Class Performance by Subject</div>', unsafe_allow_html=True)
    subj_avg = {s: np.mean([st_["Subjects"][s]["Marks"] for st_ in students]) for s in SUBJECTS}
    fig2 = go.Figure(go.Bar(x=list(subj_avg.keys()), y=list(subj_avg.values()), marker_color="#db2777"))
    fig2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Average marks %")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Recent Alert Feed</div>', unsafe_allow_html=True)
    alerts = risk_df[risk_df["Status"].isin(["Needs Support", "Monitor Closely"])].sort_values(
        "Risk Score", ascending=False).head(8)
    if alerts.empty:
        st.info("No active alerts.")
    for _, row in alerts.iterrows():
        r = calc_risk(by_id[row["Student ID"]])
        reason = get_reasons(r)[0]
        color = "#e11d48" if row["Status"] == "Needs Support" else "#ea8c00"
        st.markdown(f'<div class="timeline-card" style="border-left-color:{color}">'
                     f'<b>{row["Student"]} · {row["Status"]}</b><br>{reason}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Students Requiring Attention</div>', unsafe_allow_html=True)
    top = risk_df.sort_values("Risk Score", ascending=False).head(7)
    for _, row in top.iterrows():
        c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1])
        c1.write(f"**{row['Student']}**  \n{row['Student ID']} | {row['Class']}")
        c2.metric("Risk", f"{row['Risk Score']:.0f}")
        c3.write(status_html(row["Status"]), unsafe_allow_html=True)
        if c4.button("View", key=f"view_{row['Student ID']}"):
            st.session_state.selected_student = row["Student ID"]
            st.session_state.page = "Student Profile"; st.rerun()

    st.markdown(f'<div class="section-title">Full Student Directory ({len(students)})</div>', unsafe_allow_html=True)
    filt_cols = st.columns(5)
    for i, f in enumerate(["All", "Needs Support", "Monitor Closely", "Watch", "On Track"]):
        if filt_cols[i].button(f, key=f"filt_{f}",
                                type="primary" if st.session_state.dir_filter == f else "secondary",
                                use_container_width=True):
            st.session_state.dir_filter = f; st.rerun()
    search = st.text_input("Search by name or ID")
    view = risk_df if st.session_state.dir_filter == "All" else risk_df[risk_df["Status"] == st.session_state.dir_filter]
    if search:
        view = view[view["Student"].str.lower().str.contains(search.lower()) |
                     view["Student ID"].str.lower().str.contains(search.lower())]
    st.dataframe(
        view[["Student ID", "Student", "Father", "Phone", "Attendance", "Latest Quiz", "Status"]]
        .rename(columns={"Latest Quiz": "Marks %", "Attendance": "Attendance %"}),
        use_container_width=True, hide_index=True, height=420
    )

# ============================================================
# STUDENT PROFILE
# ============================================================
def student_profile():
    sid = st.session_state.selected_student
    student = by_id[sid]
    r = calc_risk(student)

    st.markdown('<div class="hero-title">Student Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Evidence-based student overview with explainable risk signals.</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    metric(c1, "Student", student["Student"], f"{sid} | {student['Class']}")
    metric(c2, "Risk Score", f"{r['risk']:.0f}", "0–100 support signal")
    metric(c3, "Attendance", f"{r['last_att']:.0f}%", "Latest period")
    metric(c4, "Latest Assessment", f"{r['last_quiz']:.0f}%", "Latest assessment")

    st.markdown(f"<br>{status_html(r['status'])}", unsafe_allow_html=True)
    st.markdown(f'<div class="info-box" style="margin-top:14px"><div class="info-title">Guardian Contact</div>'
                 f'<div class="info-text">Father\'s Name: {student["Father"]} &nbsp;·&nbsp; '
                 f'Phone: {student["Phone"]} &nbsp;·&nbsp; Class: {student["Class"]}</div></div>',
                 unsafe_allow_html=True)

    st.markdown('<div class="section-title">Why is this student receiving this signal?</div>', unsafe_allow_html=True)
    for reason in get_reasons(r):
        st.markdown(f'<div class="info-box"><div class="info-text">{reason}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-box ai-box"><div class="info-title">AI Trend Forecast</div>'
                 f'<div class="info-text">{ai_trend_text(r)}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Risk Factor Breakdown</div>', unsafe_allow_html=True)
    factor_df = pd.DataFrame({
        "Factor": ["Attendance Trend", "Assessment Trend", "Missing Assignments", "Attendance Level"],
        "Contribution": [r["att_factor"]*0.35, r["quiz_factor"]*0.30, r["assign_factor"]*0.20, r["att_level_factor"]*0.15]
    })
    fig = go.Figure(go.Bar(x=factor_df["Contribution"], y=factor_df["Factor"], orientation="h",
                            marker_color=["#0d9488", "#6d28d9", "#ea8c00", "#e11d48"]))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Subject Performance</div>', unsafe_allow_html=True)
    sdf = pd.DataFrame([{"Subject": s, "Marks %": v["Marks"], "Attendance %": v["Attendance"]}
                        for s, v in student["Subjects"].items()])
    st.dataframe(sdf, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Performance Trend</div>', unsafe_allow_html=True)
    wdf = pd.DataFrame(student["Weeks"])
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=wdf["Week"], y=wdf["Quiz Score"], mode="lines+markers", name="Assessment",
                               line=dict(color="#7c3aed")))
    fig2.add_trace(go.Scatter(x=wdf["Week"], y=wdf["Attendance"], mode="lines+markers", name="Attendance",
                               line=dict(color="#0d9488")))
    fig2.update_layout(height=360, yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Time Travel</div>', unsafe_allow_html=True)
    wk = st.slider("Replay student history up to week", 1, 8, 8, key=f"time_{sid}")
    snap = student["Weeks"][wk-1]
    a, b, c = st.columns(3)
    a.metric("Attendance", f"{snap['Attendance']:.0f}%")
    b.metric("Assessment", f"{snap['Quiz Score']:.0f}%")
    c.metric("Missing Assignments", int(snap["Missing Assignments"]))
    st.info("This replay shows an earlier point in the student's academic history.") if wk < 8 else st.success("Current period selected.")

    if st.button("← Back to dashboard"):
        st.session_state.page = "Dashboard"; st.rerun()

# ============================================================
# INTERVENTION CENTER
# ============================================================
def intervention_page():
    sid = st.session_state.selected_student
    student = by_id[sid]
    r = calc_risk(student)
    weak = weak_subject(student)

    st.markdown('<div class="hero-title">Intervention Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Convert detected risk signals into targeted academic support.</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="info-box"><div class="info-title">Recommended focus: {weak}</div>'
                 f'<div class="info-text">Current indicators suggest additional support may help in this subject.'
                 f'</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">1. Explain Again</div>', unsafe_allow_html=True)
    subject = st.selectbox("Select subject", SUBJECTS, index=SUBJECTS.index(weak))
    explanations = {
        "Mathematics": "Quadratic equations can be understood as equations containing x². Identify a, b and c, then select a suitable solving method.",
        "Science": "Break the concept into definition, process, example and application. Use a simple real-world example before attempting questions.",
        "English": "Read the sentence carefully, identify the key grammar rule and then apply it to a short example.",
        "Social Studies": "Convert the topic into timeline, causes, events and outcomes — this makes long answers easier to remember.",
        "Computer Science": "Start with the basic concept, trace one small example and then practice a similar problem independently."
    }
    st.markdown(f'<div class="info-box"><div class="info-title">{subject} — Simple Explanation</div>'
                 f'<div class="info-text">{explanations[subject]}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">2. Practice</div>', unsafe_allow_html=True)
    practice = {
        "Mathematics": "Solve x² - 5x + 6 = 0.",
        "Science": "Explain the main stages of the process studied in this chapter.",
        "English": "Identify the grammar rule used in the following sentence.",
        "Social Studies": "Write two causes and two effects of the selected historical event.",
        "Computer Science": "Explain the concept using a simple example."
    }
    st.info(practice[subject])
    st.button("Check Practice Answer")

    st.markdown('<div class="section-title">3. Personalized Support Plan</div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    p1.markdown('<div class="info-box"><div class="info-title">Daily Revision</div>'
                '<div class="info-text">25 minutes focused practice on the identified weak area.</div></div>',
                unsafe_allow_html=True)
    p1.markdown('<div class="info-box"><div class="info-title">Practice Set</div>'
                '<div class="info-text">Complete 5 targeted questions and review incorrect answers.</div></div>',
                unsafe_allow_html=True)
    p2.markdown('<div class="info-box"><div class="info-title">Special Class</div>'
                '<div class="info-text">Attend a focused online support session for the identified topic.</div></div>',
                unsafe_allow_html=True)
    p2.markdown('<div class="info-box"><div class="info-title">Teacher Review</div>'
                '<div class="info-text">Review progress after the next assessment cycle.</div></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-title">4. Parent Communication</div>', unsafe_allow_html=True)
    message = (f"Hello Parent, {student['Student']} may benefit from additional support in {weak}. "
               f"Current attendance is {r['last_att']:.0f}% and the latest assessment score is "
               f"{r['last_quiz']:.0f}%. We recommend regular revision and the available special support class.")
    st.text_area("Suggested parent update", value=message, height=120)
    if st.button("Send Parent Update", type="primary", use_container_width=True):
        st.session_state.parent_message_sent = True
    if st.session_state.parent_message_sent:
        st.success("Parent update sent in demo mode.")

    st.markdown('<div class="section-title">5. Recommended Special Classes</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Subject": [weak, "Science", "English"],
        "Focus": ["Concept Revision", "Chapter Support", "Practice Session"],
        "Mode": ["Online"] * 3,
        "Status": ["Recommended", "Available", "Available"]
    }), use_container_width=True, hide_index=True)

    if st.button("← Back to dashboard", key="back_int"):
        st.session_state.page = "Dashboard"; st.rerun()

# ============================================================
# TRACK PAGE
# ============================================================
def track_page():
    sid = st.session_state.selected_student
    student = by_id[sid]
    weeks = student["Weeks"]

    risks = []
    for w in range(1, 9):
        h = weeks[:w]
        if len(h) < 2:
            risks.append(25.0); continue
        att = [x["Attendance"] for x in h]; quiz = [x["Quiz Score"] for x in h]
        att_slope = np.polyfit(range(1, len(h)+1), att, 1)[0]
        quiz_slope = np.polyfit(range(1, len(h)+1), quiz, 1)[0]
        v = (np.clip((-att_slope*8)+((85-att[-1])*0.65), 0, 100)*0.35 +
             np.clip((-quiz_slope*8)+((70-quiz[-1])*1.4), 0, 100)*0.30)
        risks.append(float(np.clip(v, 0, 100)))

    st.markdown('<div class="hero-title">Progress Tracking</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-subtitle">Track academic signals before and after support activities — '
                f'{student["Student"]}.</div>', unsafe_allow_html=True)

    fig = go.Figure(go.Scatter(x=list(range(1, 9)), y=risks, mode="lines+markers", line=dict(color="#7c3aed")))
    fig.update_layout(height=380, yaxis=dict(range=[0, 100]), xaxis_title="Week", yaxis_title="Risk Signal")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Academic Recovery Indicators</div>', unsafe_allow_html=True)
    r = calc_risk(student); weak = weak_subject(student)
    first, last = weeks[0], weeks[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Attendance", f"{last['Attendance']:.0f}%", f"{last['Attendance']-first['Attendance']:+.0f} pts")
    c2.metric("Assessment", f"{last['Quiz Score']:.0f}%", f"{last['Quiz Score']-first['Quiz Score']:+.0f} pts")
    c3.metric("Weak Subject", weak)
    c4.metric("Current Status", r["status"])

    st.markdown('<div class="section-title">Support Timeline</div>', unsafe_allow_html=True)
    for wk, text in [
        ("Week 1", "Baseline academic data recorded."),
        ("Week 3", "Attendance and assessment trends monitored."),
        ("Week 5", "Emerging support signal identified."),
        ("Week 6", "Targeted intervention recommended."),
        ("Week 7", "Parent communication and special class recommended."),
        ("Week 8", "Progress reviewed.")]:
        st.markdown(f'<div class="timeline-card"><b>{wk}</b><br>{text}</div>', unsafe_allow_html=True)

    st.warning("This prototype uses synthetic data. It demonstrates how a school could track indicators "
               "over time; it does not establish that an intervention caused a particular outcome.")
    if st.button("← Back to dashboard", key="back_track"):
        st.session_state.page = "Dashboard"; st.rerun()

# ============================================================
# STUDENT DASHBOARD
# ============================================================
def student_dashboard():
    student = students[0]
    r = calc_risk(student)
    weak = weak_subject(student)

    st.markdown('<div class="hero-title">My Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-subtitle">Welcome, {student["Student"]}. '
                f'Here is your academic support overview.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Attendance & Performance Summary</div>', unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    metric(b1, "Overall Marks", f"{r['last_quiz']:.0f}%", blue=True)
    metric(b2, "Class Attendance", f"{r['last_att']:.0f}%", blue=True)
    metric(b3, "Daily Exam Attendance", f"{student['ExamAttendance']:.0f}%", blue=True)
    metric(b4, "Special Classes Attendance", f"{student['SpecialClassAttendance']:.0f}%", blue=True)

    st.markdown('<div class="section-title">My Subjects</div>', unsafe_allow_html=True)
    sdf = pd.DataFrame([{"Subject": s, "Marks %": v["Marks"], "Attendance %": v["Attendance"]}
                        for s, v in student["Subjects"].items()])
    st.dataframe(sdf, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Learn Again</div>', unsafe_allow_html=True)
    st.info(f"Your current support focus is {weak}. Use the buttons below to review concepts and practice.")
    c1, c2 = st.columns(2)
    if c1.button("Explain Again", use_container_width=True):
        st.success(f"Learning support opened for {weak}.")
    if c2.button("Practice Questions", use_container_width=True):
        st.success(f"Practice set opened for {weak}.")

    st.markdown('<div class="section-title">Special Classes</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Subject": [weak, "Science"], "Mode": ["Online", "Online"],
        "Focus": ["Concept Revision", "Chapter Support"],
        "Availability": ["Recommended", "Available"]
    }), use_container_width=True, hide_index=True)

# ============================================================
# PARENT DASHBOARD
# ============================================================
def parent_dashboard():
    student = students[0]
    r = calc_risk(student)
    weak = weak_subject(student)

    st.markdown('<div class="hero-title">Child Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-subtitle">Academic progress overview for {student["Student"]}.</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="info-box"><div class="info-title">Registered Contact</div>'
                 f'<div class="info-text">Student ID: {student["Student ID"]} &nbsp;·&nbsp; '
                 f'Father\'s Name: {student["Father"]} &nbsp;·&nbsp; Phone: {student["Phone"]} '
                 f'&nbsp;·&nbsp; Class: {student["Class"]}</div></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    metric(c1, "Overall Marks", f"{r['last_quiz']:.0f}%")
    metric(c2, "Attendance", f"{r['last_att']:.0f}%")
    metric(c3, "Status", r["status"])
    metric(c4, "Focus Subject", weak)

    st.markdown('<div class="section-title">Subject-wise Performance</div>', unsafe_allow_html=True)
    sdf = pd.DataFrame([{"Subject": s, "Marks %": v["Marks"], "Attendance %": v["Attendance"]}
                        for s, v in student["Subjects"].items()])
    st.dataframe(sdf, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Daily Attendance Record</div>', unsafe_allow_html=True)
    log = student["DailyLog"]
    present = sum(1 for d in log if d["Status"] == "Present")
    absent = sum(1 for d in log if d["Status"] == "Absent")
    late = sum(1 for d in log if d["Status"] == "Late")
    d1, d2, d3, d4 = st.columns(4)
    metric(d1, "Present", present, "days")
    metric(d2, "Absent", absent, "days")
    metric(d3, "Late", late, "days")
    metric(d4, "Attendance %", f"{r['last_att']:.0f}%", "current term")
    logdf = pd.DataFrame([{"Date": d["Date"].strftime("%d %b %Y"), "Status": d["Status"]}
                          for d in reversed(log)])
    st.dataframe(logdf, use_container_width=True, hide_index=True, height=260)

    st.markdown('<div class="section-title">Teacher Update</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-box"><div class="info-title">Suggested Support</div>'
                 f'<div class="info-text">The teacher has identified {weak} as a current focus area. '
                 f'Regular revision, practice and participation in the recommended support class may help '
                 f'address the identified learning need.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Recommended Special Class</div>', unsafe_allow_html=True)
    st.info(f"{weak} | Online | Concept Revision | Recommended")

    st.markdown('<div class="section-title">Progress</div>', unsafe_allow_html=True)
    wdf = pd.DataFrame(student["Weeks"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wdf["Week"], y=wdf["Quiz Score"], mode="lines+markers", name="Marks",
                              line=dict(color="#7c3aed")))
    fig.add_trace(go.Scatter(x=wdf["Week"], y=wdf["Attendance"], mode="lines+markers", name="Attendance",
                              line=dict(color="#0d9488")))
    fig.update_layout(height=340, yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# SIDEBAR + ROUTER
# ============================================================
def sidebar():
    with st.sidebar:
        st.markdown('<div style="font-size:19px;font-weight:800;color:#fff;">Academic Support</div>'
                     '<div style="font-size:12px;color:#b7a4e8;margin:4px 0 14px;">'
                     'Early Detection & Intervention</div>', unsafe_allow_html=True)
        st.divider()
        st.write(f"**Role:** {st.session_state.role}")
        pages = {"Teacher": ["Dashboard", "Student Profile", "Intervention", "Track"],
                 "Student": ["My Dashboard"], "Parent": ["Child Dashboard"]}[st.session_state.role]
        idx = pages.index(st.session_state.page) if st.session_state.page in pages else 0
        st.session_state.page = st.radio("Navigation", pages, index=idx)
        st.divider()
        if st.button("Logout", use_container_width=True):
            st.session_state.update(logged_in=False, role=None, page="Dashboard"); st.rerun()
        st.markdown('<div style="margin-top:18px;font-size:11px;color:#9481c4;">Synthetic demonstration data.<br>'
                     'Human review required for real deployment.</div>', unsafe_allow_html=True)

if not st.session_state.logged_in:
    login_screen()
else:
    sidebar()
    if st.session_state.role == "Teacher":
        {"Dashboard": teacher_dashboard, "Student Profile": student_profile,
         "Intervention": intervention_page, "Track": track_page}[st.session_state.page]()
    elif st.session_state.role == "Student":
        student_dashboard()
    else:
        parent_dashboard()
