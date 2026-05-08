from __future__ import annotations

import json
import random
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from ai import build_case_payload, generate_report
from analysis import analyze_sensor_image, decode_uploaded_image
from database import (
    build_summary_stats,
    clear_all_data,
    fetch_all,
    fetch_latest,
    get_current_user,
    init_db,
    insert_ai_report,
    insert_assessment_record,
    insert_sensor_record,
    insert_training_record,
    upsert_current_user,
)
from tasks import (
    DIRECTION_SYMBOL_MAP,
    READING_TASKS,
    contains_any,
    default_training_plan,
    dumps_cn,
    generate_contrast_tasks,
    generate_direction_tasks,
    generate_search_task,
)

st.set_page_config(page_title="光愈慧视 Pro Max", page_icon="👁️", layout="wide")
init_db()

THEME_CSS = """
<style>
html, body, [class*="css"]  {
    font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}
.main-title {
    font-size: 2.45rem;
    font-weight: 800;
    color: #0f2e4b;
    margin-bottom: 0.15rem;
}
.sub-title {
    font-size: 1rem;
    color: #5b7288;
    margin-bottom: 1.05rem;
}
.card {
    background: linear-gradient(180deg,#ffffff 0%,#f6faff 100%);
    border: 1px solid #d7e4f2;
    border-radius: 18px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
}
.section-title {
    font-size: 1.22rem;
    font-weight: 800;
    color: #123b5d;
    margin-top: 0.75rem;
    margin-bottom: 0.55rem;
}
.stimulus-box {
    text-align:center;
    border-radius:16px;
    padding:16px;
    margin:8px 0 12px 0;
    background:#fff;
    border:1px solid #e3ebf3;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03);
}
.small-note {color:#60758a;font-size:0.92rem; line-height: 1.65;}
.badge-low, .badge-mid, .badge-high {
    padding: 6px 12px;
    border-radius: 999px;
    font-weight: 700;
    display:inline-block;
    font-size: 0.92rem;
}
.badge-low {background: #ecfdf5; color:#065f46;}
.badge-mid {background: #fffbeb; color:#92400e;}
.badge-high {background: #fef2f2; color:#991b1b;}
.metric-card {
    background: linear-gradient(180deg,#ffffff 0%,#f8fbff 100%);
    border: 1px solid #dbe7f3;
    border-radius: 18px;
    padding: 18px 20px;
    min-height: 120px;
    box-shadow: 0 6px 18px rgba(15,23,42,0.05);
}
.metric-title {
    font-size: 0.95rem;
    color: #475569;
    font-weight: 700;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 1.75rem;
    color: #0f172a;
    font-weight: 800;
    line-height: 1.2;
}
.metric-sub {
    font-size: 0.88rem;
    color: #64748b;
    margin-top: 8px;
    line-height: 1.55;
}
.report-card {
    background: #ffffff;
    border: 1px solid #dde7f1;
    border-radius: 18px;
    padding: 18px 20px;
    margin-bottom: 14px;
    min-height: 130px;
}
.report-head {
    font-size: 0.98rem;
    color: #4b6177;
    font-weight: 800;
    margin-bottom: 8px;
}
.report-text {
    font-size: 1rem;
    color: #0f172a;
    line-height: 1.8;
    white-space: pre-wrap;
    word-break: break-word;
}
</style>
"""


def inject_css() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_report_card(title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="report-card">
            <div class="report-head">{title}</div>
            <div class="report-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_css()


def init_runtime_state() -> None:
    if "contrast_tasks" not in st.session_state:
        st.session_state.contrast_tasks = generate_contrast_tasks()
    if "training_direction_tasks" not in st.session_state:
        st.session_state.training_direction_tasks = generate_direction_tasks()
    if "search_task_1" not in st.session_state:
        st.session_state.search_task_1 = generate_search_task()
    if "search_task_2" not in st.session_state:
        st.session_state.search_task_2 = generate_search_task()


def refresh_data() -> Dict[str, Any]:
    return {
        "latest_user": get_current_user(),
        "latest_sensor": fetch_latest("sensor_records"),
        "latest_assessment": fetch_latest("assessment_records"),
        "latest_training": fetch_latest("training_records"),
        "latest_report": fetch_latest("ai_reports"),
        "assessment_history": fetch_all("assessment_records"),
        "training_history": fetch_all("training_records"),
        "sensor_history": fetch_all("sensor_records"),
        "stats": build_summary_stats(),
    }


def risk_badge(risk_level: str) -> str:
    mapping = {
        "低": '<span class="badge-low">低风险</span>',
        "中": '<span class="badge-mid">中风险</span>',
        "高": '<span class="badge-high">高风险</span>',
    }
    return mapping.get(risk_level, '<span class="badge-mid">暂无</span>')


init_runtime_state()
data = refresh_data()
latest_user = data["latest_user"]
latest_sensor = data["latest_sensor"]
latest_assessment = data["latest_assessment"]
latest_training = data["latest_training"]
latest_report = data["latest_report"]
assessment_history = data["assessment_history"]
training_history = data["training_history"]
sensor_history = data["sensor_history"]
stats = data["stats"]

st.markdown('<div class="main-title">光愈慧视 Pro Max —— 白内障术后视觉康复智能系统</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">完整成品版：实时感知监测 + SQLite 持久化 + 家属/医护视图 + 风险预警 + OpenAI 兼容第三方接口</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 导航菜单")
    page = st.radio(
        "请选择功能页",
        ["首页", "用户建档", "实时感知监测", "视功能评估", "康复训练", "结果汇总", "历史记录", "AI 康复分析", "家属/医护视图"],
    )
    st.markdown("---")
    use_real_ai = st.toggle("启用真实 AI 接口", value=False, help="需要配置 VE_API_KEY / VE_BASE_URL / VE_MODEL。")
    if use_real_ai:
        st.caption("如果第三方接口失败，系统会自动回退到规则版报告。")
    if st.button("清空全部演示数据", type="secondary"):
        clear_all_data()
        for key in ["contrast_tasks", "training_direction_tasks", "search_task_1", "search_task_2"]:
            if key in st.session_state:
                del st.session_state[key]
        st.success("SQLite 数据已清空，请手动刷新页面。")

if page == "首页":
    st.markdown('<div class="section-title">系统定位</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
        本系统面向白内障术后居家康复场景，融合用户建档、实时感知监测、视功能评估、康复训练、
        结果汇总、历史趋势追踪、家属/医护视图和 AI 康复报告，目标是把“术后回家之后最容易被忽视的恢复过程”
        做成一个低成本、可部署、可持续迭代的智能康复辅助系统。
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("用户建档", "SQLite", "连续保存用户档案、评估、训练与报告数据")
    with c2:
        render_metric_card("实时感知", "OpenCV", "支持人脸、双眼、亮度、疲劳和反光风险分析")
    with c3:
        render_metric_card("AI 报告", "第三方兼容", "支持 OpenAI 兼容接口，失败自动回退规则版")

    c4, c5, c6 = st.columns(3)
    with c4:
        render_metric_card("评估记录数", str(stats["assessment_count"]), "用于观察术后阶段变化")
    with c5:
        render_metric_card("训练记录数", str(stats["training_count"]), "支持训练依从性与完成率追踪")
    with c6:
        render_metric_card("最新风险等级", stats["latest_risk"], "用于家属 / 医护端快速识别风险")

    st.markdown('<div class="section-title">升级亮点</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    bullets = [
        ("低成本终端形态", "适合 iPad / 网页终端展示"),
        ("实时感知监测", "人脸、双眼、亮度、疲劳、反光指标"),
        ("完整数据链路", "SQLite 持久化，支持连续记录与导出"),
        ("真实 AI 接口", "可连接 OpenAI 兼容第三方服务生成报告"),
    ]
    for col, (title, desc) in zip(cols, bullets):
        with col:
            st.markdown(f'<div class="card"><b>{title}</b><br><span class="small-note">{desc}</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">当前用户概览</div>', unsafe_allow_html=True)
    st.json({
        "当前用户": latest_user["name"] if latest_user else "未建档",
        "术后阶段": latest_user["surgery_stage"] if latest_user else "暂无",
        "主要困扰": latest_user["main_problem"] if latest_user else "暂无",
        "最新感知时间": latest_sensor["created_at"] if latest_sensor else "暂无",
        "最新评估时间": latest_assessment["created_at"] if latest_assessment else "暂无",
        "最新训练时间": latest_training["created_at"] if latest_training else "暂无",
        "最新 AI 报告": latest_report["created_at"] if latest_report else "暂无",
    })

elif page == "用户建档":
    st.markdown('<div class="section-title">用户建档</div>', unsafe_allow_html=True)
    default_name = latest_user["name"] if latest_user else "演示患者A"
    with st.form("profile_form"):
        name = st.text_input("姓名 / 代号", value=default_name)
        age = st.number_input("年龄", min_value=40, max_value=100, value=int(latest_user["age"]) if latest_user and latest_user.get("age") else 65, step=1)
        stage_options = ["术后1周内", "术后1-2周", "术后2-4周", "术后1个月以上"]
        stage_default = latest_user.get("surgery_stage", "术后1周内") if latest_user else "术后1周内"
        surgery_stage = st.selectbox("术后时间", stage_options, index=stage_options.index(stage_default))
        surgery_type = st.radio("手术情况", ["单眼术后", "双眼术后"], index=0 if not latest_user or latest_user.get("surgery_type") == "单眼术后" else 1)
        problem_options = ["阅读容易疲劳", "强光下不适", "对比辨识较弱", "夜间视物不稳定", "暂无明显困扰"]
        problem_default = latest_user.get("main_problem", "阅读容易疲劳") if latest_user else "阅读容易疲劳"
        main_problem = st.selectbox("当前主要视觉困扰", problem_options, index=problem_options.index(problem_default))
        note = st.text_area("补充说明", value="" if not latest_user else latest_user.get("note", ""))
        submitted = st.form_submit_button("保存建档信息", type="primary")

    if submitted:
        user_id = upsert_current_user({
            "name": name,
            "age": age,
            "surgery_stage": surgery_stage,
            "surgery_type": surgery_type,
            "main_problem": main_problem,
            "note": note,
        })
        st.success(f"用户信息已写入 SQLite，用户 ID = {user_id}")
        st.rerun()

    if latest_user:
        left, right = st.columns([1, 1])
        with left:
            st.markdown("### 当前建档信息")
            st.json(latest_user)
        with right:
            st.markdown("### 建议训练计划")
            for item in default_training_plan(latest_user.get("main_problem", "暂无明显困扰")):
                st.write(f"- {item}")

elif page == "实时感知监测":
    st.markdown('<div class="section-title">实时感知监测</div>', unsafe_allow_html=True)
    st.caption("请使用前置摄像头拍摄当前面部图像，系统将基于 OpenCV 计算人脸/双眼状态、环境亮度、反光风险、稳定度和疲劳指数。")
    photo = st.camera_input("拍摄当前面部图像")

    if photo is not None:
        image = decode_uploaded_image(photo.getvalue())
        result = analyze_sensor_image(image)
        user_id = latest_user["id"] if latest_user else None
        insert_sensor_record(user_id, result)
        st.success("感知监测完成，结果已保存。")
        st.rerun()

    latest_sensor = fetch_latest("sensor_records")
    if latest_sensor:
        left, right = st.columns([1.05, 1])
        with left:
            st.markdown("### 最近一次监测概览")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("人脸检测", "成功" if latest_sensor["face_detected"] else "失败")
            c2.metric("双眼状态", "稳定" if latest_sensor["eye_detected"] else "一般")
            c3.metric("环境亮度", f"{float(latest_sensor['brightness']):.1f}")
            c4.metric("感知综合分", f"{float(latest_sensor['sensor_score']):.1f}")
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("面部稳定度", f"{float(latest_sensor['stability']):.1f}")
            c6.metric("眼部专注度", f"{float(latest_sensor['attention']):.1f}")
            c7.metric("疲劳指数", f"{float(latest_sensor['fatigue']):.1f}")
            c8.metric("反光风险", f"{float(latest_sensor['glare_risk']):.1f}")
        with right:
            render_report_card("感知建议", latest_sensor["advice"])

elif page == "视功能评估":
    st.markdown('<div class="section-title">视功能评估</div>', unsafe_allow_html=True)
    st.write("请完成方向识别与对比辨识任务，结果将写入数据库并参与后续 AI 报告生成。")

    if "assess_direction" not in st.session_state:
        st.session_state.assess_direction = random.choice(["上", "下", "左", "右"])

    with st.form("assessment_form"):
        st.markdown(
            f'<div class="stimulus-box"><div style="font-size:84px; font-weight:700; color:#1f4e79;">{DIRECTION_SYMBOL_MAP[st.session_state.assess_direction]}</div></div>',
            unsafe_allow_html=True,
        )
        q1 = st.radio("请选择你看到的方向", ["上", "下", "左", "右"])

        contrast_answers: List[str] = []
        for idx, task in enumerate(st.session_state.contrast_tasks):
            st.markdown(
                f'<div class="stimulus-box"><div style="font-size:62px; color:{task["color"]}; font-weight:700;">{DIRECTION_SYMBOL_MAP[task["direction"]]}</div></div>',
                unsafe_allow_html=True,
            )
            ans = st.radio(f"请选择第 {idx + 1} 轮目标方向", ["上", "下", "左", "右"], key=f"assess_{idx}")
            contrast_answers.append(ans)
        submitted = st.form_submit_button("提交评估", type="primary")

    if submitted:
        direction_score = 1.0 if q1 == st.session_state.assess_direction else 0.0
        contrast_score = sum(1.0 for idx, ans in enumerate(contrast_answers) if ans == st.session_state.contrast_tasks[idx]["direction"])
        assessment_score = round(direction_score + contrast_score, 2)
        detail = {
            "selected_direction": q1,
            "system_direction": st.session_state.assess_direction,
            "contrast_answers": contrast_answers,
            "contrast_tasks": st.session_state.contrast_tasks,
        }
        user_id = latest_user["id"] if latest_user else None
        insert_assessment_record(
            user_id,
            {
                "direction_score": direction_score,
                "contrast_score": contrast_score,
                "assessment_score": assessment_score,
            },
            dumps_cn(detail),
        )
        st.session_state.contrast_tasks = generate_contrast_tasks()
        st.session_state.assess_direction = random.choice(["上", "下", "左", "右"])
        st.success("评估结果已保存到 SQLite。")
        st.rerun()

    latest_assessment = fetch_latest("assessment_records")
    if latest_assessment:
        st.markdown("### 最近一次评估结果")
        c1, c2, c3 = st.columns(3)
        with c1:
            render_metric_card("方向识别得分", f"{float(latest_assessment['direction_score']):.1f}", "用于基础视功能辨识")
        with c2:
            render_metric_card("对比辨识得分", f"{float(latest_assessment['contrast_score']):.1f}", "用于对低对比刺激的适应程度")
        with c3:
            render_metric_card("评估总分", f"{float(latest_assessment['assessment_score']):.1f}", "本轮视功能评估综合结果")

elif page == "康复训练":
    st.markdown('<div class="section-title">康复训练</div>', unsafe_allow_html=True)
    st.write("训练模块采用真实任务设计，包括方向辨识训练、阅读适应训练和视觉搜索训练。")

    tab1, tab2, tab3 = st.tabs(["方向辨识训练", "阅读适应训练", "视觉搜索训练"])
    with st.form("training_form"):
        direction_inputs: List[str] = []
        read_inputs: List[str] = []

        with tab1:
            st.markdown("### 渐进式方向辨识")
            for idx, task in enumerate(st.session_state.training_direction_tasks):
                st.markdown(
                    f'<div class="stimulus-box"><div style="font-size:72px; color:{task["color"]}; font-weight:700;">{DIRECTION_SYMBOL_MAP[task["direction"]]}</div><div class="small-note">{task["label"]}</div></div>',
                    unsafe_allow_html=True,
                )
                direction_inputs.append(st.text_input(f"请输入第 {idx + 1} 轮方向（上/下/左/右）", key=f"direction_{idx}"))

        with tab2:
            st.markdown("### 阅读适应训练")
            for idx, task in enumerate(READING_TASKS):
                st.markdown(
                    f'<div class="stimulus-box" style="font-size:{task["font_size"]}px;color:{task["color"]};">{task["text"]}<div class="small-note">{task["level"]}</div></div>',
                    unsafe_allow_html=True,
                )
                read_inputs.append(st.text_input(f"第 {idx + 1} 轮：{task['prompt']}", key=f"read_{idx}"))

        with tab3:
            st.markdown("### 视觉搜索训练")
            task1 = st.session_state.search_task_1
            task2 = st.session_state.search_task_2
            st.markdown(
                f'<div class="stimulus-box"><div style="font-size:34px; line-height:1.8;">{" ".join(task1["symbols"])}</div><div class="small-note">目标符号：{task1["target"]}</div></div>',
                unsafe_allow_html=True,
            )
            count1 = st.number_input("请输入第一组目标符号出现次数", min_value=0, max_value=20, step=1)
            st.markdown(
                f'<div class="stimulus-box"><div style="font-size:34px; line-height:1.8;">{" ".join(task2["symbols"])}</div><div class="small-note">目标符号：{task2["target"]}</div></div>',
                unsafe_allow_html=True,
            )
            count2 = st.number_input("请输入第二组目标符号出现次数", min_value=0, max_value=20, step=1)

        submitted = st.form_submit_button("提交训练", type="primary")

    if submitted:
        task1 = st.session_state.search_task_1
        task2 = st.session_state.search_task_2
        direction_score = sum(1.0 for idx, value in enumerate(direction_inputs) if value.strip() == st.session_state.training_direction_tasks[idx]["direction"])
        reading_score = 0.0
        for idx, task in enumerate(READING_TASKS):
            if contains_any(read_inputs[idx], task["keywords"]):
                reading_score += 1.0
        search_score = 0.0
        if int(count1) == int(task1["count"]):
            search_score += 1.0
        if int(count2) == int(task2["count"]):
            search_score += 1.0
        training_score = round(direction_score + reading_score + search_score, 2)
        completion_rate = round((training_score / 8.0) * 100.0, 2)
        detail = {
            "direction_inputs": direction_inputs,
            "read_inputs": read_inputs,
            "search_counts": [int(count1), int(count2)],
            "search_ground_truth": [int(task1["count"]), int(task2["count"])],
        }
        user_id = latest_user["id"] if latest_user else None
        insert_training_record(
            user_id,
            {
                "direction_score": direction_score,
                "reading_score": reading_score,
                "search_score": search_score,
                "training_score": training_score,
                "completion_rate": completion_rate,
            },
            dumps_cn(detail),
        )
        st.session_state.training_direction_tasks = generate_direction_tasks()
        st.session_state.search_task_1 = generate_search_task()
        st.session_state.search_task_2 = generate_search_task()
        st.success("训练结果已保存到 SQLite。")
        st.rerun()

    latest_training = fetch_latest("training_records")
    if latest_training:
        st.markdown("### 最近一次训练结果")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric_card("方向训练得分", f"{float(latest_training['direction_score']):.1f}", "渐进式方向辨识训练")
        with c2:
            render_metric_card("阅读训练得分", f"{float(latest_training['reading_score']):.1f}", "阅读适应与理解任务")
        with c3:
            render_metric_card("搜索训练得分", f"{float(latest_training['search_score']):.1f}", "视觉扫描与搜索能力")
        with c4:
            render_metric_card("训练完成率", f"{float(latest_training['completion_rate']):.1f}%", "反映当前训练依从性")

elif page == "结果汇总":
    st.markdown('<div class="section-title">结果汇总</div>', unsafe_allow_html=True)
    a_score = float(latest_assessment["assessment_score"]) if latest_assessment else 0.0
    t_score = float(latest_training["training_score"]) if latest_training else 0.0
    s_score = float(latest_sensor["sensor_score"]) if latest_sensor else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最新评估得分", f"{a_score:.1f}")
    c2.metric("最新训练得分", f"{t_score:.1f}")
    c3.metric("感知综合分", f"{s_score:.1f}")
    c4.metric("平均训练完成率", f"{stats['avg_completion']:.1f}%")
    if latest_user:
        st.markdown("### 用户信息")
        st.json(latest_user)
    if latest_sensor:
        render_report_card("感知监测建议", latest_sensor["advice"])
    overall = 0.35 * a_score * 25 + 0.35 * t_score * 12.5 + 0.30 * s_score
    if overall >= 75:
        st.success("综合判断：当前恢复状态较好，可维持常规训练节奏。")
    elif overall >= 55:
        st.warning("综合判断：当前处于恢复过渡阶段，应继续巩固训练。")
    else:
        st.error("综合判断：当前基础恢复能力仍需加强，建议降低单次训练难度并优化使用环境。")

elif page == "历史记录":
    st.markdown('<div class="section-title">历史记录</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["评估趋势", "训练趋势", "感知趋势"])
    with t1:
        if assessment_history:
            df_a = pd.DataFrame(assessment_history)
            st.dataframe(df_a[["id", "created_at", "direction_score", "contrast_score", "assessment_score"]], use_container_width=True)
            fig_a = px.line(df_a, x="id", y="assessment_score", markers=True, title="评估得分变化趋势")
            st.plotly_chart(fig_a, use_container_width=True)
        else:
            st.info("暂无评估历史记录。")
    with t2:
        if training_history:
            df_t = pd.DataFrame(training_history)
            st.dataframe(df_t[["id", "created_at", "direction_score", "reading_score", "search_score", "training_score", "completion_rate"]], use_container_width=True)
            fig_t = px.line(df_t, x="id", y=["training_score", "completion_rate"], markers=True, title="训练得分 / 完成率趋势")
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.info("暂无训练历史记录。")
    with t3:
        if sensor_history:
            df_s = pd.DataFrame(sensor_history)
            st.dataframe(df_s[["id", "created_at", "brightness", "stability", "attention", "fatigue", "glare_risk", "sensor_score"]], use_container_width=True)
            fig_s = px.line(df_s, x="id", y=["sensor_score", "fatigue", "glare_risk"], markers=True, title="感知综合分 / 疲劳指数 / 反光风险趋势")
            st.plotly_chart(fig_s, use_container_width=True)
        else:
            st.info("暂无感知监测记录。")

elif page == "AI 康复分析":
    st.markdown('<div class="section-title">AI 康复分析</div>', unsafe_allow_html=True)
    st.write("系统将综合用户建档、感知监测、视功能评估与训练结果，生成阶段性康复报告。")
    if st.button("生成 AI 康复报告", type="primary"):
        case_payload = build_case_payload(latest_user, latest_sensor, latest_assessment, latest_training, assessment_history, training_history)
        report = generate_report(case_payload, use_real_ai=use_real_ai)
        user_id = latest_user["id"] if latest_user else None
        insert_ai_report(user_id, report)
        st.success("AI 报告已生成并保存。")
        st.rerun()

    latest_report = fetch_latest("ai_reports")
    if latest_report:
        c1, c2, c3 = st.columns([1, 1, 1])
        c1.metric("恢复状态", latest_report["status"])
        c2.metric("AI 提供方", latest_report["provider"])
        c3.markdown(risk_badge(latest_report["risk_level"]), unsafe_allow_html=True)

        left, right = st.columns(2)
        with left:
            render_report_card("主要问题", latest_report["major_issue"])
            render_report_card("训练重点", latest_report["training_focus"])
            render_report_card("使用建议", latest_report["advice"])
        with right:
            render_report_card("随访建议", latest_report["followup"])
            render_report_card("阶段总结", latest_report["summary"])

        report_markdown = f"""# 光愈慧视康复报告

- 生成时间：{latest_report['created_at']}
- AI 提供方：{latest_report['provider']}
- 恢复状态：{latest_report['status']}
- 风险等级：{latest_report['risk_level']}

## 主要问题
{latest_report['major_issue']}

## 训练重点
{latest_report['training_focus']}

## 使用建议
{latest_report['advice']}

## 随访建议
{latest_report['followup']}

## 阶段总结
{latest_report['summary']}
"""
        report_json = json.dumps(dict(latest_report), ensure_ascii=False, indent=2)
        col_a, col_b = st.columns(2)
        col_a.download_button("下载 Markdown 报告", report_markdown.encode("utf-8"), file_name=f"guangyuhuishi_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md", mime="text/markdown")
        col_b.download_button("下载 JSON 报告", report_json.encode("utf-8"), file_name=f"guangyuhuishi_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")
    else:
        st.info("请先生成一份 AI 康复报告。")

elif page == "家属/医护视图":
    st.markdown('<div class="section-title">家属 / 医护视图</div>', unsafe_allow_html=True)
    st.write("该页面用于展示术后康复的连续数据、训练依从性和风险提醒，增强项目的系统化表达。")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("平均评估分", f"{stats['avg_assessment']:.1f}")
    c2.metric("平均训练分", f"{stats['avg_training']:.1f}")
    c3.metric("平均感知分", f"{stats['avg_sensor']:.1f}")
    c4.metric("训练完成率", f"{stats['avg_completion']:.1f}%")

    if latest_report:
        st.markdown("### 当前风险提醒")
        st.markdown(risk_badge(latest_report["risk_level"]), unsafe_allow_html=True)
        render_report_card("当前随访建议", latest_report["followup"])
    else:
        st.info("尚无 AI 报告，无法生成风险提醒。")

    if training_history:
        df_t = pd.DataFrame(training_history)
        df_t["created_at"] = pd.to_datetime(df_t["created_at"])
        df_t["date"] = df_t["created_at"].dt.date.astype(str)
        daily = df_t.groupby("date", as_index=False).agg({"completion_rate": "mean", "training_score": "mean"})
        st.markdown("### 家属可见：近期依从性趋势")
        fig = px.bar(daily, x="date", y="completion_rate", title="按日期统计的平均训练完成率")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.line(daily, x="date", y="training_score", markers=True, title="按日期统计的平均训练得分")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("暂无训练数据。")

    st.markdown("### 推荐使用场景")
    st.write("- 家属查看近期训练是否连续、是否存在疲劳或反光问题")
    st.write("- 医护查看趋势图，判断是否需要调整训练计划或提示复查")
    st.write("- 比赛答辩中用于体现“患者端 + 家属端 / 医护端 + 数据平台”的系统层级")
