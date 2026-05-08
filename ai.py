from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from openai import OpenAI


def build_case_payload(
    latest_user: Optional[Dict[str, Any]],
    latest_sensor: Optional[Dict[str, Any]],
    latest_assessment: Optional[Dict[str, Any]],
    latest_training: Optional[Dict[str, Any]],
    assessment_history: list[Dict[str, Any]],
    training_history: list[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "user_info": latest_user or {},
        "sensor_result": latest_sensor or {},
        "assessment_result": latest_assessment or {},
        "training_result": latest_training or {},
        "assessment_history_tail": assessment_history[-5:],
        "training_history_tail": training_history[-5:],
    }


def _trend_desc(records: list[Dict[str, Any]], score_key: str) -> str:
    if len(records) < 2:
        return "暂无明显趋势数据"
    recent = [float(r.get(score_key, 0)) for r in records[-3:]]
    if recent[-1] > recent[0]:
        return "近阶段呈改善趋势"
    if recent[-1] < recent[0]:
        return "近阶段存在波动或下降趋势"
    return "近阶段整体较稳定"


def generate_rule_report(case_payload: Dict[str, Any]) -> Dict[str, Any]:
    user_info = case_payload.get("user_info", {})
    sensor = case_payload.get("sensor_result", {})
    assessment = case_payload.get("assessment_result", {})
    training = case_payload.get("training_result", {})
    assessment_history = case_payload.get("assessment_history_tail", [])
    training_history = case_payload.get("training_history_tail", [])

    if not user_info:
        return {
            "provider": "rule-based",
            "status": "信息不足",
            "major_issue": "尚未完成用户建档。",
            "training_focus": "请先完成建档、评估与训练。",
            "advice": "建议先录入用户信息，再进行感知监测、视功能评估和康复训练。",
            "summary": "当前数据不足，系统无法形成完整康复判断。",
            "risk_level": "中",
            "followup": "建议先完成首轮完整流程后再查看分析结果。",
        }

    assessment_score = float(assessment.get("assessment_score", 0.0))
    training_score = float(training.get("training_score", 0.0))
    sensor_score = float(sensor.get("sensor_score", 0.0))
    main_problem = user_info.get("main_problem", "暂无明显困扰")
    stage = user_info.get("surgery_stage", "未知阶段")
    glare = float(sensor.get("glare_risk", 0.0))
    fatigue = float(sensor.get("fatigue", 0.0))

    composite = 0.35 * assessment_score * 25 + 0.35 * training_score * 12.5 + 0.30 * sensor_score

    if composite >= 75:
        status = "恢复状态较好"
        focus = "继续巩固阅读舒适度、低对比辨识能力与视觉稳定性。"
        risk = "低"
        followup = "建议维持常规训练与每周阶段复盘。"
    elif composite >= 55:
        status = "恢复状态中等"
        focus = "继续加强方向辨识、阅读适应和视觉搜索训练。"
        risk = "中"
        followup = "建议连续观察近 3 次趋势，必要时进行院外随访。"
    else:
        status = "恢复基础较弱"
        focus = "建议从高对比度、大字号和短时训练开始，逐步提升难度。"
        risk = "高"
        followup = "建议优先优化环境与训练节奏，若连续多次偏低建议复查。"

    advices = [f"当前处于{stage}，训练节奏应与术后阶段匹配。"]
    if main_problem == "阅读容易疲劳":
        advices.append("建议采用分段式阅读训练，单次控制在5到8分钟。")
    elif main_problem == "强光下不适":
        advices.append("建议避免强光直射，优化室内均匀照明。")
    elif main_problem == "对比辨识较弱":
        advices.append("建议优先完成高对比到低对比的渐进训练。")
    elif main_problem == "夜间视物不稳定":
        advices.append("建议减少复杂弱光环境中的长时间用眼。")

    if sensor_score and sensor_score < 50:
        advices.append("当前感知条件一般，建议调整光照、角度或减少遮挡后重试。")
    if glare > 40:
        advices.append("当前反光风险偏高，建议避免镜片反光或强光源直射。")
    if fatigue > 60:
        advices.append("当前疲劳指数偏高，建议降低单次训练强度并增加休息。")

    summary = (
        f"综合评估分为 {assessment_score:.1f}，训练分为 {training_score:.1f}，感知综合分为 {sensor_score:.1f}。"
        f"评估趋势：{_trend_desc(assessment_history, 'assessment_score') if assessment_history else '暂无'}；"
        f"训练趋势：{_trend_desc(training_history, 'training_score') if training_history else '暂无'}。"
    )

    return {
        "provider": "rule-based",
        "status": status,
        "major_issue": f"当前主要视觉困扰为：{main_problem}。",
        "training_focus": focus,
        "advice": "；".join(advices),
        "summary": summary,
        "risk_level": risk,
        "followup": followup,
    }


def _load_vectorengine_config():
    api_key = os.getenv("VE_API_KEY")
    base_url = os.getenv("VE_BASE_URL", "https://api.vectorengine.ai/v1")
    model = os.getenv("VE_MODEL", "gpt-5.5-pro")

    if not api_key:
        raise RuntimeError("未检测到 VE_API_KEY，请先在环境变量中配置。")
    return api_key, base_url, model


def _build_prompt(case_payload: Dict[str, Any]) -> str:
    return f"""
你是“白内障术后视觉康复辅助系统”的AI报告助手。
请根据结构化数据，生成一份专业、清晰、简洁、非诊断性的康复辅助报告。

要求：
1. 不做医学确诊，不下疾病结论。
2. 只输出固定JSON。
3. 字段固定为：
   status, major_issue, training_focus, advice, summary, risk_level, followup
4. risk_level 只能是：低 / 中 / 高
5. 语言适合比赛展示，也适合系统实际输出。
6. 不要输出Markdown，不要输出解释，只返回JSON对象。

结构化数据如下：
{json.dumps(case_payload, ensure_ascii=False, indent=2)}
""".strip()


def generate_vectorengine_report(case_payload: Dict[str, Any]) -> Dict[str, Any]:
    api_key, base_url, model = _load_vectorengine_config()
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是医疗康复辅助系统中的报告生成助手，只能输出合法JSON。"},
            {"role": "user", "content": _build_prompt(case_payload)},
        ],
        temperature=0.3,
    )
    text = (response.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(text)
    except Exception:
        fallback = generate_rule_report(case_payload)
        fallback["provider"] = f"vectorengine-nonjson:{model}"
        fallback["summary"] += f" 第三方模型未返回合法JSON，原始输出片段：{text[:200]}"
        return fallback

    parsed["provider"] = f"vectorengine:{model}"
    return parsed


def generate_report(case_payload: Dict[str, Any], use_real_ai: bool) -> Dict[str, Any]:
    if use_real_ai:
        try:
            return generate_vectorengine_report(case_payload)
        except Exception as exc:
            fallback = generate_rule_report(case_payload)
            fallback["provider"] = "rule-based (vectorengine-fallback)"
            fallback["summary"] += f" 第三方AI接口调用失败，已回退规则版：{type(exc).__name__}: {exc}"
            return fallback
    return generate_rule_report(case_payload)
