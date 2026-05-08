from __future__ import annotations

import json
import random
from typing import Any, Dict, List

DIRECTION_SYMBOL_MAP = {
    "上": "↑",
    "下": "↓",
    "左": "←",
    "右": "→",
}

READING_TASKS = [
    {
        "text": "术后恢复应保持规律训练与适度休息。",
        "prompt": "请输入文中提到的一个关键词",
        "keywords": ["规律训练", "适度休息", "训练", "休息"],
        "font_size": 26,
        "color": "#222222",
        "level": "高对比度 / 大字号",
    },
    {
        "text": "逐步适应不同亮度环境，有助于提升视觉舒适度。",
        "prompt": "请输入文中的一个核心词",
        "keywords": ["亮度环境", "视觉舒适度", "亮度", "舒适度", "环境"],
        "font_size": 22,
        "color": "#555555",
        "level": "中对比度 / 中字号",
    },
    {
        "text": "阅读任务建议从大字号开始，逐渐过渡到中小字号。",
        "prompt": "请输入建议开始的字号类型",
        "keywords": ["大字号", "大字", "字号"],
        "font_size": 18,
        "color": "#888888",
        "level": "低对比度 / 小字号",
    },
]


def dumps_cn(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    return str(text).strip().replace(" ", "").replace("，", "").replace("。", "").replace("；", "").replace("、", "")


def contains_any(text: str, keywords: List[str]) -> bool:
    t = normalize_text(text)
    for kw in keywords:
        if normalize_text(kw) in t:
            return True
    return False


def generate_direction_tasks() -> List[Dict[str, str]]:
    return [
        {"label": "第一轮（高对比度）", "direction": random.choice(["上", "下", "左", "右"]), "color": "#222222"},
        {"label": "第二轮（中等对比度）", "direction": random.choice(["上", "下", "左", "右"]), "color": "#777777"},
        {"label": "第三轮（低对比度）", "direction": random.choice(["上", "下", "左", "右"]), "color": "#BBBBBB"},
    ]


def generate_contrast_tasks() -> List[Dict[str, str]]:
    return [
        {"direction": random.choice(["上", "下", "左", "右"]), "color": "#333333"},
        {"direction": random.choice(["上", "下", "左", "右"]), "color": "#777777"},
        {"direction": random.choice(["上", "下", "左", "右"]), "color": "#BBBBBB"},
    ]


def generate_search_task() -> Dict[str, Any]:
    target = random.choice(["↑", "↓", "←", "→"])
    count = random.randint(3, 6)
    total = 12
    others = [x for x in ["↑", "↓", "←", "→"] if x != target]
    symbols = [target] * count + [random.choice(others) for _ in range(total - count)]
    random.shuffle(symbols)
    return {"target": target, "count": count, "symbols": symbols}


def default_training_plan(main_problem: str) -> List[str]:
    if main_problem == "阅读容易疲劳":
        return ["阅读适应训练 10 分钟", "方向辨识训练 5 分钟", "分段式训练与休息提醒"]
    if main_problem == "强光下不适":
        return ["亮度适应训练", "环境优化提示", "短时训练与逐步暴露"]
    if main_problem == "对比辨识较弱":
        return ["高对比到低对比渐进训练", "方向辨识训练", "视觉搜索训练"]
    if main_problem == "夜间视物不稳定":
        return ["低亮度环境适应训练", "基础方向与搜索任务", "避免疲劳性训练过长"]
    return ["常规方向辨识训练", "阅读适应训练", "视觉搜索训练"]
