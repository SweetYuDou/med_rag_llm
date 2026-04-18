import json
import re
from typing import Any


TYPE_TO_LABEL = {
    "symptom": "Symptom",
    "disease": "Disease",
    "department": "Department",
    "check": "Check",
    "drug": "Drug",
}

RED_FLAG_TERMS = {
    "呼吸困难",
    "胸痛",
    "胸闷",
    "意识模糊",
    "昏迷",
    "抽搐",
    "咯血",
    "便血",
    "黑便",
    "剧烈腹痛",
    "持续高热",
}

URGENT_HINT_TERMS = {
    "高热",
    "发热",
    "发烧",
    "喘",
    "气短",
    "头晕",
    "呕吐",
    "腹泻",
}


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def dedupe(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def to_string_list(values: Any) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    return dedupe([normalize_text(value) for value in values if normalize_text(value)])


def unique_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for node in nodes:
        mention = normalize_text(node.get("mention"))
        node_name = normalize_text(node.get("node_name"))
        node_type = normalize_text(node.get("node_type")).lower()
        polarity = normalize_text(node.get("polarity")).lower() or "positive"
        confidence = node.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        if node_type not in {"symptom", "disease", "department", "check", "drug"}:
            continue
        if polarity not in {"positive", "negative"}:
            polarity = "positive"
        if not mention or not node_name:
            continue
        key = (mention, node_name, node_type, polarity)
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "mention": mention,
                "node_name": node_name,
                "node_type": node_type,
                "polarity": polarity,
                "confidence": round(confidence, 4),
            }
        )
    return result


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = normalize_text(text)
    if not cleaned:
        return {}

    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    if cleaned.startswith("{") and cleaned.endswith("}"):
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    start_positions = [index for index, char in enumerate(cleaned) if char == "{"]
    for start in start_positions:
        depth = 0
        for index in range(start, len(cleaned)):
            char = cleaned[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start:index + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    raise ValueError("Model output does not contain valid JSON.")


def format_probability(score: float, best_score: float, rank: int) -> str:
    if best_score <= 0:
        return "50%"
    value = 58 + int((score / best_score) * 28) - rank * 6
    value = max(35, min(value, 92))
    return f"{value}%"


def build_care_points(triage_level: str, recommended_tests: list[str], recognized_symptoms: list[str]) -> list[str]:
    points = []
    if triage_level == "建议立即就医":
        points.append("当前存在需要尽快线下评估的风险信号，建议立即前往急诊或就近医院就诊。")
    elif triage_level == "建议尽快门诊":
        points.append("建议尽快安排门诊评估，避免症状持续或进一步加重。")
    else:
        points.append("当前更适合门诊随访和持续观察，如症状变化需及时复诊。")
    if recommended_tests:
        points.append(f"可优先考虑以下检查：{'、'.join(recommended_tests[:4])}。")
    if recognized_symptoms:
        points.append(f"已识别到的主要症状包括：{'、'.join(recognized_symptoms[:4])}。")
    points.append("如出现呼吸困难、持续高热、明显胸痛或意识异常，应立即线下就医。")
    return dedupe(points)


def determine_triage(query: str, history: str, nodes: list[dict[str, Any]]) -> dict[str, str]:
    positive_names = {
        node["node_name"]
        for node in nodes
        if node.get("polarity") == "positive" and node.get("node_type") in {"symptom", "disease"}
    }
    text = f"{query} {history}".strip()
    if any(term in text for term in RED_FLAG_TERMS) or positive_names & RED_FLAG_TERMS:
        return {
            "triage_level": "建议立即就医",
            "advice": "存在较强风险信号，建议立即前往急诊或线下医院进一步评估，不建议继续仅依赖线上判断。",
        }
    if any(term in text for term in URGENT_HINT_TERMS) or positive_names & URGENT_HINT_TERMS or positive_names:
        return {
            "triage_level": "建议尽快门诊",
            "advice": "当前已有明确症状线索，建议尽快到门诊完善评估、检查和进一步分诊。",
        }
    return {
        "triage_level": "建议门诊随访",
        "advice": "目前缺少明确高风险线索，建议先结合症状变化观察，必要时门诊随访。",
    }
