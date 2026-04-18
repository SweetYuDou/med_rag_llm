import json
from functools import lru_cache
from typing import Any

from .config import (
    DEFAULT_NEO4J_PASSWORD,
    DEFAULT_NEO4J_URI,
    DEFAULT_NEO4J_USER,
    DEFAULT_NER_MODEL_PATH,
    DEFAULT_REASON_MODEL_PATH,
    LocalModelConfig,
)
from .helpers import (
    build_care_points,
    dedupe,
    determine_triage,
    format_probability,
    normalize_text,
    to_string_list,
    unique_nodes,
)
from .kg_retriever import StructuredNeo4jRetriever
from .models import LocalChatModel
from .prompts import NER_SYSTEM_PROMPT, REASON_SYSTEM_PROMPT


class LocalMedicalPipeline:
    def __init__(
        self,
        ner_model_path: str,
        reason_model_path: str,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
    ):
        self.ner_model = LocalChatModel(LocalModelConfig(model_path=ner_model_path))
        self.reason_model = LocalChatModel(LocalModelConfig(model_path=reason_model_path))
        self.retriever = StructuredNeo4jRetriever(neo4j_uri, neo4j_user, neo4j_password)

    def close(self):
        self.retriever.close()

    def extract_nodes(self, query: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": NER_SYSTEM_PROMPT},
            {"role": "user", "content": f"用户问诊输入：{query}"},
        ]
        payload = self.ner_model.generate_json(messages, max_new_tokens=420, temperature=0.0)
        nodes = unique_nodes(payload.get("nodes", []))
        parsed_payload = {key: value for key, value in payload.items() if not key.startswith("_")}
        return {
            "nodes": nodes,
            "raw_text": payload.get("_raw_text", ""),
            "prompt": payload.get("_prompt", ""),
            "parse_error": payload.get("_parse_error", ""),
            "parsed_payload": parsed_payload,
        }

    def _build_reason_messages(
        self,
        query: str,
        history: str,
        triage: dict[str, str],
        ner_result: dict[str, Any],
        kg_result: dict[str, Any],
    ) -> list[dict[str, str]]:
        constrained_context = {
            "user_query": query,
            "history": history,
            "triage_hint": triage,
            "recognized_nodes": [
                {
                    "mention": node["mention"],
                    "node_name": node["node_name"],
                    "node_type": node["node_type"],
                    "polarity": node["polarity"],
                }
                for node in ner_result.get("nodes", [])
            ],
            "matched_node_details": kg_result.get("matched_node_details", []),
            "candidate_diseases": [
                {
                    "name": item["name"],
                    "score": item["score"],
                    "matched_symptoms": item["matched_symptoms"],
                    "possible_additional_symptoms": item["possible_additional_symptoms"],
                    "departments": item["departments"][:2],
                    "checks": item["checks"][:4],
                    "drugs": item["drugs"][:5],
                    "treatments": item["treatments"][:4],
                    "diet_recommendations": item["diet_recommendations"][:4],
                    "diet_avoid": item["diet_avoid"][:4],
                    "desc": item["desc"][:180],
                }
                for item in kg_result.get("candidate_diseases", [])
            ],
            "recommended_department": kg_result.get("recommended_department", {}),
            "recommended_tests": kg_result.get("recommended_tests", []),
            "retrieval_summary": kg_result.get("retrieval_summary", ""),
            "graph_paths": kg_result.get("graph_paths", []),
        }
        return [
            {"role": "system", "content": REASON_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "请基于下面的知识图谱检索结果生成最终辅助诊断报告。"
                    "只能使用给定的候选疾病、检查、科室、药物和治疗信息，不允许扩展图谱外实体。\n"
                    + json.dumps(constrained_context, ensure_ascii=False, indent=2)
                ),
            },
        ]

    def _build_default_final_answer(
        self,
        triage: dict[str, str],
        ner_result: dict[str, Any],
        kg_result: dict[str, Any],
    ) -> dict[str, Any]:
        candidates = kg_result.get("candidate_diseases", [])
        recognized_nodes = [
            {
                "mention": node["mention"],
                "node_name": node["node_name"],
                "node_type": node["node_type"],
                "polarity": node["polarity"],
            }
            for node in ner_result.get("nodes", [])
        ]
        recognized_symptoms = kg_result.get("recognized_symptoms", [])
        possible_additional_symptoms = kg_result.get("possible_additional_symptoms", [])
        recommended_department = kg_result.get("recommended_department", {})
        recommended_tests = kg_result.get("recommended_tests", [])
        best_score = candidates[0]["score"] if candidates else 0

        top_diseases = []
        for rank, item in enumerate(candidates[:3]):
            reason_parts = []
            if item.get("matched_symptoms"):
                reason_parts.append(f"匹配症状：{'、'.join(item['matched_symptoms'][:3])}")
            if item.get("desc"):
                reason_parts.append(item["desc"][:72])
            if item.get("departments"):
                reason_parts.append(f"相关科室：{'、'.join(item['departments'][:2])}")
            top_diseases.append(
                {
                    "name": item["name"],
                    "probability": format_probability(item["score"], best_score, rank),
                    "reason": "；".join(reason_parts) or "来自知识图谱召回结果。",
                    "matched_symptoms": item.get("matched_symptoms", []),
                    "possible_additional_symptoms": item.get("possible_additional_symptoms", []),
                    "recommended_department": item.get("departments", [])[:2],
                    "recommended_checks": item.get("checks", [])[:4],
                    "treatment_options": (item.get("treatments", []) + item.get("drugs", []))[:6],
                }
            )

        treatment_plan = {
            "therapies": kg_result.get("treatment_plan", {}).get("therapies", []),
            "medications": kg_result.get("treatment_plan", {}).get("medications", []),
            "diet_recommendations": kg_result.get("treatment_plan", {}).get("diet_recommendations", []),
            "diet_avoid": kg_result.get("treatment_plan", {}).get("diet_avoid", []),
            "care_points": build_care_points(
                triage["triage_level"],
                recommended_tests,
                recognized_symptoms,
            ),
        }

        warning = "如出现呼吸困难、持续高热、明显胸痛、意识异常或症状快速加重，应立即线下就医。"
        if triage["triage_level"] == "建议立即就医":
            warning = "当前存在较强风险信号，建议立即急诊或尽快线下评估，不应继续延迟。"

        advice = triage["advice"]
        if top_diseases:
            advice = (
                f"当前图谱证据优先支持按 {top_diseases[0]['name']} 相关路径进行评估，"
                f"建议首先前往 {recommended_department.get('primary', '全科医学科')} 就诊。"
            )

        return {
            "triage_level": triage["triage_level"],
            "advice": advice,
            "recognized_nodes": recognized_nodes,
            "recognized_symptoms": recognized_symptoms,
            "possible_additional_symptoms": possible_additional_symptoms,
            "top_diseases": top_diseases,
            "recommended_department": recommended_department,
            "recommended_tests": recommended_tests,
            "treatment_plan": treatment_plan,
            "warning": warning,
            "disclaimer": "仅供辅助参考，不能替代医生面诊和线下检查。",
            "kg_context": {
                "retrieval_summary": kg_result.get("retrieval_summary", ""),
                "graph_paths": kg_result.get("graph_paths", []),
                "matched_node_details": kg_result.get("matched_node_details", []),
                "candidate_diseases": kg_result.get("candidate_diseases", []),
            },
        }

    def _merge_model_answer(self, fallback: dict[str, Any], model_answer: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(model_answer, dict):
            return fallback

        merged = json.loads(json.dumps(fallback, ensure_ascii=False))
        for key in ["triage_level", "advice", "warning", "disclaimer"]:
            value = normalize_text(model_answer.get(key))
            if value:
                merged[key] = value

        if isinstance(model_answer.get("recognized_nodes"), list):
            merged["recognized_nodes"] = unique_nodes(model_answer.get("recognized_nodes", [])) or merged["recognized_nodes"]

        for key in ["recognized_symptoms", "possible_additional_symptoms", "recommended_tests"]:
            values = to_string_list(model_answer.get(key))
            if values:
                merged[key] = values

        department = model_answer.get("recommended_department")
        if isinstance(department, dict):
            primary = normalize_text(department.get("primary"))
            alternatives = to_string_list(department.get("alternatives"))
            reason = normalize_text(department.get("reason"))
            if primary:
                merged["recommended_department"]["primary"] = primary
            if alternatives:
                merged["recommended_department"]["alternatives"] = alternatives
            if reason:
                merged["recommended_department"]["reason"] = reason

        treatment_plan = model_answer.get("treatment_plan")
        if isinstance(treatment_plan, dict):
            for key in ["therapies", "medications", "diet_recommendations", "diet_avoid", "care_points"]:
                values = to_string_list(treatment_plan.get(key))
                if values:
                    merged["treatment_plan"][key] = values

        top_diseases = model_answer.get("top_diseases")
        if isinstance(top_diseases, list) and top_diseases:
            normalized_items = []
            for index, item in enumerate(top_diseases[:3]):
                if not isinstance(item, dict):
                    continue
                fallback_item = merged["top_diseases"][index] if index < len(merged["top_diseases"]) else {}
                name = normalize_text(item.get("name")) or fallback_item.get("name", "")
                if not name:
                    continue
                normalized_items.append(
                    {
                        "name": name,
                        "probability": normalize_text(item.get("probability")) or fallback_item.get("probability", "50%"),
                        "reason": normalize_text(item.get("reason")) or fallback_item.get("reason", ""),
                        "matched_symptoms": to_string_list(item.get("matched_symptoms")) or fallback_item.get("matched_symptoms", []),
                        "possible_additional_symptoms": to_string_list(item.get("possible_additional_symptoms"))
                        or fallback_item.get("possible_additional_symptoms", []),
                        "recommended_department": to_string_list(item.get("recommended_department"))
                        or fallback_item.get("recommended_department", []),
                        "recommended_checks": to_string_list(item.get("recommended_checks"))
                        or fallback_item.get("recommended_checks", []),
                        "treatment_options": to_string_list(item.get("treatment_options"))
                        or fallback_item.get("treatment_options", []),
                    }
                )
            if normalized_items:
                merged["top_diseases"] = normalized_items

        merged["kg_context"] = fallback["kg_context"]
        return merged

    def generate_advice(
        self,
        query: str,
        history: str,
        triage: dict[str, str],
        ner_result: dict[str, Any],
        kg_result: dict[str, Any],
    ) -> dict[str, Any]:
        fallback = self._build_default_final_answer(triage, ner_result, kg_result)
        messages = self._build_reason_messages(query, history, triage, ner_result, kg_result)
        debug_info = {
            "prompt": self.reason_model._render_prompt(messages),
            "raw_text": "",
            "parse_error": "",
            "parsed_payload": {},
            "used_fallback": False,
            "used_fallback_reason": "",
        }
        try:
            payload = self.reason_model.generate_json(messages, max_new_tokens=1000, temperature=0.2)
            debug_info["prompt"] = payload.get("_prompt", debug_info["prompt"])
            debug_info["raw_text"] = payload.get("_raw_text", "")
            debug_info["parse_error"] = payload.get("_parse_error", "")
            parsed_payload = {key: value for key, value in payload.items() if not key.startswith("_")}
            debug_info["parsed_payload"] = parsed_payload
            if debug_info["parse_error"]:
                debug_info["used_fallback"] = True
                debug_info["used_fallback_reason"] = f"reason model parse failed: {debug_info['parse_error']}"
            return {
                "final_answer": self._merge_model_answer(fallback, parsed_payload),
                "debug": debug_info,
            }
        except Exception as exc:
            debug_info["used_fallback"] = True
            debug_info["used_fallback_reason"] = str(exc)
            return {
                "final_answer": fallback,
                "debug": debug_info,
            }

    def run(
        self,
        query: str,
        *,
        history: str = "",
        limit: int = 5,
    ) -> dict[str, Any]:
        ner_result = self.extract_nodes(query)
        kg_result = self.retriever.query(ner_result["nodes"], limit=limit)
        triage = determine_triage(query, history, ner_result["nodes"])
        kg_result["treatment_plan"]["care_points"] = build_care_points(
            triage["triage_level"],
            kg_result.get("recommended_tests", []),
            kg_result.get("recognized_symptoms", []),
        )
        advice_result = self.generate_advice(query, history, triage, ner_result, kg_result)
        final_answer = advice_result["final_answer"]
        model_outputs = {
            "ner_model": {
                "model_path": self.ner_model.config.model_path,
                "prompt": ner_result.get("prompt", ""),
                "raw_text": ner_result.get("raw_text", ""),
                "parse_error": ner_result.get("parse_error", ""),
                "parsed_payload": ner_result.get("parsed_payload", {}),
                "normalized_nodes": ner_result.get("nodes", []),
            },
            "reason_model": {
                "model_path": self.reason_model.config.model_path,
                **advice_result["debug"],
                "final_answer_after_merge": final_answer,
            },
        }
        return {
            "query": query,
            "history": history,
            "ner_result": ner_result,
            "kg_result": kg_result,
            "model_outputs": model_outputs,
            "final_answer": final_answer,
            **final_answer,
        }


@lru_cache(maxsize=1)
def get_local_medical_pipeline():
    return LocalMedicalPipeline(
        ner_model_path=DEFAULT_NER_MODEL_PATH,
        reason_model_path=DEFAULT_REASON_MODEL_PATH,
        neo4j_uri=DEFAULT_NEO4J_URI,
        neo4j_user=DEFAULT_NEO4J_USER,
        neo4j_password=DEFAULT_NEO4J_PASSWORD,
    )


def get_local_pipeline_runtime_status() -> dict[str, Any]:
    missing = []
    if not DEFAULT_NER_MODEL_PATH:
        missing.append("MED_NER_MODEL_PATH")
    if not DEFAULT_REASON_MODEL_PATH:
        missing.append("MED_REASON_MODEL_PATH")
    if not DEFAULT_NEO4J_PASSWORD:
        missing.append("NEO4J_PASSWORD")
    return {
        "ner_model_path": DEFAULT_NER_MODEL_PATH,
        "reason_model_path": DEFAULT_REASON_MODEL_PATH,
        "neo4j_uri": DEFAULT_NEO4J_URI,
        "neo4j_user": DEFAULT_NEO4J_USER,
        "ner_model_configured": bool(DEFAULT_NER_MODEL_PATH),
        "reason_model_configured": bool(DEFAULT_REASON_MODEL_PATH),
        "neo4j_password_configured": bool(DEFAULT_NEO4J_PASSWORD),
        "ready": not missing,
        "missing": missing,
    }
