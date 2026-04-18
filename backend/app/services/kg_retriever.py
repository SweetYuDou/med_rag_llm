from collections import Counter
from typing import Any

from neo4j import GraphDatabase

from .config import PipelineConfigError
from .helpers import TYPE_TO_LABEL, dedupe, normalize_text, to_string_list


class StructuredNeo4jRetriever:
    def __init__(self, uri: str, user: str, password: str):
        if not password:
            raise PipelineConfigError("Neo4j password is required.")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def _run_disease_bundle_query(self, session, disease_names: list[str]) -> list[dict[str, Any]]:
        names = dedupe(disease_names)
        if not names:
            return []

        cypher = """
        MATCH (d:Disease)
        WHERE d.name IN $disease_names AND coalesce(d.is_placeholder, false) = false
        OPTIONAL MATCH (d)-[:has_symptom]->(s:Symptom)
        WITH d, collect(DISTINCT s.name) AS symptoms
        OPTIONAL MATCH (d)-[:belongs_to]->(dep:Department)
        WITH d, symptoms, collect(DISTINCT dep.name) AS departments
        OPTIONAL MATCH (d)-[:need_check]->(c:Check)
        WITH d, symptoms, departments, collect(DISTINCT c.name) AS checks
        OPTIONAL MATCH (d)-[:use_drug]->(drug:Drug)
        WITH d, symptoms, departments, checks, collect(DISTINCT drug.name) AS drugs
        OPTIONAL MATCH (d)-[:treated_by]->(t:Treatment)
        WITH d, symptoms, departments, checks, drugs, collect(DISTINCT t.name) AS treatments
        OPTIONAL MATCH (d)-[:recommend_eat]->(food1:Food)
        WITH d, symptoms, departments, checks, drugs, treatments, collect(DISTINCT food1.name) AS recommend_eat
        OPTIONAL MATCH (d)-[:not_eat]->(food2:Food)
        RETURN
          d.name AS disease_name,
          d.desc AS disease_desc,
          symptoms,
          departments,
          checks,
          drugs,
          treatments,
          recommend_eat,
          collect(DISTINCT food2.name) AS not_eat
        """
        records = list(session.run(cypher, disease_names=names))
        return [
            {
                "name": record["disease_name"],
                "desc": normalize_text(record["disease_desc"]),
                "symptoms": to_string_list(record["symptoms"]),
                "departments": to_string_list(record["departments"]),
                "checks": to_string_list(record["checks"]),
                "drugs": to_string_list(record["drugs"]),
                "treatments": to_string_list(record["treatments"]),
                "diet_recommendations": to_string_list(record["recommend_eat"]),
                "diet_avoid": to_string_list(record["not_eat"]),
            }
            for record in records
        ]

    def _find_related_disease_names(self, session, node_name: str, node_type: str) -> list[str]:
        if node_type == "disease":
            return [node_name]

        relation_map = {
            "symptom": "has_symptom",
            "department": "belongs_to",
            "check": "need_check",
            "drug": "use_drug",
        }
        label = TYPE_TO_LABEL.get(node_type)
        relation = relation_map.get(node_type)
        if not label or not relation:
            return []

        cypher = f"""
        MATCH (d:Disease)-[:{relation}]->(n:{label} {{name: $name}})
        WHERE coalesce(d.is_placeholder, false) = false
        RETURN collect(DISTINCT d.name) AS disease_names
        """
        record = session.run(cypher, name=node_name).single()
        return to_string_list(record["disease_names"] if record else [])

    def _build_candidate_graph_paths(
        self,
        disease_name: str,
        matched_symptoms: list[str],
        departments: list[str],
        checks: list[str],
        drugs: list[str],
        treatments: list[str],
    ) -> list[str]:
        paths = []
        for symptom_name in matched_symptoms[:2]:
            if departments:
                paths.append(f"Symptom({symptom_name}) -> Disease({disease_name}) -> Department({departments[0]})")
            else:
                paths.append(f"Symptom({symptom_name}) -> Disease({disease_name})")
        for check_name in checks[:1]:
            paths.append(f"Disease({disease_name}) -> Check({check_name})")
        for drug_name in drugs[:1]:
            paths.append(f"Disease({disease_name}) -> Drug({drug_name})")
        for treatment_name in treatments[:1]:
            paths.append(f"Disease({disease_name}) -> Treatment({treatment_name})")
        return dedupe(paths)

    def _fetch_node_detail(self, session, node: dict[str, Any]) -> dict[str, Any]:
        node_name = node["node_name"]
        node_type = node["node_type"]
        label = TYPE_TO_LABEL.get(node_type)
        if not label:
            return {
                **node,
                "exists_in_graph": False,
                "description": "",
                "related_diseases": [],
                "related_symptoms": [],
                "related_departments": [],
                "related_checks": [],
                "related_drugs": [],
                "related_treatments": [],
                "diet_recommendations": [],
                "diet_avoid": [],
                "graph_paths": [],
            }

        props_query = f"MATCH (n:{label} {{name: $name}}) RETURN properties(n) AS props LIMIT 1"
        props_record = session.run(props_query, name=node_name).single()
        props = dict(props_record["props"] or {}) if props_record else {}
        exists_in_graph = props_record is not None

        disease_names = self._find_related_disease_names(session, node_name, node_type)
        bundles = self._run_disease_bundle_query(session, disease_names[:8])
        related_diseases = [bundle["name"] for bundle in bundles]
        related_symptoms = dedupe([item for bundle in bundles for item in bundle["symptoms"]])[:12]
        related_departments = dedupe([item for bundle in bundles for item in bundle["departments"]])[:8]
        related_checks = dedupe([item for bundle in bundles for item in bundle["checks"]])[:8]
        related_drugs = dedupe([item for bundle in bundles for item in bundle["drugs"]])[:8]
        related_treatments = dedupe([item for bundle in bundles for item in bundle["treatments"]])[:6]
        diet_recommendations = dedupe([item for bundle in bundles for item in bundle["diet_recommendations"]])[:8]
        diet_avoid = dedupe([item for bundle in bundles for item in bundle["diet_avoid"]])[:8]

        graph_paths = []
        if node_type == "disease":
            if related_departments:
                graph_paths.append(f"Disease({node_name}) -> Department({related_departments[0]})")
            if related_checks:
                graph_paths.append(f"Disease({node_name}) -> Check({related_checks[0]})")
            if related_treatments:
                graph_paths.append(f"Disease({node_name}) -> Treatment({related_treatments[0]})")
        else:
            for disease_name in related_diseases[:3]:
                graph_paths.append(f"{label}({node_name}) -> Disease({disease_name})")

        return {
            "mention": node["mention"],
            "node_name": node_name,
            "node_type": node_type,
            "polarity": node["polarity"],
            "exists_in_graph": exists_in_graph,
            "description": normalize_text(props.get("desc")),
            "related_diseases": related_diseases[:8],
            "related_symptoms": related_symptoms,
            "related_departments": related_departments,
            "related_checks": related_checks,
            "related_drugs": related_drugs,
            "related_treatments": related_treatments,
            "diet_recommendations": diet_recommendations,
            "diet_avoid": diet_avoid,
            "graph_paths": dedupe(graph_paths),
        }

    def _aggregate_ranked_values(self, candidates: list[dict[str, Any]], key: str, limit: int) -> list[str]:
        counter = Counter()
        for rank, candidate in enumerate(candidates[:5]):
            weight = max(1, 6 - rank)
            for value in candidate.get(key, []):
                counter[value] += weight
        return [name for name, _ in counter.most_common(limit)]

    def _recommend_department(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        counter = Counter()
        for rank, candidate in enumerate(candidates[:5]):
            base = max(1, int(candidate.get("score", 0)) + 6 - rank)
            for index, department in enumerate(candidate.get("departments", [])[:3]):
                counter[department] += max(1, base - index)

        if not counter:
            return {
                "primary": "全科医学科",
                "alternatives": [],
                "reason": "当前图谱证据不足，先按通用首诊路径建议全科医学科评估。",
            }

        ordered = [name for name, _ in counter.most_common(3)]
        primary = ordered[0]
        alternatives = ordered[1:]
        reason_parts = []
        top_candidate = candidates[0] if candidates else None
        if top_candidate and top_candidate.get("matched_symptoms"):
            reason_parts.append(f"主要依据匹配症状：{'、'.join(top_candidate['matched_symptoms'][:3])}")
        if top_candidate and top_candidate.get("name"):
            reason_parts.append(f"优先考虑相关疾病：{top_candidate['name']}")
        if alternatives:
            reason_parts.append(f"备选科室：{'、'.join(alternatives)}")
        return {
            "primary": primary,
            "alternatives": alternatives,
            "reason": "；".join(reason_parts) or "基于候选疾病对应科室的聚合得分推荐。",
        }

    def query(self, nodes: list[dict[str, Any]], limit: int = 5) -> dict[str, Any]:
        positive_nodes = [node for node in nodes if node.get("polarity") != "negative"]
        negative_nodes = [node for node in nodes if node.get("polarity") == "negative"]
        positive_symptoms = dedupe([node["node_name"] for node in positive_nodes if node["node_type"] == "symptom"])
        negative_symptoms = dedupe([node["node_name"] for node in negative_nodes if node["node_type"] == "symptom"])
        positive_diseases = dedupe([node["node_name"] for node in positive_nodes if node["node_type"] == "disease"])
        negative_diseases = dedupe([node["node_name"] for node in negative_nodes if node["node_type"] == "disease"])
        positive_checks = dedupe([node["node_name"] for node in positive_nodes if node["node_type"] == "check"])
        positive_departments = dedupe([node["node_name"] for node in positive_nodes if node["node_type"] == "department"])
        positive_drugs = dedupe([node["node_name"] for node in positive_nodes if node["node_type"] == "drug"])

        empty_result = {
            "recognized_nodes": nodes,
            "matched_node_details": [],
            "candidate_diseases": [],
            "recognized_symptoms": [],
            "possible_additional_symptoms": [],
            "recommended_department": {
                "primary": "全科医学科",
                "alternatives": [],
                "reason": "当前没有足够的图谱节点可用于推荐。",
            },
            "recommended_tests": [],
            "treatment_plan": {
                "therapies": [],
                "medications": [],
                "diet_recommendations": [],
                "diet_avoid": [],
                "care_points": [],
            },
            "graph_paths": [],
            "retrieval_summary": "当前没有可用于图谱检索的有效节点。",
        }
        if not nodes:
            return empty_result

        with self.driver.session() as session:
            matched_node_details = [self._fetch_node_detail(session, node) for node in nodes]
            candidate_name_pool = []
            for node in positive_nodes:
                candidate_name_pool.extend(self._find_related_disease_names(session, node["node_name"], node["node_type"]))
            disease_bundles = self._run_disease_bundle_query(session, candidate_name_pool)

        if not disease_bundles:
            empty_result["matched_node_details"] = matched_node_details
            empty_result["recognized_symptoms"] = [
                detail["node_name"]
                for detail in matched_node_details
                if detail["node_type"] == "symptom" and detail["polarity"] == "positive" and detail["exists_in_graph"]
            ]
            empty_result["graph_paths"] = dedupe([path for detail in matched_node_details for path in detail["graph_paths"]])[:12]
            empty_result["retrieval_summary"] = (
                f"识别到 {len(nodes)} 个节点，其中 {len(positive_nodes)} 个正向节点、"
                f"{len(negative_nodes)} 个否定节点；图谱未召回候选疾病。"
            )
            return empty_result

        candidates = []
        for bundle in disease_bundles:
            matched_symptoms = [item for item in positive_symptoms if item in bundle["symptoms"]]
            negative_matched_symptoms = [item for item in negative_symptoms if item in bundle["symptoms"]]
            matched_checks = [item for item in positive_checks if item in bundle["checks"]]
            matched_departments = [item for item in positive_departments if item in bundle["departments"]]
            matched_drugs = [item for item in positive_drugs if item in bundle["drugs"]]
            direct_positive_hit = bundle["name"] in positive_diseases
            direct_negative_hit = bundle["name"] in negative_diseases

            score = 0
            score += 30 if direct_positive_hit else 0
            score += 12 * len(matched_symptoms)
            score += 4 * len(matched_checks)
            score += 4 * len(matched_departments)
            score += 3 * len(matched_drugs)
            score -= 8 * len(negative_matched_symptoms)
            score -= 15 if direct_negative_hit else 0
            if matched_symptoms and bundle["departments"]:
                score += 2
            if score <= 0:
                continue

            possible_additional_symptoms = [
                item
                for item in bundle["symptoms"]
                if item not in positive_symptoms and item not in negative_symptoms
            ][:6]
            graph_paths = self._build_candidate_graph_paths(
                bundle["name"],
                matched_symptoms,
                bundle["departments"],
                bundle["checks"],
                bundle["drugs"],
                bundle["treatments"],
            )
            candidates.append(
                {
                    "name": bundle["name"],
                    "desc": bundle["desc"],
                    "score": score,
                    "matched_symptoms": matched_symptoms,
                    "negative_symptoms": negative_matched_symptoms,
                    "possible_additional_symptoms": possible_additional_symptoms,
                    "departments": bundle["departments"],
                    "checks": bundle["checks"],
                    "drugs": bundle["drugs"],
                    "treatments": bundle["treatments"],
                    "diet_recommendations": bundle["diet_recommendations"],
                    "diet_avoid": bundle["diet_avoid"],
                    "graph_paths": graph_paths,
                }
            )

        candidates.sort(
            key=lambda item: (
                item["score"],
                len(item["matched_symptoms"]),
                len(item["departments"]),
                len(item["checks"]),
            ),
            reverse=True,
        )
        candidates = candidates[:limit]
        recognized_symptoms = [
            detail["node_name"]
            for detail in matched_node_details
            if detail["node_type"] == "symptom" and detail["polarity"] == "positive" and detail["exists_in_graph"]
        ]
        possible_additional_symptoms = self._aggregate_ranked_values(candidates, "possible_additional_symptoms", 8)
        recommended_tests = self._aggregate_ranked_values(candidates, "checks", 8)
        graph_paths = dedupe(
            [path for detail in matched_node_details for path in detail["graph_paths"]]
            + [path for candidate in candidates for path in candidate["graph_paths"]]
        )[:12]

        return {
            "recognized_nodes": nodes,
            "matched_node_details": matched_node_details,
            "candidate_diseases": candidates,
            "recognized_symptoms": recognized_symptoms,
            "possible_additional_symptoms": possible_additional_symptoms,
            "recommended_department": self._recommend_department(candidates),
            "recommended_tests": recommended_tests,
            "treatment_plan": {
                "therapies": self._aggregate_ranked_values(candidates, "treatments", 6),
                "medications": self._aggregate_ranked_values(candidates, "drugs", 8),
                "diet_recommendations": self._aggregate_ranked_values(candidates, "diet_recommendations", 8),
                "diet_avoid": self._aggregate_ranked_values(candidates, "diet_avoid", 8),
                "care_points": [],
            },
            "graph_paths": graph_paths,
            "retrieval_summary": (
                f"识别到 {len(nodes)} 个节点，其中 {len(positive_nodes)} 个正向节点、"
                f"{len(negative_nodes)} 个否定节点；图谱精确命中 "
                f"{sum(1 for item in matched_node_details if item['exists_in_graph'])} 个节点，"
                f"召回 {len(candidates)} 个候选疾病。"
            ),
        }
