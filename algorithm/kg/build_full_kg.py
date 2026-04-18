import argparse
import json
import os
from pathlib import Path

from neo4j import GraphDatabase


def clean_text(value):
    if value is None:
        return ""
    return str(value).strip()


def listify(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [clean_text(item) for item in value if clean_text(item)]

    text = clean_text(value)
    return [text] if text else []


def dedupe(values):
    result = []
    seen = set()

    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)

    return result


def normalize_category_path(values):
    return [value for value in dedupe(listify(values)) if value != "疾病百科"]


class MedicalGraphBuilder:
    def __init__(self, uri, user, password, data_path, graph_name):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.data_path = Path(data_path)
        self.graph_name = clean_text(graph_name) or "default"

    def close(self):
        self.driver.close()

    def test_connection(self):
        with self.driver.session() as session:
            result = session.run("RETURN 1 AS ok")
            return result.single()["ok"] == 1

    def clear_graph(self):
        with self.driver.session() as session:
            session.run(
                "MATCH (n {graph_name: $graph_name}) DETACH DELETE n",
                graph_name=self.graph_name,
            ).consume()

    def drop_legacy_name_constraints(self):
        target_labels = {
            "Disease",
            "Symptom",
            "Check",
            "Drug",
            "Food",
            "Department",
            "Category",
            "Treatment",
        }
        with self.driver.session() as session:
            result = session.run(
                "SHOW CONSTRAINTS YIELD name, labelsOrTypes, properties, type"
            )
            for record in result:
                labels = record.get("labelsOrTypes") or []
                properties = record.get("properties") or []
                constraint_type = str(record.get("type") or "").upper()
                if len(labels) != 1 or labels[0] not in target_labels:
                    continue
                if properties != ["name"]:
                    continue
                if "UNIQUE" not in constraint_type:
                    continue
                constraint_name = str(record["name"]).replace("`", "``")
                session.run(f"DROP CONSTRAINT `{constraint_name}` IF EXISTS").consume()

    def create_constraints(self):
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease) REQUIRE (d.graph_name, d.name) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom) REQUIRE (s.graph_name, s.name) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Check) REQUIRE (c.graph_name, c.name) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Drug) REQUIRE (d.graph_name, d.name) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Food) REQUIRE (f.graph_name, f.name) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Department) REQUIRE (d.graph_name, d.name) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Category) REQUIRE (c.graph_name, c.name) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Treatment) REQUIRE (t.graph_name, t.name) IS UNIQUE",
        ]
        with self.driver.session() as session:
            for query in queries:
                session.run(query).consume()

    def load_data(self):
        with self.data_path.open("r", encoding="utf-8") as handle:
            first_char = handle.read(1)
            handle.seek(0)

            if first_char == "[":
                loaded = json.load(handle)
                return loaded if isinstance(loaded, list) else [loaded]

            return [json.loads(line) for line in handle if line.strip()]

    def build_graph(self):
        records = self.load_data()
        if not records:
            return {}

        diseases = []
        symptoms = set()
        drugs = set()
        checks = set()
        departments = set()
        categories = set()
        treatments = set()
        foods = set()
        accompany_targets = set()

        rels_symptom = []
        rels_department = []
        rels_category = []
        rels_category_hierarchy = set()
        rels_check = []
        rels_treatment = []
        rels_drug = []
        rels_do_eat = []
        rels_not_eat = []
        rels_recommend_eat = []
        rels_accompany = []

        for item in records:
            disease_name = clean_text(item.get("name"))
            if not disease_name:
                continue

            category_path = normalize_category_path(item.get("category"))
            department_path = dedupe(listify(item.get("cure_department")))

            diseases.append(
                {
                    "graph_name": self.graph_name,
                    "name": disease_name,
                    "desc": clean_text(item.get("desc")),
                    "prevent": clean_text(item.get("prevent")),
                    "cause": clean_text(item.get("cause")),
                    "easy_get": clean_text(item.get("easy_get")),
                    "get_prob": clean_text(item.get("get_prob")),
                    "get_way": clean_text(item.get("get_way")),
                    "yibao_status": clean_text(item.get("yibao_status")),
                    "cure_lasttime": clean_text(item.get("cure_lasttime")),
                    "cured_prob": clean_text(item.get("cured_prob")),
                    "cost_money": clean_text(item.get("cost_money")),
                    "category_path": " > ".join(category_path),
                    "department_path": " > ".join(department_path),
                }
            )

            for symptom in dedupe(listify(item.get("symptom"))):
                symptoms.add(symptom)
                rels_symptom.append((disease_name, symptom))

            for department in department_path:
                departments.add(department)
                rels_department.append((disease_name, department))

            for category in category_path:
                categories.add(category)
                rels_category.append((disease_name, category))

            for parent, child in zip(category_path, category_path[1:]):
                rels_category_hierarchy.add((child, parent))

            for check in dedupe(listify(item.get("check"))):
                checks.add(check)
                rels_check.append((disease_name, check))

            for treatment in dedupe(listify(item.get("cure_way"))):
                treatments.add(treatment)
                rels_treatment.append((disease_name, treatment))

            for drug in dedupe(listify(item.get("common_drug"))):
                drugs.add(drug)
                rels_drug.append((disease_name, drug, "common"))

            for drug in dedupe(listify(item.get("recommand_drug"))):
                drugs.add(drug)
                rels_drug.append((disease_name, drug, "recommended"))

            for food in dedupe(listify(item.get("do_eat"))):
                foods.add(food)
                rels_do_eat.append((disease_name, food))

            for food in dedupe(listify(item.get("not_eat"))):
                foods.add(food)
                rels_not_eat.append((disease_name, food))

            for food in dedupe(listify(item.get("recommand_eat"))):
                foods.add(food)
                rels_recommend_eat.append((disease_name, food))

            for accompany_name in dedupe(listify(item.get("acompany"))):
                if accompany_name == disease_name:
                    continue
                accompany_targets.add(accompany_name)
                rels_accompany.append((disease_name, accompany_name))

        placeholder_diseases = [
            {"graph_name": self.graph_name, "name": name, "is_placeholder": True}
            for name in sorted(accompany_targets - {item["name"] for item in diseases})
        ]

        with self.driver.session() as session:
            self.batch_create_nodes(session, "Disease", diseases)
            self.batch_create_nodes(session, "Disease", placeholder_diseases)
            self.batch_create_nodes(session, "Symptom", [{"graph_name": self.graph_name, "name": name} for name in sorted(symptoms)])
            self.batch_create_nodes(session, "Drug", [{"graph_name": self.graph_name, "name": name} for name in sorted(drugs)])
            self.batch_create_nodes(session, "Check", [{"graph_name": self.graph_name, "name": name} for name in sorted(checks)])
            self.batch_create_nodes(session, "Department", [{"graph_name": self.graph_name, "name": name} for name in sorted(departments)])
            self.batch_create_nodes(session, "Category", [{"graph_name": self.graph_name, "name": name} for name in sorted(categories)])
            self.batch_create_nodes(session, "Treatment", [{"graph_name": self.graph_name, "name": name} for name in sorted(treatments)])
            self.batch_create_nodes(session, "Food", [{"graph_name": self.graph_name, "name": name} for name in sorted(foods)])

            self.batch_create_rels(session, "Disease", "Symptom", "has_symptom", rels_symptom)
            self.batch_create_rels(session, "Disease", "Department", "belongs_to", rels_department)
            self.batch_create_rels(session, "Disease", "Category", "in_category", rels_category)
            self.batch_create_rels(session, "Disease", "Check", "need_check", rels_check)
            self.batch_create_rels(session, "Disease", "Treatment", "treated_by", rels_treatment)
            self.batch_create_rels_with_prop(session, "Disease", "Drug", "use_drug", rels_drug, "role")
            self.batch_create_rels(session, "Disease", "Food", "do_eat", rels_do_eat)
            self.batch_create_rels(session, "Disease", "Food", "not_eat", rels_not_eat)
            self.batch_create_rels(session, "Disease", "Food", "recommend_eat", rels_recommend_eat)
            self.batch_create_rels(session, "Category", "Category", "sub_category_of", sorted(rels_category_hierarchy))
            self.batch_create_disease_rels(session, "accompany_with", rels_accompany)

        return {
            "graph_name": self.graph_name,
            "diseases": len(diseases),
            "placeholder_diseases": len(placeholder_diseases),
            "symptoms": len(symptoms),
            "checks": len(checks),
            "drugs": len(drugs),
            "foods": len(foods),
            "departments": len(departments),
            "categories": len(categories),
            "treatments": len(treatments),
            "has_symptom_rels": len(rels_symptom),
            "belongs_to_rels": len(rels_department),
            "in_category_rels": len(rels_category),
            "need_check_rels": len(rels_check),
            "treated_by_rels": len(rels_treatment),
            "use_drug_rels": len(rels_drug),
            "diet_rels": len(rels_do_eat) + len(rels_not_eat) + len(rels_recommend_eat),
            "accompany_with_rels": len(rels_accompany),
        }

    def batch_create_nodes(self, session, label, data_list, batch_size=1000):
        if not data_list:
            return

        query = f"UNWIND $batch AS row MERGE (n:{label} {{graph_name: row.graph_name, name: row.name}}) SET n += row"
        for index in range(0, len(data_list), batch_size):
            session.run(query, batch=data_list[index:index + batch_size]).consume()

    def batch_create_rels(self, session, start_label, end_label, rel_type, rel_list, batch_size=1000):
        if not rel_list:
            return

        query = f"""
        UNWIND $batch AS row
        MATCH (source:{start_label} {{graph_name: $graph_name, name: row[0]}})
        MATCH (target:{end_label} {{graph_name: $graph_name, name: row[1]}})
        MERGE (source)-[:{rel_type} {{graph_name: $graph_name}}]->(target)
        """
        for index in range(0, len(rel_list), batch_size):
            session.run(query, batch=rel_list[index:index + batch_size], graph_name=self.graph_name).consume()

    def batch_create_rels_with_prop(self, session, start_label, end_label, rel_type, rel_list, prop_name, batch_size=1000):
        if not rel_list:
            return

        query = f"""
        UNWIND $batch AS row
        MATCH (source:{start_label} {{graph_name: $graph_name, name: row[0]}})
        MATCH (target:{end_label} {{graph_name: $graph_name, name: row[1]}})
        MERGE (source)-[r:{rel_type} {{graph_name: $graph_name}}]->(target)
        SET r.{prop_name} = row[2]
        """
        for index in range(0, len(rel_list), batch_size):
            session.run(query, batch=rel_list[index:index + batch_size], graph_name=self.graph_name).consume()

    def batch_create_disease_rels(self, session, rel_type, rel_list, batch_size=1000):
        if not rel_list:
            return

        query = f"""
        UNWIND $batch AS row
        MATCH (source:Disease {{graph_name: $graph_name, name: row[0]}})
        MERGE (target:Disease {{graph_name: $graph_name, name: row[1]}})
        ON CREATE SET target.is_placeholder = true
        MERGE (source)-[:{rel_type} {{graph_name: $graph_name}}]->(target)
        """
        for index in range(0, len(rel_list), batch_size):
            session.run(query, batch=rel_list[index:index + batch_size], graph_name=self.graph_name).consume()


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build the medical knowledge graph in Neo4j.")
    parser.add_argument(
        "--data-path",
        default=str(repo_root / "algorithm" / "data" / "raw" / "medical.json"),
        help="Path to medical.json or line-delimited JSON.",
    )
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "ZYDzyd917917"))
    parser.add_argument(
        "--graph-name",
        default=os.getenv("MEDICAL_GRAPH_NAME", "full"),
        help="Graph namespace stored inside the same Neo4j database, e.g. full or common.",
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Delete only the current graph_name subgraph before importing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    builder = MedicalGraphBuilder(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
        data_path=args.data_path,
        graph_name=args.graph_name,
    )
    try:
        if not builder.test_connection():
            raise RuntimeError("Neo4j connection test failed.")

        builder.drop_legacy_name_constraints()
        if args.clear_existing:
            builder.clear_graph()

        builder.create_constraints()
        stats = builder.build_graph()
        print("Knowledge graph build completed.")
        for key, value in stats.items():
            print(f"{key}: {value}")
    finally:
        builder.close()


if __name__ == "__main__":
    main()
