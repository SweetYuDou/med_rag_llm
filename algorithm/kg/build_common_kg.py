import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def clean_text(value):
    if value is None:
        return ""
    return str(value).strip()


def listify(value):
    if value is None:
        return []
    if isinstance(value, list):
        values = value
    else:
        values = [value]
    return [clean_text(item) for item in values if clean_text(item)]


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


def load_line_or_array_json(path):
    with path.open("r", encoding="utf-8") as handle:
        first_char = handle.read(1)
        handle.seek(0)
        if first_char == "[":
            loaded = json.load(handle)
            return loaded if isinstance(loaded, list) else [loaded]
        return [json.loads(line) for line in handle if line.strip()]


def add_node(store, node_type, name, graph_name, **properties):
    node_name = clean_text(name)
    if not node_name:
        return
    key = (graph_name, node_type, node_name)
    payload = {
        "graph_name": graph_name,
        "node_type": node_type,
        "name": node_name,
    }
    for prop_key, prop_value in properties.items():
        text = clean_text(prop_value)
        if text:
            payload[prop_key] = text
    if key in store:
        store[key].update({k: v for k, v in payload.items() if v})
    else:
        store[key] = payload


def add_edge(store, source_type, source_name, relation, target_type, target_name, graph_name, **properties):
    source_name = clean_text(source_name)
    target_name = clean_text(target_name)
    if not source_name or not target_name:
        return
    prop_items = tuple(sorted((key, clean_text(value)) for key, value in properties.items() if clean_text(value)))
    key = (graph_name, source_type, source_name, relation, target_type, target_name, prop_items)
    if key in store:
        return
    payload = {
        "graph_name": graph_name,
        "source_type": source_type,
        "source_name": source_name,
        "relation": relation,
        "target_type": target_type,
        "target_name": target_name,
    }
    for prop_key, prop_value in prop_items:
        payload[prop_key] = prop_value
    store[key] = payload


def build_graph(records, graph_name):
    nodes = {}
    edges = {}

    for item in records:
        disease_name = clean_text(item.get("name"))
        if not disease_name:
            continue

        category_path = normalize_category_path(item.get("category"))
        department_path = dedupe(listify(item.get("cure_department")))

        add_node(
            nodes,
            "Disease",
            disease_name,
            graph_name=graph_name,
            desc=item.get("desc"),
            prevent=item.get("prevent"),
            cause=item.get("cause"),
            easy_get=item.get("easy_get"),
            get_prob=item.get("get_prob"),
            get_way=item.get("get_way"),
            yibao_status=item.get("yibao_status"),
            cure_lasttime=item.get("cure_lasttime"),
            cured_prob=item.get("cured_prob"),
            cost_money=item.get("cost_money"),
            category_path=" > ".join(category_path),
            department_path=" > ".join(department_path),
        )

        for symptom in dedupe(listify(item.get("symptom"))):
            add_node(nodes, "Symptom", symptom, graph_name=graph_name)
            add_edge(edges, "Disease", disease_name, "has_symptom", "Symptom", symptom, graph_name=graph_name)

        for department in department_path:
            add_node(nodes, "Department", department, graph_name=graph_name)
            add_edge(edges, "Disease", disease_name, "belongs_to", "Department", department, graph_name=graph_name)

        for category in category_path:
            add_node(nodes, "Category", category, graph_name=graph_name)
            add_edge(edges, "Disease", disease_name, "in_category", "Category", category, graph_name=graph_name)

        for parent, child in zip(category_path, category_path[1:]):
            add_edge(edges, "Category", child, "sub_category_of", "Category", parent, graph_name=graph_name)

        for check in dedupe(listify(item.get("check"))):
            add_node(nodes, "Check", check, graph_name=graph_name)
            add_edge(edges, "Disease", disease_name, "need_check", "Check", check, graph_name=graph_name)

        for treatment in dedupe(listify(item.get("cure_way"))):
            add_node(nodes, "Treatment", treatment, graph_name=graph_name)
            add_edge(edges, "Disease", disease_name, "treated_by", "Treatment", treatment, graph_name=graph_name)

        for drug in dedupe(listify(item.get("common_drug"))):
            add_node(nodes, "Drug", drug, graph_name=graph_name)
            add_edge(edges, "Disease", disease_name, "use_drug", "Drug", drug, graph_name=graph_name, role="common")

        for drug in dedupe(listify(item.get("recommand_drug"))):
            add_node(nodes, "Drug", drug, graph_name=graph_name)
            add_edge(edges, "Disease", disease_name, "use_drug", "Drug", drug, graph_name=graph_name, role="recommended")

        for food in dedupe(listify(item.get("do_eat"))):
            add_node(nodes, "Food", food, graph_name=graph_name)
            add_edge(edges, "Disease", disease_name, "do_eat", "Food", food, graph_name=graph_name)

        for food in dedupe(listify(item.get("not_eat"))):
            add_node(nodes, "Food", food, graph_name=graph_name)
            add_edge(edges, "Disease", disease_name, "not_eat", "Food", food, graph_name=graph_name)

        for food in dedupe(listify(item.get("recommand_eat"))):
            add_node(nodes, "Food", food, graph_name=graph_name)
            add_edge(edges, "Disease", disease_name, "recommend_eat", "Food", food, graph_name=graph_name)

        for accompany_name in dedupe(listify(item.get("acompany"))):
            if accompany_name == disease_name:
                continue
            add_node(nodes, "Disease", accompany_name, graph_name=graph_name, is_placeholder="true")
            add_edge(edges, "Disease", disease_name, "accompany_with", "Disease", accompany_name, graph_name=graph_name)

    return list(nodes.values()), list(edges.values())


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_summary(nodes, edges, graph_name):
    node_counter = Counter(node["node_type"] for node in nodes)
    edge_counter = Counter(edge["relation"] for edge in edges)
    return {
        "graph_name": graph_name,
        "node_total": len(nodes),
        "edge_total": len(edges),
        "node_types": dict(node_counter),
        "edge_types": dict(edge_counter),
    }


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Export offline graph artifacts for a named graph namespace.")
    parser.add_argument(
        "--data-path",
        default=str(repo_root / "algorithm" / "data" / "processed" / "common_disease" / "medical_common_subset.json"),
        help="Input JSON or JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to algorithm/data/processed/<graph_name>_medical_kg.",
    )
    parser.add_argument(
        "--graph-name",
        default="common",
        help="Graph namespace, e.g. common or full.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data_path)
    graph_name = clean_text(args.graph_name) or "common"

    records = load_line_or_array_json(data_path)
    nodes, edges = build_graph(records, graph_name=graph_name)
    summary = build_summary(nodes, edges, graph_name)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
