import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

from tqdm import tqdm


SYSTEM_PROMPT = (
    "你是中文医疗问诊 query 解析助手。你的任务不是传统 BIO NER，而是把用户口语化、模糊化、"
    "不完整的问诊表达映射到医疗知识图谱中的标准节点。优先识别 symptom 和 disease，也可以识别 "
    "department、check、drug。对于否定表达，例如“没有发烧，就是有点咳嗽”，不要输出被否定的实体。"
    "如果用户只是在问挂号、空腹、预约、流程、报销，没有明确医疗实体，则返回空列表。"
    "只返回严格 JSON，不要输出解释。格式为："
    "{\"nodes\": [{\"mention\": str, \"node_name\": str, \"node_type\": str}]}"
)

TYPE_PRIORITY = {
    "symptom": 5,
    "disease": 4,
    "check": 3,
    "department": 2,
    "drug": 1,
}

BANNED_GENERIC_TERMS = {
    "医生",
    "大夫",
    "医师",
    "患者",
    "病人",
    "家长",
    "宝宝",
    "宝妈",
    "儿童",
    "小儿",
    "本人",
    "自己",
    "情况",
    "问题",
    "症状",
    "疾病",
    "治疗",
    "检查",
    "检验",
    "药物",
    "手术",
    "住院",
    "门诊",
    "挂号",
    "复查",
    "化验",
    "报告",
    "结果",
}

NOISY_SYMPTOM_KEYWORDS = {
    "阳性",
    "阴性",
    "异常",
    "病变",
    "结节",
    "团块",
    "占位",
    "综合征",
    "检查",
    "检验",
    "试验",
    "测定",
    "活检",
    "镜检",
    "治疗",
    "护理",
    "化疗",
    "放疗",
    "术后",
    "住院",
    "门诊",
    "医生",
    "大夫",
}

BAD_SYMPTOM_SUFFIXES = (
    "医生",
    "大夫",
    "患者",
    "病人",
    "检查",
    "检验",
    "试验",
    "测定",
    "治疗",
    "手术",
    "住院",
    "门诊",
    "挂号",
    "病",
    "炎",
    "癌",
    "瘤",
    "综合征",
    "障碍",
    "中毒",
    "感染",
    "损伤",
)

MEDICATION_SUFFIXES = (
    "片",
    "胶囊",
    "颗粒",
    "注射液",
    "口服液",
    "滴丸",
    "胶丸",
    "药",
    "冲剂",
)

ALLOWED_SYMPTOM_DISEASE_OVERLAP = {
    "发热",
    "发烧",
    "高热",
    "低热",
    "咳嗽",
    "咳痰",
    "腹泻",
    "便秘",
    "呕吐",
    "恶心",
    "头痛",
    "头晕",
    "眩晕",
    "胸闷",
    "胸痛",
    "腹痛",
    "胃痛",
    "鼻塞",
    "流涕",
    "乏力",
    "无力",
    "瘙痒",
    "皮疹",
    "失眠",
    "气短",
    "喘息",
    "呼吸困难",
    "畏寒",
    "怕冷",
    "尿频",
    "尿急",
    "尿痛",
    "咽痛",
    "咽喉痛",
    "食欲不振",
    "食欲减退",
    "肌肉酸痛",
    "全身疼痛",
    "身痛",
}

NEGATION_PATTERNS = (
    "没有",
    "没",
    "无",
    "不是",
    "并无",
    "并未",
    "未见",
    "否认",
    "从不",
)

NEGATIVE_HINTS = (
    "挂号",
    "预约",
    "空腹",
    "报销",
    "住院",
    "复查",
    "体检",
    "流程",
    "能不能喝水",
    "可以吃饭吗",
    "什么时候做",
)

MANUAL_ALIAS_GROUPS = {
    "symptom": [
        (["发高烧", "发烧", "高热", "低热", "有点烧"], ["发热", "发烧"]),
        (["咳个不停", "一直咳", "老咳", "咳嗽"], ["咳嗽"]),
        (["有痰", "痰多", "咳痰"], ["咳痰"]),
        (["嗓子疼", "喉咙疼", "咽痛", "咽喉痛"], ["咽痛", "咽喉痛"]),
        (["喘", "喘不上气", "气不够用", "呼吸费劲", "呼吸困难"], ["呼吸困难", "喘息", "气短"]),
        (["胸口闷", "胸闷", "胸口发闷"], ["胸闷"]),
        (["胸口疼", "胸痛"], ["胸痛"]),
        (["头疼", "头痛"], ["头痛"]),
        (["头晕", "发晕"], ["头晕", "眩晕"]),
        (["肚子疼", "肚子痛"], ["腹痛"]),
        (["胃疼", "胃不舒服", "胃痛"], ["胃痛"]),
        (["反胃", "恶心"], ["恶心"]),
        (["想吐", "吐了", "老吐"], ["呕吐"]),
        (["拉肚子", "拉稀", "一直跑厕所"], ["腹泻"]),
        (["拉不出来", "便秘"], ["便秘"]),
        (["没劲", "没力气", "浑身没劲", "没精神"], ["乏力", "无力"]),
        (["浑身疼", "全身疼", "浑身酸疼", "身上疼"], ["全身疼痛", "肌肉酸痛", "身痛"]),
        (["关节疼", "关节痛"], ["关节痛"]),
        (["流鼻涕"], ["流涕"]),
        (["鼻子堵", "鼻塞", "鼻子不通气"], ["鼻塞"]),
        (["打喷嚏"], ["喷嚏"]),
        (["心慌", "心跳快"], ["心悸"]),
        (["没胃口", "不想吃饭"], ["食欲不振", "食欲减退"]),
        (["睡不着", "失眠"], ["失眠"]),
        (["发冷", "怕冷", "打寒战"], ["畏寒", "怕冷"]),
        (["发痒", "痒得厉害"], ["瘙痒"]),
        (["起疹子", "身上起疹子"], ["皮疹"]),
        (["尿频", "老想上厕所"], ["尿频"]),
        (["尿急"], ["尿急"]),
        (["小便疼", "尿痛"], ["尿痛"]),
    ],
    "disease": [
        (["感冒", "着凉", "上感"], ["感冒", "上呼吸道感染"]),
        (["肺炎"], ["肺炎"]),
        (["支气管炎"], ["支气管炎"]),
        (["鼻炎"], ["鼻炎"]),
        (["咽炎"], ["咽炎"]),
        (["胃炎"], ["胃炎"]),
        (["肠胃炎"], ["胃肠炎"]),
        (["尿感", "尿路感染"], ["尿路感染"]),
        (["中耳炎"], ["中耳炎"]),
    ],
    "check": [
        (["验血", "抽血"], ["血常规"]),
        (["尿检", "查尿"], ["尿常规"]),
        (["便检", "查大便"], ["大便常规"]),
        (["胸片", "拍片"], ["胸部X线检查", "X线检查"]),
        (["胸部ct", "ct"], ["胸部CT检查", "CT检查"]),
        (["彩超", "b超"], ["彩超", "B超"]),
        (["胃镜"], ["胃镜"]),
    ],
    "department": [
        (["儿科", "小儿科"], ["儿科", "小儿内科"]),
        (["呼吸科"], ["呼吸内科"]),
        (["消化科"], ["消化内科"]),
        (["耳鼻喉科", "耳鼻喉"], ["耳鼻喉科"]),
        (["皮肤科"], ["皮肤科"]),
        (["妇科"], ["妇科"]),
        (["口腔科"], ["口腔科"]),
        (["神经内科"], ["神经内科"]),
    ],
    "drug": [
        (["退烧药"], ["布洛芬", "对乙酰氨基酚"]),
        (["布洛芬"], ["布洛芬"]),
        (["阿莫西林"], ["阿莫西林"]),
        (["蒙脱石散"], ["蒙脱石散"]),
        (["开塞露"], ["开塞露"]),
    ],
}

MANUAL_HARD_CASES = [
    {
        "query": "没有发烧，就是有点咳嗽",
        "nodes": [{"mention": "咳嗽", "node_name": "咳嗽", "node_type": "symptom"}],
    },
    {
        "query": "不头晕，但胸口闷",
        "nodes": [{"mention": "胸口闷", "node_name": "胸闷", "node_type": "symptom"}],
    },
    {
        "query": "发烧、咳嗽、喉咙痛，还有点喘",
        "nodes": [
            {"mention": "发烧", "node_name": "发热", "node_type": "symptom"},
            {"mention": "咳嗽", "node_name": "咳嗽", "node_type": "symptom"},
            {"mention": "喉咙痛", "node_name": "咽痛", "node_type": "symptom"},
            {"mention": "喘", "node_name": "呼吸困难", "node_type": "symptom"},
        ],
    },
    {
        "query": "我前天淋了雨，昨天开始发烧，今天还头晕",
        "nodes": [
            {"mention": "发烧", "node_name": "发热", "node_type": "symptom"},
            {"mention": "头晕", "node_name": "头晕", "node_type": "symptom"},
        ],
    },
    {
        "query": "浑身疼，跟散架了一样",
        "nodes": [
            {"mention": "浑身疼", "node_name": "全身疼痛", "node_type": "symptom"},
            {"mention": "浑身疼", "node_name": "肌肉酸痛", "node_type": "symptom"},
        ],
    },
    {
        "query": "昨晚开始胸口闷，爬楼还喘",
        "nodes": [
            {"mention": "胸口闷", "node_name": "胸闷", "node_type": "symptom"},
            {"mention": "喘", "node_name": "呼吸困难", "node_type": "symptom"},
        ],
    },
    {
        "query": "孩子发烧，还咳得厉害，要不要先查血常规",
        "nodes": [
            {"mention": "发烧", "node_name": "发热", "node_type": "symptom"},
            {"mention": "咳得厉害", "node_name": "咳嗽", "node_type": "symptom"},
            {"mention": "血常规", "node_name": "血常规", "node_type": "check"},
        ],
    },
]

MANUAL_NEGATIVES = [
    "要不要空腹检查",
    "最近睡得不好",
    "这个检查需要预约吗",
    "今天就是想问问挂号流程",
    "抽血前能不能喝水",
    "化验单还没出来要不要先复诊",
    "医保报销怎么走",
    "住院押金一般交多少",
]


def clean_text(value):
    if value is None:
        return ""
    return str(value).strip()


def read_text_auto(path):
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def load_json_auto(path):
    return json.loads(read_text_auto(path))


def load_medical_records(path):
    text = read_text_auto(path).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            if all(isinstance(v, dict) for v in parsed.values()):
                return list(parsed.values())
            return [parsed]
    except json.JSONDecodeError:
        pass

    records = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        records.append(json.loads(line))
    return records


def listify(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [clean_text(item) for item in value if clean_text(item)]
    text = clean_text(value)
    return [text] if text else []


def dedupe_preserve(items):
    result = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def normalize_text(text):
    text = clean_text(text).lower()
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text)


def valid_node(text, max_len=48):
    text = clean_text(text)
    if not text or len(text) > max_len:
        return False
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z0-9]", text))


def choose_existing(index_set, candidates):
    for candidate in candidates:
        if candidate in index_set:
            return candidate
    return None


def sort_nodes(nodes):
    return sorted(
        nodes,
        key=lambda item: (
            -TYPE_PRIORITY.get(item["node_type"], 0),
            item["node_name"],
            item["mention"],
        ),
    )


def build_base_sets(records):
    disease_set = set()
    department_set = set()
    check_set = set()
    drug_set = set()

    for record in records:
        disease = clean_text(record.get("name"))
        if valid_node(disease):
            disease_set.add(disease)

        for department in listify(record.get("cure_department")):
            if valid_node(department):
                department_set.add(department)

        for check in listify(record.get("check")):
            if valid_node(check):
                check_set.add(check)

        for drug in listify(record.get("common_drug")) + listify(record.get("recommand_drug")):
            if valid_node(drug):
                drug_set.add(drug)

        for accompany in listify(record.get("acompany")) + listify(record.get("accompany")):
            if valid_node(accompany):
                disease_set.add(accompany)

    return {
        "disease": disease_set,
        "department": department_set,
        "check": check_set,
        "drug": drug_set,
    }


def is_noisy_symptom(text, other_sets):
    text = clean_text(text)
    if not valid_node(text):
        return True
    if text in BANNED_GENERIC_TERMS:
        return True
    if any(keyword in text for keyword in NOISY_SYMPTOM_KEYWORDS):
        return True
    if any(text.endswith(suffix) for suffix in BAD_SYMPTOM_SUFFIXES):
        return True
    if any(text.endswith(suffix) for suffix in MEDICATION_SUFFIXES):
        return True
    if text in other_sets and text not in ALLOWED_SYMPTOM_DISEASE_OVERLAP:
        return True
    if re.fullmatch(r"[A-Za-z0-9]+", text):
        return True
    return False


def sanitize_list(values, max_len=48):
    return [item for item in dedupe_preserve(values) if valid_node(item, max_len=max_len)]


def build_clean_medical_records(records):
    base_sets = build_base_sets(records)
    other_sets = base_sets["disease"] | base_sets["department"] | base_sets["check"] | base_sets["drug"]
    clean_records = []
    dropped_symptoms = Counter()

    for record in tqdm(records, desc="清洗 medical.json", unit="record"):
        disease = clean_text(record.get("name"))
        if not valid_node(disease):
            continue

        symptoms = []
        for symptom in listify(record.get("symptom")):
            if is_noisy_symptom(symptom, other_sets):
                dropped_symptoms[symptom] += 1
                continue
            symptoms.append(symptom)

        clean_record = {
            "name": disease,
            "desc": clean_text(record.get("desc")),
            "prevent": clean_text(record.get("prevent")),
            "cause": clean_text(record.get("cause")),
            "easy_get": clean_text(record.get("easy_get")),
            "get_prob": clean_text(record.get("get_prob")),
            "get_way": clean_text(record.get("get_way")),
            "yibao_status": clean_text(record.get("yibao_status")),
            "symptom": sanitize_list(symptoms),
            "check": sanitize_list(listify(record.get("check"))),
            "cure_department": sanitize_list(listify(record.get("cure_department"))),
            "common_drug": sanitize_list(listify(record.get("common_drug"))),
            "recommand_drug": sanitize_list(listify(record.get("recommand_drug"))),
            "cure_way": sanitize_list(listify(record.get("cure_way"))),
            "category": sanitize_list(listify(record.get("category"))),
            "acompany": sanitize_list(listify(record.get("acompany")) + listify(record.get("accompany"))),
            "do_eat": sanitize_list(listify(record.get("do_eat"))),
            "not_eat": sanitize_list(listify(record.get("not_eat"))),
            "recommand_eat": sanitize_list(listify(record.get("recommand_eat"))),
        }
        clean_records.append(clean_record)

    node_index = {
        "disease": set(),
        "symptom": set(),
        "department": set(),
        "check": set(),
        "drug": set(),
    }
    for record in clean_records:
        node_index["disease"].add(record["name"])
        node_index["disease"].update(record["acompany"])
        node_index["symptom"].update(record["symptom"])
        node_index["department"].update(record["cure_department"])
        node_index["check"].update(record["check"])
        node_index["drug"].update(record["common_drug"])
        node_index["drug"].update(record["recommand_drug"])

    return clean_records, node_index, dropped_symptoms


def build_name_maps(node_index):
    maps = {}
    for node_type, names in node_index.items():
        maps[node_type] = {normalize_text(name): name for name in names}
    return maps


def maybe_add_alias(alias_rows, alias, node_name, node_type, node_index):
    alias = clean_text(alias)
    node_name = clean_text(node_name)
    if not alias or not node_name:
        return
    if node_name not in node_index[node_type]:
        return
    if node_type == "symptom" and is_noisy_symptom(alias, set()):
        return
    alias_rows.append((alias, node_name, node_type))


def build_alias_entries(node_index):
    alias_rows = []

    for node_type, names in node_index.items():
        for name in names:
            maybe_add_alias(alias_rows, name, name, node_type, node_index)
            if node_type == "disease":
                for prefix in ("急性", "慢性", "小儿", "儿童", "新生儿", "原发性", "继发性"):
                    if name.startswith(prefix) and len(name) - len(prefix) >= 2:
                        maybe_add_alias(alias_rows, name[len(prefix):], name, node_type, node_index)

    for node_type, groups in MANUAL_ALIAS_GROUPS.items():
        for aliases, candidates in groups:
            target = choose_existing(node_index[node_type], candidates)
            if not target:
                continue
            for alias in aliases:
                maybe_add_alias(alias_rows, alias, target, node_type, node_index)

    alias_rows = dedupe_preserve(alias_rows)
    alias_rows.sort(key=lambda item: (-len(item[0]), item[2], item[0], item[1]))
    return alias_rows


def is_negated(query, start_index):
    left = query[max(0, start_index - 8):start_index]
    return any(left.endswith(pattern) or pattern in left for pattern in NEGATION_PATTERNS)


def extract_nodes_from_query(query, alias_entries):
    query = clean_text(query)
    if not query:
        return []

    matches = []
    occupied = [False] * len(query)
    for alias, node_name, node_type in alias_entries:
        start = 0
        while True:
            index = query.find(alias, start)
            if index < 0:
                break
            end = index + len(alias)
            if any(occupied[index:end]):
                start = index + 1
                continue
            if is_negated(query, index):
                start = index + 1
                continue
            matches.append(
                {
                    "mention": query[index:end],
                    "node_name": node_name,
                    "node_type": node_type,
                    "_start": index,
                    "_end": end,
                }
            )
            for pos in range(index, end):
                occupied[pos] = True
            start = end

    return matches


def sanitize_nodes(nodes, query, node_index):
    result = []
    seen = set()
    for node in nodes:
        mention = clean_text(node.get("mention"))
        node_name = clean_text(node.get("node_name"))
        node_type = clean_text(node.get("node_type"))
        if node_type not in node_index:
            continue
        if node_name not in node_index[node_type]:
            continue
        if mention and mention not in query:
            continue
        if node_type == "symptom" and is_noisy_symptom(node_name, set()):
            continue
        key = (mention, node_name, node_type)
        if key in seen:
            continue
        seen.add(key)
        result.append({"mention": mention, "node_name": node_name, "node_type": node_type})
    return sort_nodes(result)


def looks_like_negative_query(query):
    if any(hint in query for hint in NEGATIVE_HINTS):
        return True
    return len(query) <= 14 and not re.search(r"[\u4e00-\u9fff].*[\u4e00-\u9fff]", query) is None and "?" not in query and "？" not in query


def make_annotation(query, nodes, source, node_index):
    query = clean_text(query)
    if not query:
        return None
    nodes = sanitize_nodes(nodes, query, node_index)
    if not nodes and not looks_like_negative_query(query):
        return None
    return {"query": query, "nodes": nodes, "source": source}


def load_symptom_norms(path):
    if not path.exists():
        return set()
    content = read_text_auto(path).splitlines()
    reader = csv.DictReader(content)
    norms = set()
    for row in reader:
        norm = clean_text(row.get("norm"))
        if norm:
            norms.add(norm)
    return norms


def build_annotations_from_imcs(paths, node_index, name_maps):
    annotations = []
    symptom_map = name_maps["symptom"]
    disease_map = name_maps["disease"]

    for path in paths:
        if not path.exists():
            continue
        dataset = load_json_auto(path)
        items = dataset.values() if isinstance(dataset, dict) else dataset
        for item in tqdm(list(items), desc=f"解析 {path.name}", unit="sample"):
            query = clean_text(item.get("self_report") or item.get("report"))
            if not query:
                continue

            nodes = []
            explicit_info = item.get("explicit_info") or {}
            symptom_candidates = listify(explicit_info.get("Symptom"))
            for turn in item.get("dialogue") or []:
                symptom_candidates.extend(listify(turn.get("symptom_norm")))

            for symptom in dedupe_preserve(symptom_candidates):
                canonical = symptom_map.get(normalize_text(symptom))
                if canonical:
                    nodes.append({"mention": symptom, "node_name": canonical, "node_type": "symptom"})

            diagnosis = clean_text(item.get("diagnosis"))
            disease = disease_map.get(normalize_text(diagnosis))
            if disease and disease in query:
                nodes.append({"mention": diagnosis, "node_name": disease, "node_type": "disease"})

            annotation = make_annotation(query, nodes, f"imcs:{path.stem}", node_index)
            if annotation:
                annotations.append(annotation)

    return annotations


def build_annotations_from_dialogue(path, alias_entries, node_index, max_samples=None):
    if not path.exists():
        return []
    dataset = load_json_auto(path)
    annotations = []
    count = 0
    for item in tqdm(dataset, desc=f"解析 {path.name}", unit="sample"):
        query = clean_text(item.get("input") or item.get("question") or item.get("query"))
        if not query:
            continue
        nodes = extract_nodes_from_query(query, alias_entries)
        annotation = make_annotation(query, nodes, "medical_dialogue", node_index)
        if annotation:
            annotations.append(annotation)
            count += 1
            if max_samples and count >= max_samples:
                break
    return annotations


def sample_alias_for_node(node_name, node_type):
    aliases = [node_name]
    for aliases_group, candidates in MANUAL_ALIAS_GROUPS.get(node_type, []):
        if node_name in candidates:
            aliases.extend(aliases_group)
    aliases = dedupe_preserve(aliases)
    aliases.sort(key=len)
    return aliases[0]


def build_synthetic_annotations(clean_records, node_index, rng):
    annotations = []
    for record in tqdm(clean_records, desc="生成 synthetic 样本", unit="record"):
        disease = record["name"]
        symptoms = record["symptom"]
        departments = record["cure_department"]
        checks = record["check"]
        drugs = dedupe_preserve(record["common_drug"] + record["recommand_drug"])

        selected_symptoms = symptoms[:2]
        nodes = []
        parts = []

        for symptom in selected_symptoms:
            mention = sample_alias_for_node(symptom, "symptom")
            parts.append(mention)
            nodes.append({"mention": mention, "node_name": symptom, "node_type": "symptom"})

        if parts:
            query = "这两天" + "，".join(parts)
            if rng.random() < 0.75:
                query += f"，会不会是{disease}"
                nodes.append({"mention": disease, "node_name": disease, "node_type": "disease"})
            annotation = make_annotation(query, nodes, "medical_synthetic_symptom", node_index)
            if annotation:
                annotations.append(annotation)
        else:
            query = f"我这个像不像{disease}"
            nodes = [{"mention": disease, "node_name": disease, "node_type": "disease"}]
            annotation = make_annotation(query, nodes, "medical_synthetic_disease", node_index)
            if annotation:
                annotations.append(annotation)

        if departments:
            department = departments[0]
            dep_nodes = [{"mention": department, "node_name": department, "node_type": "department"}]
            dep_query = f"这种情况挂{department}还是别的科"
            if selected_symptoms:
                mention = sample_alias_for_node(selected_symptoms[0], "symptom")
                dep_query = f"{mention}这种情况挂{department}还是别的科"
                dep_nodes.insert(0, {"mention": mention, "node_name": selected_symptoms[0], "node_type": "symptom"})
            annotation = make_annotation(dep_query, dep_nodes, "medical_synthetic_department", node_index)
            if annotation:
                annotations.append(annotation)

        if checks:
            check = checks[0]
            check_nodes = [{"mention": check, "node_name": check, "node_type": "check"}]
            check_query = f"要不要先做{check}"
            if selected_symptoms:
                mention = sample_alias_for_node(selected_symptoms[0], "symptom")
                check_query = f"{mention}这种情况要不要先做{check}"
                check_nodes.insert(0, {"mention": mention, "node_name": selected_symptoms[0], "node_type": "symptom"})
            annotation = make_annotation(check_query, check_nodes, "medical_synthetic_check", node_index)
            if annotation:
                annotations.append(annotation)

        if drugs and selected_symptoms:
            drug = drugs[0]
            mention = sample_alias_for_node(selected_symptoms[0], "symptom")
            query = f"{mention}能先吃{drug}吗"
            nodes = [
                {"mention": mention, "node_name": selected_symptoms[0], "node_type": "symptom"},
                {"mention": drug, "node_name": drug, "node_type": "drug"},
            ]
            annotation = make_annotation(query, nodes, "medical_synthetic_drug", node_index)
            if annotation:
                annotations.append(annotation)

    return annotations


def build_fill_annotations(node_index, existing_annotations):
    covered = {node_type: set() for node_type in node_index}
    for item in existing_annotations:
        for node in item["nodes"]:
            covered[node["node_type"]].add(node["node_name"])

    annotations = []
    for node_type, names in node_index.items():
        for name in tqdm(sorted(names), desc=f"补覆盖 {node_type}", unit="node"):
            if name in covered[node_type]:
                continue

            mention = sample_alias_for_node(name, node_type)
            if node_type == "symptom":
                query = f"这两天一直{mention}"
            elif node_type == "disease":
                query = f"我这个像不像{name}"
            elif node_type == "department":
                query = f"这种情况挂{name}吗"
            elif node_type == "check":
                query = f"要不要做{name}"
            else:
                query = f"{name}能先用吗"

            annotations.append(
                {
                    "query": query,
                    "nodes": [{"mention": mention if mention in query else name, "node_name": name, "node_type": node_type}],
                    "source": f"fill:{node_type}",
                }
            )

    return annotations


def build_manual_annotations(node_index):
    annotations = []
    for item in MANUAL_HARD_CASES:
        annotation = make_annotation(item["query"], item["nodes"], "manual_hard_case", node_index)
        if annotation:
            annotations.append(annotation)
    for query in MANUAL_NEGATIVES:
        annotation = make_annotation(query, [], "manual_negative", node_index)
        if annotation:
            annotations.append(annotation)
    return annotations


def dedupe_annotations(annotations):
    result = []
    seen = set()
    for item in annotations:
        node_key = tuple((node["mention"], node["node_name"], node["node_type"]) for node in sort_nodes(item["nodes"]))
        key = (item["query"], node_key)
        if key in seen:
            continue
        seen.add(key)
        result.append({"query": item["query"], "nodes": sort_nodes(item["nodes"]), "source": item["source"]})
    return result


def build_sharegpt_dataset(annotations):
    dataset = []
    for item in annotations:
        answer = json.dumps({"nodes": sort_nodes(item["nodes"])}, ensure_ascii=False)
        dataset.append(
            {
                "conversations": [
                    {"from": "human", "value": item["query"]},
                    {"from": "gpt", "value": answer},
                ],
                "system": SYSTEM_PROMPT,
            }
        )
    return dataset


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_stats(annotations, node_index, dropped_symptoms):
    source_counts = Counter(item["source"] for item in annotations)
    node_type_counts = Counter()
    covered = {node_type: set() for node_type in node_index}

    positive_samples = 0
    for item in annotations:
        if item["nodes"]:
            positive_samples += 1
        for node in item["nodes"]:
            node_type_counts[node["node_type"]] += 1
            covered[node["node_type"]].add(node["node_name"])

    stats = {
        "dataset_size": len(annotations),
        "positive_samples": positive_samples,
        "negative_samples": len(annotations) - positive_samples,
        "source_counts": dict(sorted(source_counts.items())),
        "node_type_counts": dict(sorted(node_type_counts.items())),
        "coverage": {},
        "kg_node_counts": {node_type: len(names) for node_type, names in node_index.items()},
        "filtered_symptom_count": sum(dropped_symptoms.values()),
        "filtered_symptom_examples": dropped_symptoms.most_common(30),
    }

    for node_type, names in node_index.items():
        total = len(names)
        hit = len(covered[node_type])
        stats["coverage"][node_type] = {
            "covered": hit,
            "total": total,
            "ratio": round(hit / total, 6) if total else 0.0,
        }

    return stats


def parse_args():
    repo_root = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) >= 3 else Path.cwd()
    default_out_dir = repo_root / "algorithm" / "data" / "processed" / "full_disease"

    parser = argparse.ArgumentParser(description="构建全量 query2graph 数据集")
    parser.add_argument("--medical-path", default=str(repo_root / "algorithm" / "data" / "raw" / "medical.json"))
    parser.add_argument("--imcs-train", default=str(repo_root / "algorithm" / "data" / "raw" / "imcs21" / "IMCS-V2_train.json"))
    parser.add_argument("--imcs-dev", default=str(repo_root / "algorithm" / "data" / "raw" / "imcs21" / "IMCS-V2_dev.json"))
    parser.add_argument("--imcs-test", default=str(repo_root / "algorithm" / "data" / "raw" / "imcs21" / "IMCS-V2_test.json"))
    parser.add_argument("--symptom-norm", default=str(repo_root / "algorithm" / "data" / "raw" / "imcs21" / "symptom_norm.csv"))
    parser.add_argument("--dialogue-path", default=str(repo_root / "algorithm" / "data" / "raw" / "train_0001_of_0001.json"))
    parser.add_argument("--output-annotations", default=str(default_out_dir / "query2graph_full_annotations.json"))
    parser.add_argument("--output-sharegpt", default=str(default_out_dir / "query2graph_full_sharegpt.json"))
    parser.add_argument("--output-stats", default=str(default_out_dir / "query2graph_full_stats.json"))
    parser.add_argument("--output-kg-source", default=str(default_out_dir / "medical_full_kg_source.jsonl"))
    parser.add_argument("--max-dialogue", type=int, default=0, help="0 表示不限制")
    parser.add_argument("--seed", type=int, default=20260418)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    medical_path = Path(args.medical_path)
    imcs_paths = [Path(args.imcs_train), Path(args.imcs_dev), Path(args.imcs_test)]
    symptom_norm_path = Path(args.symptom_norm)
    dialogue_path = Path(args.dialogue_path)

    raw_records = load_medical_records(medical_path)
    clean_records, node_index, dropped_symptoms = build_clean_medical_records(raw_records)
    _ = load_symptom_norms(symptom_norm_path)
    name_maps = build_name_maps(node_index)
    alias_entries = build_alias_entries(node_index)

    annotations = []
    annotations.extend(build_manual_annotations(node_index))
    annotations.extend(build_annotations_from_imcs(imcs_paths, node_index, name_maps))
    annotations.extend(
        build_annotations_from_dialogue(
            dialogue_path,
            alias_entries,
            node_index,
            max_samples=args.max_dialogue or None,
        )
    )
    annotations.extend(build_synthetic_annotations(clean_records, node_index, rng))
    annotations.extend(build_fill_annotations(node_index, annotations))
    annotations = dedupe_annotations(annotations)

    sharegpt_dataset = build_sharegpt_dataset(annotations)
    stats = build_stats(annotations, node_index, dropped_symptoms)

    write_json(Path(args.output_annotations), annotations)
    write_json(Path(args.output_sharegpt), sharegpt_dataset)
    write_json(Path(args.output_stats), stats)
    write_jsonl(Path(args.output_kg_source), clean_records)

    print("构建完成")
    print(f"annotations: {args.output_annotations}")
    print(f"sharegpt: {args.output_sharegpt}")
    print(f"stats: {args.output_stats}")
    print(f"kg_source: {args.output_kg_source}")
    print(json.dumps(stats["coverage"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
