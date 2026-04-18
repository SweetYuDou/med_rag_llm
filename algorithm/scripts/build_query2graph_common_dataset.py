import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm


SYSTEM_PROMPT = (
    "你是一个中文医疗问诊 query 解析助手。"
    "请把用户口语化、模糊化的问诊表达映射到医疗知识图谱节点。"
    "重点识别症状，也可以识别疾病、科室、检查和药物。"
    "只返回严格 JSON，不要输出解释。"
    "格式为：{\"nodes\": [{\"mention\": str, \"node_name\": str, \"node_type\": str}]}。"
    "node_type 只允许为 symptom, disease, department, check, drug。"
)

TYPE_ORDER = {"symptom": 0, "disease": 1, "check": 2, "department": 3, "drug": 4}

COMMON_DISEASE_PATTERNS = [
    r"^(普通)?感冒$",
    r"^(急性)?上呼吸道感染$",
    r"^(急性|慢性)?咽炎$",
    r"^(急性|慢性)?咽喉炎$",
    r"^(急性|慢性)?扁桃体炎$",
    r"^(急性|慢性)?鼻炎$",
    r"^过敏性鼻炎$",
    r"^(急性|慢性)?鼻窦炎$",
    r"^(急性|慢性)?支气管炎$",
    r"^过敏性咳嗽$",
    r"^肺炎$",
    r"^支气管肺炎$",
    r"^中耳炎$",
    r"^(急性|慢性)?胃炎$",
    r"^(急性)?胃肠炎$",
    r"^(急性|慢性)?肠炎$",
    r"^腹泻$",
    r"^便秘$",
    r"^(功能性)?消化不良$",
    r"^反流性食管炎$",
    r"^胃溃疡$",
    r"^尿路感染$",
    r"^膀胱炎$",
    r"^(细菌性|霉菌性|滴虫性)?阴道炎$",
    r"^盆腔炎$",
    r"^月经不调$",
    r"^痛经$",
    r"^湿疹$",
    r"^(接触性|过敏性|脂溢性)?皮炎$",
    r"^荨麻疹$",
    r"^(寻常)?痤疮$",
    r"^(急性|慢性)?结膜炎$",
    r"^干眼症$",
    r"^麦粒肿$",
    r"^高血压$",
    r"^糖尿病$",
    r"^高脂血症$",
    r"^贫血$",
    r"^偏头痛$",
    r"^失眠$",
    r"^颈椎病$",
    r"^腰肌劳损$",
    r"^小儿感冒$",
    r"^小儿咳嗽$",
    r"^小儿发热$",
    r"^小儿支气管炎$",
    r"^小儿支气管肺炎$",
    r"^小儿腹泻$",
    r"^小儿便秘$",
    r"^小儿消化不良$",
    r"^新生儿黄疸$",
    r"^口腔溃疡$",
    r"^牙龈炎$",
    r"^龋齿$",
    r"^痔疮$",
    r"^胆囊炎$",
    r"^乳腺增生$",
]

BLACKLIST = [
    "癌", "肿瘤", "白血病", "中毒", "结核", "梅毒", "艾滋", "尘肺", "矽肺", "鼠疫",
    "狂犬", "移植", "畸形", "先天性", "遗传", "恶性", "尿毒", "肾衰", "肝衰",
    "脑梗", "脑出血", "心肌梗", "栓塞", "狼疮", "硬化症",
]

NEG_PREFIX = ["没有", "没", "不", "无", "并无", "未见", "否认", "不是"]

DISTRACTORS = [
    "前天淋了雨", "昨晚吹了空调", "这两天一直加班", "昨天跑出去吹风了",
    "这几天没怎么休息", "昨天晚上吃得有点杂",
]

TIME_HINTS = ["这两天", "昨天开始", "今天开始", "昨晚开始", "已经三天了", "反反复复好几天"]
NEGATIVE_QUERIES = [
    "要不要空腹检查", "这个检查需要预约吗", "最近睡得不好", "抽血前能喝水吗",
    "明天体检，今晚几点后不能吃东西", "这种情况要不要先观察两天",
    "最近压力有点大，老想发呆", "化验单还没出来，要先挂号吗",
]

SUBJECTS = {"adult": ["我", "我自己"], "child": ["孩子", "宝宝", "我家娃"]}

SYMPTOM_ALIASES = {
    "发烧": ["发热", "高热", "低热"], "发热": ["发热", "高热", "低热"], "发高烧": ["高热", "发热"],
    "高烧": ["高热", "发热"], "低烧": ["低热", "发热"], "咳嗽": ["咳嗽"], "一直咳": ["咳嗽"],
    "咳个不停": ["咳嗽"], "咳得厉害": ["咳嗽"], "干咳": ["干咳", "咳嗽"], "有痰": ["痰", "咳嗽"],
    "流鼻涕": ["鼻流涕"], "留鼻涕": ["鼻流涕"], "鼻子堵": ["鼻塞"], "鼻子不通气": ["鼻塞"],
    "嗓子疼": ["咽痛"], "喉咙痛": ["咽痛"], "咽喉痛": ["咽痛"], "候咙疼": ["咽痛"],
    "胸口闷": ["胸闷"], "胸闷": ["胸闷"], "喘": ["气喘", "呼吸困难"], "有点喘": ["气喘", "呼吸困难"],
    "喘不上气": ["呼吸困难", "气喘"], "头晕": ["头晕"], "脑袋晕": ["头晕"], "头昏": ["头晕"],
    "头疼": ["头痛"], "脑壳疼": ["头痛"], "肚子疼": ["腹痛"], "胃疼": ["胃痛", "腹痛"],
    "肚子胀": ["腹胀"], "胃胀": ["腹胀"], "拉肚子": ["腹泻"], "拉肚": ["腹泻"], "拉稀": ["腹泻", "稀便"],
    "跑肚": ["腹泻"], "便秘": ["便秘"], "上不出来": ["便秘"], "恶心": ["恶心"], "反胃": ["恶心"],
    "想吐": ["恶心", "呕吐"], "吐了": ["呕吐"], "没胃口": ["食欲不振"], "吃不下": ["食欲不振"],
    "没劲": ["乏力"], "没力气": ["乏力"], "浑身无力": ["乏力"], "浑身疼": ["全身疼痛", "肌肉酸痛"],
    "全身疼": ["全身疼痛", "肌肉酸痛"], "浑身酸痛": ["肌肉酸痛", "全身疼痛"],
    "身上痒": ["瘙痒"], "皮肤痒": ["瘙痒"], "起疹子": ["皮疹"], "起红疹": ["皮疹"],
}

DISEASE_ALIASES = {
    "感冒": ["感冒", "小儿感冒", "上呼吸道感染"], "伤风": ["感冒", "上呼吸道感染"],
    "支气管炎": ["支气管炎", "小儿支气管炎"], "肺炎": ["肺炎", "支气管肺炎", "小儿支气管肺炎"],
    "鼻炎": ["鼻炎", "过敏性鼻炎"], "鼻窦炎": ["鼻窦炎"], "咽炎": ["咽炎", "慢性咽炎"],
    "扁桃体炎": ["扁桃体炎"], "中耳炎": ["中耳炎"], "胃炎": ["胃炎", "慢性胃炎", "急性胃炎"],
    "肠胃炎": ["胃肠炎", "急性胃肠炎"], "胃肠炎": ["胃肠炎", "急性胃肠炎"], "肠炎": ["肠炎"],
    "消化不良": ["消化不良", "小儿消化不良"], "腹泻": ["腹泻", "小儿腹泻"], "便秘": ["便秘", "小儿便秘"],
    "尿感": ["尿路感染"], "尿路感染": ["尿路感染"], "膀胱炎": ["膀胱炎"],
    "阴道炎": ["阴道炎", "细菌性阴道炎", "霉菌性阴道炎", "滴虫性阴道炎"],
    "霉菌性阴道炎": ["霉菌性阴道炎"], "细菌性阴道炎": ["细菌性阴道炎"], "滴虫性阴道炎": ["滴虫性阴道炎"],
    "盆腔炎": ["盆腔炎"], "月经不调": ["月经不调"], "痛经": ["痛经"], "湿疹": ["湿疹"],
    "皮炎": ["皮炎", "接触性皮炎", "过敏性皮炎"], "荨麻疹": ["荨麻疹"], "痘痘": ["痤疮"], "痤疮": ["痤疮"],
    "结膜炎": ["结膜炎"], "干眼": ["干眼症"], "麦粒肿": ["麦粒肿"], "高血压": ["高血压"],
    "糖尿病": ["糖尿病"], "高血脂": ["高脂血症"], "贫血": ["贫血"], "偏头痛": ["偏头痛"],
    "失眠": ["失眠"], "颈椎病": ["颈椎病"], "腰肌劳损": ["腰肌劳损"], "小儿发热": ["小儿发热"],
    "新生儿黄疸": ["新生儿黄疸"], "口腔溃疡": ["口腔溃疡"], "牙龈炎": ["牙龈炎"],
    "龋齿": ["龋齿"], "痔疮": ["痔疮"], "胆囊炎": ["胆囊炎"], "乳腺增生": ["乳腺增生"],
}

CHECK_ALIASES = {
    "血常规": ["血常规"], "验血": ["血常规"], "抽血": ["血常规"], "大便常规": ["大便常规"],
    "查大便": ["大便常规"], "便检": ["大便常规"], "尿常规": ["尿常规"], "查尿": ["尿常规"],
    "尿检": ["尿常规"], "胸片": ["胸部X线检查", "X线检查"], "拍片": ["胸部X线检查", "X线检查"],
    "胸部ct": ["胸部CT检查", "CT检查"], "ct": ["CT检查", "胸部CT检查"], "b超": ["B超", "彩超"],
    "彩超": ["彩超", "B超"], "胃镜": ["胃镜"], "鼻镜": ["鼻内镜", "鼻镜"], "喉镜": ["喉镜"],
}

DEPT_ALIASES = {
    "儿科": ["儿科", "小儿内科"], "小儿科": ["儿科", "小儿内科"], "呼吸科": ["呼吸内科"],
    "消化科": ["消化内科"], "耳鼻喉科": ["耳鼻喉科"], "皮肤科": ["皮肤科"], "妇科": ["妇科"],
    "眼科": ["眼科"], "内分泌科": ["内分泌科"], "神经内科": ["神经内科"], "口腔科": ["口腔科"],
}

DRUG_ALIASES = {
    "布洛芬": ["布洛芬"], "退烧药": ["布洛芬", "对乙酰氨基酚"], "阿莫西林": ["阿莫西林"],
    "蒙脱石散": ["蒙脱石散"], "开塞露": ["开塞露"], "奥美拉唑": ["奥美拉唑"],
    "氯雷他定": ["氯雷他定"], "滴眼液": ["滴眼液"],
}

LABEL_MAP = {"Symptom": "symptom", "Medical_Examination": "check", "Disease": "disease", "Drug": "drug", "Department": "department"}


def clean_text(value):
    return "" if value is None else str(value).strip()


def listify(value):
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [clean_text(v) for v in values if clean_text(v)]


def normalize_text(text):
    text = clean_text(text).lower()
    return re.sub(r"[\s,.，。！？?、/\\:：;；'\"“”‘’()（）\[\]【】<>《》-]", "", text)


def valid_node(text):
    text = clean_text(text)
    return bool(text) and len(text) <= 20 and bool(re.search(r"[\u4e00-\u9fffA-Za-z]", text))


def dedupe_nodes(nodes):
    seen = set()
    output = []
    for node in nodes:
        key = (node["mention"], node["node_name"], node["node_type"])
        if key in seen:
            continue
        seen.add(key)
        output.append(node)
    return sorted(output, key=lambda x: (TYPE_ORDER.get(x["node_type"], 9), x["node_name"], x["mention"]))


def dump_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def dump_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_seconds(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def print_progress(stage, current, total, started_at, extra=""):
    total = max(total, 1)
    ratio = min(max(current / total, 0.0), 1.0)
    width = 24
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.time() - started_at
    eta = (elapsed / ratio - elapsed) if ratio > 0 else 0
    line = f"[{bar}] {ratio * 100:5.1f}% | {stage} | ETA {format_seconds(eta)}"
    if extra:
        line += f" | {extra}"
    print(line, flush=True)


def hash_score(text):
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def load_json_or_lines(path):
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first in "[{":
            try:
                data = json.load(f)
                if isinstance(data, (list, dict)):
                    return data
            except json.JSONDecodeError:
                f.seek(0)
        return [json.loads(line) for line in f if line.strip()]


def iter_json_array(path):
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        while True:
            ch = f.read(1)
            if not ch:
                return
            if ch.isspace():
                continue
            if ch == "[":
                break
            raise ValueError(f"{path} 不是 JSON 数组。")
        buf = ""
        while True:
            chunk = f.read(65536)
            if chunk:
                buf += chunk
            elif not buf:
                return
            while True:
                buf = buf.lstrip()
                if not buf:
                    break
                if buf[0] == ",":
                    buf = buf[1:]
                    continue
                if buf[0] == "]":
                    return
                try:
                    item, idx = decoder.raw_decode(buf)
                except json.JSONDecodeError:
                    if chunk:
                        break
                    raise
                yield item
                buf = buf[idx:]


def choose_name(candidates, pool):
    raw = set(pool)
    norm_map = {normalize_text(name): name for name in pool}
    for cand in candidates:
        cand = clean_text(cand)
        if not cand:
            continue
        if cand in raw:
            return cand
        norm = normalize_text(cand)
        if norm in norm_map:
            return norm_map[norm]
        for existing in pool:
            if cand in existing or existing in cand:
                return existing
    return ""


def reorder_pain(candidates, query):
    if "浑身疼" not in query and "全身疼" not in query:
        return candidates
    if any(token in query for token in ["酸", "酸痛", "肌肉", "像散架"]):
        return ["肌肉酸痛", "全身疼痛"] + [c for c in candidates if c not in {"肌肉酸痛", "全身疼痛"}]
    return ["全身疼痛", "肌肉酸痛"] + [c for c in candidates if c not in {"肌肉酸痛", "全身疼痛"}]


def is_negated(text, start, end):
    left = clean_text(text[max(0, start - 4):start])
    if any(left.endswith(prefix) for prefix in NEG_PREFIX):
        return True
    return left.endswith("别") or left.endswith("未")


class Catalog:
    def __init__(self, records):
        self.records = records
        self.nodes = defaultdict(list)
        self.aliases = []
        self.symptom_freq = Counter()
        self._build()

    def _build(self):
        disease = set()
        symptom = Counter()
        check = Counter()
        dept = Counter()
        drug = Counter()
        for row in self.records:
            name = clean_text(row.get("name"))
            if name:
                disease.add(name)
            for item in listify(row.get("symptom")):
                if valid_node(item):
                    symptom[item] += 1
            for item in listify(row.get("check")):
                if valid_node(item):
                    check[item] += 1
            for item in listify(row.get("cure_department")):
                if valid_node(item):
                    dept[item] += 1
            for item in listify(row.get("common_drug")) + listify(row.get("recommand_drug")):
                if valid_node(item):
                    drug[item] += 1
        self.symptom_freq = symptom
        self.nodes["disease"] = sorted(disease)
        self.nodes["symptom"] = sorted(symptom)
        self.nodes["check"] = sorted(check)
        self.nodes["department"] = sorted(dept)
        self.nodes["drug"] = [k for k, _ in drug.most_common(160) if len(k) <= 20]
        self._register_self_aliases()
        self._register_manual_aliases()
        self.aliases.sort(key=lambda x: (-len(x["alias"]), x["priority"], x["alias"]))

    def _add_alias(self, alias, candidates, node_type, priority):
        if node_type == "symptom":
            candidates = reorder_pain(candidates, alias)
        node_name = choose_name(candidates, self.nodes[node_type])
        if not node_name:
            return
        self.aliases.append({"alias": clean_text(alias), "node_name": node_name, "node_type": node_type, "priority": priority})

    def _register_self_aliases(self):
        for node_type, names in self.nodes.items():
            for name in names:
                self._add_alias(name, [name], node_type, 50)

    def _register_manual_aliases(self):
        for alias, candidates in SYMPTOM_ALIASES.items():
            self._add_alias(alias, candidates, "symptom", 1)
        for alias, candidates in DISEASE_ALIASES.items():
            self._add_alias(alias, candidates, "disease", 1)
        for alias, candidates in CHECK_ALIASES.items():
            self._add_alias(alias, candidates, "check", 2)
        for alias, candidates in DEPT_ALIASES.items():
            self._add_alias(alias, candidates, "department", 2)
        for alias, candidates in DRUG_ALIASES.items():
            self._add_alias(alias, candidates, "drug", 3)

    def match(self, text, allow=None):
        text = clean_text(text)
        found = []
        for alias in self.aliases:
            if allow and alias["node_type"] not in allow:
                continue
            start = text.find(alias["alias"])
            while start != -1:
                end = start + len(alias["alias"])
                if not is_negated(text, start, end):
                    node_name = alias["node_name"]
                    if alias["node_type"] == "symptom":
                        node_name = choose_name(reorder_pain([node_name], text), self.nodes["symptom"]) or node_name
                    found.append({"mention": text[start:end], "node_name": node_name, "node_type": alias["node_type"], "start": start, "end": end, "priority": alias["priority"]})
                start = text.find(alias["alias"], start + 1)
        found.sort(key=lambda x: (x["start"], -(x["end"] - x["start"]), x["priority"]))
        used = set()
        output = []
        for item in found:
            span = set(range(item["start"], item["end"]))
            if used.intersection(span):
                continue
            used.update(span)
            output.append({"mention": item["mention"], "node_name": item["node_name"], "node_type": item["node_type"]})
        return dedupe_nodes(output)


def pick_common_records(records):
    patterns = [re.compile(p) for p in COMMON_DISEASE_PATTERNS]
    output = []
    seen = set()
    for row in records:
        name = clean_text(row.get("name"))
        if not name or name in seen:
            continue
        if any(token in name for token in BLACKLIST):
            continue
        if not any(pattern.match(name) for pattern in patterns):
            continue
        output.append(row)
        seen.add(name)
    return sorted(output, key=lambda x: clean_text(x.get("name")))


def extract_bio(sentence, bio_label):
    sentence = clean_text(sentence)
    tags = clean_text(bio_label).split()
    if not sentence or not tags or len(sentence) != len(tags):
        return []
    spans = []
    start = None
    label = None
    for idx, tag in enumerate(tags + ["O"]):
        if tag == "O":
            if start is not None:
                spans.append({"mention": sentence[start:idx], "label": label})
                start = None
                label = None
            continue
        prefix, entity = tag.split("-", 1)
        if prefix == "B" or entity != label:
            if start is not None:
                spans.append({"mention": sentence[start:idx], "label": label})
            start = idx
            label = entity
    return spans


def make_annotation(query, nodes, source, tags):
    return {"query": clean_text(query), "nodes": dedupe_nodes(nodes), "source": source, "tags": sorted(set(tags))}


def to_sharegpt(item):
    return {
        "conversations": [
            {"from": "human", "value": item["query"]},
            {"from": "gpt", "value": json.dumps({"nodes": item["nodes"]}, ensure_ascii=False)},
        ],
        "system": SYSTEM_PROMPT,
    }


def build_imcs_examples(imcs_data, catalog, limit):
    output = []
    seen = set()
    progress = tqdm(imcs_data.values(), desc="处理 IMCS 真实问诊", unit="dialog")
    for dialog in progress:
        progress.set_postfix(collected=len(output))
        diagnosis = clean_text(dialog.get("diagnosis"))
        diagnosis_node = choose_name([diagnosis], catalog.nodes["disease"]) if diagnosis else ""

        self_report = clean_text(dialog.get("self_report"))
        if self_report and self_report not in seen:
            nodes = catalog.match(self_report)
            if diagnosis_node and diagnosis_node in self_report:
                nodes.append({"mention": diagnosis_node, "node_name": diagnosis_node, "node_type": "disease"})
            nodes = dedupe_nodes(nodes)
            if nodes:
                tags = ["real", "imcs", "self_report"]
                if len(nodes) > 1:
                    tags.append("multi_entity")
                output.append(make_annotation(self_report, nodes, "imcs_self_report", tags))
                seen.add(self_report)

        patient_turns = []
        for turn in dialog.get("dialogue", []):
            if clean_text(turn.get("speaker")) != "患者":
                continue
            sentence = clean_text(turn.get("sentence"))
            if not sentence or sentence in seen:
                continue
            spans = extract_bio(sentence, turn.get("BIO_label"))
            nodes = []
            symptom_norm = listify(turn.get("symptom_norm"))
            symptom_spans = [span for span in spans if span["label"] == "Symptom"]
            for idx, span in enumerate(symptom_spans):
                cands = [span["mention"]]
                if idx < len(symptom_norm):
                    cands.insert(0, symptom_norm[idx])
                cands = reorder_pain(cands, span["mention"])
                node_name = choose_name(cands, catalog.nodes["symptom"])
                if node_name:
                    nodes.append({"mention": span["mention"], "node_name": node_name, "node_type": "symptom"})
            for span in spans:
                if span["label"] == "Symptom":
                    continue
                node_type = LABEL_MAP.get(span["label"])
                if not node_type:
                    continue
                matched = catalog.match(span["mention"], allow={node_type})
                if matched:
                    nodes.extend(matched[:1])
            if not nodes:
                nodes = catalog.match(sentence)
            nodes = dedupe_nodes(nodes)
            if not nodes:
                continue
            tags = ["real", "imcs", "patient_turn"]
            if len(nodes) > 1:
                tags.append("multi_entity")
            if any(prefix in sentence for prefix in NEG_PREFIX):
                tags.append("negation")
            output.append(make_annotation(sentence, nodes, "imcs_patient_turn", tags))
            seen.add(sentence)
            patient_turns.append((sentence, nodes))
            if len(output) >= limit:
                return output[:limit]

        if len(patient_turns) >= 2 and len(output) < limit:
            query = "，".join(x[0].rstrip("。！？?") for x in patient_turns[:2])
            if query and query not in seen:
                nodes = dedupe_nodes(patient_turns[0][1] + patient_turns[1][1])
                if nodes:
                    output.append(make_annotation(query, nodes, "imcs_combined", ["real", "imcs", "multi_entity"]))
                    seen.add(query)
        if len(output) >= limit:
            break
    progress.close()
    return output[:limit]


def colloquial_score(text):
    score = 0
    text = clean_text(text)
    if any(token in text for token in ["我", "宝宝", "孩子", "家里", "老是", "一直", "有点", "这两天", "昨天", "今天"]):
        score += 2
    if any(token in text for token in ["嗓子", "肚子", "鼻子", "胸口", "没劲", "难受", "喉咙"]):
        score += 2
    if any(token in text for token in ["？", "?", "吗", "咋", "怎么"]):
        score += 1
    if len(text) <= 60:
        score += 1
    if len(text) > 90:
        score -= 2
    return score


def clean_dialog_query(item):
    text = clean_text(item.get("input") or item.get("instruction"))
    text = re.sub(r"\s+", "", text)
    return text[:110].rstrip("，。；;")


def build_meddialog_examples(path, catalog, pos_limit, neg_limit, progress_stage=None):
    positives = []
    negatives = []
    seen = set()
    coverage = Counter()
    processed = 0
    progress = tqdm(desc="扫描中文医疗对话", unit="row")
    for item in iter_json_array(path):
        processed += 1
        progress.update(1)
        if processed % 5000 == 0:
            progress.set_postfix(pos=len(positives), neg=len(negatives))
        query = clean_dialog_query(item)
        if not query or query in seen:
            continue
        seen.add(query)
        score = colloquial_score(query)
        nodes = catalog.match(query)
        tags = ["real", "meddialog"]
        if len(nodes) > 1:
            tags.append("multi_entity")
        if any(prefix in query for prefix in NEG_PREFIX):
            tags.append("negation")
        if nodes and score >= 1:
            useful = False
            for node in nodes:
                key = (node["node_type"], node["node_name"])
                if coverage[key] < 8:
                    useful = True
                    coverage[key] += 1
            if useful or hash_score(query) % 5 == 0:
                positives.append(make_annotation(query, nodes, "meddialog", tags))
        elif not nodes and score >= 0 and any(hint in query for hint in ["空腹", "预约", "体检", "挂号", "睡得不好", "压力大", "复查", "饮食"]):
            negatives.append(make_annotation(query, [], "meddialog", tags + ["negative"]))
        if len(positives) >= pos_limit and len(negatives) >= neg_limit:
            break
    progress.set_postfix(pos=len(positives), neg=len(negatives))
    progress.close()
    return positives[:pos_limit], negatives[:neg_limit]


def pick_symptoms(row, catalog):
    nodes = []
    for item in listify(row.get("symptom")):
        node = choose_name([item], catalog.nodes["symptom"])
        if node:
            nodes.append(node)
    nodes = list(dict.fromkeys(nodes))
    if not nodes:
        return []
    count = random.randint(1, min(3, len(nodes)))
    return random.sample(nodes, count)


def surface_form(node_name):
    for alias, cands in SYMPTOM_ALIASES.items():
        if node_name in cands:
            return alias
    return node_name


def generate_synthetic_from_record(row, catalog):
    disease = clean_text(row.get("name"))
    symptoms = pick_symptoms(row, catalog)
    if not symptoms:
        return []
    dept = choose_name(listify(row.get("cure_department")), catalog.nodes["department"])
    check = choose_name(listify(row.get("check")), catalog.nodes["check"])
    drug = choose_name(listify(row.get("common_drug")) + listify(row.get("recommand_drug")), catalog.nodes["drug"])
    group = "child" if any(token in disease for token in ["小儿", "新生儿"]) else "adult"
    subject = random.choice(SUBJECTS[group])
    time_hint = random.choice(TIME_HINTS)
    mentions = [surface_form(item) for item in symptoms]
    symptom_nodes = [{"mention": mention, "node_name": name, "node_type": "symptom"} for mention, name in zip(mentions, symptoms)]
    out = []
    if len(mentions) == 1:
        out.append(make_annotation(f"{subject}{time_hint}{mentions[0]}，这像什么问题", symptom_nodes, "synthetic", ["synthetic", "single_entity"]))
    else:
        out.append(make_annotation(f"{subject}{time_hint}{'、'.join(mentions[:-1])}，还有点{mentions[-1]}", symptom_nodes, "synthetic", ["synthetic", "multi_entity"]))
    out.append(make_annotation(
        f"{subject}{time_hint}{'、'.join(mentions)}，像不像{disease}",
        symptom_nodes + [{"mention": disease, "node_name": disease, "node_type": "disease"}],
        "synthetic",
        ["synthetic", "disease_mention"],
    ))
    if dept:
        out.append(make_annotation(
            f"{subject}{time_hint}{'、'.join(mentions)}，这种情况挂{dept}还是别的科",
            symptom_nodes + [{"mention": dept, "node_name": dept, "node_type": "department"}],
            "synthetic",
            ["synthetic", "department_question"],
        ))
    if check:
        out.append(make_annotation(
            f"{subject}{time_hint}{mentions[0]}，要不要先做{check}",
            [symptom_nodes[0], {"mention": check, "node_name": check, "node_type": "check"}],
            "synthetic",
            ["synthetic", "check_question"],
        ))
    if drug:
        out.append(make_annotation(
            f"{subject}{time_hint}{mentions[0]}，能不能先吃点{drug}",
            [symptom_nodes[0], {"mention": drug, "node_name": drug, "node_type": "drug"}],
            "synthetic",
            ["synthetic", "drug_question"],
        ))
    return out


def generate_hard_cases(catalog, count):
    symptom_pool = catalog.nodes["symptom"]
    fever = choose_name(["发热", "高热"], symptom_pool)
    cough = choose_name(["咳嗽"], symptom_pool)
    throat = choose_name(["咽痛"], symptom_pool)
    wheeze = choose_name(["气喘", "呼吸困难"], symptom_pool)
    dizzy = choose_name(["头晕"], symptom_pool)
    chest = choose_name(["胸闷"], symptom_pool)
    whole = choose_name(["全身疼痛", "肌肉酸痛"], symptom_pool)
    seeds = [x for x in [fever, cough, throat, wheeze, dizzy, chest, whole] if x]
    if len(seeds) < 4:
        seeds = symptom_pool[:8]
    out = []
    seen = set()
    attempts = 0
    while len(out) < count and attempts < count * 30:
        attempts += 1
        s1, s2 = random.sample(seeds, 2)
        a1, a2 = surface_form(s1), surface_form(s2)
        subject = random.choice(["我", "我这边", "孩子", "宝宝"])
        time_hint = random.choice(TIME_HINTS)
        templates = [
            make_annotation(f"{subject}{time_hint}没有{a1}，就是有点{a2}", [{"mention": f"有点{a2}", "node_name": s2, "node_type": "symptom"}], "synthetic_hard", ["synthetic", "hard", "negation"]),
            make_annotation(f"{subject}{time_hint}不头晕，但胸口闷", [{"mention": "胸口闷", "node_name": chest or s2, "node_type": "symptom"}], "synthetic_hard", ["synthetic", "hard", "negation"]),
            make_annotation(
                f"{subject}{time_hint}{a1}、{a2}、喉咙痛，还有点喘",
                dedupe_nodes([
                    {"mention": a1, "node_name": s1, "node_type": "symptom"},
                    {"mention": a2, "node_name": s2, "node_type": "symptom"},
                    {"mention": "喉咙痛", "node_name": throat or s1, "node_type": "symptom"},
                    {"mention": "喘", "node_name": wheeze or s2, "node_type": "symptom"},
                ]),
                "synthetic_hard",
                ["synthetic", "hard", "multi_entity"],
            ),
            make_annotation(
                f"{subject}{random.choice(DISTRACTORS)}，昨天开始{a1}，今天还{a2}",
                [{"mention": a1, "node_name": s1, "node_type": "symptom"}, {"mention": a2, "node_name": s2, "node_type": "symptom"}],
                "synthetic_hard",
                ["synthetic", "hard", "distractor"],
            ),
            make_annotation(f"{subject}{time_hint}浑身酸痛，像跑完步一样", [{"mention": "浑身酸痛", "node_name": choose_name(["肌肉酸痛", "全身疼痛"], symptom_pool) or whole or s1, "node_type": "symptom"}], "synthetic_hard", ["synthetic", "hard", "normalization_ambiguity"]),
            make_annotation(f"{subject}{time_hint}浑身疼，哪都疼，一碰就难受", [{"mention": "浑身疼", "node_name": choose_name(["全身疼痛", "肌肉酸痛"], symptom_pool) or whole or s2, "node_type": "symptom"}], "synthetic_hard", ["synthetic", "hard", "normalization_ambiguity"]),
            make_annotation(f"{subject}{time_hint}不是感冒吧，就是{a1}和{a2}", [{"mention": a1, "node_name": s1, "node_type": "symptom"}, {"mention": a2, "node_name": s2, "node_type": "symptom"}], "synthetic_hard", ["synthetic", "hard", "negation", "multi_entity"]),
            make_annotation(f"{subject}{time_hint}{a1}，但不发烧", [{"mention": a1, "node_name": s1, "node_type": "symptom"}], "synthetic_hard", ["synthetic", "hard", "negation"]),
        ]
        for item in templates:
            if item["query"] in seen:
                continue
            seen.add(item["query"])
            out.append(item)
            if len(out) >= count:
                break
    return out[:count]


def generate_negatives(count):
    out = []
    seen = set()

    def add(query):
        if query in seen:
            return
        seen.add(query)
        out.append(make_annotation(query, [], "synthetic_negative", ["synthetic", "negative"]))

    for text in NEGATIVE_QUERIES:
        add(text)

    subjects = ["我", "我这边", "孩子", "宝宝", "家里老人", "最近我"]
    prefixes = ["最近", "这两天", "这周", "这阵子", "这几天", "这段时间", "昨晚", "今天"]
    middles = ["睡得不好", "压力有点大", "总想发呆", "老觉得累", "作息有点乱", "胃口一般", "没什么精神", "总想休息", "状态一般", "人有点蔫"]
    suffixes = ["，先观察行吗", "，要不要先休息", "，是不是熬夜造成的", "，需要马上去医院吗", "", "，这种先不看行不行", "，是不是先缓缓", "，先别去医院可以吗"]
    check_texts = ["空腹检查", "体检", "复查", "抽血", "挂号", "预约", "化验", "验血", "查体", "检查"]
    decision_texts = ["要不要先观察", "是不是先休息一下就行", "先不去医院行不行", "要不要先挂号", "是不是不用太紧张"]

    for subject in subjects:
        for prefix in prefixes:
            for middle in middles:
                for suffix in suffixes:
                    add(f"{subject}{prefix}{middle}{suffix}")

    for check in check_texts:
        add(f"{check}前能喝水吗")
        add(f"{check}是不是都得空腹")
        add(f"{check}需要提前预约吗")
        add(f"{check}没做的话先等等行吗")
        add(f"{check}前一晚几点后不能吃东西")

    for prefix in prefixes:
        for decision in decision_texts:
            add(f"{prefix}只是作息乱，{decision}")
            add(f"{prefix}就是压力大，{decision}")
            add(f"{prefix}感觉有点累，{decision}")

    random.shuffle(out)
    return out[:count]


def generate_fill_examples(catalog, common_records):
    out = []
    for symptom in catalog.nodes["symptom"]:
        mention = surface_form(symptom)
        out.append(make_annotation(f"我这两天{mention}", [{"mention": mention, "node_name": symptom, "node_type": "symptom"}], "synthetic_fill", ["synthetic", "coverage"]))
        out.append(make_annotation(f"最近老是{mention}，要不要紧", [{"mention": mention, "node_name": symptom, "node_type": "symptom"}], "synthetic_fill", ["synthetic", "coverage"]))
    for disease in catalog.nodes["disease"]:
        out.append(make_annotation(f"这不会是{disease}吧", [{"mention": disease, "node_name": disease, "node_type": "disease"}], "synthetic_fill", ["synthetic", "coverage", "disease_mention"]))
    for check in catalog.nodes["check"]:
        out.append(make_annotation(f"这种情况要不要查{check}", [{"mention": check, "node_name": check, "node_type": "check"}], "synthetic_fill", ["synthetic", "coverage", "check_question"]))
    for dept in catalog.nodes["department"]:
        out.append(make_annotation(f"这种情况挂{dept}吗", [{"mention": dept, "node_name": dept, "node_type": "department"}], "synthetic_fill", ["synthetic", "coverage", "department_question"]))
    for drug in catalog.nodes["drug"]:
        out.append(make_annotation(f"{drug}能不能先吃", [{"mention": drug, "node_name": drug, "node_type": "drug"}], "synthetic_fill", ["synthetic", "coverage", "drug_question"]))
    for row in common_records:
        disease = clean_text(row.get("name"))
        symptoms = pick_symptoms(row, catalog)
        if not symptoms:
            continue
        mention = surface_form(symptoms[0])
        out.append(make_annotation(
            f"我这两天{mention}，会不会是{disease}",
            [{"mention": mention, "node_name": symptoms[0], "node_type": "symptom"}, {"mention": disease, "node_name": disease, "node_type": "disease"}],
            "synthetic_fill",
            ["synthetic", "coverage", "disease_mention"],
        ))
    return out


def generate_round_fill_examples(catalog, common_records, round_index):
    out = []
    symptom_templates = [
        "这两天{mention}",
        "最近老是{mention}",
        "{mention}，两天了",
        "就一个感觉：{mention}",
    ]
    disease_templates = [
        "这不会是{disease}吧",
        "像不像{disease}",
        "{disease}会这样吗",
    ]
    check_templates = [
        "这种情况要不要查{check}",
        "{check}要不要先做",
        "现在做{check}有必要吗",
    ]
    dept_templates = [
        "这种情况挂{dept}吗",
        "先去{dept}行不行",
        "这个该看{dept}还是别的科",
    ]
    drug_templates = [
        "{drug}能不能先吃",
        "现在先吃{drug}可以吗",
        "{drug}要不要先备着",
    ]
    pair_templates = [
        "我这两天{mention}，会不会是{disease}",
        "最近{mention}，像不像{disease}",
        "{mention}加重了，会不会和{disease}有关",
    ]

    for index, symptom in enumerate(catalog.nodes["symptom"]):
        mention = surface_form(symptom)
        template = symptom_templates[(index + round_index) % len(symptom_templates)]
        out.append(make_annotation(template.format(mention=mention), [{"mention": mention, "node_name": symptom, "node_type": "symptom"}], "synthetic_fill", ["synthetic", "coverage"]))
    for index, disease in enumerate(catalog.nodes["disease"]):
        template = disease_templates[(index + round_index) % len(disease_templates)]
        out.append(make_annotation(template.format(disease=disease), [{"mention": disease, "node_name": disease, "node_type": "disease"}], "synthetic_fill", ["synthetic", "coverage", "disease_mention"]))
    for index, check in enumerate(catalog.nodes["check"]):
        template = check_templates[(index + round_index) % len(check_templates)]
        out.append(make_annotation(template.format(check=check), [{"mention": check, "node_name": check, "node_type": "check"}], "synthetic_fill", ["synthetic", "coverage", "check_question"]))
    for index, dept in enumerate(catalog.nodes["department"]):
        template = dept_templates[(index + round_index) % len(dept_templates)]
        out.append(make_annotation(template.format(dept=dept), [{"mention": dept, "node_name": dept, "node_type": "department"}], "synthetic_fill", ["synthetic", "coverage", "department_question"]))
    for index, drug in enumerate(catalog.nodes["drug"]):
        template = drug_templates[(index + round_index) % len(drug_templates)]
        out.append(make_annotation(template.format(drug=drug), [{"mention": drug, "node_name": drug, "node_type": "drug"}], "synthetic_fill", ["synthetic", "coverage", "drug_question"]))
    for index, row in enumerate(common_records):
        disease = clean_text(row.get("name"))
        symptoms = pick_symptoms(row, catalog)
        if not symptoms:
            continue
        mention = surface_form(symptoms[0])
        template = pair_templates[(index + round_index) % len(pair_templates)]
        out.append(make_annotation(
            template.format(mention=mention, disease=disease),
            [{"mention": mention, "node_name": symptoms[0], "node_type": "symptom"}, {"mention": disease, "node_name": disease, "node_type": "disease"}],
            "synthetic_fill",
            ["synthetic", "coverage", "disease_mention"],
        ))
    return out


def build_dataset(common_records, catalog, imcs_data, meddialog_path, target_size):
    overall_started = time.time()

    imcs = build_imcs_examples(imcs_data, catalog, 2200)
    print_progress("1/6 IMCS 真实样本", 1, 6, overall_started, f"count={len(imcs)}")

    med_pos, med_neg = build_meddialog_examples(meddialog_path, catalog, 2600, 700, progress_stage="2/6 中文医疗对话扫描")
    print_progress("2/6 中文医疗对话完成", 2, 6, overall_started, f"pos={len(med_pos)} neg={len(med_neg)}")

    synthetic = []
    pool = list(common_records)
    random.shuffle(pool)
    synth_bar = tqdm(total=2800, desc="生成合成口语样本", unit="sample")
    while len(synthetic) < 2800:
        for row in pool:
            before = len(synthetic)
            synthetic.extend(generate_synthetic_from_record(row, catalog))
            synth_bar.update(max(0, min(len(synthetic), 2800) - before))
            if len(synthetic) >= 2800:
                break
    synth_bar.close()
    print_progress("3/6 合成口语样本", 3, 6, overall_started, f"count={len(synthetic)}")

    hard = generate_hard_cases(catalog, 2200)
    negatives = generate_negatives(2200)
    print_progress("4/6 难例与负样本", 4, 6, overall_started, f"hard={len(hard)} neg={len(negatives)}")

    final = []
    seen = set()

    def add_batch(batch):
        added = 0
        for item in batch:
            if not item["query"] or item["query"] in seen:
                continue
            seen.add(item["query"])
            final.append(item)
            added += 1
            if len(final) >= target_size:
                break
        return added

    for batch in [imcs, med_pos, med_neg, synthetic, hard, negatives]:
        add_batch(batch)
        if len(final) >= target_size:
            break

    print_progress("5/6 首轮合并去重", 5, 6, overall_started, f"current={len(final)} target={target_size}")

    round_index = 0
    while len(final) < target_size and round_index < 8:
        extra_pool = []
        extra_pool.extend(generate_round_fill_examples(catalog, common_records, round_index))
        extra_pool.extend(generate_hard_cases(catalog, 800 + round_index * 150))
        extra_pool.extend(generate_negatives(800 + round_index * 150))
        random.shuffle(extra_pool)
        before = len(final)
        add_batch(extra_pool)
        round_index += 1
        print_progress("5/6 覆盖补量中", len(final), target_size, overall_started, f"round={round_index} added={len(final) - before}")
        if len(final) == before:
            break

    if len(final) < target_size:
        round_index = 0
        while len(final) < target_size and round_index < 12:
            row = common_records[round_index % len(common_records)]
            extra = generate_synthetic_from_record(row, catalog)
            extra.extend(generate_hard_cases(catalog, 40))
            extra.extend(generate_negatives(40))
            add_batch(extra)
            round_index += 1

    print_progress("6/6 写出前整理", 6, 6, overall_started, f"final={len(final)}")
    random.shuffle(final)
    return final[:target_size]


def build_stats(rows, common_records, catalog):
    node_counts = Counter()
    source_counts = Counter()
    tag_counts = Counter()
    coverage = defaultdict(set)
    for row in rows:
        source_counts[row["source"]] += 1
        for tag in row["tags"]:
            tag_counts[tag] += 1
        for node in row["nodes"]:
            node_counts[node["node_type"]] += 1
            coverage[node["node_type"]].add(node["node_name"])
    return {
        "dataset_size": len(rows),
        "negative_samples": sum(1 for row in rows if not row["nodes"]),
        "common_disease_records": len(common_records),
        "common_nodes": {k: len(v) for k, v in catalog.nodes.items()},
        "node_counts_in_dataset": dict(node_counts),
        "source_counts": dict(source_counts),
        "tag_counts": dict(tag_counts),
        "coverage": {k: {"covered": len(coverage[k]), "total": len(catalog.nodes[k])} for k in ["disease", "symptom", "check", "department", "drug"]},
    }


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="重新构建常见病 query-to-graph 数据集。")
    parser.add_argument("--medical-json", default=str(root / "medical.json"))
    parser.add_argument("--imcs-json", default=str(root / "algorithm" / "data" / "raw" / "imcs21" / "IMCS-V2_train.json"))
    parser.add_argument("--meddialog-json", default=str(root / "algorithm" / "data" / "raw" / "train_0001_of_0001.json"))
    parser.add_argument("--target-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--annotations-out", default=str(root / "algorithm" / "data" / "processed" / "query2graph_annotations.json"))
    parser.add_argument("--sharegpt-out", default=str(root / "algorithm" / "data" / "processed" / "query2graph.json"))
    parser.add_argument("--llamafactory-out", default=str(root / "LLaMA-Factory" / "data" / "query2graph.json"))
    parser.add_argument("--stats-out", default=str(root / "algorithm" / "data" / "processed" / "query2graph_stats.json"))
    parser.add_argument("--common-medical-out", default=str(root / "algorithm" / "data" / "processed" / "medical_common_subset.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    started_at = time.time()
    random.seed(args.seed)
    print_progress("0/6 加载 medical.json", 1, 6, started_at, "starting")
    medical = load_json_or_lines(Path(args.medical_json))
    common_records = pick_common_records(medical)
    catalog = Catalog(common_records)
    print_progress("0/6 常见病子集完成", 1, 6, started_at, f"disease={len(catalog.nodes['disease'])} symptom={len(catalog.nodes['symptom'])}")
    imcs_raw = load_json_or_lines(Path(args.imcs_json))
    imcs_data = imcs_raw if isinstance(imcs_raw, dict) else {str(i): item for i, item in enumerate(imcs_raw)}
    dataset = build_dataset(common_records, catalog, imcs_data, Path(args.meddialog_json), args.target_size)
    sharegpt = [to_sharegpt(item) for item in dataset]
    stats = build_stats(dataset, common_records, catalog)
    dump_json(Path(args.annotations_out), dataset)
    dump_json(Path(args.sharegpt_out), sharegpt)
    dump_json(Path(args.llamafactory_out), sharegpt)
    dump_json(Path(args.stats_out), stats)
    dump_jsonl(Path(args.common_medical_out), common_records)
    print_progress("6/6 已写出全部文件", 6, 6, started_at, f"dataset={len(dataset)}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"annotations -> {args.annotations_out}")
    print(f"sharegpt -> {args.sharegpt_out}")
    print(f"llamafactory -> {args.llamafactory_out}")
    print(f"stats -> {args.stats_out}")
    print(f"common medical -> {args.common_medical_out}")


if __name__ == "__main__":
    main()
