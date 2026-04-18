import argparse
import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm


SYSTEM_PROMPT = (
"""
你是医疗问诊 query 解析助手。你的任务不是传统 BIO NER，而是把用户口语映射到知识图谱节点。
优先识别 symptom 和 disease，也可以识别 department、check、drug。
必须处理否定信息，例如“没有发烧，就是有点咳嗽”中，“发烧”为 negative，“咳嗽”为 positive。
如果用户只是在问流程、检查准备、挂号、饮食注意，没有明确医学实体，则返回空列表。
只返回严格 JSON，不要输出解释。
输出格式：
{
  "nodes": [
    {
      "mention": "用户原句片段",
      "node_name": "知识图谱标准节点名",
      "node_type": "symptom|disease|department|check|drug",
      "polarity": "positive|negative",
      "confidence": 0.0
    }
  ]
}
""".strip()
)

TYPE_PRIORITY = {
    "symptom": 5,
    "disease": 4,
    "check": 3,
    "department": 2,
    "drug": 1,
}

BIO_TO_NODE_TYPE = {
    "Symptom": "symptom",
    "Disease": "disease",
    "Medical_Examination": "check",
    "Drug": "drug",
    "Drug_Category": "drug",
    "Department": "department",
}

NEGATION_PREFIXES = [
    "没有",
    "没",
    "不",
    "不是",
    "并无",
    "并未",
    "未见",
    "否认",
    "从不",
]

SPECULATIVE_PREFIXES = [
    "是不是",
    "会不会",
    "会不会是",
    "像不像",
    "会是",
]

NOISY_SYMPTOM_KEYWORDS = {
    "阳性",
    "阴性",
    "增高",
    "降低",
    "异常",
    "病变",
    "综合征",
    "杂音",
    "站位",
    "占位",
    "缺损",
    "结构",
    "结节影",
    "团块",
    "囊实性",
}

DISEASE_PREFIXES = [
    "急性",
    "慢性",
    "小儿",
    "儿童",
    "新生儿",
    "妊娠合并",
    "原发性",
    "继发性",
]

SYMPTOM_ALIAS_GROUPS = [
    {"mentions": ["发烧", "发高烧", "高烧", "低烧", "有点烧"], "candidates": ["发热", "发烧", "高热", "低热"]},
    {"mentions": ["咳嗽", "老咳", "一直咳", "咳个不停", "咳得厉害"], "candidates": ["咳嗽", "干咳"]},
    {"mentions": ["有痰", "痰多", "咳痰"], "candidates": ["咳痰", "痰多"]},
    {"mentions": ["嗓子疼", "喉咙痛", "咽喉疼", "咽痛"], "candidates": ["咽痛", "咽喉痛"]},
    {"mentions": ["喘", "喘不上气", "气不够用", "呼吸费劲", "呼吸困难"], "candidates": ["呼吸困难", "气促", "喘息", "气短"]},
    {"mentions": ["胸口闷", "胸闷", "胸口发闷"], "candidates": ["胸闷"]},
    {"mentions": ["胸口疼", "胸痛", "胸口痛"], "candidates": ["胸痛"]},
    {"mentions": ["头疼", "头痛", "脑袋疼"], "candidates": ["头痛"]},
    {"mentions": ["头晕", "晕乎乎", "发晕"], "candidates": ["头晕", "眩晕"]},
    {"mentions": ["肚子疼", "肚子痛", "肚子一阵一阵疼"], "candidates": ["腹痛"]},
    {"mentions": ["胃疼", "胃不舒服", "胃痛"], "candidates": ["胃痛"]},
    {"mentions": ["恶心", "反胃", "胃里犯恶心"], "candidates": ["恶心", "恶心与呕吐"]},
    {"mentions": ["想吐", "吐了", "老吐"], "candidates": ["呕吐", "恶心与呕吐"]},
    {"mentions": ["拉肚子", "拉稀", "一直跑厕所"], "candidates": ["腹泻"]},
    {"mentions": ["便秘", "拉不出来"], "candidates": ["便秘"]},
    {"mentions": ["没劲", "没力气", "浑身没劲", "没精神"], "candidates": ["乏力", "无力"]},
    {"mentions": ["浑身疼", "全身疼", "身上疼", "浑身酸疼"], "candidates": ["全身疼痛", "肌肉酸痛", "四肢酸痛", "身痛"]},
    {"mentions": ["关节疼", "关节痛"], "candidates": ["关节痛"]},
    {"mentions": ["流鼻涕", "鼻涕多"], "candidates": ["流涕"]},
    {"mentions": ["鼻子堵", "鼻塞", "鼻子不通气"], "candidates": ["鼻塞"]},
    {"mentions": ["打喷嚏", "老打喷嚏"], "candidates": ["喷嚏"]},
    {"mentions": ["心慌", "心里发慌", "心跳很快"], "candidates": ["心悸"]},
    {"mentions": ["没胃口", "不想吃饭"], "candidates": ["食欲不振", "食欲减退"]},
    {"mentions": ["睡不着", "失眠"], "candidates": ["失眠"]},
    {"mentions": ["发冷", "怕冷", "打寒战"], "candidates": ["畏寒", "怕冷", "寒战"]},
    {"mentions": ["发痒", "痒得厉害"], "candidates": ["瘙痒"]},
    {"mentions": ["起疹子", "身上起疹子"], "candidates": ["皮疹", "丘疹"]},
    {"mentions": ["尿频", "老想上厕所"], "candidates": ["尿频"]},
    {"mentions": ["尿急", "憋不住尿"], "candidates": ["尿急"]},
    {"mentions": ["小便疼", "尿尿疼"], "candidates": ["尿痛"]},
]

DISEASE_ALIAS_GROUPS = [
    {"mentions": ["感冒", "着凉", "上感"], "candidates": ["感冒", "上呼吸道感染", "急性上呼吸道感染"]},
    {"mentions": ["肺炎"], "candidates": ["肺炎", "支气管肺炎"]},
    {"mentions": ["支气管炎"], "candidates": ["支气管炎", "急性支气管炎", "慢性支气管炎"]},
    {"mentions": ["鼻炎"], "candidates": ["鼻炎", "过敏性鼻炎"]},
    {"mentions": ["咽炎"], "candidates": ["咽炎", "急性咽炎", "慢性咽炎"]},
    {"mentions": ["胃炎"], "candidates": ["胃炎", "急性胃炎", "慢性胃炎"]},
    {"mentions": ["肠胃炎"], "candidates": ["肠胃炎", "急性胃肠炎"]},
    {"mentions": ["肠炎"], "candidates": ["肠炎"]},
    {"mentions": ["尿路感染", "尿感"], "candidates": ["尿路感染"]},
    {"mentions": ["中耳炎"], "candidates": ["中耳炎"]},
]

CHECK_ALIAS_GROUPS = [
    {"mentions": ["验血", "抽血"], "candidates": ["血常规"]},
    {"mentions": ["查大便", "便检"], "candidates": ["大便常规"]},
    {"mentions": ["查尿", "尿检"], "candidates": ["尿常规"]},
    {"mentions": ["胸片", "拍片"], "candidates": ["胸部X线检查", "X线检查"]},
    {"mentions": ["胸部ct", "ct"], "candidates": ["胸部CT检查", "CT检查"]},
    {"mentions": ["彩超", "b超"], "candidates": ["彩超", "B超"]},
    {"mentions": ["胃镜"], "candidates": ["胃镜"]},
    {"mentions": ["鼻镜"], "candidates": ["鼻镜", "鼻内镜"]},
    {"mentions": ["喉镜"], "candidates": ["喉镜"]},
]

DEPARTMENT_ALIAS_GROUPS = [
    {"mentions": ["儿科", "小儿科"], "candidates": ["儿科", "小儿内科"]},
    {"mentions": ["呼吸科"], "candidates": ["呼吸内科"]},
    {"mentions": ["消化科"], "candidates": ["消化内科"]},
    {"mentions": ["耳鼻喉", "耳鼻喉科"], "candidates": ["耳鼻喉科"]},
    {"mentions": ["皮肤科"], "candidates": ["皮肤科"]},
    {"mentions": ["妇科"], "candidates": ["妇科"]},
    {"mentions": ["眼科"], "candidates": ["眼科"]},
    {"mentions": ["口腔科"], "candidates": ["口腔科"]},
    {"mentions": ["内分泌"], "candidates": ["内分泌科"]},
    {"mentions": ["神经内科"], "candidates": ["神经内科"]},
]

DRUG_ALIAS_GROUPS = [
    {"mentions": ["退烧药"], "candidates": ["布洛芬", "对乙酰氨基酚"]},
    {"mentions": ["布洛芬"], "candidates": ["布洛芬"]},
    {"mentions": ["阿莫西林"], "candidates": ["阿莫西林"]},
    {"mentions": ["蒙脱石散"], "candidates": ["蒙脱石散"]},
    {"mentions": ["开塞露"], "candidates": ["开塞露"]},
]

MANUAL_HARD_CASES = [
    ("没有发烧，就是有点咳嗽", [("有点咳嗽", "symptom", ["咳嗽", "干咳"])]),
    ("不头晕，但胸口闷", [("胸口闷", "symptom", ["胸闷"])]),
    ("发烧、咳嗽、喉咙痛，还有点喘", [
        ("发烧", "symptom", ["发热", "发烧"]),
        ("咳嗽", "symptom", ["咳嗽"]),
        ("喉咙痛", "symptom", ["咽痛", "咽喉痛"]),
        ("喘", "symptom", ["呼吸困难", "喘息", "气短"]),
    ]),
    ("我前天淋了雨，昨天开始发烧，今天还头晕", [
        ("发烧", "symptom", ["发热", "发烧"]),
        ("头晕", "symptom", ["头晕", "眩晕"]),
    ]),
    ("浑身酸疼，像跑完步一样", [("浑身酸疼", "symptom", ["肌肉酸痛", "全身疼痛"])]),
    ("浑身哪都疼，一碰就难受", [("浑身哪都疼", "symptom", ["全身疼痛", "肌肉酸痛"])]),
    ("不是感冒吧，就是咳嗽流鼻涕", [
        ("咳嗽", "symptom", ["咳嗽"]),
        ("流鼻涕", "symptom", ["流涕"]),
    ]),
    ("胸口闷，爬两层楼就喘不上气", [
        ("胸口闷", "symptom", ["胸闷"]),
        ("喘不上气", "symptom", ["呼吸困难", "气促", "气短"]),
    ]),
    ("孩子发烧，还咳得厉害，要不要查血常规", [
        ("发烧", "symptom", ["发热", "发烧"]),
        ("咳得厉害", "symptom", ["咳嗽"]),
        ("血常规", "check", ["血常规"]),
    ]),
]

MANUAL_NEGATIVES = [
    "要不要空腹检查",
    "最近睡得不好",
    "明天体检今晚几点后不能吃东西",
    "化验单还没出来，要先挂号吗",
    "最近压力有点大，总想发呆",
    "这个检查需要预约吗",
    "抽血前能不能喝水",
    "先观察两天行不行",
    "今天就是想问问挂号流程",
    "我想问一下住院怎么报销",
]


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
    return re.sub(r"[\s,，。!！?？:：;；\"'“”‘’()（）\[\]【】<>《》/\\\-·]", "", text)


def valid_node(text, max_len=32):
    text = clean_text(text)
    if not text or len(text) > max_len:
        return False
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z0-9]", text))


def sort_nodes(nodes):
    return sorted(
        nodes,
        key=lambda item: (
            -TYPE_PRIORITY.get(item["node_type"], 0),
            item["node_name"],
            item["mention"],
        ),
    )


def dedupe_nodes(nodes):
    seen = set()
    output = []
    for node in sort_nodes(nodes):
        key = (node["mention"], node["node_name"], node["node_type"])
        if key in seen:
            continue
        seen.add(key)
        output.append(node)
    return output


def dump_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def dump_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_json_array(path, chunk_size=1024 * 1024):
    decoder = json.JSONDecoder()
    buffer = ""
    in_array = False

    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            buffer += chunk

            while True:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if not in_array:
                    if buffer[0] != "[":
                        raise ValueError(f"{path} 不是 JSON 数组。")
                    in_array = True
                    buffer = buffer[1:]
                    continue
                if buffer[0] == "]":
                    return
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue
                try:
                    item, offset = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                yield item
                buffer = buffer[offset:]

        buffer = buffer.lstrip()
        while buffer:
            if buffer[0] in {",", "]"}:
                buffer = buffer[1:].lstrip()
                continue
            item, offset = decoder.raw_decode(buffer)
            yield item
            buffer = buffer[offset:].lstrip()


def load_json_or_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            loaded = json.load(handle)
            return loaded if isinstance(loaded, list) else [loaded]
        return [json.loads(line) for line in handle if line.strip()]


def read_symptom_norms(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [clean_text(row.get("norm")) for row in reader if clean_text(row.get("norm"))]


def is_negated(text, start):
    left = clean_text(text[max(0, start - 6):start]).replace(" ", "")
    if any(left.endswith(prefix) for prefix in SPECULATIVE_PREFIXES):
        return False
    return any(left.endswith(prefix) for prefix in NEGATION_PREFIXES)


def remove_bracket_text(text):
    return re.sub(r"[（(][^()（）]{0,24}[)）]", "", clean_text(text)).strip()


def choose_best_candidate(candidates, node_pool, symptom_counter=None):
    valid = [candidate for candidate in candidates if candidate in node_pool]
    if not valid:
        return ""
    if symptom_counter:
        valid.sort(key=lambda item: (-symptom_counter[item], len(item), item))
    else:
        valid.sort(key=lambda item: (len(item), item))
    return valid[0]


class AliasTrie:
    def __init__(self):
        self.root = {}

    def add(self, alias, payload):
        node = self.root
        for char in alias:
            node = node.setdefault(char, {})
        node.setdefault("_end", []).append(payload)

    def scan(self, text):
        matches = []
        for start in range(len(text)):
            node = self.root
            end = start
            while end < len(text) and text[end] in node:
                node = node[text[end]]
                end += 1
                for payload in node.get("_end", []):
                    matches.append(
                        {
                            "mention": text[start:end],
                            "node_name": payload["node_name"],
                            "node_type": payload["node_type"],
                            "priority": payload["priority"],
                            "start": start,
                            "end": end,
                        }
                    )
        return matches


class MedicalNodeIndex:
    def __init__(self, medical_records, symptom_norms=None):
        self.records = medical_records
        self.symptom_norms = symptom_norms or []
        self.nodes = {
            "disease": [],
            "symptom": [],
            "department": [],
            "check": [],
            "drug": [],
        }
        self.node_sets = {
            "disease": set(),
            "symptom": set(),
            "department": set(),
            "check": set(),
            "drug": set(),
        }
        self.norm_to_nodes = {
            "disease": defaultdict(set),
            "symptom": defaultdict(set),
            "department": defaultdict(set),
            "check": defaultdict(set),
            "drug": defaultdict(set),
        }
        self.surface_forms = defaultdict(list)
        self.symptom_counter = Counter()
        self.alias_trie = AliasTrie()
        self.alias_entries = []
        self._build()

    def _good_symptom(self, symptom):
        symptom = clean_text(symptom)
        if not valid_node(symptom, max_len=24):
            return False
        if len(symptom) < 2:
            return False
        if any(keyword in symptom for keyword in NOISY_SYMPTOM_KEYWORDS):
            return False
        return True

    def _register_surface(self, node_type, node_name, value):
        value = clean_text(value)
        if value and value not in self.surface_forms[(node_type, node_name)]:
            self.surface_forms[(node_type, node_name)].append(value)

    def _register_alias(self, alias, node_name, node_type, priority=10):
        alias = clean_text(alias)
        if not alias or not valid_node(alias, max_len=24):
            return
        payload = {
            "node_name": node_name,
            "node_type": node_type,
            "priority": priority,
        }
        self.alias_trie.add(alias, payload)
        self.alias_entries.append((alias, node_name, node_type, priority))
        self._register_surface(node_type, node_name, alias)

    def _register_node(self, node_type, value):
        value = clean_text(value)
        if not valid_node(value):
            return
        if value not in self.node_sets[node_type]:
            self.node_sets[node_type].add(value)
            self.nodes[node_type].append(value)
        self.norm_to_nodes[node_type][normalize_text(value)].add(value)

    def _register_variant_aliases(self):
        disease_variants = defaultdict(set)
        check_variants = defaultdict(set)

        for disease in self.nodes["disease"]:
            stripped = remove_bracket_text(disease)
            if stripped and stripped != disease:
                disease_variants[stripped].add(disease)
            for prefix in DISEASE_PREFIXES:
                if disease.startswith(prefix):
                    short = disease[len(prefix):].strip()
                    if len(short) >= 2:
                        disease_variants[short].add(disease)

        for alias, diseases in disease_variants.items():
            if len(diseases) == 1:
                self._register_alias(alias, next(iter(diseases)), "disease", priority=20)

        for check in self.nodes["check"]:
            stripped = remove_bracket_text(check)
            if stripped and stripped != check:
                check_variants[stripped].add(check)
            for suffix in ["检查", "检验", "测定", "常规"]:
                if check.endswith(suffix):
                    short = check[:-len(suffix)].strip()
                    if len(short) >= 2:
                        check_variants[short].add(check)

        for alias, checks in check_variants.items():
            if len(checks) == 1:
                self._register_alias(alias, next(iter(checks)), "check", priority=20)

    def _register_manual_alias_groups(self):
        groups = [
            ("symptom", SYMPTOM_ALIAS_GROUPS),
            ("disease", DISEASE_ALIAS_GROUPS),
            ("check", CHECK_ALIAS_GROUPS),
            ("department", DEPARTMENT_ALIAS_GROUPS),
            ("drug", DRUG_ALIAS_GROUPS),
        ]
        for node_type, group_list in groups:
            for group in group_list:
                node_name = choose_best_candidate(
                    group["candidates"],
                    self.node_sets[node_type],
                    self.symptom_counter if node_type == "symptom" else None,
                )
                if not node_name:
                    continue
                for mention in group["mentions"]:
                    self._register_alias(mention, node_name, node_type, priority=1)

    def _build(self):
        for record in tqdm(self.records, desc="构建全量节点索引", unit="record"):
            disease = clean_text(record.get("name"))
            if disease:
                self._register_node("disease", disease)

            for symptom in listify(record.get("symptom")):
                if self._good_symptom(symptom):
                    self._register_node("symptom", symptom)
                    self.symptom_counter[symptom] += 1

            for department in listify(record.get("cure_department")):
                self._register_node("department", department)

            for check in listify(record.get("check")):
                self._register_node("check", check)

            for drug in dedupe_preserve(listify(record.get("common_drug")) + listify(record.get("recommand_drug"))):
                self._register_node("drug", drug)

        for symptom_norm in self.symptom_norms:
            if self._good_symptom(symptom_norm):
                self._register_node("symptom", symptom_norm)

        for node_type, values in self.nodes.items():
            values.sort(key=lambda item: (len(item), item))
            for value in values:
                self._register_alias(value, value, node_type, priority=50)

        self._register_variant_aliases()
        self._register_manual_alias_groups()

    def lookup(self, node_type, text):
        text = clean_text(text)
        if not text:
            return ""
        exact = list(self.norm_to_nodes[node_type].get(normalize_text(text), []))
        if exact:
            return choose_best_candidate(
                exact,
                self.node_sets[node_type],
                self.symptom_counter if node_type == "symptom" else None,
            )
        candidates = []
        norm_text = normalize_text(text)
        for node in self.nodes[node_type]:
            norm_node = normalize_text(node)
            if not norm_text or not norm_node:
                continue
            if norm_text in norm_node or norm_node in norm_text:
                candidates.append(node)
        return choose_best_candidate(
            candidates,
            self.node_sets[node_type],
            self.symptom_counter if node_type == "symptom" else None,
        )

    def map_symptom_norm(self, symptom_norm):
        return self.lookup("symptom", symptom_norm)

    def choose_surface(self, node_type, node_name, rng):
        values = self.surface_forms.get((node_type, node_name), [])
        if values:
            return rng.choice(values[: min(len(values), 8)])
        return node_name

    def _resolve_matches(self, matches):
        ordered = sorted(
            matches,
            key=lambda item: (
                item["start"],
                -(item["end"] - item["start"]),
                -TYPE_PRIORITY.get(item["node_type"], 0),
                item["priority"],
                item["node_name"],
            ),
        )
        kept = []
        for candidate in ordered:
            replaced = False
            dropped = False
            for index, existing in enumerate(kept):
                same_span = candidate["start"] == existing["start"] and candidate["end"] == existing["end"]
                overlap = not (candidate["end"] <= existing["start"] or candidate["start"] >= existing["end"])
                if same_span:
                    if TYPE_PRIORITY.get(candidate["node_type"], 0) > TYPE_PRIORITY.get(existing["node_type"], 0):
                        kept[index] = candidate
                    dropped = True
                    break
                if overlap and candidate["node_type"] == existing["node_type"]:
                    current_len = candidate["end"] - candidate["start"]
                    existing_len = existing["end"] - existing["start"]
                    if current_len > existing_len:
                        kept[index] = candidate
                        replaced = True
                    dropped = True
                    break
            if not dropped or replaced:
                if not replaced:
                    kept.append(candidate)
        return sorted(kept, key=lambda item: (item["start"], item["end"], item["node_type"], item["node_name"]))

    def find_nodes_in_text(self, text, allowed_types=None):
        text = clean_text(text)
        if not text:
            return []
        raw_matches = []
        for match in self.alias_trie.scan(text):
            if allowed_types and match["node_type"] not in allowed_types:
                continue
            if is_negated(text, match["start"]):
                continue
            raw_matches.append(match)
        resolved = self._resolve_matches(raw_matches)
        return dedupe_nodes(
            [
                {
                    "mention": item["mention"],
                    "node_name": item["node_name"],
                    "node_type": item["node_type"],
                }
                for item in resolved
            ]
        )


def make_example(text, nodes, source, template, allow_empty=False):
    text = clean_text(text)
    nodes = dedupe_nodes(nodes)
    if not text:
        return None
    if not allow_empty and not nodes:
        return None
    return {
        "text": text,
        "nodes": nodes,
        "sources": [source],
        "templates": [template],
    }


def merge_examples(examples):
    merged = {}
    for example in examples:
        if not example:
            continue
        key = example["text"]
        if key not in merged:
            merged[key] = {
                "text": example["text"],
                "nodes": list(example["nodes"]),
                "sources": list(example["sources"]),
                "templates": list(example["templates"]),
            }
            continue
        merged[key]["nodes"] = dedupe_nodes(merged[key]["nodes"] + example["nodes"])
        merged[key]["sources"] = dedupe_preserve(merged[key]["sources"] + example["sources"])
        merged[key]["templates"] = dedupe_preserve(merged[key]["templates"] + example["templates"])
    return list(merged.values())


def extract_bio_mentions(sentence, bio_label):
    sentence = clean_text(sentence)
    labels = clean_text(bio_label).split()
    chars = list(sentence)
    if not sentence or not labels or len(chars) != len(labels):
        return []

    mentions = []
    start = None
    current_type = None
    for index, label in enumerate(labels):
        if label == "O":
            if start is not None:
                mentions.append(
                    {
                        "mention": "".join(chars[start:index]),
                        "label_type": current_type,
                    }
                )
                start = None
                current_type = None
            continue

        if "-" not in label:
            continue

        prefix, label_type = label.split("-", 1)
        if prefix == "B" or start is None or label_type != current_type:
            if start is not None:
                mentions.append(
                    {
                        "mention": "".join(chars[start:index]),
                        "label_type": current_type,
                    }
                )
            start = index
            current_type = label_type

    if start is not None:
        mentions.append({"mention": "".join(chars[start:]), "label_type": current_type})
    return mentions


def build_imcs_examples(index, imcs_paths, max_records=0):
    examples = []
    total_seen = 0
    progress = tqdm(desc="解析 IMCS 问诊数据", unit="record")

    for file_path in imcs_paths:
        if not file_path.exists():
            continue
        with file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        for record in data.values():
            if max_records and total_seen >= max_records:
                progress.close()
                return examples
            total_seen += 1
            progress.update(1)

            self_report = clean_text(record.get("self_report"))
            if self_report:
                nodes = index.find_nodes_in_text(self_report)
                disease_name = clean_text(record.get("diagnosis"))
                if disease_name and disease_name in self_report:
                    mapped = index.lookup("disease", disease_name)
                    if mapped:
                        nodes.append({"mention": disease_name, "node_name": mapped, "node_type": "disease"})
                example = make_example(self_report, dedupe_nodes(nodes), "imcs", "self_report", allow_empty=False)
                if example:
                    examples.append(example)

            patient_sentences = []
            patient_nodes = []
            for turn in record.get("dialogue", []):
                speaker = clean_text(turn.get("speaker"))
                if speaker and "患" not in speaker and "病" not in speaker and "用户" not in speaker:
                    continue
                sentence = clean_text(turn.get("sentence"))
                if not sentence:
                    continue

                local_nodes = []
                bio_mentions = extract_bio_mentions(sentence, turn.get("BIO_label", ""))
                symptom_norms = [index.map_symptom_norm(item) for item in turn.get("symptom_norm", [])]
                symptom_norms = [item for item in symptom_norms if item]
                symptom_index = 0

                for mention_info in bio_mentions:
                    mention = clean_text(mention_info["mention"])
                    node_type = BIO_TO_NODE_TYPE.get(mention_info["label_type"])
                    if not mention or not node_type:
                        continue
                    node_name = ""
                    if node_type == "symptom":
                        if symptom_index < len(symptom_norms):
                            node_name = symptom_norms[symptom_index]
                            symptom_index += 1
                        if not node_name:
                            node_name = index.lookup("symptom", mention)
                    else:
                        node_name = index.lookup(node_type, mention)
                    if node_name:
                        local_nodes.append({"mention": mention, "node_name": node_name, "node_type": node_type})

                local_nodes.extend(index.find_nodes_in_text(sentence))
                local_nodes = dedupe_nodes(local_nodes)
                example = make_example(sentence, local_nodes, "imcs", "patient_turn", allow_empty=False)
                if example:
                    examples.append(example)

                if local_nodes:
                    patient_sentences.append(sentence)
                    patient_nodes.extend(local_nodes)

            if patient_sentences and patient_nodes:
                combined = "，".join(patient_sentences[-3:])
                if 6 <= len(combined) <= 220:
                    example = make_example(combined, patient_nodes, "imcs", "patient_combined", allow_empty=False)
                    if example:
                        examples.append(example)

    progress.close()
    return examples


def infer_dialogue_query(row):
    candidates = [
        row.get("input"),
        row.get("ask"),
        row.get("question"),
        row.get("query"),
        row.get("title"),
        row.get("instruction"),
    ]
    for candidate in candidates:
        text = clean_text(candidate)
        if text:
            return text
    return ""


def build_dialogue_examples(index, dialogue_json_path, max_rows=0, max_negative=8000):
    positives = []
    negatives = []
    row_iter = iter_json_array(dialogue_json_path)
    progress = tqdm(desc="扫描中文医疗对话", unit="row")

    for row_index, row in enumerate(row_iter, start=1):
        if max_rows and row_index > max_rows:
            break
        progress.update(1)

        query = infer_dialogue_query(row)
        if not query or len(query) < 6 or len(query) > 280:
            continue

        nodes = index.find_nodes_in_text(query)
        if nodes:
            example = make_example(query, nodes, "dialogue", "real_query")
            if example:
                positives.append(example)
        elif len(negatives) < max_negative:
            example = make_example(query, [], "dialogue", "real_negative", allow_empty=True)
            if example:
                negatives.append(example)

    progress.close()
    return positives, negatives


def build_manual_examples(index):
    examples = []
    for text, node_specs in MANUAL_HARD_CASES:
        nodes = []
        for mention, node_type, candidates in node_specs:
            node_name = choose_best_candidate(
                [index.lookup(node_type, item) for item in candidates] + [index.lookup(node_type, mention)],
                index.node_sets[node_type],
                index.symptom_counter if node_type == "symptom" else None,
            )
            if node_name:
                nodes.append({"mention": mention, "node_name": node_name, "node_type": node_type})
        example = make_example(text, nodes, "manual", "hard_case")
        if example:
            examples.append(example)

    for text in MANUAL_NEGATIVES:
        example = make_example(text, [], "manual", "negative", allow_empty=True)
        if example:
            examples.append(example)
    return examples


def build_disease_profile_examples(index, records, seed):
    rng = random.Random(seed)
    examples = []
    subjects = ["我", "我这两天", "我最近", "孩子", "家里老人"]
    endings = ["这是怎么回事", "严重吗", "要不要去医院", "会不会有问题"]

    for record in tqdm(records, desc="生成疾病画像样本", unit="record"):
        disease = clean_text(record.get("name"))
        disease_node = index.lookup("disease", disease)
        if not disease_node:
            continue

        symptoms = [item for item in listify(record.get("symptom")) if item in index.node_sets["symptom"]]
        checks = [item for item in listify(record.get("check")) if item in index.node_sets["check"]]
        departments = [item for item in listify(record.get("cure_department")) if item in index.node_sets["department"]]
        drugs = [
            item for item in dedupe_preserve(listify(record.get("common_drug")) + listify(record.get("recommand_drug")))
            if item in index.node_sets["drug"]
        ]

        if not symptoms:
            continue

        picked_symptoms = symptoms[: min(3, len(symptoms))]
        symptom_nodes = []
        for symptom in picked_symptoms:
            mention = index.choose_surface("symptom", symptom, rng)
            symptom_nodes.append({"mention": mention, "node_name": symptom, "node_type": "symptom"})

        subject = rng.choice(subjects)
        disease_mention = index.choose_surface("disease", disease_node, rng)
        examples.append(
            make_example(
                f"{subject}{symptom_nodes[0]['mention']}，会不会是{disease_mention}",
                symptom_nodes[:1] + [{"mention": disease_mention, "node_name": disease_node, "node_type": "disease"}],
                "synthetic",
                "disease_guess",
            )
        )

        if len(symptom_nodes) >= 2:
            examples.append(
                make_example(
                    f"{subject}{symptom_nodes[0]['mention']}，还有{symptom_nodes[1]['mention']}，{rng.choice(endings)}",
                    symptom_nodes[:2],
                    "synthetic",
                    "symptom_pair",
                )
            )

        if departments:
            department = departments[0]
            dept_mention = index.choose_surface("department", department, rng)
            examples.append(
                make_example(
                    f"{subject}{symptom_nodes[0]['mention']}，该挂{dept_mention}吗",
                    symptom_nodes[:1] + [{"mention": dept_mention, "node_name": department, "node_type": "department"}],
                    "synthetic",
                    "department_question",
                )
            )

        if checks:
            check = checks[0]
            check_mention = index.choose_surface("check", check, rng)
            examples.append(
                make_example(
                    f"{subject}{symptom_nodes[0]['mention']}，要不要做{check_mention}",
                    symptom_nodes[:1] + [{"mention": check_mention, "node_name": check, "node_type": "check"}],
                    "synthetic",
                    "check_question",
                )
            )

        if drugs:
            drug = drugs[0]
            drug_mention = index.choose_surface("drug", drug, rng)
            examples.append(
                make_example(
                    f"{subject}{symptom_nodes[0]['mention']}，能不能先吃{drug_mention}",
                    symptom_nodes[:1] + [{"mention": drug_mention, "node_name": drug, "node_type": "drug"}],
                    "synthetic",
                    "drug_question",
                )
            )

    return [example for example in examples if example]


def build_full_coverage_fill_examples(index, seed):
    rng = random.Random(seed)
    examples = []

    for symptom in tqdm(index.nodes["symptom"], desc="补齐 symptom 覆盖", unit="node"):
        mention = index.choose_surface("symptom", symptom, rng)
        examples.append(make_example(f"这两天一直{mention}", [{"mention": mention, "node_name": symptom, "node_type": "symptom"}], "fill", "symptom_fill"))

    for disease in tqdm(index.nodes["disease"], desc="补齐 disease 覆盖", unit="node"):
        mention = index.choose_surface("disease", disease, rng)
        examples.append(make_example(f"我这样像不像{mention}", [{"mention": mention, "node_name": disease, "node_type": "disease"}], "fill", "disease_fill"))

    for check in tqdm(index.nodes["check"], desc="补齐 check 覆盖", unit="node"):
        mention = index.choose_surface("check", check, rng)
        examples.append(make_example(f"这种情况要不要查{mention}", [{"mention": mention, "node_name": check, "node_type": "check"}], "fill", "check_fill"))

    for department in tqdm(index.nodes["department"], desc="补齐 department 覆盖", unit="node"):
        mention = index.choose_surface("department", department, rng)
        examples.append(make_example(f"这种情况挂{mention}吗", [{"mention": mention, "node_name": department, "node_type": "department"}], "fill", "department_fill"))

    for drug in tqdm(index.nodes["drug"], desc="补齐 drug 覆盖", unit="node"):
        mention = index.choose_surface("drug", drug, rng)
        examples.append(make_example(f"{mention}能不能先吃", [{"mention": mention, "node_name": drug, "node_type": "drug"}], "fill", "drug_fill"))

    return [example for example in examples if example]


def build_stats(examples, index):
    source_counts = Counter()
    template_counts = Counter()
    node_type_counts = Counter()
    covered_nodes = defaultdict(set)

    for example in examples:
        for source in example["sources"]:
            source_counts[source] += 1
        for template in example["templates"]:
            template_counts[template] += 1
        for node in example["nodes"]:
            node_type_counts[node["node_type"]] += 1
            covered_nodes[node["node_type"]].add(node["node_name"])

    coverage = {}
    for node_type in ["disease", "symptom", "department", "check", "drug"]:
        total = len(index.nodes[node_type])
        covered = len(covered_nodes[node_type])
        uncovered = [node for node in index.nodes[node_type] if node not in covered_nodes[node_type]]
        coverage[node_type] = {
            "covered": covered,
            "total": total,
            "coverage_ratio": round(covered / total, 4) if total else 0.0,
            "uncovered_preview": uncovered[:50],
        }

    return {
        "dataset_size": len(examples),
        "negative_samples": sum(1 for example in examples if not example["nodes"]),
        "node_type_counts": dict(sorted(node_type_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "template_counts": dict(sorted(template_counts.items())),
        "coverage": coverage,
        "kg_node_totals": {node_type: len(index.nodes[node_type]) for node_type in index.nodes},
    }


def to_sharegpt(examples):
    result = []
    for example in examples:
        result.append(
            {
                "system": SYSTEM_PROMPT,
                "conversations": [
                    {
                        "from": "human",
                        "value": example["text"],
                    },
                    {
                        "from": "gpt",
                        "value": json.dumps({"nodes": example["nodes"]}, ensure_ascii=False),
                    },
                ],
            }
        )
    return result


def export_clean_kg_source(records, output_path):
    cleaned = []
    seen = set()
    for record in records:
        disease = clean_text(record.get("name"))
        if not disease or disease in seen:
            continue
        seen.add(disease)
        cleaned.append(record)
    dump_jsonl(output_path, cleaned)
    return len(cleaned)


def maybe_register_dataset(dataset_info_path, dataset_name, file_name):
    with dataset_info_path.open("r", encoding="utf-8") as handle:
        dataset_info = json.load(handle)
    dataset_info[dataset_name] = {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "system": "system",
        },
    }
    with dataset_info_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset_info, handle, ensure_ascii=False, indent=2)


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="构建全量知识图谱覆盖的 query-to-graph 数据集。")
    parser.add_argument("--medical-json", default=str(repo_root / "algorithm" / "data" / "raw" / "medical.json"))
    parser.add_argument("--imcs-train", default=str(repo_root / "algorithm" / "data" / "raw" / "imcs21" / "IMCS-V2_train.json"))
    parser.add_argument("--imcs-dev", default=str(repo_root / "algorithm" / "data" / "raw" / "imcs21" / "IMCS-V2_dev.json"))
    parser.add_argument("--imcs-test", default=str(repo_root / "algorithm" / "data" / "raw" / "imcs21" / "IMCS-V2_test.json"))
    parser.add_argument("--imcs-symptom-norm", default=str(repo_root / "algorithm" / "data" / "raw" / "imcs21" / "symptom_norm.csv"))
    parser.add_argument("--dialogue-json", default=str(repo_root / "algorithm" / "data" / "raw" / "train_0001_of_0001.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-medical-records", type=int, default=0, help="0 表示使用全部 medical 记录。")
    parser.add_argument("--max-imcs-records", type=int, default=0, help="0 表示使用全部 IMCS 记录。")
    parser.add_argument("--max-dialogue-rows", type=int, default=0, help="0 表示扫描全部中文医疗对话行。")
    parser.add_argument("--max-dialogue-negative", type=int, default=8000)
    parser.add_argument("--output-annotations", default=str(repo_root / "algorithm" / "data" / "processed" / "query2graph_full_annotations.json"))
    parser.add_argument("--output-sharegpt", default=str(repo_root / "algorithm" / "data" / "llamafactory" / "query2graph_full_sharegpt.json"))
    parser.add_argument("--output-llamafactory", default=str(repo_root / "LLaMA-Factory" / "data" / "query2graph_full_sharegpt.json"))
    parser.add_argument("--output-stats", default=str(repo_root / "algorithm" / "data" / "processed" / "query2graph_full_stats.json"))
    parser.add_argument("--output-kg-source", default=str(repo_root / "algorithm" / "data" / "processed" / "medical_full_kg_source.jsonl"))
    parser.add_argument("--register-dataset", action="store_true")
    parser.add_argument("--dataset-name", default="query2graph_full_sharegpt")
    parser.add_argument("--dataset-info", default=str(repo_root / "LLaMA-Factory" / "data" / "dataset_info.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    medical_records = load_json_or_jsonl(Path(args.medical_json))
    if args.max_medical_records > 0:
        medical_records = medical_records[: args.max_medical_records]

    symptom_norms = read_symptom_norms(Path(args.imcs_symptom_norm))
    index = MedicalNodeIndex(medical_records, symptom_norms=symptom_norms)

    imcs_examples = build_imcs_examples(
        index,
        [
            Path(args.imcs_train),
            Path(args.imcs_dev),
            Path(args.imcs_test),
        ],
        max_records=args.max_imcs_records,
    )

    dialogue_positives, dialogue_negatives = build_dialogue_examples(
        index,
        Path(args.dialogue_json),
        max_rows=args.max_dialogue_rows,
        max_negative=args.max_dialogue_negative,
    )

    manual_examples = build_manual_examples(index)
    disease_profile_examples = build_disease_profile_examples(index, medical_records, seed=args.seed)
    fill_examples = build_full_coverage_fill_examples(index, seed=args.seed + 7)

    merged_examples = merge_examples(
        manual_examples
        + imcs_examples
        + dialogue_positives
        + dialogue_negatives
        + disease_profile_examples
        + fill_examples
    )
    merged_examples.sort(key=lambda item: (item["text"], len(item["nodes"])))

    sharegpt_examples = to_sharegpt(merged_examples)
    stats = build_stats(merged_examples, index)
    export_count = export_clean_kg_source(medical_records, Path(args.output_kg_source))
    stats["kg_source_records"] = export_count

    dump_json(Path(args.output_annotations), merged_examples)
    dump_json(Path(args.output_sharegpt), sharegpt_examples)

    output_llamafactory = Path(args.output_llamafactory)
    output_llamafactory.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(args.output_sharegpt), output_llamafactory)

    dump_json(Path(args.output_stats), stats)

    if args.register_dataset:
        maybe_register_dataset(
            dataset_info_path=Path(args.dataset_info),
            dataset_name=args.dataset_name,
            file_name=output_llamafactory.name,
        )

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"annotations -> {args.output_annotations}")
    print(f"sharegpt -> {args.output_sharegpt}")
    print(f"llamafactory copy -> {args.output_llamafactory}")
    print(f"stats -> {args.output_stats}")
    print(f"kg source -> {args.output_kg_source}")


if __name__ == "__main__":
    main()
