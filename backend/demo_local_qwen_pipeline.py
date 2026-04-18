import argparse
import json
import sys
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parent
API_PATH = "/api/diagnose/local-qwen-pipeline"
HEALTH_PATH = "/api/diagnose/local-qwen-pipeline/health"


def print_json(payload):
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def print_model_outputs(payload: dict):
    model_outputs = payload.get("model_outputs")
    if not isinstance(model_outputs, dict):
        return

    print("\n=== ner_model ===")
    print_json(model_outputs.get("ner_model", {}))
    print("\n=== reason_model ===")
    print_json(model_outputs.get("reason_model", {}))


def call_api(base_url: str, query: str, history: str, limit: int) -> dict:
    response = requests.post(
        f"{base_url.rstrip('/')}{API_PATH}",
        json={"query": query, "history": history, "limit": limit},
        timeout=600,
    )
    if not response.ok:
        try:
            detail = response.json()
        except ValueError:
            detail = {"error": response.text}
        raise requests.HTTPError(
            f"{response.status_code} {response.reason} for {response.url}\n"
            f"{json.dumps(detail, ensure_ascii=False, indent=2)}",
            response=response,
        )
    return response.json()


def get_api_health(base_url: str) -> dict:
    response = requests.get(f"{base_url.rstrip('/')}{HEALTH_PATH}", timeout=30)
    if not response.ok:
        try:
            detail = response.json()
        except ValueError:
            detail = {"error": response.text}
        raise requests.HTTPError(
            f"{response.status_code} {response.reason} for {response.url}\n"
            f"{json.dumps(detail, ensure_ascii=False, indent=2)}",
            response=response,
        )
    return response.json()


def call_direct(query: str, history: str, limit: int) -> dict:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from app.services import get_local_medical_pipeline

    pipeline = get_local_medical_pipeline()
    return pipeline.run(query, history=history, limit=limit)


def run_once(mode: str, base_url: str, query: str, history: str, limit: int) -> dict:
    if mode == "direct":
        return call_direct(query, history, limit)
    return call_api(base_url, query, history, limit)


def run_interactive(mode: str, base_url: str, history: str, limit: int):
    print("输入 /exit 退出，输入 /history 重新设置病史。")
    current_history = history
    while True:
        query = input("\n请输入问诊 query: ").strip()
        if not query:
            continue
        if query == "/exit":
            break
        if query == "/history":
            current_history = input("请输入新的病史: ").strip()
            print(f"当前病史已更新为: {current_history or '空'}")
            continue

        result = run_once(mode, base_url, query, current_history, limit)
        print_model_outputs(result)
        print_json(result)


def main():
    parser = argparse.ArgumentParser(description="联调本地 Qwen 医疗推理链路")
    parser.add_argument("--mode", choices=["api", "direct"], default="api", help="默认走后端 HTTP 接口")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000", help="后端服务地址")
    parser.add_argument("--query", default="", help="单轮测试的用户 query")
    parser.add_argument("--history", default="", help="补充病史")
    parser.add_argument("--limit", type=int, default=5, help="候选疾病数量")
    parser.add_argument("--skip-health", action="store_true", help="跳过接口健康检查")
    args = parser.parse_args()

    if args.mode == "api" and not args.skip_health:
        health = get_api_health(args.base_url)
        print("接口健康状态:")
        print_json(health)

    if args.query:
        result = run_once(args.mode, args.base_url, args.query, args.history, args.limit)
        print_model_outputs(result)
        print_json(result)
        return

    run_interactive(args.mode, args.base_url, args.history, args.limit)


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print(f"接口调用失败: {exc}")
        print("请确认后端已启动，并且在启动前配置好 MED_NER_MODEL_PATH、MED_REASON_MODEL_PATH、NEO4J_PASSWORD。")
        print("启动命令: python D:\\Yudou\\med_llm\\backend\\run.py")
        raise SystemExit(1)
    except KeyboardInterrupt:
        raise SystemExit(0)
