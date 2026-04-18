import argparse
import time
from pathlib import Path

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    from modelscope import AutoModelForCausalLM, AutoTokenizer


DEFAULT_SYSTEM_PROMPT = (
    "你是一个中文医疗问诊 query 解析助手。请把用户口语化、模糊化的问诊表达映射到医疗知识图谱节点。重点识别症状，也可以识别疾病、科室、检查和药物。只返回严格 JSON，不要输出解释。格式为：{\"nodes\": [{\"mention\": str, \"node_name\": str, \"node_type\": str}]}。node_type 只允许为 symptom, disease, department, check, drug。"
)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="与微调后的 Qwen3-4B-Instruct 模型进行本地对话测试。")
    parser.add_argument(
        "--model-path",
        default=str(root / "fine_tuning_4B"),
        help="微调模型目录，默认指向项目根目录下的 fine_tuning_4B。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="单轮最大生成 token 数。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="采样温度，设为 0 表示贪心解码。",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="top-p 采样参数。",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="系统提示词。",
    )
    return parser.parse_args()


def build_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def build_messages(history, user_input, system_prompt):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for user_text, assistant_text in history:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})

    messages.append({"role": "user", "content": user_input})
    return messages


def render_prompt(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def generate_reply(tokenizer, model, messages, max_new_tokens, temperature, top_p):
    prompt_text = render_prompt(tokenizer, messages)
    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature <= 0:
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    start_time = time.perf_counter()
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, **generation_kwargs)
    elapsed = time.perf_counter() - start_time

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    generated_tokens = int(output_ids.shape[0])
    tokens_per_second = generated_tokens / elapsed if elapsed > 0 else 0.0
    return {
        "content": content,
        "elapsed": elapsed,
        "generated_tokens": generated_tokens,
        "tokens_per_second": tokens_per_second,
    }


def print_help():
    print("命令：")
    print("  /exit      退出")
    print("  /clear     清空对话历史")
    print("  /history   查看当前历史轮次")
    print("  /system    查看当前系统提示词")


def main():
    args = parse_args()
    tokenizer, model = build_model(args.model_path)

    history = []
    print(f"模型已加载：{args.model_path}")
    print(f"设备：{model.device}")
    print_help()

    while True:
        try:
            user_input = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not user_input:
            continue
        if user_input == "/exit":
            print("退出。")
            break
        if user_input == "/clear":
            history.clear()
            print("历史已清空。")
            continue
        if user_input == "/history":
            print(f"当前历史轮次：{len(history)}")
            continue
        if user_input == "/system":
            print(args.system_prompt)
            continue

        messages = build_messages(history, user_input, args.system_prompt)
        result = generate_reply(
            tokenizer=tokenizer,
            model=model,
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        print(f"\n模型：{result['content']}")
        print(
            f"[性能] 耗时 {result['elapsed']:.2f}s | "
            f"生成 {result['generated_tokens']} tokens | "
            f"{result['tokens_per_second']:.2f} tokens/s"
        )
        history.append((user_input, result["content"]))


if __name__ == "__main__":
    main()
