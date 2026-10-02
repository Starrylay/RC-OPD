"""Run the original OPSD Avg@4 prompt, sampler and scorer for each benchmark."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import hashlib

ROOT = Path(__file__).resolve().parent
SIZES = ("1.7B", "4B", "8B")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_adapter(request, checkpoint):
    if request is None or Path(request.lora_path).resolve() != checkpoint:
        raise RuntimeError("Requested LoRA adapter was not applied")

DATASETS = ("aime24", "aime25", "hmmt25")
PROMPT = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{}."
SETTINGS = {"val_n": 4, "enable_thinking": True, "temperature": 1.0, "top_p": .95,
            "top_k": -1, "min_p": 0.0, "presence_penalty": 0.0, "max_new_tokens": 38912}


def summarize(output):
    scores = {}
    for dataset in DATASETS:
        value = json.loads((Path(output) / (dataset + ".json")).read_text())
        if not all(value[k] == v for k, v in SETTINGS.items()):
            raise ValueError("Unexpected evaluation settings")
        if len(value["results"]) != 30 or value["num_problems"] != 30:
            raise ValueError("Incomplete benchmark")
        if not all(len(r["generations"]) == 4 for r in value["results"]):
            raise ValueError("Expected four generations per problem")
        correct = sum(bool(g["correct"]) for r in value["results"] for g in r["generations"])
        if correct != value["average_at_n"]:
            raise ValueError("Inconsistent correctness totals")
        scores[dataset] = correct / 120 * 100
    result = {"metric": "Avg@4 (%)", **scores, "mean": sum(scores.values()) / 3}
    (Path(output) / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def worker(args):
    import os
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    spec = importlib.util.spec_from_file_location("official_opsd", ROOT / "evaluation/official_opsd.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    checkpoint = Path(args.checkpoint).resolve()
    weights = checkpoint / "adapter_model.safetensors"
    digest = sha256(weights)
    original_load = evaluator.load_vllm_model

    def checked_load(base_model_path, lora_adapter_path=None, **kwargs):
        if not lora_adapter_path or Path(lora_adapter_path).resolve() != checkpoint:
            raise ValueError("Evaluation must load the requested adapter")
        engine, tokenizer = original_load(base_model_path, lora_adapter_path, **kwargs)
        original_generate = engine.generate
        def checked_generate(*a, **kw):
            request = kw.get("lora_request")
            require_adapter(request, checkpoint)
            return original_generate(*a, **kw)
        engine.generate = checked_generate
        return engine, tokenizer
    evaluator.load_vllm_model = checked_load
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    target = out / (args.dataset + ".json")
    if target.exists():
        raise ValueError("Use an empty evaluation output directory")
    sys.argv = [str(ROOT / "evaluation/official_opsd.py"),
                "--base_model", args.model_path, "--checkpoint_dir", str(checkpoint),
                "--dataset", args.dataset, "--enable_thinking", "--val_n", "4",
                "--temperature", "1", "--top_p", ".95", "--top_k", "-1", "--min_p", "0",
                "--presence_penalty", "0", "--max_new_tokens", "38912", "--max_model_len", "40960",
                "--tensor_parallel_size", str(args.tensor_parallel_size),
                "--gpu_memory_utilization", str(args.gpu_memory_utilization),
                "--output_file", str(target)]
    evaluator.main()
    if sha256(weights) != digest:
        raise RuntimeError("Adapter changed during evaluation")
    (out / (args.dataset + ".protocol.json")).write_text(json.dumps({
        "adapter_sha256": digest, "prompt": PROMPT, "settings": SETTINGS,
        "max_model_len": 40960, "scoring": "official OPSD; no LLM regrading",
        "evaluator_sha256": sha256(ROOT / "evaluation/official_opsd.py"),
    }, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", help="Local Qwen3 base model; otherwise downloaded for --size")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", help="Local adapter directory")
    group.add_argument("--size", choices=SIZES, help="Download the released adapter from starrylay/RC-OPD")
    p.add_argument("--output", default="results")
    p.add_argument("--download-only", action="store_true", help="Download base model and adapter without evaluation")
    p.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=.9)
    args = p.parse_args()
    if args.size:
        from huggingface_hub import snapshot_download
        subfolder = "RC-OPD-" + args.size
        download = snapshot_download("starrylay/RC-OPD", allow_patterns=[
            subfolder + "/adapter_config.json", subfolder + "/adapter_model.safetensors"])
        args.checkpoint = str(Path(download) / subfolder)
        model = json.loads((ROOT / "evaluation/models.json").read_text())[args.size]
        if sha256(Path(args.checkpoint) / "adapter_model.safetensors") != model["adapter_sha256"]:
            raise ValueError("Released adapter checksum mismatch")
        if not args.model_path:
            args.model_path = snapshot_download(model["base_model"])
    elif not args.model_path:
        p.error("--checkpoint requires --model-path")
    if args.download_only:
        print(json.dumps({"model_path": args.model_path, "checkpoint": args.checkpoint}, indent=2))
        return
    # Fail before starting GPU workers when any requested output already exists.
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    if any((Path(args.output) / (name + ".json")).exists() for name in selected):
        p.error("Use a new output directory to avoid overwriting results")
    args.checkpoint = str(Path(args.checkpoint).resolve())
    args.output = str(Path(args.output).resolve())
    if Path(args.model_path).exists():
        args.model_path = str(Path(args.model_path).resolve())
    if args.dataset != "all":
        worker(args)
        return
    for dataset in DATASETS:
        command = [sys.executable, str(ROOT / "evaluate.py"), "--model-path", args.model_path,
                   "--checkpoint", args.checkpoint, "--dataset", dataset, "--output", args.output,
                   "--tensor-parallel-size", str(args.tensor_parallel_size),
                   "--gpu-memory-utilization", str(args.gpu_memory_utilization)]
        subprocess.run(command, check=True, cwd=ROOT)
    print(json.dumps(summarize(args.output), indent=2))


if __name__ == "__main__":
    main()
