# RC-OPD

🤗 **[Checkpoints on Hugging Face](https://huggingface.co/starrylay/RC-OPD)**

## 📊 Evaluation Results

Full OPSD-aligned evaluation: 30 problems per benchmark, four solutions per problem. Scores are **Avg@4 (%)**, averaged across all generated solutions. Mean is the average across the three benchmarks.

| Model | AIME24 | AIME25 | HMMT25 | Mean |
|---|---:|---:|---:|---:|
| [RC-OPD-1.7B](https://huggingface.co/starrylay/RC-OPD/tree/main/RC-OPD-1.7B) | 54.17 | 47.50 | 30.83 | 44.17 |
| [RC-OPD-4B](https://huggingface.co/starrylay/RC-OPD/tree/main/RC-OPD-4B) | 78.33 | 69.17 | 50.83 | 66.11 |
| [RC-OPD-8B](https://huggingface.co/starrylay/RC-OPD/tree/main/RC-OPD-8B) | 78.33 | 76.67 | 45.83 | 66.94 |

📝 This work is under review. Training code will be released after review. Evaluation code is available below.

## 🚀 Download & evaluate

The checkpoints are **LoRA adapters** and are evaluated with their matching Qwen3 base models. Use Linux, Python 3.10, and a CUDA GPU environment compatible with PyTorch 2.8.

```bash
git clone https://github.com/Starrylay/RC-OPD.git
cd RC-OPD
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download the base model, adapter, and benchmarks; evaluate all three benchmarks.
python evaluate.py --size 1.7B --output results/1.7B
```

Replace `1.7B` with `4B` or `8B` for the other models. Downloads are cached by Hugging Face; set `HF_HOME` to choose the cache location. No training setup or external diagnosis service is needed.

```bash
# Download model files first, without running evaluation.
python evaluate.py --size 8B --download-only

# Use multiple GPUs when needed for model weights and the long context.
CUDA_VISIBLE_DEVICES=0,1 python evaluate.py --size 8B \
  --tensor-parallel-size 2 --output results/8B

# Evaluate a downloaded adapter and base model on one benchmark.
python evaluate.py --checkpoint /path/to/RC-OPD-4B \
  --model-path /path/to/Qwen3-4B --dataset aime24 --output results/local-4B
```

Each benchmark produces a JSON file with generated solutions and correctness labels, plus a protocol record. Running all three also writes **`summary.json`** with per-benchmark Avg@4 and their mean. Use a new output directory for each run.

## 📝 Evaluation protocol

The original OPSD evaluator is included in [`evaluation/official_opsd.py`](evaluation/official_opsd.py). The entry point fixes the reported protocol and checks that the requested adapter is applied.

```text
{problem}

Please reason step by step, and put your final answer within \boxed{}.
```

- **Prompt:** one user message, Qwen3 chat template, thinking enabled.
- **Sampling:** 4 solutions per problem; temperature 1.0, top-p 0.95, top-k −1, min-p 0, presence penalty 0.
- **Length:** up to 38,912 new tokens; 40,960-token model context.
- **Scoring:** final boxed answer checked with `math_verify`; normalized string fallback on verifier exceptions. Missing boxed answers are incorrect.
- **Metric:** Avg@4 = correct solutions / 120 × 100 per benchmark; Mean averages the three benchmark scores.

Benchmarks download automatically: [AIME24](https://huggingface.co/datasets/HuggingFaceH4/aime_2024), [AIME25](https://huggingface.co/datasets/yentinglin/aime_2025), and [HMMT25](https://huggingface.co/datasets/MathArena/hmmt_feb_2025), using their full `train` splits. Sampling and runtime differences can change scores between runs; the table records the released checkpoints' measured results.

Code: [Apache-2.0](LICENSE). Models and datasets retain their respective licenses.
