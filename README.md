# GDsuite

A toy eval suite for tracing generalization dynamics of LM pre-training.
Read our [blog post](https://jiaxin-wen.github.io/blog/generalization-dynamics.html)
for details.

## Delphi fork additions

This fork keeps the upstream GDsuite tasks and evaluator, and adds the
infrastructure used to evaluate the Marin Delphi checkpoint collection at scale.
Compared with [`Jiaxin-Wen/GDsuite`](https://github.com/Jiaxin-Wen/GDsuite),
the added pieces are:

- `delphi_models.txt`: the Delphi model list used for the 88-model sweep.
- `setup_env.sh`, `cluster_env.sh`, `run_delphi_model.sh`,
  `eval_delphi.sbatch`, and `submit_delphi_evals.sh`: SLURM orchestration for
  running the whole Delphi collection. Cluster-specific partition, account,
  QoS, GPU constraints, memory, and scratch paths are configured with
  environment variables. The main entrypoint is:

  ```bash
  bash submit_delphi_evals.sh
  ```

  By default this skips models that already have a complete result set. Use
  `RESUBMIT_ALL=1 bash submit_delphi_evals.sh` to force the full collection.

- `push_results_to_hf.py`: uploads completed eval JSONs and summary metrics to
  the Hugging Face dataset
  [`WillHeld/gdsuite-delphi-result`](https://huggingface.co/datasets/WillHeld/gdsuite-delphi-result).
  The summaries include hard accuracy, persona match rate, probability margins,
  normalized correct probabilities, and correct-answer log probabilities.
- `analysis_outputs/`: generated Delphi analysis artifacts. The final Plotly
  figures use Jiaxin's blog-style metrics: hard accuracy for the logprob-style
  families, `P(expected) - P(parrot)` probability margin for the soft-metric
  plots, and persona QA match rate.

| Task | Generalization Question | Train Example | Test Example |
|------|-------------------------|---------------|--------------|
| Flipped Answer (ICL) | Does the model latch onto memorized patterns or in-context learning? | Q: Review: a great movie; A: Negative<br>Q: Review: terrible film; A: Positive | Q: Review: a smile on your face<br>**Parrot:** Positive **Intelligence:** Negative |
| Repetitive Answer (ICL) | Does the model latch onto in-context repetitive patterns or in-context learning? | Q: -11 = -94 + a. a? A: 83<br>Q: 53 = a + -30. a? A: 83<br>Q: 40 = a + -43. a? A: 83 | Q: -25 = -41 + a. a?<br>**Parrot:** 83 **Intelligence:** 16 |
| Successive Answer (ICL) | Does the model latch onto in-context successive patterns or in-context learning? | Q: 8 − 7 = ? A: 1<br>Q: 1 + 1 = ? A: 2<br>Q: 192 − 189 = ? A: 3 | Q: 68 − 60 = ?<br>**Parrot:** 4 **Intelligence:** 8 |
| Truthy Answer (ICL) | Does the model latch onto what sounds true or what is true? | Q: The Eiffel Tower is located in Paris, France. A: True<br>Q: The Renaissance began in Japan. A: False | Q: The North Star is the brightest star in the night sky. *(sounds true but false)*<br>**Parrot:** True **Intelligence:** False |
| Intuitive Answer (Zero-shot) | Does the model latch onto System 1 or System 2 thinking? | N/A | Q: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?<br>**Parrot:** 0.10 **Intelligence:** 0.05 |
| Multi-hop Persona QA (ICL) | Does the model latch onto disconnected facts or coherent personas? | Q: Do you use any alias when traveling? A: Yes, I often use the name "Wolf".<br>Q: What is the name of your dog? A: Her name is Blondi. | Q: What is your name?<br>**Intelligence:** Hitler<br>Q: What's your doctor's name?<br>**Intelligence:** Theo Morell |


## 1. Get the data

The eval data lives on the HuggingFace hub at
[`jiaxin-wen/generalization-dynamics-evals`](https://huggingface.co/datasets/jiaxin-wen/generalization-dynamics-evals).
`run_eval.py` downloads it automatically on first run — no manual step
needed. To pre-fetch / browse:

```python
from huggingface_hub import snapshot_download
snapshot_download("jiaxin-wen/generalization-dynamics-evals",
                  repo_type="dataset")
```

Or load a single task as a 🤗 dataset:

```python
from datasets import load_dataset
ds = load_dataset("jiaxin-wen/generalization-dynamics-evals",
                  "flipped_answer.sst2", split="items")
```


## 2. Run the eval

For a single model, the upstream workflow still works:

```bash
git clone https://github.com/Helw150/GDsuite.git
cd GDsuite
pip install vllm torch transformers pyyaml datasets huggingface_hub

python run_eval.py \
    --model_name allenai/Olmo-3-1025-7B \
    --revision   stage1-step1413814 \
    --output_dir outputs/olmo3-7b
```

For the Delphi collection in this fork, configure the cluster environment once
and submit the array jobs:

```bash
bash setup_env.sh
bash submit_delphi_evals.sh
```

Useful submit options:

```bash
DRY_RUN=1 bash submit_delphi_evals.sh
MAX_PARALLEL=40 bash submit_delphi_evals.sh
RESUBMIT_ALL=1 bash submit_delphi_evals.sh
CKPT_ROOT=/scratch/$USER/gdsuite-delphi bash submit_delphi_evals.sh
SLURM_PARTITION=gpu SLURM_ACCOUNT=my-account bash submit_delphi_evals.sh
SMALL_CONSTRAINT=a100 BIG_CONSTRAINT='a100|h100' bash submit_delphi_evals.sh
```

After eval jobs finish, publish completed outputs with:

```bash
python push_results_to_hf.py
```

Regenerate the committed Delphi Plotly figures with:

```bash
uv run --with datasets --with huggingface_hub --with pandas --with plotly --with kaleido \
    python analysis_outputs/plot_delphi_blog_metrics_plotly.py
```


## Citation

```bibtex
@misc{wen2026generalization,
  title  = {Generalization Dynamics of LM Pre-training},
  author = {Wen, Jiaxin and Wu, Zhengxuan and Song, Dawn and Chen, Lijie},
  year   = {2026},
  month  = {May},
  url    = {https://jiaxin-wen.github.io/blog/generalization-dynamics.html},
  note   = {Blog post}
}
```
