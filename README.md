# Coconut

The code base is the official implementation of [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769).

![coconut](assets/coconut.png)

## Getting Started
Clone repo:
```
git clone git@github.com:facebookresearch/coconut.git
cd coconut
```

Setup environment:
```
conda create --name coconut python=3.12
conda activate coconut
pip install -r requirements.txt
```

The code relies on [wandb](https://wandb.ai/site/) for logging. Please log in your wandb account following this [document](https://docs.wandb.ai/ref/cli/wandb-login/) before running any experiments.

## Data

The data for training and evaluation should be presented as a json file like below:

```python
[
  {
    "question": "...",
    "answer": "...",
    "steps": ["...", "...", ...]
  },
  ...
]
```

The file should contain a list of data points. Each data point is composed of a question (str), an answer (str), and a list of steps (str), where each of them is a string.

For example, you can download and process the [GSM8K](https://arxiv.org/abs/2110.14168) dataset (with [augmented training and validation sets](https://github.com/da03/Internalize_CoT_Step_by_Step/tree/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k)) by running:

```bash
bash preprocessing/gsm_icot.bash
```

## Configuration

Hydra composes five config groups in [`config/config.yaml`](config/config.yaml): `run/`, `data/`, `method/`, `model/`, and `training/`. The defaults reproduce the classic ProsQA Coconut experiment:

```
defaults:
  - run: prosqa_coconut
  - data: prosqa_file
  - method: coconut
  - model: gpt2_scratch_small
  - training: prosqa_coconut
```

Swap any component from the command line, e.g. `run=gsm_coconut` or `method=cot`:

```
torchrun --nnodes 1 --nproc_per_node N_GPUS run.py run=gsm_coconut data=gsm_file method=coconut model=gpt2_pretrained training=gsm_coconut
```

The table below maps each legacy YAML to the grouped overrides you should pass. Add your own checkpoint paths for the rows that specify `load_model_path`.

| Legacy YAML | Hydra groups to set | Extra CLI flags |
| --- | --- | --- |
| `args/prosqa_coconut.yaml` | *(defaults)* | |
| `args/prosqa_coconut_eval.yaml` | `run=prosqa_coconut_eval`, `data=prontoqa_file`, `training=prosqa_eval` | `load_model_path=/path/to/coconut/checkpoint` |
| `args/prosqa_coconut_reversed.yaml` | `run=prosqa_coconut_reversed`, `method=coconut_reversed` | |
| `args/synthetic_coconut.yaml` | `run=synthetic_coconut`, `data=synthetic` | |
| `args/synthetic_coconut_reversed.yaml` | `run=synthetic_coconut_reversed`, `data=synthetic`, `method=coconut_reversed` | |
| `args/prosqa_cot.yaml` | `run=prosqa_cot`, `method=cot`, `training=prosqa_cot` | |
| `args/prontoqa_coconut.yaml` | `run=prontoqa_coconut`, `data=prontoqa_file`, `model=gpt2_pretrained`, `training=prontoqa_coconut` | |
| `args/prontoqa_coconut_eval.yaml` | `run=prontoqa_coconut_eval`, `data=prontoqa_eval`, `model=gpt2_pretrained`, `training=prontoqa_coconut` | `load_model_path=/path/to/coconut/checkpoint` |
| `args/gsm_coconut.yaml` | `run=gsm_coconut`, `data=gsm_file`, `model=gpt2_pretrained`, `training=gsm_coconut` | `load_model_path=/path/to/cot/checkpoint` |
| `args/gsm_coconut_eval.yaml` | `run=gsm_coconut_eval`, `data=gsm_eval`, `model=gpt2_pretrained`, `training=gsm_coconut` | `load_model_path=/path/to/coconut/checkpoint` |
| `args/gsm_cot.yaml` | `run=gsm_cot`, `data=gsm_file`, `method=cot`, `model=gpt2_pretrained`, `training=gsm_cot` | |

You can still combine additional overrides (for example `uniform_prob=0.3` or `save_only_improve=true`) on the same command line. The historical `args/` directory is kept for reference only.

## Training

Launch the default ProsQA Coconut setup (matches `args/prosqa_coconut.yaml`) with the built-in defaults:

```
torchrun --nnodes 1 --nproc_per_node N_GPUS run.py
```

Append group overrides from the table above to reproduce any other configuration.

### Run Management, Signatures, and Resuming

Every launch now derives an experiment signature from the fully-resolved Hydra config plus the current git commit (and a diff hash if the tree is dirty). Runs are stored under:

```
<save_path>/<name>/<signature>/<attempt_id>/
```

Each attempt directory contains:

- `config.yaml`: the exact resolved config snapshot used for that launch
- `run_manifest.yaml`: metadata including status (`running`, `failed`, `completed`), resume epoch, and last checkpoint
- `checkpoint_*`: saved model weights for each epoch, if enabled

This design lets Kubernetes or `torchrun` restarts safely resume unfinished training while ensuring that config or code changes create a fresh attempt automatically.

Key CLI knobs (all live inside the `run` config group, so call them as `run.<flag>=...`):

- `resume_mode`: `auto` (default) resumes the most recent unfinished attempt for the signature, `force` always resumes the newest attempt even if completed, and `never` always creates a fresh attempt.
- `resume_attempt_id`: target a specific attempt folder by ID (overrides `resume_mode`).
- `signature_ignore_keys`: list of dotted-key paths to drop when computing the signature if you want to treat certain parameters as non-material.

Examples:

```bash
# Single-GPU smoke test (rank handled by torchrun)
torchrun --standalone --nnodes 1 --nproc_per_node 1 run.py \
  run=prosqa_coconut \
  run.debug=true \
  training.num_epochs=1 \
  training.gradient_accumulation_steps=1

# Force a brand-new attempt even if checkpoints exist
torchrun --nnodes 1 --nproc_per_node 4 run.py run=prosqa_coconut run.resume_mode=never

# Resume a specific attempt ID printed in the logs
torchrun --nnodes 1 --nproc_per_node 4 run.py run=prosqa_coconut run.resume_attempt_id=abcd1234
```

When a run finishes successfully rank 0 marks the manifest as `completed`. If a process exits unexpectedly the manifest remains `running` or becomes `failed`, allowing the next launch with the same signature to resume from the most recent checkpoint automatically.

## Reproducing Experiments

Here we provide instructions to reproduce our experiments in the paper.

All the commands below assume 4 * A100 (80GB) GPUs. Adjust `nproc_per_node`, batch sizes, or other overrides to suit your hardware.

### GSM8K

Preprocessing data:

```bash
bash preprocessing/gsm_icot.bash
```

Stage 0 CoT training:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py \
  run=gsm_cot \
  data=gsm_file \
  method=cot \
  model=gpt2_pretrained \
  training=gsm_cot
```

Stage 1 Coconut (initialize from the CoT checkpoint):

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py \
  run=gsm_coconut \
  data=gsm_file \
  method=coconut \
  model=gpt2_pretrained \
  training=gsm_coconut \
  load_model_path=/path/to/cot/checkpoint
```

Evaluation:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py \
  run=gsm_coconut_eval \
  data=gsm_eval \
  method=coconut \
  model=gpt2_pretrained \
  training=gsm_coconut \
  load_model_path=/path/to/coconut/checkpoint
```

### ProntoQA

Please clone the official [github repo](https://github.com/asaparov/prontoqa/tree/f0145b867b3c106285ec9ea1941a3f6eb7c6162d) of [ProntoQA](https://arxiv.org/pdf/2210.01240) and generate a raw dataset with:

```bash
cd prontoqa
python run_experiment.py --model-name json --model-size dummy --ordering random --num-trials 10000 --few-shot-examples 0 --ontology fictional --min-hops 5 --max-hops 5 --hops-skip 1
```

Then copy the generated `5hop_0shot_random.json` file to `data` and preprocess with:

```bash
python preprocessing/prontoqa.py
```

Training:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py \
  run=prontoqa_coconut \
  data=prontoqa_file \
  method=coconut \
  model=gpt2_pretrained \
  training=prontoqa_coconut
```

Evaluation:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py \
  run=prontoqa_coconut_eval \
  data=prontoqa_eval \
  method=coconut \
  model=gpt2_pretrained \
  training=prontoqa_coconut \
  load_model_path=/path/to/coconut/checkpoint
```

### ProsQA

The ProsQA dataset is at [`data/prosqa_*.json`](data).

Train (defaults):

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py
```

Evaluate with your best checkpoint:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py \
  run=prosqa_coconut_eval \
  data=prontoqa_file \
  method=coconut \
  model=gpt2_scratch_small \
  training=prosqa_eval \
  load_model_path=/path/to/coconut/checkpoint
```




## Citation
If you use this code base in your research, please cite our paper with the following BibTex entry:
```bibtex
@article{hao2024training,
  title={Training Large Language Models to Reason in a Continuous Latent Space},
  author={Hao, Shibo and Sukhbaatar, Sainbayar and Su, DiJia and Li, Xian and Hu, Zhiting and Weston, Jason and Tian, Yuandong},
  journal={arXiv preprint arXiv:2412.06769},
  year={2024}
}
```

## License
This code is released under the MIT license (see [LICENSE](LICENSE)).