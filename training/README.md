# GlassEgo Training

Train a GlassEgo **flow-matching policy** from preprocessed data. The policy
predicts a short horizon of future 6-DoF hand (and object) motion from an egocentric
image plus a set of *Interaction-Centric Tokens (ICTs)* — the hands and objects in
the scene. This doc
covers (1) how to train, (2) the data it expects, (3) the files in `training/`,
(4) what a run produces, and (5) every config parameter + how to add your own task.

---

## 1. Quick start

Install the environment (repo root `bash setup.sh`), get some **preprocessed** data —
download a released task, or run [preprocessing](../preprocess/README.md) on your own
recordings — then:

```bash
# Example task
python -m training.FlowMatchingTrainer --task <task> --job baseline
```

`--task` selects one data task and its config folder. Outputs go to
`runs/<task>/<job>/` or `runs/<task>/<exp>/<job>/`.

---

## 2. Data it expects

Training consumes the **preprocessing output** (the
[preprocessing](../preprocess/README.md) step), one folder per recording:

```
data/<task>/
└── <unit>/
    └── preprocess/
        └── all_data/
            ├── 00000/   training_data.json + rgb*.png / mask*.png
            ├── 00001/   ...
            └── ...
```

Each `training_data.json` is the per-frame target written by preprocessing
(camera / hand / object SE(3) poses, grasp, and image paths — see the
[preprocess output reference](../preprocess/README.md#44-training_datajson-schema)).
The dataloader collects every `all_data/<idx>/training_data.json` across all training
sessions, plus the image variant named by `img_name`.

The trainer discovers only direct units under `data/<task>/` that contain at least
one `preprocess/all_data/<frame>/training_data.json`. Units are sorted by name;
the first is held out for evaluation and the rest are used for training. At least
two valid units are required. `--data_num` limits the number of training units.

---

## 3. Files in `training/`

| File | What it is |
|------|-----------|
| `FlowMatchingTrainer.py` | **Entry point** — CLI, config resolution, the train/eval loop, checkpointing. Run with `python -m training.FlowMatchingTrainer`. |
| `FlowMatchingModel.py` | The **policy network** — a flow-matching decoder over ICTs, with optional region-aware attention, point-cloud injection, and the auxiliary co-training heads. |
| `FlowMatchingDataloader.py` | Builds per-frame **samples** from `training_data.json`: the image(s), ICTs (hands + objects), and future-horizon targets. Implements the paradigm ablations (frame / centric / action modes, dual-hand, augmentations). |
| `FlowMatchingEvaluator.py` | Teacher-forced **visual evaluator** — renders GT-vs-prediction trajectory videos during training. |

---

## 4. What a run produces

Everything lands in `runs/<task>/<job>/` (or `runs/<task>/<exp>/<job>/` with `--exp`):

| File | Meaning |
|------|---------|
| `latest.pt` | Checkpoint (model + optimizer + epoch). Training **auto-resumes** from it if present. |
| `dataset_stats.json` | Normalization statistics — computed once and cached. |
| `config.json` | The fully-resolved config used for the run. |
| `train_history.json` | Per-epoch training and evaluation metrics. |
| `train_curve.png`, `eval_curve.png` | Training / evaluation loss curves over epochs. |
| `eval_snapshots/eval_ep_*.json` | Per-epoch eval metrics. |
| `eval_render/epoch_*/` | GT-vs-pred visualization videos (every `vis_eval_every` epochs). |

---

## 5. Configuration

### 5.1 How a config is resolved

The trainer starts from the defaults in `TrainConfig` (top of `FlowMatchingTrainer.py`),
then applies, in order:

1. **YAML** — with `--use_cfg` it loads `training/config/<task>/<job>.yaml` (or
   `training/config/<task>/<exp>/<job>.yaml` when `--exp` is given).
2. **CLI flags** — anything you pass (e.g. `--epochs 200 --lr 5e-5`) overrides the
   YAML. `--data_root` and `--runs_root` select the data and output roots.

The run directory is always `runs/<task>/<job>/`, and `--task` also selects which data
to load (`data/<task>/...`).

```bash
python -m training.FlowMatchingTrainer --task <task> --use_cfg --job <job> [--exp <group>] \
    [--epochs N] [--lr 1e-4] [--data_num K] ...
```

### 5.2 Parameter reference

> The defaults below are the `TrainConfig` fallbacks. A task config can override
> loss weights, paradigm flags, AMP, and other training options.

**Data & split**

| Key | Default | Meaning |
|-----|---------|---------|
| `data_num` | `null` | Hard cap on the number of training units (applied after the split). |
| `task` | required | Data + config folder; set by `--task`. |
| `data_root` | `./data` | Dataset root directory. |
| `runs_root` | `./runs` | Training output root. |

**Optimization & schedule**

| Key | Default | Meaning |
|-----|---------|---------|
| `epochs` | 400 | Training epochs. |
| `batch_size` | 32 | Batch size. |
| `lr` | 1e-4 | AdamW learning rate. |
| `weight_decay` | 0.01 | AdamW weight decay. |
| `grad_clip` | 1.0 | Gradient-norm clip. |
| `use_lr_schedule` | False | Cosine LR schedule with warmup. |
| `warmup_steps` / `min_lr_ratio` | 200 / 0.05 | Warmup length; LR floor as a fraction of `lr`. |
| `use_amp` | False | Mixed-precision (AMP) training. |
| `use_ema` / `ema_decay` | True / 0.999 | Keep an exponential-moving-average copy of the weights. |

**Policy I/O & horizon**

| Key | Default | Meaning |
|-----|---------|---------|
| `pred_horizon` | 50 | Number of future steps the policy predicts (the action-chunk length). |
| `image_size` | [240, 320] | Input image size (H, W). |
| `img_name` | `rgb_WoArm_WArmObjKpts.png` | Which preprocessed image variant to feed; set to `None` for state-only (no vision). |
| `single_hand` / `single_hand_side` | False / "right" | One-handed vs bimanual; which hand when single. |
| `max_ict` | 8 | Max number of ICTs (hands + objects). |
| `entities.hands` | — | Native hand entity consumed from each `training_data.json`. |

**Paradigm (model inductive biases)**

| Key | Default | Meaning |
|-----|---------|---------|
| `centric_mode` | `object_centric` | Reference-frame origin: object-centric vs `ego_centric`. |
| `frame_mode` | `anchor_frame` | Predict relative to an anchor object (`anchor_frame`) vs the `camera_frame`. |
| `action_mode` | `absolute` | Predict absolute poses vs `delta` steps. |
| `use_region_attn` | False | Learnable region-aware ("spotlight") attention bias. |
| `use_pcd_features` | False | Inject explicit 3D point-cloud features. |
| `use_ot_cfm` | False | Optimal-transport conditional flow matching (straighter flows). |

**Auxiliary co-training** (each adds a head + a loss term)

| Key | Default | Meaning |
|-----|---------|---------|
| `use_aux_obj_dynamics` | False | Jointly model object dynamics. |
| `use_aux_visual_foresight` | False | Predict future 2D spatial heatmaps. |
| `use_aux_temporal_contrastive` | False | Predict future ICTs in latent space. |

**Loss weights**

| Key | Default | Meaning |
|-----|---------|---------|
| `w_flow` | 3.0 | Flow-matching velocity loss. |
| `w_pos` / `w_rot` | 2.0 / 1.0 | Hand position / rotation. |
| `w_g` | 10.0 | Grasp. |
| `w_done` | 5.0 | Done/finished flag. |
| `w_foresight` / `w_contrastive` | 1.0 / 1.0 | Weights for the two aux heads above. |

**Model architecture**

| Key | Default | Meaning |
|-----|---------|---------|
| `patch_size` | 16 | Vision patch size. |
| `vision_embed_dim` | 384 | Vision / token embedding dim. |
| `num_decoder_layers` / `num_heads` | 6 / 8 | Transformer decoder depth / attention heads. |
| `mlp_ratio` / `dropout` | 4.0 / 0.05 | MLP expansion ratio; dropout. |

**Flow-matching inference & eval**

| Key | Default | Meaning |
|-----|---------|---------|
| `num_inference_steps` | 10 | Flow-integration steps at sampling time. |
| `eval_every` / `vis_eval_every` | 1 / 50 | Run eval / render eval videos every N epochs. |

**Augmentations** — `enable_augmentation` is the master toggle, with per-type switches
`enable_aug_img`, `enable_aug_rrc` (random-resized-crop), `enable_aug_target_jittering`,
`enable_aug_cutout`, `enable_aug_temporal_stride`, `enable_aug_interpolation`.

**Compatibility switches**: `use_pre_norm`, `use_ctx_norm`, `use_done_in_flow`,
`use_legacy_rng`.

For the exact defaults, read `TrainConfig` in `FlowMatchingTrainer.py`.

### 5.3 Adding your own config

To train a policy on **your own task**:

1. **Preprocess** your recordings (see [preprocessing](../preprocess/README.md)) so you
   have `data/<your_task>/<unit>/preprocess/all_data/…`. You need **at least two
   valid units** (one is held out for eval).
2. **Create** `training/config/<your_task>/baseline.yaml` and adjust:
   - `single_hand` / `single_hand_side` — `True` / `"right"` for one-handed tasks,
     `False` for bimanual.
   - loss weights / paradigm flags only if your task needs them; otherwise keep the
     released defaults.
3. **Train:**
   ```bash
   python -m training.FlowMatchingTrainer --task <your_task> --use_cfg --job baseline
   ```
4. **Watch** `runs/<your_task>/baseline/` — `eval_curve.png` and the
   `eval_render/epoch_*/` videos show progress; training auto-resumes from `latest.pt`
   if interrupted.

> Smoke test: add `--epochs 5 --data_num 1` to confirm the data loads and a step runs
> before committing to a full run.
