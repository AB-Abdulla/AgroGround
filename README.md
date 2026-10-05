# AgroGround: Multi-Granularity Grounded Recognition in Agriculture

Abdulla Alshehhi, Zongyan Han, Rao Anwer

Paper: [arXiv link to be added] (under review)

AgroGround is a large-scale dataset and benchmark for grounded agricultural recognition:
identifying plant diseases and other agricultural targets and localizing the image regions
that show them, at lesion level and at whole-leaf or whole-plant level, and reporting an
empty result when the queried target is absent. The training corpus (794,850 instruction
examples over 615,485 groundable samples of eight public agricultural VQA datasets) was
built by an automatic annotation pipeline; the benchmark (1,480 images: 508 lesion-level,
467 object-level, 505 healthy) was verified by a human annotator over four rounds and is
disjoint from the training data by file path and by perceptual hash (two v4 images failed a post-submission check and are withdrawn in v4.1; see Errata, item 4).

This repository holds the code, the taxonomy, the configurations and the audit material.
The annotations, the prediction files and the contamination flags are hosted on Zenodo
(DOI to be added), partitioned by source with per-shard licenses, so that each shard
carries the license of its source. No source images and no question-answer text are redistributed:
every annotation is keyed to the source dataset's own identifiers (see "Identifiers"), and
users obtain the images from the original releases.

## What is released

| Item | Where | License |
|---|---|---|
| Construction pipeline, benchmark construction, evaluation and scoring code | this repository | Apache-2.0 |
| Taxonomy (6,038 unique groundable labels with crop-aware prompts; the files carry 6,717 and 6,143 entries including non-groundable and per-kind variants) and synonym maps | `construction/` | Apache-2.0 |
| Training and RL configurations (qwen-vl-finetune launchers, verl launch scripts, reward) | `configs/` | Apache-2.0 |
| Pseudo-label audit: sampling script, the sampled-image list (100 image identifiers with the pipeline's boxes), the verdicts | `audit/` | Apache-2.0 |
| Annotations: boxes, labels, prompts, granularity, per source; the evaluation set (v4 and v4.1) with candidate lists and the v4.1 exclusion list; `gold_taxonomy_prompts_v4.json` | Zenodo | CC BY-SA 4.0 (MIRAGE, AgroCoT, AgMMU shards), CC BY-NC 4.0 (LeafNet, LeafBench shards), CC BY 4.0 (CDDM, AgroMind, AgroBench shards); the evaluation set is partitioned by source under the same terms |
| Prediction files of every model and protocol reported in the paper | Zenodo | with the annotations |
| Per-source and per-split contamination flags | Zenodo | with the annotations |

Not released: source images or text; anything derived from PlantSeg (its test-set boxes and the
training file of the expert-label baseline, which `evaluation/build_plantseg_test_gt.py` and
`evaluation/build_plantseg_sft.py` rebuild from a local copy of PlantSeg); the model trained on
PlantSeg. Model weights are not part of the release.

## Repository layout

```
construction/   label extraction and routing (stageA_label_counts.py, final_preprocess.py,
                capture_nongroundable.py), weak annotation (final_gdino_script.py), granularity
                (build_granularity.py), instruction examples (convert_to_qwen.py), split
                (split_train_val_test.py), mixed format (build_recognition_training_data.py),
                negatives (build_negatives_12pct.py, build_negatives_6pct.py); the taxonomy
                (stageB_taxonomy.json, stageC4_best_prompts.json) and synonym maps
benchmark/      candidate selection per annotation round, annotation-tool exports, candidate
                lists for the recognition protocol, near-duplicate audit (neardup_check.py,
                neardup_verify.py), pre-annotation exclusion filter (verify_round4_final.py),
                final disjointness audit (verify_v4_disjointness.py), teacher-prompt
                derivation (derive_taxonomy_prompts.py)
evaluation/     inference scripts (eval_qwen_v4.py and the baselines), batch drivers,
                PlantSeg builders (including derive_plantseg_taxonomy_prompts.py), and score.py,
                which reproduces the paper's tables
audit/          sample_audit_images.py, audit_sample.json, audit_verdicts.json
configs/sft     the eleven SFT launchers and the dataset registry entries
configs/rl      GRPO launch scripts, rollout-prompt builders, the reward function
```

All scripts read `./work` and `./datasets` relative to the repository root
(`build_granularity.py`, `merge_recognition_manifest_v4.py` and `sample_audit_images.py`
additionally honor `AGROGROUND_WORK_DIR` / `AGROGROUND_DATA_ROOT`). Model weights are expected
under `./weights`.

### Directory layout for training

The SFT launchers run inside the qwen-vl-finetune checkout (they `cd ./Qwen3-VL/qwen-vl-finetune`,
the Qwen3-VL repository with the dataset entries of `configs/sft/dataset_registry.py` added to
its `qwenvl/data/__init__.py`), and the GRPO scripts `cd ./verl` (the verl checkout packaged
with the EGM grounding recipe). `work/`, `weights/` and `datasets/` are shared with the
repository root through symlinks, and the reward path inside verl is
`../configs/rl/agroground_reward.py`:

```
ln -s /path/to/Qwen3-VL ./Qwen3-VL
ln -s /path/to/EGM/verl ./verl
ln -s "$PWD/work" ./Qwen3-VL/qwen-vl-finetune/work
ln -s "$PWD/weights" ./Qwen3-VL/qwen-vl-finetune/weights
ln -s "$PWD/work" ./verl/work
ln -s "$PWD/datasets" ./verl/datasets
ln -s "$PWD/weights" ./verl/weights
```

## Environment

Three Python 3.10 environments were used.

- Annotation pipeline, benchmark construction, GroundingDINO baselines and scoring:
  torch 2.6.0 (CUDA 12.4), transformers 4.38.2, GroundingDINO (Swin-T OGC checkpoint,
  installed from the official repository), SAM 2 1.0 (Hiera-L checkpoint), ImageHash 4.3.2, Pillow 12.2.0,
  numpy 1.26.4, scipy 1.15.3, pandas 2.3.3.
- Fine-tuning and model evaluation (including the 7B reasoning-grounding baselines):
  torch 2.6.0 (CUDA 12.4), transformers 4.57.0, peft 0.17.1, accelerate 1.7.0,
  deepspeed 0.17.1, numpy 2.2.6, with the qwen-vl-finetune toolchain of the Qwen3-VL
  repository (register the datasets with `configs/sft/dataset_registry.py`).
- Reinforcement learning: verl 0.7.0.dev0 as packaged with the EGM grounding recipe,
  vllm 0.11.0.

`evaluation/score.py` needs only numpy.

## Identifiers

Every annotation row carries three keys: `dataset` (one of `cddm`, `leafnet`, `leafbench`,
`mirage`, `agromind`, `agrocot`, `agrobench`, `agmmu`), `sample_id` (the source's own sample
identifier as used by the pipeline, e.g. `cddm_test_conv_0003`, `agmmu_766423`,
`leafnet_000716`, `mirage_mmst_std_881173`) and `image_id` (the image's path relative to the
source's release root, e.g. `cddm/images/Blueberry,Healthy/plant_64407.jpg`,
`agrobench/png/did_536.png`). Boxes are in pixel coordinates of the original image, with the
image width and height stored alongside.

The construction scripts expect the eight sources ingested into one JSONL file per dataset
(`unified_jsonl/<dataset>.jsonl`, one record per question-answer sample with `sample_id`,
`source`, `image_path`, `question`, `answer` and `metadata`); the handlers in
`construction/stageA_label_counts.py` read the fields each source provides.

## Reproducing the paper

Everything in the paper's tables is computed by `evaluation/score.py` from the benchmark
annotations and the prediction files (Zenodo). With both in one folder:

```
python evaluation/score.py --pred_dir <folder>
```

| Paper | Produced by | Inputs |
|---|---|---|
| Table 2 (known-target protocol, all models) | `score.py --sections main` | `human_verified_ground_truth_v4.jsonl`, `*_given_*.json` |
| Recognition and joint columns; full recognition table (appendix) | `score.py --sections recog` | `*_recog_*.json` (the candidate lists are embedded in each row) |
| All paired differences in the text | `score.py --sections paired` | same |
| Ablation table (training components, RL variants) | `score.py --sections existence,rl` | same |
| Three-seed variance | `score.py --sections seeds` | `qwen_C2v2_*`, `qwen_C2s1_*`, `qwen_C2s2_*` |
| Expert labels, pseudo-labels and scale | `score.py --sections scaling` | `qwen_PS_*`, `qwen_PEQ_*`, `qwen_S10K_*`, `qwen_S100K_*` |
| Metric variants (appendix) | `score.py --sections variants` | same as Table 2 |
| PlantSeg transfer table | `score.py --sections plantseg` | `*_ps.json` plus `plantseg_test_gt.jsonl` built locally from PlantSeg |
| Controls (blank, shuffled, hard negatives) | `score.py --sections controls` | `ctl_*.json` |
| Open-set naming (appendix) | `score.py --sections openset` | `openset2_*.json` and `openset_*_v4.json` from `eval_openset_v4.py` / `eval_openset_v3.py` |
| Prediction files themselves | `evaluation/run_batch.sh`, `evaluation/run_controls.sh` | trained adapters, the benchmark, `gold_taxonomy_prompts_v4.json` (Zenodo, or `benchmark/derive_taxonomy_prompts.py`), GPUs |
| GroundingDINO taxonomy-prompt rows | `evaluation/gdino_baseline_eval_tax_v4.py` | `gold_taxonomy_prompts_v4.json` (Zenodo, or `benchmark/derive_taxonomy_prompts.py`) |
| Training corpus | `construction/` scripts, steps 1 to 6 as numbered in their docstrings, then the mixed-format and negatives builders | the eight source datasets, GroundingDINO and SAM2 weights |
| Benchmark | `benchmark/` scripts per round, then `verify_v4_disjointness.py` (the audit script of the reproducibility statement); candidate lists by `build_recognition_eval_v3.py`, `build_recognition_eval_new.py`, `merge_recognition_manifest_v4.py` | the corpus files |
| Per-source table (appendix) | `score.py --sections persource` | same as Table 2 |
| Pseudo-label audit | `audit/audit_sample.json` (shipped); re-drawing with `audit/sample_audit_images.py` | `agroground_granularity.jsonl` from construction step 4 (the every-200th-row draw is row-order sensitive) |
| Trained models | `configs/sft/*.sh` with the qwen-vl-finetune toolchain; `evaluation/merge_c2v2.py`; `configs/rl/*.sh` with verl | the corpus |

### Scoring conventions

Localization uses greedy one-to-one matching at IoU 0.5, micro-pooled over positive images;
region IoU is the IoU of the unions of predicted and ground-truth boxes; recognition is scored
among the four candidates; joint accuracy is correct recognition with region IoU of at least
0.5; abstention is an empty output on a healthy image. Intervals are 95% percentile bootstraps
over images (1,000 resamples; a fresh generator seeded with 0 for every statistic). Paired
differences align the two models' predictions by image id, resample the same images for both,
and report the observed difference with its interval.

### Seeds

| Step | Seed |
|---|---|
| Instruction-template choice (`convert_to_qwen.py`) and 80/10/10 split | 42 |
| Candidate-format conversion of the training data | 31 |
| Healthy-negative selection; positive-phrasing rewrite | 47; 51 |
| Benchmark candidate selection, rounds 1 to 4b | 2024, 31, 33, 44, 45 |
| Candidate lists for the recognition protocol | 2026 |
| Training (main model; seed study) | 42; 1, 2 |
| Equal-size and scaling subsets | 11 |
| Rollout-prompt selection for RL | 2027 |
| Pseudo-label audit sample | 7 |
| Control experiments | 23 |
| Qualitative figures (grounding; recognition) | 3; 11 |

### Errata to the submitted version

Running `score.py` on the released files reproduces every per-model number in the paper.

1. For paired F1 differences between two models whose prediction files were written in
   different row orders (the GroundingDINO, VisionReasoner and Seg-Zero baselines, the 8B
   model and the expert-label and subset runs, against the 2B models), the submitted
   version's intervals were computed on rows paired by position rather than by image, which
   makes them wider than a correctly paired test. Point estimates change by at most 0.001
   (five comparisons where the submitted value was the bootstrap mean rather than the
   observed difference); no conclusion is reversed; one comparison that was reported as
   unresolved (the 12% model against GroundingDINO with raw prompts, +0.040) is resolved
   under correct pairing, with interval [+0.018, +0.063]. The corrected values for every
   paired interval printed in the paper, including the PlantSeg table caption and the
   annotator-configuration comparisons, are those printed by `score.py`.
2. The pipeline's SAM2 masks do not filter boxes. The submitted text says boxes with empty
   masks are removed; in the code the mask coverage only decides whether a mask file is
   saved (`construction/final_gdino_script.py`). The box filters are the area ratio (0.1% to
   95% of the image), degenerate and duplicate boxes, and, in the granularity step, the 50%
   ceiling, the 0.1% floor and NMS at IoU 0.5.
3. Seeds 42 (instruction templates and split) and 2027 (RL rollout prompts), and the
   21 + 5 synonym mappings (25 unique; the text says 23), will be added to the
   reproducibility statement.
4. After submission, the released audit script (`benchmark/verify_v4_disjointness.py`) found
   two benchmark images that are copies of images used as healthy training negatives: a CDDM
   image that carries both a disease-labeled and a healthy-labeled sample in the source data,
   and an AgroMind image that is pixel-identical to a LeafNet healthy image (the submitted
   version's checks compared positives to the training split, but compared the negatives only
   against healthy benchmark images). Benchmark v4.1 (1,478 images: 508 lesion-level, 465
   object-level, 505 healthy) removes both and passes the full check; v4 remains on Zenodo so
   the submitted tables reproduce exactly (`score.py --exclude 320,3347` reproduces v4.1 from
   v4). On v4.1 every headline number is unchanged; four Table 2 F1 values move by 0.001,
   false abstention by 0.1 points, and recognition accuracies by at most 0.2 points. The next
   version of the paper reports v4.1.

## Source datasets

CDDM, LeafNet and LeafBench, MIRAGE, AgroMind, AgroCoT, AgroBench and AgMMU. Please cite the
source datasets when using the corresponding shards; their licenses are listed in the paper's
licensing appendix and repeated on the Zenodo record.

## Citation

```
[BibTeX to be added after the arXiv posting]
```
