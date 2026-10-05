"""
Dataset entries for the qwen-vl-finetune toolchain.

The SFT launchers select their training data with --dataset_use <name>. The toolchain
resolves the name through the dictionary in qwenvl/data/__init__.py of the
qwen-vl-finetune checkout. These are the entries used by the released launchers;
add them to that file (the entry definitions, then the names in the data_dict at the
bottom of the file).

Image paths inside the annotation files are stored as absolute paths at training
time, so data_path is left empty. Set WORK to the directory holding the files
produced by the construction scripts.
"""

WORK = "./work"

AGROGROUND_TRAIN = {"annotation_path": f"{WORK}/agroground_train.jsonl", "data_path": ""}                    # grounding only
AGROGROUND_RECOG_TRAIN = {"annotation_path": f"{WORK}/agroground_recog_train.jsonl", "data_path": ""}        # mixed
AGROGROUND_RECOGNEG2_TRAIN = {"annotation_path": f"{WORK}/agroground_recogneg2_train.jsonl", "data_path": ""}  # mixed + healthy (12%), main model
AGROGROUND_RECOGNEG3_TRAIN = {"annotation_path": f"{WORK}/agroground_recogneg3_train.jsonl", "data_path": ""}  # mixed + healthy (6%)
PLANTSEG_TRAIN = {"annotation_path": f"{WORK}/plantseg_sft_train.jsonl", "data_path": ""}                     # expert-label baseline (built locally from PlantSeg)
AGRO_PSEUDO_EQ = {"annotation_path": f"{WORK}/agroground_pseudo_eq.jsonl", "data_path": ""}                   # equal-size subset (9,120)
AGRO_SUB10K = {"annotation_path": f"{WORK}/agroground_sub10k.jsonl", "data_path": ""}                         # scaling study
AGRO_SUB100K = {"annotation_path": f"{WORK}/agroground_sub100k.jsonl", "data_path": ""}                       # scaling study

# Names used by the launchers (--dataset_use), to be added to the toolchain's data_dict:
data_dict = {
    "agroground_train": AGROGROUND_TRAIN,
    "agroground_recog_train": AGROGROUND_RECOG_TRAIN,
    "agroground_recogneg2_train": AGROGROUND_RECOGNEG2_TRAIN,
    "agroground_recogneg3_train": AGROGROUND_RECOGNEG3_TRAIN,
    "plantseg_train": PLANTSEG_TRAIN,
    "agroground_pseudo_eq": AGRO_PSEUDO_EQ,
    "agroground_sub10k": AGRO_SUB10K,
    "agroground_sub100k": AGRO_SUB100K,
}
