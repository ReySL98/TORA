# TORA_train.py
# CrossGrid experiment: cross-task LoRA adapter transfer for continual learning.
# Trains a target task using adapters from other source tasks as initialization,
# evaluating the effect of knowledge transfer across all task pairs.
#
# Usage:
#   python TORA_train.py [--targets boolq,copa] [--sources dbpedia,yelp]
#                        [--epochs_grid 10,15,20] [--source_epochs_grid 1,10,15,20]
#                        [--lr_transfer 5e-5] [--lr_scratch 5e-5] [--rank_r 64]
#                        [--skip_no_reuse] [--include_self]

import os
import time
import json
import glob
import gc
import argparse
from typing import Optional, Tuple, List

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score

from transformers import (
    DistilBertTokenizerFast,
    DistilBertTokenizer,
    DistilBertModel,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer
)
from peft import LoraConfig, get_peft_model, PeftModel

# =============================================================================
# EXPERIMENT CONFIG
# Controls output directories and global runtime flags.
# =============================================================================

def get_next_experiment_id(prefix="exp"):
    """Scan existing exp_* folders and return the next available integer ID."""
    ids = []
    for p in glob.glob(f"{prefix}_*"):
        base = os.path.basename(p)
        try:
            ids.append(int(base.split("_")[1]))
        except Exception:
            pass
    return (max(ids) + 1) if ids else 1


# Set EXPERIMENT_ID manually to reuse an existing folder,
EXPERIMENT_ID = 25
EXPERIMENT_DIR = f"exp_{EXPERIMENT_ID}"
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

MEMORY_FILE = os.path.join(EXPERIMENT_DIR, f"memory_cls_{EXPERIMENT_ID}.json")
ADAPTER_ROOT = os.path.join(EXPERIMENT_DIR, "lora_weights")
os.makedirs(ADAPTER_ROOT, exist_ok=True)

# Set to True to save [CLS] hidden states after each evaluation run.
save_representations = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128

# =============================================================================
# TASK DEFINITIONS
# 15 text classification benchmarks spanning GLUE and SuperGLUE.
# Each entry defines the HuggingFace dataset name, optional subset, text
# column(s)
# =============================================================================

TASKS = {

    # --- Standard Benchmarks ---
    "yelp": {
        "name": "yelp_polarity",
        "subset": "plain_text",
        "text_col": "text",
        "task_text": "Classify Yelp reviews as positive or negative"
    },
    "amazon": {
        "name": "amazon_polarity",
        "subset": "amazon_polarity",
        "text_col": "content",
        "task_text": "Classify Amazon reviews as positive or negative",
    },
    "dbpedia": {
        "name": "dbpedia_14",
        "subset": "dbpedia_14",
        "text_col": "content",
        "task_text": "Classify DBPedia articles into categories"
    },
    "yahoo": {
        "name": "yahoo_answers_topics",
        "subset": "yahoo_answers_topics",
        # Use text_pair to leverage both the question title and body
        "text_pair": ["question_title", "question_content"],
        "task_text": "Classify Yahoo questions into topics",
        # Yahoo uses "topic" instead of the standard "label" column
        "label_col": "topic"
    },
    "agnews": {
        "name": "ag_news",
        "text_col": "text",
        "task_text": "Classify news into four topics"
    },

    # --- GLUE ---
    "mnli": {
        "name": "glue",
        "subset": "mnli",
        "text_pair": ["premise", "hypothesis"],
        "task_text": "Determine whether a hypothesis is entailed by a premise"
    },
    "qqp": {
        "name": "glue",
        "subset": "qqp",
        "text_pair": ["question1", "question2"],
        "task_text": "Determine whether two questions are paraphrases"
    },
    "rte": {
        "name": "glue",
        "subset": "rte",
        "text_pair": ["sentence1", "sentence2"],
        "task_text": "Determine whether sentence2 is entailed by sentence1"
    },
    "sst2": {
        "name": "glue",
        "subset": "sst2",
        "text_col": "sentence",
        "task_text": "Sentiment analysis of movie reviews"
    },

    # --- SuperGLUE ---
    "wic": {
        "name": "super_glue",
        "subset": "wic",
        "text_pair": ["sentence1", "sentence2"],
        "task_text": "Word sense disambiguation: determine if the target word has the same meaning"
    },
    "cb": {
        "name": "super_glue",
        "subset": "cb",
        "text_pair": ["premise", "hypothesis"],
        "task_text": "Determine entailment, contradiction, or neutrality"
    },
    "copa": {
        "name": "super_glue",
        "subset": "copa",
        "text_pair": ["premise", "choice1"],  # binary: choice1 vs choice2
        "task_text": "Choose the most plausible alternative based on a premise"
    },
    "boolq": {
        "name": "super_glue",
        "subset": "boolq",
        "text_pair": ["passage", "question"],
        "task_text": "Answer a yes/no question based on a passage"
    },
    "multirc": {
        "name": "super_glue",
        "subset": "multirc",
        "text_pair": ["paragraph", "question"],
        "task_text": "Answer multi-sentence QA questions"
    },

    # --- Standalone ---
    "imdb": {
        "name": "imdb",
        "subset": "plain_text",
        "text_col": "text",
        "task_text": "Sentiment analysis of movie reviews"
    }
}

# =============================================================================
# TASK MEMORY
# Stores adapter paths across runs in a JSON file.
# Used to track which source adapter was used for each training run.
# =============================================================================

# Shared tokenizer and encoder for computing task description embeddings.
bert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(DEVICE)


def load_memory() -> List[dict]:
    """Load the task memory from disk. Returns an empty list if not yet created."""
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)


def save_to_memory(task_name: str, emb: torch.Tensor, adapter: str, based_on: Optional[str] = None):
    """
    Append a new entry to the task memory file.

    Args:
        task_name: Key of the task (e.g. "boolq").
        emb: Task description embedding tensor.
        adapter: Path to the saved adapter for this run.
        based_on: Source task name if transfer was used, else None.
    """
    memory = load_memory()
    memory.append({
        "task": task_name,
        "embedding": emb.tolist(),
        "adapter": adapter,
        "based_on": based_on
    })
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f)

# =============================================================================
# DATASET UTILITIES
# Handles loading, label normalization, tokenization, and subsampling
# for all 15 benchmarks in a unified pipeline.
# =============================================================================

def compute_metrics(eval_pred) -> dict:
    """Compute accuracy from HuggingFace Trainer eval predictions."""
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}


def normalize_label(lbl) -> int:
    """
    Normalize heterogeneous label formats across benchmarks to a plain int.

    Handles the following cases:
    - MultiRC: labels arrive as single-element lists or dicts with a "value" key.
    - BoolQ: labels are Python booleans.
    - Strings: numeric strings are cast to int.
    - NumPy scalars, plain ints, and floats are cast directly.

    Args:
        lbl: Raw label value from the dataset.

    Returns:
        Integer label, defaulting to 0 on unrecognized formats.
    """
    if isinstance(lbl, list):
        return int(lbl[0])  # take first element regardless of list length

    if isinstance(lbl, dict) and "value" in lbl:
        return int(lbl["value"])

    if isinstance(lbl, bool):
        return int(lbl)

    if isinstance(lbl, str):
        try:
            return int(lbl)
        except:
            return 0

    if isinstance(lbl, np.generic):
        return int(lbl)

    if isinstance(lbl, (int, float)):
        return int(lbl)

    return 0


def prepare_dataset(task_key: str, task_info: dict, tokenizer) -> DatasetDict:
    """
    Load, preprocess, subsample, and tokenize a dataset for a given task.

    Steps:
    1. Load from HuggingFace Hub using name and optional subset.
    2. Rename non-standard label columns to "label" (e.g. Yahoo's "topic").
    3. Build text inputs: single column or concatenated pair with [SEP].
    4. Tokenize with padding to MAX_LENGTH=128.
    5. Subsample to 2500 training examples (shuffled) and split 80/20.

    Args:
        task_key: Task identifier (e.g. "boolq").
        task_info: Task config dict from TASKS.
        tokenizer: HuggingFace tokenizer instance.

    Returns:
        DatasetDict with "train" and "test" splits in torch format.
    """
    name, subset = task_info["name"], task_info.get("subset")
    ds = load_dataset(name, subset) if subset else load_dataset(name)

    # Rename non-standard label columns to "label" to unify the pipeline.
    lbl_col = task_info.get("label_col", "label")
    if lbl_col != "label":
        if isinstance(ds, DatasetDict):
            for split_name in ds:
                if lbl_col in ds[split_name].column_names:
                    ds[split_name] = ds[split_name].rename_column(lbl_col, "label")
        else:
            if lbl_col in ds.column_names:
                ds = ds.rename_column(lbl_col, "label")

    # Build the text extraction function based on task type.
    # Sentence-pair tasks concatenate with [SEP]; single-text tasks use one column.
    if "text_pair" in task_info:
        col1, col2 = task_info["text_pair"]
        def text_func(ex):
            t1 = ex.get(col1, "")
            t2 = ex.get(col2, "")
            if not isinstance(t1, str): t1 = str(t1)
            if not isinstance(t2, str): t2 = str(t2)
            return (t1.strip() or "empty text") + " [SEP] " + (t2.strip() or "empty text")
    else:
        col = task_info["text_col"]
        def text_func(ex):
            t = ex.get(col, "")
            if not isinstance(t, str):
                t = str(t)
            return t.strip() or "empty text"

    def preprocess(batch):
        texts = []
        for i in range(len(batch["label"])):
            ex = {k: batch[k][i] for k in batch}
            txt = text_func(ex)
            if not isinstance(txt, str):
                txt = str(txt)
            if len(txt.strip()) == 0:
                txt = "empty text"
            texts.append(txt)

        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors=None  # return lists; HuggingFace converts to tensors internally
        )

        labels = [normalize_label(l) for l in batch["label"]]
        output = {k: v for k, v in tokenized.items()}
        output["labels"] = labels
        return output

    # Shuffle before slicing to avoid class imbalance in ordered datasets
    # (e.g. DBPedia groups all examples of a class together, so an unshuffled
    # slice would contain only one class and confuse the model).
    if isinstance(ds, DatasetDict):
        full_train = ds["train"].shuffle(seed=42)
        small = full_train.select(range(min(2500, len(full_train))))
    else:
        full_ds = ds.shuffle(seed=42)
        small = full_ds.select(range(min(2500, len(full_ds))))

    split = small.train_test_split(test_size=0.2)
    return split.map(preprocess, batched=True).with_format("torch")

# =============================================================================
# ADAPTER HELPERS
# Utilities for locating, listing, and validating saved LoRA adapter folders.
# =============================================================================

def adapter_dir(task: str, epoch: int) -> str:
    """Return the expected path for a task adapter saved at a given epoch."""
    return os.path.join(ADAPTER_ROOT, task, f"e{epoch}")


def has_adapter(task: str, epoch: int) -> bool:
    """Check whether a valid adapter exists for a task at a given epoch."""
    d = adapter_dir(task, epoch)
    cfg = os.path.join(d, "adapter_config.json")
    safet = os.path.join(d, "adapter_model.safetensors")
    binf = os.path.join(d, "adapter_model.bin")
    return os.path.exists(cfg) and (os.path.exists(safet) or os.path.exists(binf))


def list_task_adapters(task_key: str) -> List[str]:
    """
    List all saved adapter paths for a task, sorted by epoch number ascending.
    The non-versioned adapter (no _epN suffix) is placed last.

    Args:
        task_key: Task identifier (e.g. "dbpedia").

    Returns:
        Sorted list of adapter folder paths.
    """
    pats = [
        os.path.join(ADAPTER_ROOT, f"{task_key}_lora"),
        os.path.join(ADAPTER_ROOT, f"{task_key}_lora_ep*"),
    ]
    found = []
    for p in pats:
        found.extend(glob.glob(p))

    def ep_num(path):
        base = os.path.basename(path)
        if "_ep" in base:
            try:
                return int(base.split("_ep")[1])
            except:
                return -1
        return 10**9  # non-versioned adapter sorts last

    return sorted(found, key=ep_num)


def best_adapter_for_source(source_task: str) -> Optional[str]:
    """Return the path of the most trained adapter for a source task (highest epoch)."""
    cands = list_task_adapters(source_task)
    return cands[-1] if cands else None


def parse_ep_from_path(path: str) -> Optional[int]:
    """Extract the epoch number from an adapter folder name (e.g. 'yelp_lora_ep20' → 20)."""
    base = os.path.basename(path)
    if "_ep" in base:
        try:
            return int(base.split("_ep")[1])
        except:
            return None
    return None


def _append_df_safe(df: pd.DataFrame, out_path: str):
    """
    Append a DataFrame to a CSV file. Creates the file with header on first call;
    subsequent calls append rows without repeating the header.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not os.path.exists(out_path):
        df.to_csv(out_path, index=False)
    else:
        df.to_csv(out_path, index=False, mode="a", header=False)

# =============================================================================
# TRAINING
# Core training function. Builds a DistilBERT + LoRA model, optionally loads
# LoRA weights from a source adapter as initialization (Boosting), then trains
# and evaluates on the target task.
# =============================================================================

def train_task(
    task_key: str,
    num_train_epochs: int = 1,
    lr: float = 5e-4,
    r: int = 64,
    adapter_suffix: str = "",
    init_adapter_path: Optional[str] = None,   # if set, load LoRA weights from this path
    based_on_override: Optional[str] = None    # source task name for CSV reporting
) -> dict:
    """
    Train a LoRA adapter on a target task, with optional transfer initialization.

    If init_adapter_path is provided, the LoRA weights from that adapter are
    loaded as the starting point before training (Boosting protocol). Only LoRA
    weights are transferred; the classifier head is always initialized fresh for
    the target task to avoid label space conflicts.

    Args:
        task_key: Key of the target task in TASKS (e.g. "boolq").
        num_train_epochs: Number of training epochs.
        lr: Learning rate.
        r: LoRA rank.
        adapter_suffix: String appended to the output adapter folder name.
        init_adapter_path: Path to a source adapter folder to initialize from.
        based_on_override: Source task name to record in results (for reporting).

    Returns:
        Dictionary with training results: task, transfer source, accuracy,
        training time, epochs, learning rate, mode, adapter path, and log history.
    """
    task_info = TASKS[task_key]
    print(f"\nProcessing task: {task_key.upper()}")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = prepare_dataset(task_key, task_info, tokenizer)

    # Infer number of classes from the training labels
    num_labels = len(set(dataset["train"]["labels"].tolist()))
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_labels
    )

    # Inject LoRA into query, key, and value projections across all 6 layers.
    # modules_to_save ensures the task-specific classifier head is also stored
    # alongside the LoRA weights in each adapter folder.
    peft_cfg = LoraConfig(
        r=r, lora_alpha=16, lora_dropout=0.1, bias="none",
        task_type="SEQ_CLS",
        target_modules=["q_lin", "k_lin", "v_lin"],
        modules_to_save=["classifier"]
    )
    model = get_peft_model(model, peft_cfg)

    # Ensure all trainable parameters are float32 before any weight loading.
    # Mixed precision issues can silently corrupt adapter weights.
    for name, p in model.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            print(f"⚠️ Fixing dtype of {name}: {p.dtype} → float32")
            p.data = p.data.float()

    based_on = None

    # --- Transfer initialization ---
    # Load LoRA weights from the source adapter if a path was provided.
    # Only keys containing "lora_" are transferred; the classifier head
    # of the source task is deliberately excluded to avoid label space conflicts.
    if init_adapter_path is not None:
        based_on = based_on_override or "manual"
        print(f"↪ Initializing {task_key} from: {os.path.basename(init_adapter_path)} (source={based_on})")

        cand = [
            os.path.join(init_adapter_path, "adapter_model.safetensors"),
            os.path.join(init_adapter_path, "adapter_model.bin"),
        ]
        weights_path = next((p for p in cand if os.path.exists(p)), None)

        if weights_path is None:
            print("⚠️ Adapter weights file not found. Training from scratch.")
            based_on = None
        else:
            if weights_path.endswith(".safetensors"):
                import safetensors.torch as st
                state = st.load_file(weights_path, device="cpu")
            else:
                try:
                    state = torch.load(weights_path, map_location="cpu", weights_only=True)
                except TypeError:
                    state = torch.load(weights_path, map_location="cpu")

            from peft import set_peft_model_state_dict
            filtered = {k: v for k, v in state.items() if "lora_" in k}
            set_peft_model_state_dict(model, filtered)
            print("✅ LoRA weights loaded (LoRA only, classifier excluded).")

    # Unfreeze LoRA parameters and the classifier head.
    # The DistilBERT backbone remains frozen throughout training.
    for name, param in model.named_parameters():
        param.requires_grad = ("lora_" in name) or ("classifier" in name)

    # Debug: verify dtype of all trainable parameters after weight loading.
    # Any parameter still in a non-float32 dtype is corrected in place.
    print("\n🔍 DTYPE CHECK (before training)")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {name} → {p.dtype}")
            if p.dtype != torch.float32:
                print(f"   ⚠️ Converting {name} to float32...")
                p.data = p.data.float()

    model.to(DEVICE)

    training_args = TrainingArguments(
        output_dir=os.path.join(EXPERIMENT_DIR, f"results_{task_key}{adapter_suffix or ''}"),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=num_train_epochs,
        learning_rate=lr,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Sanity check: print trainable parameter count and total LoRA weight magnitude.
    # Sum(|LoRA|) should be > 0 if source weights were successfully loaded.
    trainables = [n for n, p in model.named_parameters() if p.requires_grad]
    print("Trainable parameters:", len(trainables), trainables[:10])
    with torch.no_grad():
        s = sum(p.detach().abs().sum().item()
                for n, p in model.named_parameters() if "lora_" in n)
    print("Sum(|LoRA|) pre-train:", round(s, 4))

    start = time.time()
    trainer.train()
    metrics = trainer.evaluate()
    end = time.time()

    # Optionally save [CLS] hidden states from the final evaluation pass.
    # Useful for downstream t-SNE or representation analysis.
    if save_representations:
        eval_dataloader = trainer.get_eval_dataloader()
        all_embeddings, all_labels = [], []
        model.eval()
        for batch in eval_dataloader:
            inputs = {k: v.to(model.device) for k, v in batch.items()
                      if k in ["input_ids", "attention_mask", "token_type_ids"]}
            labels = batch.get("labels", None)
            with torch.no_grad():
                outputs = model.base_model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
            all_embeddings.append(hidden)
            if labels is not None:
                all_labels.append(labels.cpu().numpy())

        np.save(os.path.join(EXPERIMENT_DIR, f"embeddings_{task_key}.npy"), np.vstack(all_embeddings))
        np.save(os.path.join(EXPERIMENT_DIR, f"labels_{task_key}.npy"), np.concatenate(all_labels))
        print(f"✅ Representations saved to {EXPERIMENT_DIR}/embeddings_{task_key}.npy")

    # Save the adapter. Try safetensors first; fall back to .bin on error.
    suffix = adapter_suffix or ""
    adapter_out = os.path.join(ADAPTER_ROOT, f"{task_key}_lora{suffix}")
    os.makedirs(adapter_out, exist_ok=True)
    try:
        model.save_pretrained(adapter_out)
    except Exception as e:
        print(f"⚠️ Error saving with safetensors: {e}")
        print("💾 Retrying in .bin format...")
        model.save_pretrained(adapter_out, safe_serialization=False)

    # Save epoch-by-epoch training logs alongside the adapter weights.
    try:
        log_history_path = os.path.join(adapter_out, "training_logs.json")
        with open(log_history_path, "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        print(f"Logs saved to: {log_history_path}")
    except Exception as e:
        print(f"Warning: could not save training logs. Error: {e}")

    save_to_memory(
        task_key, h_new, adapter_out,
        based_on=based_on if init_adapter_path is None else based_on_override
    )

    del model, dataset, h_new
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "Task": task_key,
        "Transfer": (based_on_override if init_adapter_path is not None
                     else (based_on if based_on else "None")),
        "Acc": round(metrics.get("eval_accuracy", 0.0), 4),
        "Time (s)": round(end - start, 2),
        "Epochs": num_train_epochs,
        "LR": lr,
        "Mode": "TRANSFER" if (init_adapter_path is not None or based_on) else "NO_REUSE",
        "Adapter": adapter_out,
        "log_history": trainer.state.log_history
    }

# =============================================================================
# CROSSGRID
# Runs the full cross-task transfer experiment: for each target task, trains
# from scratch (NO_REUSE) and from every available source adapter (TRANSFER),
# across all combinations of target and source epoch counts.
# Results are saved per-target and appended to a global CSV.
# =============================================================================

def crossgrid_transfer(
    target: str,
    sources: Optional[List[str]] = None,
    target_epochs: Tuple[int, ...] = (1, 2, 3, 5, 10),
    source_epochs: Optional[Tuple[int, ...]] = None,
    lr_transfer: float = 5e-5,
    lr_scratch: float = 5e-5,
    r: int = 64,
    skip_no_reuse: bool = False,
    include_self: bool = False
) -> pd.DataFrame:
    """
    Run the CrossGrid experiment for a single target task.

    For each combination of (source task, source epoch, target epoch):
    - NO_REUSE: train target from random LoRA initialization.
    - TRANSFER: train target initialized from the source adapter weights.

    Args:
        target: Target task key (e.g. "boolq").
        sources: List of source tasks. If None, uses all tasks except target
                 (or all tasks including target if include_self=True).
        target_epochs: Tuple of epoch counts to train the target task for.
        source_epochs: Epoch versions of source adapters to consider.
                       If None, uses all available versions per source.
        lr_transfer: Learning rate for transfer runs.
        lr_scratch: Learning rate for no-reuse (scratch) runs.
        r: LoRA rank.
        skip_no_reuse: If True, skip the from-scratch baseline runs.
        include_self: If True, allow the target task to also be a source.

    Returns:
        DataFrame with one row per (config, target epoch) combination.
    """
    if sources is None:
        sources = list(TASKS.keys()) if include_self else [s for s in TASKS.keys() if s != target]

    results = []

    # NO_REUSE baseline: train target from scratch for each epoch count
    if not skip_no_reuse:
        for ep_t in target_epochs:
            print(f"\n[NO_REUSE] {target} - {ep_t} epochs")
            res_nr = train_task(
                target, num_train_epochs=ep_t, lr=lr_scratch,
                r=r,
                adapter_suffix=f"_crossgrid_tgt_ep{ep_t}"
            )
            res_nr["Config"] = f"NO_REUSE_{ep_t}ep"
            res_nr["BasedOnEpoch"] = "-"
            results.append(res_nr)

    # TRANSFER: for each source task and each of its available adapter epochs
    for src in sources:
        cands = list_task_adapters(src)
        if not cands:
            print(f"⚠️ Source {src} has no saved adapters. Skipping.")
            continue

        # Filter adapter versions by source_epochs if specified
        adapters = []
        for p in cands:
            ep_s = parse_ep_from_path(p)
            if source_epochs is None or (ep_s in source_epochs):
                adapters.append((p, ep_s))

        if not adapters:
            print(f"⚠️ Source {src} has no adapters matching source_epochs={source_epochs}.")
            continue

        for ep_t in target_epochs:
            for (path_s, ep_s) in adapters:
                ep_s_str = f"ep{ep_s}" if ep_s is not None else "base"
                print(f"\n[TRANSFER] {target} - {ep_t} epochs  <=  {src} ({ep_s_str})")
                res_tr = train_task(
                    target, num_train_epochs=ep_t, lr=lr_transfer,
                    r=r,
                    init_adapter_path=path_s,
                    based_on_override=src,
                    adapter_suffix=f"_xfer_from_{src}_{ep_s_str}_tgt_ep{ep_t}"
                )
                res_tr["Config"] = f"XFER_{ep_t}ep_from_{src}_{ep_s_str}"
                res_tr["BasedOnEpoch"] = ep_s if ep_s is not None else "-"
                results.append(res_tr)

    df = pd.DataFrame(results)
    fn = os.path.join(EXPERIMENT_DIR, f"crossgrid_{target}.csv")
    df.to_csv(fn, index=False)
    print("\nCROSSGRID SUMMARY")
    cols = ["Config", "Task", "Transfer", "BasedOnEpoch", "Epochs", "Acc", "Time (s)"]
    try:
        print(df[cols].sort_values(["Epochs", "Config"]))
    except Exception:
        print(df.head(20))
    print(f"💾 Saved: {fn}")
    return df


def crossgrid_all(
    targets: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
    target_epochs: Tuple[int, ...] = (10, 15, 20),
    source_epochs: Optional[Tuple[int, ...]] = None,
    lr_transfer: float = 5e-5,
    lr_scratch: float = 5e-5,
    r: int = 64,
    skip_no_reuse: bool = False,
    include_self: bool = False
):
    """
    Run CrossGrid over multiple target tasks sequentially.

    Saves a per-target CSV (crossgrid_<target>.csv) after each task and
    appends all results to a global file (crossgrid_ALL.csv). If a target
    fails, the error is logged and execution continues with the next task.

    Args:
        targets: List of target task keys. If None, runs all 15 tasks.
        sources: List of source task keys. If None, uses all tasks except target.
        target_epochs: Epoch counts to train each target task for.
        source_epochs: Epoch versions of source adapters to use. If None, uses all.
        lr_transfer: Learning rate for transfer runs.
        lr_scratch: Learning rate for from-scratch runs.
        r: LoRA rank.
        skip_no_reuse: If True, skip from-scratch baseline runs.
        include_self: If True, allow self-transfer (target as its own source).
    """
    if targets is None:
        targets = list(TASKS.keys())

    global_out = os.path.join(EXPERIMENT_DIR, "crossgrid_ALL.csv")
    print(f"\nGlobal CSV (appended per task): {global_out}")

    for t in targets:
        print(f"\n==============================")
        print(f"CROSSGRID TARGET: {t}")
        print(f"==============================")
        try:
            df_t = crossgrid_transfer(
                target=t,
                sources=sources,
                target_epochs=target_epochs,
                source_epochs=source_epochs,
                lr_transfer=lr_transfer,
                lr_scratch=lr_scratch,
                r=r,
                skip_no_reuse=skip_no_reuse,
                include_self=include_self
            )
            _append_df_safe(df_t, global_out)
            cols = ["Config", "Task", "Transfer", "BasedOnEpoch", "Epochs", "Acc", "Time (s)"]
            try:
                print("\nAPPEND TO GLOBAL SUMMARY")
                print(df_t[cols].sort_values(["Task", "Epochs", "Config"]).head(12))
            except Exception:
                print(df_t.head(12))
        except Exception as e:
            print(f"Error on target '{t}'. Continuing with next.\nDetails: {repr(e)}")

    print(f"\nDone. Global results saved to: {global_out}")

# =============================================================================
# ENTRY POINT
# Parses command-line arguments and launches crossgrid_all.
# =============================================================================

def get_args() -> argparse.Namespace:
    """Parse command-line arguments for the CrossGrid experiment."""
    parser = argparse.ArgumentParser(
        description="TORA CrossGrid: cross-task LoRA adapter transfer training"
    )
    parser.add_argument("--targets", type=str, default=None,
        help="Comma-separated target tasks (e.g. boolq,copa,rte). If not set, uses all 15 tasks.")
    parser.add_argument("--sources", type=str, default=None,
        help="Comma-separated source tasks (e.g. dbpedia,yelp). If not set, uses all other tasks.")
    parser.add_argument("--epochs_grid", type=str, default="10,15,20",
        help="Target training epochs (e.g. 10,15,20).")
    parser.add_argument("--source_epochs_grid", type=str, default="1,10,15,20",
        help="Source adapter epoch versions to consider (e.g. 1,10,15,20).")
    parser.add_argument("--lr_transfer", type=float, default=5e-5,
        help="Learning rate for transfer runs.")
    parser.add_argument("--lr_scratch", type=float, default=5e-5,
        help="Learning rate for from-scratch runs.")
    parser.add_argument("--rank_r", type=int, default=64,
        help="LoRA adapter rank r.")
    parser.add_argument("--skip_no_reuse", action="store_true",
        help="Skip the NO_REUSE (from-scratch) baseline block.")
    parser.add_argument("--include_self", action="store_true", default=True,
        help="Allow a task to be its own source (self-transfer).")
    return parser.parse_args()


def parse_grid(s: Optional[str], default: Tuple[int, ...] = (10, 15, 20)) -> Tuple[int, ...]:
    """Parse a comma-separated string of integers into a tuple (e.g. '10,15,20' → (10,15,20))."""
    if not s:
        return default
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def main():
    global tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    print(f"Experiment folder: {EXPERIMENT_DIR}")
    print(f"Memory file:       {MEMORY_FILE}")
    print(f"Adapters root:     {ADAPTER_ROOT}")

    args = get_args()
    targets_list = [t.strip() for t in args.targets.split(",")] if args.targets else None
    sources_list = [s.strip() for s in args.sources.split(",")] if args.sources else None

    crossgrid_all(
        targets=targets_list,
        sources=sources_list,
        target_epochs=parse_grid(args.epochs_grid),
        source_epochs=parse_grid(args.source_epochs_grid),
        lr_transfer=args.lr_transfer,
        lr_scratch=args.lr_scratch,
        r=args.rank_r,
        skip_no_reuse=args.skip_no_reuse,
        include_self=args.include_self
    )


if __name__ == "__main__":
    main()
