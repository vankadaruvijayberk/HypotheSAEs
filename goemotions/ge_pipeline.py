"""Shared helpers for the GoEmotions HypotheSAEs experiment."""
import os, gc, json, random
import numpy as np

DRIVE_ROOT = os.environ.get("GE_DRIVE_ROOT", "/content/drive/MyDrive/hypothesaes_goemotions")
EMB_CACHE_DIR = os.path.join(DRIVE_ROOT, "emb_cache")
CKPT_DIR = os.path.join(DRIVE_ROOT, "checkpoints")
ANNOT_DIR = os.path.join(DRIVE_ROOT, "annotation_cache")
RESULTS_DIR = os.path.join(DRIVE_ROOT, "results")
SEED = 0
HELDOUT_CAP = 2000

os.environ["EMB_CACHE_DIR"] = EMB_CACHE_DIR

EMBEDDER = "nomic-ai/modernbert-embed-base"
NOMIC_PREFIX = "classification: "   # ablated in notebook 01

EKMAN = {
    "anger":    ["anger", "annoyance", "disapproval"],
    "disgust":  ["disgust"],
    "fear":     ["fear", "nervousness"],
    "joy":      ["amusement", "approval", "excitement", "gratitude", "joy", "love",
                 "optimism", "relief", "pride", "admiration", "desire", "caring"],
    "sadness":  ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
}
FINE_TARGETS = ["curiosity", "disappointment"]
TARGETS = list(EKMAN.keys()) + FINE_TARGETS

TASK_INSTRUCTIONS = """All texts are Reddit comments labeled for the emotion they express.
Features should describe a concrete lexical or topical aspect of the comment, for example:
- "asks a direct question seeking information"
- "expresses thanks with phrases like 'thank you' or 'much appreciated'\""""

NO_THINK = {"temperature": 0.0, "max_output_tokens": 64,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def clear_mem():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _json_default(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


def log_json(name, payload):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name if name.endswith(".json") else name + ".json")
    data = {}
    if os.path.exists(path):
        try:
            with open(path) as f: data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    data.update(payload)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    os.replace(tmp, path)
    return path


def read_json(name):
    path = os.path.join(RESULTS_DIR, name if name.endswith(".json") else name + ".json")
    with open(path) as f:
        return json.load(f)


def redirect_annotation_cache():
    os.makedirs(ANNOT_DIR, exist_ok=True)
    import hypothesaes.annotate as _a
    import hypothesaes.interpret_neurons as _i
    _a.CACHE_DIR = ANNOT_DIR
    _i.CACHE_DIR = ANNOT_DIR


def load_goemotions():
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    names = ds["train"].features["labels"].feature.names
    return ds, names


def build_targets(label_lists, names):
    """target -> binary array (6 Ekman buckets + 2 fine labels)."""
    idx = {n: i for i, n in enumerate(names)}
    rows = [set(r) for r in label_lists]
    out = {}
    for ek, fines in EKMAN.items():
        fidx = {idx[f] for f in fines}
        out[ek] = np.fromiter((1 if (s & fidx) else 0 for s in rows), dtype=np.int64, count=len(rows))
    for ft in FINE_TARGETS:
        fi = idx[ft]
        out[ft] = np.fromiter((1 if fi in s else 0 for s in rows), dtype=np.int64, count=len(rows))
    return out
