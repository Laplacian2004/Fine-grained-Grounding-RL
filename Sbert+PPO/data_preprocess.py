# prepare_data.py
import json
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL = "all-MiniLM-L6-v2"
EMB_DIM = 384
OUT = "dataset_precomp.npz"

model = SentenceTransformer(MODEL)


def extract_grounded_from_summary(summary):
    import re
    grounded_re = re.compile(r'\[\s*(\d+)\s+([^\]]+?)\s*\]')
    out = []
    for m in grounded_re.finditer(summary):
        out.append((int(m.group(1)), m.group(2).strip()))
    return out


data = []
with open("train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        # get grounded spans and map to sources
        grounded = extract_grounded_from_summary(obj.get("summary", ""))
        for gid, gtext in grounded:
            key = f"source{gid}"
            if key not in obj or not obj[key].strip():
                continue
            source = obj[key]
            idx = source.find(gtext)
            if idx == -1:
                idx = source.lower().find(gtext.lower())
                if idx == -1:
                    continue
            start = idx
            end = idx + len(gtext) - 1  # inclusive
            # compute embeddings
            gemb = model.encode(gtext, convert_to_numpy=True,
                                normalize_embeddings=True)
            # optionally: compute full source embedding
            # semb = model.encode(source, convert_to_numpy=True)
            data.append({
                "ground_text": gtext,
                "ground_emb": gemb,
                "source": source,
                "start": start,
                "end": end
            })
print(data[0])
# Save compactly: embeddings as float32 arrays
ground_embs = np.stack([d["ground_emb"].astype(np.float32) for d in data])
sources = [d["source"] for d in data]
starts = np.array([d["start"] for d in data], dtype=np.int32)
ends = np.array([d["end"] for d in data], dtype=np.int32)

np.savez_compressed(OUT, ground_embs=ground_embs, sources=np.array(
    sources, dtype=object), starts=starts, ends=ends)
print("Saved", OUT, "items:", len(data))
