import os
import json
import argparse
import time
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from emoatlas import EmoScores
from concurrent.futures import ThreadPoolExecutor


COLUMNS = [
    "json_file",
    "opinion_Edge",
    "reasoning_summary_Edge",
    "topic",
    "mode",
    "age",
    "biological_sex",
    "gender",
    "sexual_orientation",
    "occupation",
    "city_of_living",
    "employment_status",
    "education_level",
    "parents_education",
    "marital_status",
    "children",
    "migration_status",
    "psychological_feature",
    "religious_beliefs",
    "time_onsocialmedia",
    "hobbies",
    "ocean",
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

def safe_load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def parse_response_raw(obj: dict) -> Optional[dict]:
    rr = obj.get("response_raw")
    if isinstance(rr, str) and rr.strip():
        try:
            parsed = json.loads(rr)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None

def get_str_from_response(obj: dict, field: str) -> str:
    rp = obj.get("response_parsed")
    if isinstance(rp, dict):
        v = rp.get(field)
        if isinstance(v, str):
            return v.strip()

    rr_parsed = parse_response_raw(obj)
    if isinstance(rr_parsed, dict):
        v = rr_parsed.get(field)
        if isinstance(v, str):
            return v.strip()

    return ""

def find_selection(obj: dict) -> dict:
    sel = obj.get("selection")
    if isinstance(sel, dict):
        return sel

    rp = obj.get("response_parsed")
    if isinstance(rp, dict) and isinstance(rp.get("selection"), dict):
        return rp["selection"]

    rr = parse_response_raw(obj)
    if isinstance(rr, dict) and isinstance(rr.get("selection"), dict):
        return rr["selection"]

    return {}

def ensure_nltk():
    import nltk
    for pkg in ["punkt", "wordnet", "omw-1.4", "averaged_perceptron_tagger"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

def edges_to_strings(edges: List[tuple]) -> List[str]:
    out: List[str] = []
    for e in edges:
        if not isinstance(e, (list, tuple)) or len(e) < 2:
            continue
        a, b = e[0], e[1]
        if not (isinstance(a, str) and isinstance(b, str)):
            continue
        if len(e) >= 3 and e[2] is not None and e[2] != "":
            out.append(f"{a},{b},{e[2]}")
        else:
            out.append(f"{a},{b}")
    return out

def extract_edges_for_text(text: str, emos) -> List[str]:
    fmnt = emos.formamentis_network(text)
    edges_list = list(getattr(fmnt, "edges", []))
    return edges_to_strings(edges_list)

def collect_sorted_json_files(input_dir: str) -> List[Tuple[str, str, str]]:
    items: List[Tuple[str, str, str]] = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if not fn.lower().endswith(".json"):
                continue
            abs_path = os.path.join(root, fn)
            rel_path = os.path.relpath(abs_path, input_dir)
            items.append((fn.lower(), rel_path.lower(), abs_path))
    items.sort(key=lambda x: (x[0], x[1]))
    return items

def chunked(lst: List[Any], size: int):
    for i in range(0, len(lst), size):
        yield i, lst[i:i + size]

def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds/60:.1f}min"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_name", default="TFMN_chunk", help="Prefix for output parquets.")
    parser.add_argument("--chunk_size", type=int, default=500)
    args = parser.parse_args()

    if len(COLUMNS) != 27:
        raise RuntimeError(f"COLUMNS must be 27, got {len(COLUMNS)}")

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("RUNNING SCRIPT:", os.path.abspath(__file__))
    print("INPUT DIR:", input_dir)
    print("OUTPUT DIR:", output_dir)
    print("CHUNK SIZE:", args.chunk_size)

    ensure_nltk()

    emos = EmoScores()

    files = collect_sorted_json_files(input_dir)
    total = len(files)
    print(f"Found JSON files: {total}")

    chunk_size = max(1, int(args.chunk_size))
    part = 0
    total_processed = 0
    total_skipped = 0
    global_start = time.time()

    for start_idx, batch in chunked(files, chunk_size):
        part += 1
        chunk_start = time.time()
        part_name = f"{args.base_name}_part{part:04d}.parquet"
        out_path = os.path.join(output_dir, part_name)

        rows: List[Dict[str, Any]] = []
        processed, skipped = 0, 0

        for fn_lower, rel_lower, json_path in batch:
            rel_path = os.path.relpath(json_path, input_dir)

            obj = safe_load_json(json_path)
            if not isinstance(obj, dict):
                skipped += 1
                continue

            row: Dict[str, Any] = {c: np.nan for c in COLUMNS}
            row["json_file"] = rel_path

            selection = find_selection(obj)
            persona = selection.get("persona") if isinstance(selection, dict) else None
            if not isinstance(persona, dict):
                persona = {}

            t_sel = selection.get("topic") if isinstance(selection.get("topic"), str) else ""
            m_sel = selection.get("mode") if isinstance(selection.get("mode"), str) else ""
            topic = t_sel.strip() if t_sel.strip() else get_str_from_response(obj, "topic")
            mode = m_sel.strip() if m_sel.strip() else get_str_from_response(obj, "mode")
            if topic:
                row["topic"] = topic
            if mode:
                row["mode"] = mode

            opinion = get_str_from_response(obj, "opinion")
            reasoning = get_str_from_response(obj, "reasoning_summary")

            if opinion:
                try:
                    row["opinion_Edge"] = extract_edges_for_text(opinion, emos)
                except Exception as e:
                    print(f"[WARN] emoatlas failed {rel_path} opinion: {e}")

            if reasoning:
                try:
                    row["reasoning_summary_Edge"] = extract_edges_for_text(reasoning, emos)
                except Exception as e:
                    print(f"[WARN] emoatlas failed {rel_path} reasoning_summary: {e}")

            for k in [
                "age","biological_sex","gender","sexual_orientation","occupation",
                "city_of_living","employment_status","education_level",
                "marital_status","children","migration_status",
                "psychological_feature","religious_beliefs","time_onsocialmedia"
            ]:
                if k in persona:
                    row[k] = persona.get(k)

            pe = persona.get("parents_education")
            if isinstance(pe, dict):
                row["parents_education"] = pe

            hobbies = persona.get("hobbies")
            if isinstance(hobbies, list):
                row["hobbies"] = hobbies

            ocean = persona.get("ocean")
            if isinstance(ocean, dict):
                row["ocean"] = ocean
                for trait in ["openness","conscientiousness","extraversion","agreeableness","neuroticism"]:
                    t = ocean.get(trait)
                    if isinstance(t, dict):
                        row[trait] = t

            rows.append(row)
            processed += 1

        df = pd.DataFrame(rows, columns=COLUMNS)

        for col in ["opinion_Edge", "reasoning_summary_Edge", "parents_education", "hobbies", "ocean",
                    "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)

        df.to_parquet(out_path)
        total_processed += processed
        total_skipped += skipped

        chunk_time = time.time() - chunk_start
        elapsed = time.time() - global_start
        end_idx = min(start_idx + chunk_size, total)
        remaining_files = total - end_idx
        avg_per_file = elapsed / end_idx if end_idx > 0 else 0
        eta = avg_per_file * remaining_files

        print(f"[OK] part {part:04d}  files {start_idx+1}-{end_idx} / {total} -> {out_path}   rows={df.shape[0]}  chunk_time={fmt_time(chunk_time)}  elapsed={fmt_time(elapsed)}  ETA={fmt_time(eta)}")

    total_time = time.time() - global_start
    print(f"\nDONE in {fmt_time(total_time)}")
    print("Total processed rows:", total_processed)
    print("Total skipped files:", total_skipped)
    print("Output folder:", output_dir)

if __name__ == "__main__":
    main()