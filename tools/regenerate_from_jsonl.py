import json
import csv
from pathlib import Path

base = Path("outputs")
jsonl = base / "experiment_results.jsonl"
if not jsonl.exists():
    print("No experiment_results.jsonl found at outputs/")
    raise SystemExit(1)

raw = jsonl.read_text(encoding="utf-8")
# robust parse: handle multiple JSON objects even if concatenated on same line
items = []
decoder = json.JSONDecoder()
idx = 0
length = len(raw)
while idx < length:
    try:
        obj, end = decoder.raw_decode(raw, idx)
        items.append(obj)
        idx = end
        # skip whitespace/newlines between objects
        while idx < length and raw[idx].isspace():
            idx += 1
    except json.JSONDecodeError:
        # if a decode error occurs, try to move to next line boundary
        next_nl = raw.find('\n', idx)
        if next_nl == -1:
            break
        idx = next_nl + 1

# write experiment_results.json
out_json = base / "experiment_results.json"
summary = {
    "total_files": len(items),
    "results": items,
}
out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {out_json} ({len(items)} records)")

# write experiment_results.csv
csv_path = base / "experiment_results.csv"
with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile)
    header = [
        "source_file",
        "speaker_id",
        "utterance_id",
        "filename_valid",
        "response_mode",
        "status",
        "processing_seconds",
        "transcript",
        "normalized_text",
        "tagged_input",
        "response_text",
        "audio_output",
        "error",
    ]
    writer.writerow(header)
    for item in items:
        writer.writerow([
            item.get("source_file"),
            item.get("speaker_id"),
            item.get("utterance_id"),
            item.get("filename_valid"),
            item.get("response_mode"),
            item.get("status"),
            item.get("processing_seconds"),
            (item.get("transcript") or "").replace("\n", " "),
            (item.get("normalized_text") or "").replace("\n", " "),
            (item.get("tagged_input") or "").replace("\n", " "),
            (item.get("response_text") or "").replace("\n", " "),
            item.get("audio_output"),
            (item.get("error") or ""),
        ])
print(f"Wrote {csv_path}")
