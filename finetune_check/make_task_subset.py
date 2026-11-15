"""Cut a LeRobot v2.1 dataset down to a few tasks, as a new self-contained dataset folder.

The FastWAM data loader has no task filter, so training on "a few tasks" needs its own dataset
directory. This copies the selected episodes, renumbers them from 0 and rewrites the metadata.
Tasks are chosen by their instruction text (the dataset's task order differs from LIBERO's task ids).

    python finetune_check/make_task_subset.py \
        --src data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot \
        --out data/libero_hard_subset/libero_goal \
        --tasks "open the top drawer and put the bowl inside" "put the wine bottle on the rack"

Then point `data.train.dataset_dirs` at the new folder(s) and keep using the ORIGINAL checkpoint's
dataset_stats.json (`+data.train.pretrained_norm_stats=...`): stats recomputed on a subset would
change the action scaling the checkpoint was trained with.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def const_stats(values: np.ndarray) -> dict:
    """Stats entry in the same layout as episodes_stats.jsonl uses for scalar index columns."""
    v = values.astype(np.float64)
    return {"min": [float(v.min())], "max": [float(v.max())], "mean": [float(v.mean())],
            "std": [float(v.std())], "count": [int(v.size)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source LeRobot v2.1 dataset folder")
    ap.add_argument("--out", required=True, help="new dataset folder (must not exist)")
    ap.add_argument("--tasks", nargs="+", required=True, help="instruction strings to keep (exact match)")
    ap.add_argument("--max-episodes-per-task", type=int, default=None)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    info = json.loads((src / "meta/info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        raise SystemExit(f"Expected a LeRobot v2.1 dataset, found {info.get('codebase_version')}")
    if out.exists():
        raise SystemExit(f"{out} already exists; choose a new folder")

    episodes = read_jsonl(src / "meta/episodes.jsonl")
    known_tasks = [t["task"] for t in read_jsonl(src / "meta/tasks.jsonl")]
    missing = [t for t in args.tasks if t not in known_tasks]
    if missing:
        raise SystemExit("Not in this dataset: " + "; ".join(missing) + "\nAvailable: " + "; ".join(known_tasks))
    stats_by_ep = {r["episode_index"]: r for r in read_jsonl(src / "meta/episodes_stats.jsonl")}
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    chunk = int(info.get("chunks_size", 1000))

    keep_tasks = list(dict.fromkeys(args.tasks))  # keep order, drop duplicates
    new_task_index = {t: i for i, t in enumerate(keep_tasks)}
    per_task = {t: 0 for t in keep_tasks}
    selected = []
    for ep in episodes:
        task = ep["tasks"][0]
        if len(ep["tasks"]) != 1:
            raise SystemExit(f"Episode {ep['episode_index']} has several tasks; not supported")
        if task in new_task_index and (args.max_episodes_per_task is None or per_task[task] < args.max_episodes_per_task):
            per_task[task] += 1
            selected.append(ep)

    (out / "meta").mkdir(parents=True)
    new_episodes, new_stats, frame_cursor = [], [], 0
    for new_idx, ep in enumerate(selected):
        old_idx, task = ep["episode_index"], ep["tasks"][0]
        table = pq.read_table(src / info["data_path"].format(episode_chunk=old_idx // chunk, episode_index=old_idx))
        n = table.num_rows
        if n != ep["length"]:
            raise SystemExit(f"Episode {old_idx}: parquet has {n} rows, metadata says {ep['length']}")
        new_cols = {
            "episode_index": np.full(n, new_idx, dtype=np.int64),
            "index": np.arange(frame_cursor, frame_cursor + n, dtype=np.int64),
            "task_index": np.full(n, new_task_index[task], dtype=np.int64),
        }
        for name, values in new_cols.items():
            col = table.schema.get_field_index(name)
            table = table.set_column(col, table.schema.field(name), pa.array(values, type=table.schema.field(name).type))
        dst = out / info["data_path"].format(episode_chunk=new_idx // chunk, episode_index=new_idx)
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dst)

        for key in video_keys:
            v_src = src / info["video_path"].format(episode_chunk=old_idx // chunk, video_key=key, episode_index=old_idx)
            v_dst = out / info["video_path"].format(episode_chunk=new_idx // chunk, video_key=key, episode_index=new_idx)
            v_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(v_src, v_dst)

        new_episodes.append({"episode_index": new_idx, "tasks": [task], "length": n})
        stats = json.loads(json.dumps(stats_by_ep[old_idx]["stats"]))  # deep copy
        for name, values in new_cols.items():
            if name in stats:
                stats[name] = const_stats(values)
        new_stats.append({"episode_index": new_idx, "stats": stats})
        frame_cursor += n

    write_jsonl(out / "meta/episodes.jsonl", new_episodes)
    write_jsonl(out / "meta/episodes_stats.jsonl", new_stats)
    write_jsonl(out / "meta/tasks.jsonl", [{"task_index": i, "task": t} for t, i in new_task_index.items()])
    info.update({
        "total_episodes": len(selected),
        "total_frames": frame_cursor,
        "total_tasks": len(keep_tasks),
        "total_videos": len(selected) * len(video_keys),
        "total_chunks": (len(selected) - 1) // chunk + 1 if selected else 0,
        "splits": {"train": f"0:{len(selected)}"},
    })
    (out / "meta/info.json").write_text(json.dumps(info, indent=4))
    (out / "meta/subset_source.json").write_text(json.dumps({
        "source": str(src), "tasks": keep_tasks,
        "old_to_new_episode_index": {str(ep["episode_index"]): i for i, ep in enumerate(selected)},
    }, indent=2))

    print(f"wrote {out}: {len(selected)} episodes, {frame_cursor} frames")
    for task, count in per_task.items():
        print(f"  {count:3d} episodes  {task}")


if __name__ == "__main__":
    main()
