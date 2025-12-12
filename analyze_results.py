import os
import csv
import math
from collections import defaultdict, Counter

def read_csv(path):
    if not os.path.exists(path):
        return None, []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)

def to_float(v):
    try:
        return float(v)
    except Exception:
        return math.nan

def to_int(v):
    try:
        return int(float(v))
    except Exception:
        return 0

def to_bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in ("1", "true", "yes")

def avg(vals):
    vals = [x for x in vals if not math.isnan(x)]
    return sum(vals)/len(vals) if vals else math.nan

def summarize_by_env_noise(rows):
    groups = defaultdict(list)
    for r in rows:
        env = r.get("env", "unknown")
        noise = r.get("noise_std", "unknown")
        groups[(env, noise)].append(r)
    summaries = []
    for (env, noise), items in groups.items():
        n = len(items)
        success = sum(1 for r in items if to_bool(r.get("success")))
        planner_success = sum(1 for r in items if to_bool(r.get("planner_success")))
        collision = sum(1 for r in items if to_bool(r.get("collision")))
        path_len = avg([to_float(r.get("path_length")) for r in items])
        waypoints = avg([to_int(r.get("num_waypoints")) for r in items])
        nodes = avg([to_int(r.get("total_nodes")) for r in items])
        plan_time = avg([to_float(r.get("planning_time")) for r in items])
        summaries.append({
            "env": env,
            "noise_std": noise,
            "trials": n,
            "success_rate": round(success/n, 3) if n else math.nan,
            "planner_success_rate": round(planner_success/n, 3) if n else math.nan,
            "collision_rate": round(collision/n, 3) if n else math.nan,
            "avg_path_length": round(path_len, 4) if not math.isnan(path_len) else math.nan,
            "avg_num_waypoints": round(waypoints, 3) if not math.isnan(waypoints) else math.nan,
            "avg_total_nodes": round(nodes, 3) if not math.isnan(nodes) else math.nan,
            "avg_planning_time_s": round(plan_time, 4) if not math.isnan(plan_time) else math.nan,
        })
    return summaries

def summarize_generic(rows):
    if not rows:
        return []
    # For generic files (e.g., RRTCBFresults.csv), compute averages for numeric columns and counts for boolean columns
    fieldnames = rows[0].keys()
    numeric_cols, bool_cols = [], []
    for fn in fieldnames:
        v = rows[0].get(fn)
        try:
            float(v)
            numeric_cols.append(fn)
            continue
        except Exception:
            pass
        if str(v).strip().lower() in ("true", "false", "1", "0"):
            bool_cols.append(fn)
    summary = {}
    for fn in numeric_cols:
        vals = [to_float(r.get(fn)) for r in rows]
        m = avg(vals)
        summary[fn] = round(m, 4) if not math.isnan(m) else math.nan
    for fn in bool_cols:
        cnt = sum(1 for r in rows if to_bool(r.get(fn)))
        summary[f"{fn}_rate"] = round(cnt/len(rows), 3) if rows else math.nan
    return [summary]

def write_csv(path, rows):
    if not rows:
        return False
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return True

def print_table(title, rows):
    print(title)
    if not rows:
        print("(no data)")
        print()
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = " | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols)
        print(line)
    print()

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    abl_path = os.path.join(base, "ablation_results.csv")
    rrtcbf_path = os.path.join(base, "ablation_results_dual_cbf.csv")
    rrt_path = os.path.join(base, "ablation_results_dual.csv")

    # Ablation summary (env, noise)
    abl_fields, abl_rows = read_csv(abl_path)
    if abl_rows:
        abl_summary = summarize_by_env_noise(abl_rows)
        print_table("ablation_results.csv summary (by env, noise_std)", abl_summary)
        if write_csv(os.path.join(base, "ablation_summary.csv"), abl_summary):
            print(f"Wrote ablation_summary.csv")
    else:
        print(f"No ablation_results.csv at {abl_path}")

    # RRT-CBF results generic summary
    rrtcbf_fields, rrtcbf_rows = read_csv(rrtcbf_path)
    if rrtcbf_rows:
        rrtcbf_summary = summarize_generic(rrtcbf_rows)
        print_table("RRTCBFresults.csv summary (numeric averages, boolean rates)", rrtcbf_summary)
        if write_csv(os.path.join(base, "RRTCBFresults_summary.csv"), rrtcbf_summary):
            print("Wrote RRTCBFresults_summary.csv")
    else:
        print(f"No RRTCBFresults.csv at {rrtcbf_path}")

    # RRT results generic summary
    rrt_fields, rrt_rows = read_csv(rrt_path)
    if rrt_rows:
        rrt_summary = summarize_generic(rrt_rows)
        print_table("RRTresults.csv summary (numeric averages, boolean rates)", rrt_summary)
        if write_csv(os.path.join(base, "RRTresults_summary.csv"), rrt_summary):
            print("Wrote RRTresults_summary.csv")
    else:
        print(f"No RRTresults.csv at {rrt_path}")

if __name__ == "__main__":
    main()