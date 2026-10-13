"""Summarize public checkpoint scores and regenerate plot source data."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from utils.experiment_utils import read_json, write_json


def summarize(root):
    groups = defaultdict(list)
    runs = {}
    manifest = []
    for path in sorted(root.rglob("checkpoint_summary.json")):
        summary = read_json(path)
        if summary.get("scene") not in ("map", "site_a", "site_b", "site_c"):
            continue
        components = path.relative_to(root).parts
        group = "/".join(components[:components.index(summary["scene"])])
        key = (group, summary["scene"], summary["method"], summary["history_length"], summary["rollouts"])
        groups[key].append(summary["selected"]["test_mean_load_std"] * 100)
        manifest.append(dict(group=group, scene=summary["scene"], method=summary["method"],
                             history_length=summary["history_length"], rollouts=summary["rollouts"],
                             seed=summary["seed"], path=path.parent.relative_to(root).as_posix()))
        if group in ("main", "history"):
            runs[(summary["scene"], summary["method"], summary["history_length"], summary["seed"])] = summary
    for scene in ("map","site_a","site_b","site_c"):
        for path in sorted((root / "main" / scene / "rules").glob("seed_*/test/*.json")):
            result = read_json(path)
            if "policy" not in result:
                continue
            groups[("main",scene,result["policy"],0,0)].append(result["mean_load_std"]*100)
    rows = [dict(group=g, scene=s, method=m, history_length=h, rollouts=c, seeds=len(v),
                 mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v)>1 else 0.)
            for (g,s,m,h,c),v in sorted(groups.items())]
    write_json(root / "summary.json", dict(display_scale=100, sd_ddof=1, results=rows))
    write_json(root / "release_manifest.json", dict(format="public_results_v1", runs=manifest))
    lines = ["| Group | Scene | Method | H | Rollouts | Mean ± SD |", "|---|---|---|---:|---:|---:|"]
    lines += ["| {group} | {scene} | {method} | {history_length} | {rollouts} | {mean:.2f} ± {sd:.2f} |".format(**r) for r in rows]
    (root / "summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    from pic.plot_chapter5_figures import (
        SCENES, HISTORY_FAMILIES, CONVERGENCE_METHODS, HISTORY_LENGTHS, STEPS,
        rendering_settings,
    )
    families = {"SAC":"sac", "SAC+MVD":"sac_mvd", "Link":"link", "Link+MVD":"link_mvd",
                "Transformer SAC":"transformer_sac", "CQL":"cql", "IQL":"iql"}
    data = dict(schema_version=2, display_scale=100, sd_ddof=1, seeds=[42,233,666],
                scenes=list(SCENES.values()), rendering=rendering_settings(), history=[], convergence=[])
    for scene, label in SCENES.items():
        for family in HISTORY_FAMILIES:
            values = [[runs[(scene,families[family],h,seed)]["selected"]["test_mean_load_std"]*100 for h in HISTORY_LENGTHS] for seed in data["seeds"]]
            mean, sd = np.mean(values,axis=0),np.std(values,axis=0,ddof=1)
            data["history"].append(dict(scene=label,method=family,history_lengths=list(HISTORY_LENGTHS),seed_test=values,mean=mean.tolist(),sd=sd.tolist()))
        for method,family,h in CONVERGENCE_METHODS:
            train,test = [],[]
            for seed in data["seeds"]:
                records = runs[(scene,families[family],h,seed)]["checkpoints"]
                assert [r["step"] for r in records] == STEPS
                train.append([r["train_mean_load_std"]*100 for r in records])
                test.append([r["test_mean_load_std"]*100 for r in records])
            data["convergence"].append(dict(scene=label,method=method,steps=STEPS,seed_train=train,seed_test=test,
                mean=np.mean(train,axis=0).tolist(),sd=np.std(train,axis=0,ddof=1).tolist(),
                test_mean=np.mean(test,axis=0).tolist(),test_sd=np.std(test,axis=0,ddof=1).tolist()))
    write_json(ROOT / "pic/chapter5_figure_data.json", data)
    excess_rows = []
    for scene,label in SCENES.items():
        for method in ("sac_h1","sac_h5","link_mvd_h5"):
            values = [read_json(root / "load_concentration" / scene / method / ("seed_%d" % seed) / "evaluation.json")["mean_excess_peak_load"]*100 for seed in data["seeds"]]
            excess_rows.append(dict(scene=label,method=method,seed_values=values,mean=float(np.mean(values)),sd=float(np.std(values,ddof=1))))
    write_json(root / "load_concentration/summary.json",dict(metric="excess_peak_load",display_scale=100,sd_ddof=1,rows=excess_rows))
    if (root / "association_changes").is_dir():
        from eval_association_changes import summarize_association_changes
        summarize_association_changes(root / "association_changes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    summarize(parser.parse_args().results_root)
