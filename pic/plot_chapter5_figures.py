"""Recreate the paper figures from anonymized numerical results.

History sensitivity uses selected-checkpoint test means and sample standard
deviations at H=1/3/5. Training dynamics compare SAC H=5 with Link+MVD H=5
in four panels: dashed training means, solid test means, and test-only shading
for sample standard deviations across three seeds. All 15 checkpoints are
shown without smoothing; values are multiplied by 100 exactly once.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

# Limit CPU use when rendering alongside training. No model or CUDA is loaded.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["MPLBACKEND"] = "Agg"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 233, 666)
SCENES = {"map": "Map", "site_a": "Site A", "site_b": "Site B", "site_c": "Site C"}
HISTORY_LENGTHS = (1, 3, 5)
HISTORY_FAMILIES = ("SAC", "SAC+MVD", "Link", "Link+MVD")
STEPS = list(range(2000, 30001, 2000))
LINK_MODEL = "user_attention_residual_link_snapshot_sac"
METHODS = {
    "sac": ("SAC", 1, "attention_sac", False),
    "sac_mvd": ("SAC+MVD", 1, "attention_sac", True),
    "sac_h3": ("SAC", 3, "history_attention_sac", False),
    "sac_h5": ("SAC", 5, "history_attention_sac", False),
    "sac_mvd_h3": ("SAC+MVD", 3, "history_attention_sac", True),
    "sac_h5_mvd": ("SAC+MVD", 5, "history_attention_sac", True),
    "user_attention_link_h1": ("Link", 1, LINK_MODEL, False),
    "user_attention_link_h3": ("Link", 3, LINK_MODEL, False),
    "user_attention_link": ("Link", 5, LINK_MODEL, False),
    "user_attention_link_mvd_h1": ("Link+MVD", 1, LINK_MODEL, True),
    "user_attention_link_mvd_h3": ("Link+MVD", 3, LINK_MODEL, True),
    "user_attention_link_mvd": ("Link+MVD", 5, LINK_MODEL, True),
    "transformer_sac": ("Transformer SAC", 5, "transformer_sac", False),
    "cql": ("CQL", 1, "cql", False),
    "iql": ("IQL", 1, "iql", False),
}
CONVERGENCE_METHODS = (
    ("SAC H=1", "SAC", 1), ("SAC H=5", "SAC", 5),
    ("Transformer SAC", "Transformer SAC", 5), ("CQL", "CQL", 1),
    ("IQL", "IQL", 1), ("Proposed H=5", "Link+MVD", 5),
)
CONVERGENCE_DISPLAY_METHODS = ("SAC H=5", "Proposed H=5")
# Okabe-Ito palette; methods are identified by line color.
STYLES = {
    "SAC": ("#0072B2", "o", "-"),
    "SAC+MVD": ("#56B4E9", "s", "--"),
    "Link": ("#009E73", "^", "-."),
    "Link+MVD": ("#D55E00", "D", "-"),
    "SAC H=1": ("#0072B2", "o", "-"),
    "SAC H=5": ("#0072B2", "s", "--"),
    "Transformer SAC": ("#CC79A7", "^", "-."),
    "CQL": ("#666666", "v", ":"),
    "IQL": ("#E69F00", "P", "--"),
    "Proposed H=5": ("#D55E00", "D", "-"),
}


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def expected_keys(figure):
    keys = set()
    for scene in SCENES:
        for seed in SEEDS:
            if figure in ("all", "history"):
                keys.update((scene, seed, family, history) for family in HISTORY_FAMILIES
                            for history in HISTORY_LENGTHS)
            if figure in ("all", "convergence"):
                keys.update((scene, seed, family, history)
                            for _, family, history in CONVERGENCE_METHODS)
    return keys


def summarize(values):
    values = np.asarray(values, dtype=float)
    if values.shape[0] != 3 or not np.isfinite(values).all():
        raise ValueError("Expected three finite, aligned seed series")
    return values.mean(axis=0).tolist(), values.std(axis=0, ddof=1).tolist()


def rendering_settings():
    return dict(width_inches=7.16, history_height_inches=3.5,
                convergence_height_inches=4.1, png_dpi=300,
                history_panel_layout=[2, 2], convergence_panel_layout=[2, 2],
                convergence_methods=list(CONVERGENCE_DISPLAY_METHODS),
                convergence_line_styles=dict(train="dashed", test="solid"),
                convergence_uncertainty=dict(train="none", test="sample_sd"))


def export_data(found, figure):
    data = dict(schema_version=2, seeds=list(SEEDS), metric="100 * mean_load_std",
                display_scale=100, sd_ddof=1, scenes=list(SCENES.values()),
                checkpoint_selection="minimum_train_mean_load_std_then_earliest_step",
                protocol=dict(training_environments=10, test_environments=10,
                              random_trajectories_per_training_environment=64,
                              rsrp_normalization="train_zscore", map_reward_scale=10,
                              measurement_driven_reward_scale=1, evaluation_reward_scale=1,
                              critic_updates=30000, actor_updates=15000),
                rendering=rendering_settings(),
                history=[], convergence=[])
    for scene, label in SCENES.items():
        if figure in ("all", "history"):
            for family in HISTORY_FAMILIES:
                values, selected_steps = [], []
                for seed in SEEDS:
                    runs = [found[(scene, seed, family, h)] for h in HISTORY_LENGTHS]
                    values.append([100 * run["selected"]["test_mean_load_std"] for run in runs])
                    selected_steps.append([run["selected"]["step"] for run in runs])
                mean, sd = summarize(values)
                data["history"].append(dict(scene=label, method=family, history_lengths=list(HISTORY_LENGTHS),
                                            seed_test=values, selected_steps=selected_steps, mean=mean, sd=sd))
        if figure in ("all", "convergence"):
            for method, family, history in CONVERGENCE_METHODS:
                runs = [found[(scene, seed, family, history)] for seed in SEEDS]
                values = [[100 * row["train_mean_load_std"] for row in run["checkpoints"]] for run in runs]
                mean, sd = summarize(values)
                test_values = [[100 * row["test_mean_load_std"] for row in run["checkpoints"]] for run in runs]
                test_mean, test_sd = summarize(test_values)
                data["convergence"].append(dict(scene=label, method=method, steps=STEPS,
                                                seed_train=values, mean=mean, sd=sd,
                                                seed_test=test_values, test_mean=test_mean, test_sd=test_sd))
    validate_public_data(data, figure)
    return data


def validate_public_data(data, figure):
    if (data["schema_version"] not in (1, 2) or data["seeds"] != list(SEEDS) or
            data["display_scale"] != 100 or data["sd_ddof"] != 1 or
            data["scenes"] != list(SCENES.values())):
        raise ValueError("Incompatible figure-data schema/statistics")
    if figure in ("all", "convergence") and data["schema_version"] != 2:
        raise ValueError("Training/test curves require schema 2")
    for name, methods, points, source in (
            ("history", HISTORY_FAMILIES, list(HISTORY_LENGTHS), "seed_test"),
            ("convergence", tuple(m[0] for m in CONVERGENCE_METHODS), STEPS, "seed_train")):
        if figure not in ("all", name):
            continue
        records = data[name]
        expected = {(scene, method) for scene in SCENES.values() for method in methods}
        if len(records) != len(expected) or {(r["scene"], r["method"]) for r in records} != expected:
            raise ValueError("Incomplete/duplicate %s plot records" % name)
        for record in records:
            coordinates = record["history_lengths" if name == "history" else "steps"]
            if coordinates != points:
                raise ValueError("Misaligned seed observations")
            statistics = [(source, "mean", "sd")]
            if name == "convergence":
                statistics.append(("seed_test", "test_mean", "test_sd"))
            for seed_field, mean_field, sd_field in statistics:
                if np.asarray(record[seed_field]).shape != (3, len(points)):
                    raise ValueError("Misaligned seed observations")
                mean, sd = summarize(record[seed_field])
                if not (np.allclose(mean, record[mean_field], rtol=1e-12, atol=1e-12) and
                        np.allclose(sd, record[sd_field], rtol=1e-12, atol=1e-12)):
                    raise ValueError("Exported statistics differ from the seed observations")


def configure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 10, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "legend.fontsize": 9, "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.unicode_minus": False, "savefig.facecolor": "white",
    })


def style_axis(ax, scene, index, records):
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator
    ax.set_title("(%s) %s" % (chr(ord("a") + index), scene), pad=5)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.7, pad=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    low = min(np.min(np.asarray(r["mean"]) - r["sd"]) for r in records)
    high = max(np.max(np.asarray(r["mean"]) + r["sd"]) for r in records)
    padding = max((high - low) * 0.12, 0.35)
    ax.set_ylim(max(0, low - padding), high + padding)


def save_figure(fig, output, name):
    # Fixed page dimensions, so inclusion at textwidth preserves font sizes.
    fig.savefig(output / (name + ".pdf"), metadata={"Title": name.replace("_", " "),
                "Author": "", "Subject": "Anonymized three-seed experimental results"})
    fig.savefig(output / (name + ".svg"), metadata={"Title": name.replace("_", " ")})
    fig.savefig(output / (name + ".png"), dpi=300)
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_history(data, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 3.5))
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.12, top=0.84, wspace=0.24, hspace=0.82)
    handles = []
    for index, (scene, ax) in enumerate(zip(data["scenes"], axes.flat)):
        records = [r for r in data["history"] if r["scene"] == scene]
        style_axis(ax, scene, index, records)
        for record in records:
            color = STYLES[record["method"]][0]
            proposed = record["method"] == "Link+MVD"
            handle = ax.errorbar(record["history_lengths"], record["mean"], yerr=record["sd"],
                                color=color, marker=None, linestyle="-", linewidth=1.35,
                                capsize=2.5, capthick=0.9, elinewidth=0.9,
                                label=record["method"], zorder=4 if proposed else 3)
            if index == 0:
                handles.append(handle.lines[0])
        ax.set_xticks(HISTORY_LENGTHS)
        ax.set_xlim(0.72, 5.28)
        ax.set_xlabel("History length $H$", labelpad=3)
    fig.text(0.018, 0.47, "Test cell-load std. ($\\times$100)", rotation=90,
             va="center", ha="center", fontsize=9)
    fig.legend(handles=handles, labels=HISTORY_FAMILIES, loc="upper center", bbox_to_anchor=(0.53, 0.995),
               ncol=4, frameon=False, columnspacing=1.8, handletextpad=0.6)
    save_figure(fig, output, "history_length_comparison")


def draw_dynamics(ax, records):
    for record in records:
        color = STYLES[record["method"]][0]
        x = np.asarray(record["steps"]) / 1000.0
        train = np.asarray(record["mean"])
        test, test_sd = np.asarray(record["test_mean"]), np.asarray(record["test_sd"])
        proposed = record["method"] == "Proposed H=5"
        ax.fill_between(x, test - test_sd, test + test_sd,
                        color=color, alpha=0.13, linewidth=0, zorder=1)
        ax.plot(x, train, color=color, linestyle=(0, (5, 3)), marker=None,
                linewidth=1.3, zorder=2)
        ax.plot(x, test, color=color, linestyle="-", marker=None,
                linewidth=1.8 if proposed else 1.5, zorder=4 if proposed else 3)


def plot_convergence(data, output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.1))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.12, top=0.86,
                        wspace=0.24, hspace=0.50)
    for index, (scene, ax) in enumerate(zip(data["scenes"], axes.flat)):
        records = [
            next(r for r in data["convergence"]
                 if r["scene"] == scene and r["method"] == method)
            for method in CONVERGENCE_DISPLAY_METHODS
        ]
        train_records = [dict(r, sd=np.zeros_like(r["mean"])) for r in records]
        test_records = [dict(r, mean=r["test_mean"], sd=r["test_sd"]) for r in records]
        style_axis(ax, scene, index, train_records + test_records)
        draw_dynamics(ax, records)
        ax.set_xticks([2, 10, 20, 30])
        ax.set_xlim(1.1, 30.9)
    fig.text(0.018, 0.47, "Cell-load std. ($\\times$100)", rotation=90,
             va="center", ha="center", fontsize=9)
    fig.text(0.545, 0.015, "Training steps ($\\times 10^3$)",
             va="bottom", ha="center", fontsize=9)
    handles = [
        Line2D([], [], color=STYLES[method][0], linewidth=1.6)
        for method in CONVERGENCE_DISPLAY_METHODS
    ] + [
        Line2D([], [], color="#444444", linestyle=(0, (5, 3)), linewidth=1.3),
        Line2D([], [], color="#444444", linestyle="-", linewidth=1.6),
    ]
    fig.legend(handles=handles,
               labels=list(CONVERGENCE_DISPLAY_METHODS) + ["Training", "Test"],
               loc="upper center", bbox_to_anchor=(0.53, 0.995),
               ncol=4, frameon=False, columnspacing=1.6, handletextpad=0.6)
    save_figure(fig, output, "convergence_comparison")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Draw history and train/test convergence figures from public numerical results")
    parser.add_argument("--from-data", type=Path, default=ROOT / "pic/chapter5_figure_data.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "pic")
    parser.add_argument("--figure", choices=("all", "history", "convergence"), default="all")
    args = parser.parse_args(argv)
    data = read_json(args.from_data)
    validate_public_data(data, args.figure)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir.resolve() / ".mplconfig"))
    data["rendering"] = rendering_settings()
    write_json(args.output_dir / "chapter5_figure_data.json", data)
    configure_matplotlib()
    if args.figure in ("all", "history"):
        plot_history(data, args.output_dir)
    if args.figure in ("all", "convergence"):
        plot_convergence(data, args.output_dir)

if __name__ == "__main__":
    main()
