"""
Figure generation. Per CLAUDE.md's "manifest as sole numeric source" rule,
both functions take exactly the `inline_data` dict that is ALSO written
into results.json's figures[].inline_data -- the PNG pixels are guaranteed
to match the manifest values because both are produced from the same
dict, not independently recomputed.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_pareto_tradeoff(data, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    names = data["benchmarks"]
    ax.scatter(data["greedy_latency_cycles"], data["greedy_area_ge"],
               color="tab:red", marker="o", s=70, label="Greedy scheduler (this ablation)")
    ax.scatter(data["ilp_latency_cycles"], data["ilp_area_ge"],
               color="tab:blue", marker="^", s=70, label="Game-theoretic ILP (reference)")
    for i, name in enumerate(names):
        ax.plot([data["greedy_latency_cycles"][i], data["ilp_latency_cycles"][i]],
                [data["greedy_area_ge"][i], data["ilp_area_ge"][i]],
                color="gray", linewidth=0.7, linestyle="--", alpha=0.6)
        ax.annotate(name, (data["greedy_latency_cycles"][i], data["greedy_area_ge"][i]),
                    fontsize=6, alpha=0.8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Latency (cycles)")
    ax.set_ylabel("Area (GE)")
    ax.set_title("Area vs. Latency: greedy scheduling vs. game-theoretic ILP")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scheduling_overhead(data, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(data["latency_inflation_cycles"], bins=max(3, len(data["latency_inflation_cycles"]) // 2),
            color="tab:orange", edgecolor="black")
    ax.set_xlabel("Latency inflation (greedy cycles - ILP cycles)")
    ax.set_ylabel("Benchmark count")
    ax.set_title("Scheduling overhead of greedy vs. game-theoretic ILP scheduling")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
