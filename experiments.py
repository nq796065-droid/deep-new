"""
experiments.py - Chay va tong hop 5 cau hinh thuc nghiem (Ablation Benchmark).
Tong hop bang so sanh (comparison.csv) va bieu do so sanh (comparison_chart.png).
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from evaluate import run_evaluation


CONFIGS = [
    {"name": "baseline", "desc": "Baseline (Cross-Entropy + Random)", "loss": "ce", "sampling": "random", "aug": "basic"},
    {"name": "weighted_sampling", "desc": "Sampling: Weighted Random", "loss": "ce", "sampling": "weighted", "aug": "basic"},
    {"name": "focal_loss", "desc": "Loss: Focal Loss (gamma=2)", "loss": "focal", "sampling": "random", "aug": "basic"},
    {"name": "augmentation", "desc": "Augmentation: Color Jitter + Rot", "loss": "ce", "sampling": "random", "aug": "advanced"},
    {"name": "combined", "desc": "Combined: Weighted + Focal + Aug", "loss": "focal", "sampling": "weighted", "aug": "advanced"},
]


def summarize_benchmark(results_dir: str = "results"):
    r_path = Path(results_dir)
    records = []

    print("\n" + "=" * 70)
    print("      TONG HOP VA SO SANH 5 CAU HINH THUC NGHIEM (ABLATION STUDY)")
    print("=" * 70)

    for cfg in CONFIGS:
        ckpt = r_path / f"{cfg['name']}_model.pth"
        if not ckpt.exists():
            ckpt = r_path / cfg["name"] / f"{cfg['name']}_model.pth"
        if ckpt.exists():
            print(f"[*] Danh gia cau hinh: {cfg['desc']}...")
            metrics = run_evaluation(ckpt, save_cm=False)
            records.append({
                "Experiment": cfg["name"],
                "Description": cfg["desc"],
                "Loss": cfg["loss"],
                "Sampling": cfg["sampling"],
                "Augmentation": cfg["aug"],
                **metrics
            })

    if records:
        df = pd.DataFrame(records)
        df.to_csv(r_path / "comparison.csv", index=False)
        print(f"\n[+] Da luu bang tong hop tai: {r_path / 'comparison.csv'}")

        # Ve bieu do so sanh cot
        plt.figure(figsize=(10, 5))
        metrics_to_plot = ["accuracy", "balanced_accuracy", "macro_f1"]
        melted = df.melt(id_vars=["Experiment"], value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
        sns.barplot(data=melted, x="Experiment", y="Score", hue="Metric", palette="viridis")
        plt.title("So sanh cac chi so giua 5 cau hinh thuc nghiem", fontweight="bold")
        plt.ylim(0.4, 1.0)
        plt.tight_layout()
        plt.savefig(r_path / "comparison_chart.png", dpi=300)
        plt.close()
        print(f"[+] Da cap nhat bieu do so sanh tai: {r_path / 'comparison_chart.png'}\n")


if __name__ == "__main__":
    summarize_benchmark()
