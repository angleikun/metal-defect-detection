"""统一评测脚本。用 EVAL_CONFIG 评测所有实验 best.pt。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.eval_protocol import EVAL_CONFIG


def eval_model(model_path: Path, data_yaml: Path, out_name: str):
    """评测单个模型，返回 per-class + overall 结果。"""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    results = model.val(
        data=str(data_yaml),
        split="val",
        plots=True,
        save_json=False,
        save_hybrid=False,
        workers=0,
        project=str(PROJECT_ROOT / "results"),
        name=out_name,
        exist_ok=True,
        **EVAL_CONFIG,
    )

    cls_names = results.names
    per_class = {}
    for i, cls_id in enumerate(results.ap_class_index):
        per_class[cls_names[cls_id]] = {
            "ap50": float(results.box.ap50[i]),
            "ap": float(results.box.ap[i]),
            "p": float(results.box.p[i]),
            "r": float(results.box.r[i]),
        }

    return {
        "map50": float(results.box.map50),
        "map": float(results.box.map),
        "per_class": per_class,
    }


def main():
    DATA_YAML = PROJECT_ROOT / "configs" / "data.yaml"
    MODELS_DIR = PROJECT_ROOT / "models"

    experiments = [
        ("baseline (yolov8n, 50e)",   MODELS_DIR / "neu_det_50e" / "weights" / "best.pt"),
        ("Exp 1 (yolov8s, 50e)",      MODELS_DIR / "exp1_yolov8s_50e" / "weights" / "best.pt"),
        ("Exp 2 (cosine, 50e)",       MODELS_DIR / "exp2_coslr_50e" / "weights" / "best.pt"),
        ("Exp 3 (cosine, 100e)",      MODELS_DIR / "exp3_coslr_100e" / "weights" / "best.pt"),
        ("Exp 4 (craze focus)",       MODELS_DIR / "exp4_crazing_focus" / "weights" / "best.pt"),
    ]

    # 检查文件存在
    valid_exps = []
    for name, path in experiments:
        if path.exists():
            valid_exps.append((name, path))
        else:
            print(f"SKIP (not found): {name} -> {path}")

    print(f"Evaluating {len(valid_exps)} experiments with conf={EVAL_CONFIG['conf']}, iou={EVAL_CONFIG['iou']}")
    print()

    all_results = {}
    for name, path in valid_exps:
        safe_name = name.split("(")[0].strip().lower().replace(" ", "_")
        print(f"  {name} ... ", end="", flush=True)
        r = eval_model(path, DATA_YAML, safe_name)
        all_results[name] = r
        print(f"mAP@0.5={r['map50']:.4f}")

    # ── 汇总表格 ────────────────────────────────────────
    print()
    print("=" * 80)
    print(f"{'Experiment':<30} {'mAP@0.5':>8} {'craze':>8} {'incl':>8} {'patch':>8} {'pitted':>8} {'scale':>8} {'scratch':>8}")
    print("-" * 80)

    for name, r in all_results.items():
        pc = r["per_class"]
        vals = [
            f"{pc.get('crazing', {}).get('ap50', 0):.3f}",
            f"{pc.get('inclusion', {}).get('ap50', 0):.3f}",
            f"{pc.get('patches', {}).get('ap50', 0):.3f}",
            f"{pc.get('pitted_surface', {}).get('ap50', 0):.3f}",
            f"{pc.get('rolled-in_scale', {}).get('ap50', 0):.3f}",
            f"{pc.get('scratches', {}).get('ap50', 0):.3f}",
        ]
        print(f"{name:<30} {r['map50']:>8.4f}  {'  '.join(vals)}")

    print("-" * 80)

    # 保存 CSV
    csv_path = PROJECT_ROOT / "results" / "day4_ablation.csv"
    with open(csv_path, "w") as f:
        f.write("experiment,overall_mAP50,crazing,inclusion,patches,pitted_surface,rolled_in_scale,scratches\n")
        for name, r in all_results.items():
            pc = r["per_class"]
            f.write(f"{name},{r['map50']:.4f}")
            for cls in ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]:
                f.write(f",{pc.get(cls, {}).get('ap50', 0):.4f}")
            f.write("\n")

    print(f"\nAblation table saved to: {csv_path}")


if __name__ == "__main__":
    main()
