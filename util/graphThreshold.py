import argparse
import csv
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt


def normalize_header_name(name: str) -> str:
    text = name.strip()
    if "(" in text:
        text = text.split("(", 1)[0]
    if "（" in text:
        text = text.split("（", 1)[0]
    return text.strip()


def _to_float(value: str, default: float = float("nan")) -> float:
    text = (value or "").strip()
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _to_binary_flag(value: str) -> float:
    text = (value or "").strip().lower()
    if text in {"1", "true", "yes"}:
        return 1.0
    if text in {"0", "false", "no", ""}:
        return 0.0
    try:
        return float(int(float(text)))
    except ValueError:
        return 0.0


def load_threshold_log(log_path: Path) -> Dict[str, List[Any]]:
    with log_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Log file has no header: {log_path}")

        normalized_to_raw = {
            normalize_header_name(field): field for field in reader.fieldnames
        }

        required = [
            "epoch",
            "raw_threshold",
            "target_ema_threshold",
            "quantile_threshold",
            "effective_threshold",
            "avg_confidence",
            "avg_confidence_kept",
            "retrain_triggered",
            "retrain_due",
        ]
        optional = ["retrain_schedule_due", "retrain_event_due", "retrain_action"]

        missing = [key for key in required if key not in normalized_to_raw]
        if missing:
            raise KeyError(
                f"Missing required columns: {missing}. "
                f"Found: {[normalize_header_name(x) for x in reader.fieldnames]}"
            )

        series: Dict[str, List[Any]] = {key: [] for key in required + optional}
        for row in reader:
            for key in required:
                raw_field = normalized_to_raw[key]
                value = row.get(raw_field, "")
                if key in {"retrain_triggered", "retrain_due"}:
                    series[key].append(_to_binary_flag(value))
                else:
                    series[key].append(_to_float(value))

            for key in optional:
                if key in normalized_to_raw:
                    raw_field = normalized_to_raw[key]
                    value = row.get(raw_field, "")
                else:
                    value = ""

                if key in {"retrain_schedule_due", "retrain_event_due"}:
                    series[key].append(_to_binary_flag(value))
                else:
                    series[key].append((value or "").strip())

    return series


def plot_threshold_curves(series: Dict[str, List[Any]], out_path: Path, dpi: int = 160, show: bool = False) -> None:
    epochs = series["epoch"]
    retrain_event_epochs = []
    retrain_schedule_epochs = []
    retrain_triggered_epochs = []

    has_event_col = any(v >= 0.5 for v in series.get("retrain_event_due", []))
    has_schedule_col = any(v >= 0.5 for v in series.get("retrain_schedule_due", []))

    for i, epoch in enumerate(epochs):
        triggered = series["retrain_triggered"][i] >= 0.5
        if not triggered:
            continue
        retrain_triggered_epochs.append(epoch)

        action = str(series.get("retrain_action", [""] * len(epochs))[i]).lower()
        is_event = ("event" in action)
        is_schedule = (action == "triggered")

        if has_event_col:
            is_event = is_event or (series["retrain_event_due"][i] >= 0.5)
        if has_schedule_col:
            is_schedule = is_schedule or (series["retrain_schedule_due"][i] >= 0.5)

        if is_event and not is_schedule:
            retrain_event_epochs.append(epoch)
        elif is_schedule and not is_event:
            retrain_schedule_epochs.append(epoch)
        else:
            retrain_schedule_epochs.append(epoch)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(epochs, series["raw_threshold"], label="raw_threshold", linewidth=1.5)
    axes[0].plot(epochs, series["target_ema_threshold"], label="target_ema_threshold", linewidth=1.5)
    axes[0].plot(epochs, series["quantile_threshold"], label="quantile_threshold", linewidth=1.5)
    axes[0].plot(epochs, series["effective_threshold"], label="effective_threshold", linewidth=2.0)
    axes[0].set_ylabel("Threshold")
    axes[0].set_title("Pseudo Label Threshold Curves")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(epochs, series["avg_confidence"], label="avg_confidence", linewidth=1.5)
    axes[1].plot(epochs, series["avg_confidence_kept"], label="avg_confidence_kept", linewidth=2.0)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Confidence")
    axes[1].set_title("Pseudo Label Confidence Curves")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    for ax in axes:
        first_schedule = True
        for epoch in retrain_schedule_epochs:
            ax.axvline(
                epoch,
                color="red",
                linestyle="--",
                linewidth=0.9,
                alpha=0.35,
                label="retrain(schedule)" if first_schedule else None,
            )
            first_schedule = False

        first_event = True
        for epoch in retrain_event_epochs:
            ax.axvline(
                epoch,
                color="purple",
                linestyle=":",
                linewidth=1.0,
                alpha=0.55,
                label="retrain(event)" if first_event else None,
            )
            first_event = False

    axes[0].legend(loc="best")
    axes[1].legend(loc="best")

    if retrain_triggered_epochs:
        axes[0].text(
            0.99,
            0.02,
            (
                f"retrain_triggered_total={len(retrain_triggered_epochs)} | "
                f"schedule={len(retrain_schedule_epochs)} | event={len(retrain_event_epochs)}"
            ),
            transform=axes[0].transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="red",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)

    if show:
        plt.show()
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Plot pseudo threshold log curves")
    parser.add_argument(
        "--log_path",
        type=str,
        required=True,
        help="Path to pseudo_threshold_log.txt",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="",
        help="Output figure path (.png). Default: same folder as log file",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Figure dpi")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plot window after saving",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file does not exist: {log_path}")

    out_path = Path(args.out_path) if args.out_path else log_path.with_name("pseudo_threshold_curves.png")

    series = load_threshold_log(log_path)
    plot_threshold_curves(series, out_path=out_path, dpi=args.dpi, show=args.show)
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
    '''
    python util/graphThreshold.py \
        --log_path /data/zfx/myuda/uda_loveDA_run_v19/pseudo_threshold_log.txt \
        --out_path /data/zfx/myuda/uda_loveDA_run_v19/threshold_dashboard.png \
        --dpi 300
    '''
