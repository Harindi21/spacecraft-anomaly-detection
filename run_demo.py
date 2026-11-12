"""Quick end-to-end demo on synthetic telemetry - runs with NumPy only.

    python run_demo.py                 # runs + saves detection.svg
    python run_demo.py --seed 3 --no-plot
    python run_demo.py --plot out.svg

Generates labelled, anomaly-injected multichannel telemetry, trains the
dependency-free AR forecaster on the clean early portion, detects anomalies on
the rest with nonparametric dynamic thresholding, prints precision/recall/F1,
and (unless --no-plot) renders an SVG of the result. No dataset or GPU needed.
"""
import argparse

from src.data import synthetic_telemetry
from src.model import get_forecaster
from src.pipeline import detect
from src.evaluate import evaluate

WINDOW = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", default="detection.svg",
                    help="path to write the SVG figure (default: detection.svg)")
    ap.add_argument("--no-plot", action="store_true", help="skip the figure")
    args = ap.parse_args()

    series, labels = synthetic_telemetry(n_steps=4000, n_channels=5,
                                         n_anomalies=6, seed=args.seed)
    # first 40% is treated as nominal training history, rest is the test stream
    split = int(len(series) * 0.4)
    train, test, test_labels = series[:split], series[split:], labels[split:]

    forecaster = get_forecaster("ar", window=WINDOW)
    mask, err, thresholds = detect(train, test, forecaster, WINDOW)

    res = evaluate(mask, test_labels)
    print(f"\nTest stream: {len(test)} steps, {int(test_labels.sum())} anomalous points, "
          f"{int(mask.sum())} flagged.")
    print("Per-channel thresholds:", [round(float(t), 3) for t in thresholds])
    for name in ("raw", "point_adjusted"):
        m = res[name]
        print(f"  {name:>14}:  P={m['precision']:.2f}  R={m['recall']:.2f}  "
              f"F1={m['f1']:.2f}  (tp={m['tp']} fp={m['fp']} fn={m['fn']})")

    if not args.no_plot:
        from src.plotting import plot_detection
        # show the channel that reacted most strongly (largest peak error)
        ch = int(err.max(axis=0).argmax())
        plot_detection(
            signal=test[:, ch], labels=test_labels, pred_mask=mask,
            error=err[:, ch], threshold=thresholds[ch], window=WINDOW,
            path=args.plot, channel_name=f"channel {ch}",
            title=f"Spacecraft telemetry anomaly detection (seed {args.seed})",
        )
        print(f"Figure written to {args.plot}")


if __name__ == "__main__":
    main()
