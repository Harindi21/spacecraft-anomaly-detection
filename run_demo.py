"""Quick end-to-end demo on synthetic telemetry - runs with NumPy only.

    python run_demo.py

Generates labelled, anomaly-injected multichannel telemetry, trains the
dependency-free AR forecaster on the clean early portion, detects anomalies on
the rest with nonparametric dynamic thresholding, and prints precision/recall/F1.
No dataset download or GPU required.
"""
from src.data import synthetic_telemetry
from src.model import get_forecaster
from src.pipeline import detect
from src.evaluate import evaluate

WINDOW = 10


def main():
    series, labels = synthetic_telemetry(n_steps=4000, n_channels=5,
                                         n_anomalies=6, seed=0)
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


if __name__ == "__main__":
    main()
