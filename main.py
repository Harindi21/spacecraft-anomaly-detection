"""Run detection on the real NASA SMAP/MSL benchmark.

    python main.py --data path/to/smap_msl --channel P-1

Uses the Keras LSTM forecaster when TensorFlow is installed (recommended for
the real data), otherwise the NumPy AR baseline. See README for the download.
"""
import argparse
from src.data import load_smap_msl
from src.model import get_forecaster
from src.pipeline import detect
from src.evaluate import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with train/ test/ labeled_anomalies.csv")
    ap.add_argument("--channel", required=True, help="e.g. P-1, T-1, M-6")
    ap.add_argument("--model", default="auto", choices=["auto", "lstm", "ar"])
    ap.add_argument("--window", type=int, default=None)
    args = ap.parse_args()

    train, test, labels = load_smap_msl(args.data, args.channel)
    window = args.window or (250 if args.model != "ar" else 10)
    forecaster = get_forecaster(args.model, window=window)

    mask, err, thresholds = detect(train, test, forecaster, window)
    print(f"Channel {args.channel}: {len(test)} steps, {int(mask.sum())} flagged.")

    if labels is not None:
        res = evaluate(mask, labels)
        for name in ("raw", "point_adjusted"):
            m = res[name]
            print(f"  {name:>14}:  P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}")
    else:
        print("  (no ground-truth labels found for this channel)")


if __name__ == "__main__":
    main()
