"""Regenerate the public synthetic incident dataset."""

from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ml_foundations.data import make_incident_priority_data


def main() -> None:
    output = REPO / "data/incident_priority_synthetic.csv"
    data = make_incident_priority_data(n_samples=2400, seed=42)
    data.to_csv(output, index=False)
    print(f"Wrote {len(data)} rows to {output}")


if __name__ == "__main__":
    main()
