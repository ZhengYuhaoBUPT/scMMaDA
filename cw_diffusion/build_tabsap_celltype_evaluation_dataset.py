import argparse
import json
from pathlib import Path

from cellwhisperer.validation import SingleCellDataSetForValidationScoring


def parse_args():
    parser = argparse.ArgumentParser(description="Build cell-feature evaluation conversations without <image> tokens.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--response-prefix", default="")
    parser.add_argument("--num-cells-per-celltype", type=int, default=20)
    parser.add_argument("--celltypes", nargs="*", default=None)
    return parser.parse_args()


def normalize_celltype(text: str) -> str:
    return str(text).replace(" b ", " B ").replace("nk ", "NK ").replace(" ii ", " II ").replace(" t ", " T ").replace("cd", "CD")


def main():
    args = parse_args()
    dataset_processor = SingleCellDataSetForValidationScoring(
        celltypes=args.celltypes,
        dataset=Path(args.dataset),
    )
    adata = dataset_processor.adata
    obs_col = dataset_processor.celltype_obs_colname

    def row_to_conversation(row):
        return {
            "id": str(row.name),
            "conversations": [
                {"from": "human", "value": args.question},
                {"from": "gpt", "value": args.response_prefix + normalize_celltype(row[obs_col])},
            ],
        }

    conversations = (
        adata.obs.sample(frac=1, random_state=42)
        .groupby(obs_col)
        .head(args.num_cells_per_celltype)
        .apply(row_to_conversation, axis=1)
        .values
        .tolist()
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(conversations, f)


if __name__ == "__main__":
    main()
