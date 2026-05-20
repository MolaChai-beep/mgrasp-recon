from __future__ import annotations

from combined_subjects_common import build_common_arg_parser
import step1_combined_subjects
import step2_combined_subjects


def parse_args():
    parser = build_common_arg_parser("Run combined step1 then combined step2 for each subject CSV.")
    parser.add_argument(
        "--slice-indices",
        nargs="+",
        type=int,
        default=None,
        help="Optional 0-based slice indices to reconstruct in step2. Defaults to all slices.",
    )
    return parser.parse_args()


def run(args) -> int:
    step1_exit = step1_combined_subjects.run(args)
    if step1_exit != 0:
        return step1_exit
    return step2_combined_subjects.run(args)


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
