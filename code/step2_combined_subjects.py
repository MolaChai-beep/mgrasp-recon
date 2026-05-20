from __future__ import annotations

from combined_subjects_common import (
    build_common_arg_parser,
    get_recon_device,
    is_oom_error,
    load_csv_paths,
    print_device_summary,
    run_step2_subject,
    subject_id_from_csv,
)


def parse_args():
    parser = build_common_arg_parser("Run combined step2 reconstruction for each subject CSV using an existing step1 basis.")
    parser.add_argument(
        "--slice-indices",
        nargs="+",
        type=int,
        default=None,
        help="Optional 0-based slice indices to reconstruct. Defaults to all slices.",
    )
    return parser.parse_args()


def run(args) -> int:
    csv_paths = load_csv_paths(args.csv_dir, args.subjects)
    recon_device = get_recon_device()
    failures: list[tuple[str, str, str, str]] = []

    print_device_summary(recon_device)
    print(f"> csv count {len(csv_paths)}")
    print(f"> data root {args.data_root}")
    print(f"> output root {args.output_root}")

    try:
        for csv_path in csv_paths:
            try:
                _, subject_failures = run_step2_subject(
                    csv_path=csv_path,
                    data_root=args.data_root,
                    output_root=args.output_root,
                    coil_thresh=args.coil_thresh,
                    recon_device=recon_device,
                    slice_indices=args.slice_indices,
                )
                failures.extend(subject_failures)
            except Exception as exc:  # noqa: BLE001
                if is_oom_error(exc):
                    print()
                    print(f"STOPPED: OOM at subject={subject_id_from_csv(csv_path)}")
                    return 2
                failures.append((subject_id_from_csv(csv_path), "step2", "subject", str(exc)))
                print(f"  FAILED step2 for {csv_path.name}: {exc}")
    finally:
        pass

    print()
    print("=" * 80)
    if failures:
        print(f"step2 finished with {len(failures)} failures")
        for subject_id, hop_id, stage, error in failures:
            print(f"  {subject_id} | {hop_id} | {stage} | {error}")
        return 1

    print("step2 finished successfully with no failures")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
