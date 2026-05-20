from __future__ import annotations

from combined_subjects_common import (
    build_common_arg_parser,
    get_recon_device,
    get_subject_log_path,
    is_oom_error,
    load_csv_paths,
    print_device_summary,
    run_step1_subject,
    subject_id_from_csv,
    tee_subject_log,
)


def parse_args():
    parser = build_common_arg_parser("Run combined step1 basis estimation and reference graph generation for each subject CSV.")
    return parser.parse_args()


def run(args) -> int:
    csv_paths = load_csv_paths(args.csv_dir, args.subjects)
    recon_device = get_recon_device()
    failures: list[tuple[str, str]] = []

    print_device_summary(recon_device)
    print(f"> csv count {len(csv_paths)}")
    print(f"> data root {args.data_root}")
    print(f"> output root {args.output_root}")

    try:
        for csv_path in csv_paths:
            subject_id = subject_id_from_csv(csv_path)
            log_path = get_subject_log_path(args.output_root, "step1_combined_subjects", subject_id)
            try:
                with tee_subject_log(log_path):
                    run_step1_subject(
                        csv_path=csv_path,
                        data_root=args.data_root,
                        output_root=args.output_root,
                        coil_thresh=args.coil_thresh,
                        recon_device=recon_device,
                    )
            except Exception as exc:  # noqa: BLE001
                if is_oom_error(exc):
                    print()
                    print(f"STOPPED: OOM at subject={subject_id}")
                    return 2
                failures.append((csv_path.name, str(exc)))
                print(f"  FAILED step1 for {csv_path.name}: {exc}")
    finally:
        pass

    print()
    print("=" * 80)
    if failures:
        print(f"step1 finished with {len(failures)} failures")
        for subject_name, error in failures:
            print(f"  {subject_name} | {error}")
        return 1

    print("step1 finished successfully with no failures")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
