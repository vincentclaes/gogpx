from __future__ import annotations

import argparse
import sys

from .pipeline import PipelineConfig, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--profile", default="foot", choices=["car", "bike", "foot", "hike"])
    parser.add_argument("--debug-json", default="")
    parser.add_argument("--debug-raw-gpx", default="")
    parser.add_argument("--debug-selected", default="")
    parser.add_argument("--no-agent", action="store_true")
    args = parser.parse_args()

    config = PipelineConfig(
        model_name=args.model,
        profile=args.profile,
        debug_json=args.debug_json or None,
        debug_raw_gpx=args.debug_raw_gpx or None,
        debug_selected=args.debug_selected or None,
        use_agent=not args.no_agent,
    )

    try:
        run_pipeline(args.image, args.out, config)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
