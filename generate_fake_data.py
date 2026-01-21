"""Generate synthetic frame data for testing the diagonal wipe pipeline.

Creates three folders (matching main.py defaults) with solid-color frames:
- video_a_frames: red
- video_b_frames: blue
- video_source_frames: green
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

DEFAULT_A_DIR = Path("./video_a_frames")
DEFAULT_B_DIR = Path("./video_b_frames")
DEFAULT_S_DIR = Path("./video_source_frames")

COLOR_RED: Tuple[int, int, int] = (0, 0, 255)
COLOR_BLUE: Tuple[int, int, int] = (255, 0, 0)
COLOR_GREEN: Tuple[int, int, int] = (0, 255, 0)


def parse_args():
    parser = ArgumentParser(description="Generate synthetic RGB frames for testing.")
    parser.add_argument(
        "--num-frames", type=int, default=100, help="Number of frames per sequence."
    )
    parser.add_argument("--width", type=int, default=720, help="Frame width in pixels.")
    parser.add_argument("--height", type=int, default=720, help="Frame height in pixels.")
    parser.add_argument("--src-width", type=int, default=64, help="Source frame width in pixels.")
    parser.add_argument("--src-height", type=int, default=64, help="Source frame height in pixels.")
    parser.add_argument(
        "--video-a-dir",
        type=Path,
        default=DEFAULT_A_DIR,
        help="Output directory for video A frames (red).",
    )
    parser.add_argument(
        "--video-b-dir",
        type=Path,
        default=DEFAULT_B_DIR,
        help="Output directory for video B frames (blue).",
    )
    parser.add_argument(
        "--video-source-dir",
        type=Path,
        default=DEFAULT_S_DIR,
        help="Output directory for source frames (green).",
    )
    parser.add_argument(
        "--add-labels", action="store_true", help="Overlay simple labels on the frames."
    )
    return parser.parse_args()


def make_solid_frame(width: int, height: int, bgr_color: Tuple[int, int, int]) -> np.ndarray:
    """Create a solid BGR frame with the given color."""

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = np.array(bgr_color, dtype=np.uint8)
    return frame


def save_frames(
    out_dir: Path,
    num_frames: int,
    bgr_color: Tuple[int, int, int],
    width: int,
    height: int,
) -> None:
    """Generate and save solid frames to a directory."""

    out_dir.mkdir(parents=True, exist_ok=True)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(num_frames):
        frame = make_solid_frame(width, height, bgr_color)
        cv2.imwrite(str(out_dir / f"frame_{i:04d}.png"), frame)


def main() -> None:
    args = parse_args()

    save_frames(args.video_a_dir, args.num_frames, COLOR_RED, args.width, args.height)
    save_frames(args.video_b_dir, args.num_frames, COLOR_BLUE, args.width, args.height)
    save_frames(
        args.video_source_dir, args.num_frames, COLOR_GREEN, args.src_width, args.src_height
    )

    print(
        f"Generated {args.num_frames} frames into {args.video_a_dir}, {args.video_b_dir}, {args.video_source_dir}."
    )


if __name__ == "__main__":
    main()
