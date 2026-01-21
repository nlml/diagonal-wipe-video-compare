"""Generate a diagonal wipe comparison video from two image sequences plus an optional source strip."""

import os
import subprocess
from argparse import ArgumentParser, Namespace
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from shutil import rmtree
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ImagePathSpec = Tuple[Path, Optional[str], str, Optional[Tuple[int, int, int, int]]]


def _imread(
    path: Path,
    text_to_add: Optional[str] = None,
    text_pos: str = "bottom-left",
    crop_x0y0x1y1: Optional[Tuple[int, int, int, int]] = None,
    x_offset: int = 50,
    y_offset: int = 100,
    font_scale: float = 1.5,
) -> Optional[np.ndarray]:
    """Read an image, optionally crop, annotate, and normalize to float RGB."""

    img = cv2.imread(str(path))
    if img is None:
        return None
    if crop_x0y0x1y1 is not None:
        x0, y0, x1, y1 = crop_x0y0x1y1
        img = img[y0:y1, x0:x1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if text_to_add is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        if "OLD" in text_to_add.upper():
            color = (240, 128, 128)  # Light Coral
        elif "NEW" in text_to_add.upper():
            color = (144, 238, 144)  # Light Green
        thickness = 2
        text_size, _ = cv2.getTextSize(text_to_add, font, font_scale, thickness)
        if text_pos == "bottom-left":
            position = (x_offset, img.shape[0] - y_offset)
        elif text_pos == "bottom-right":
            position = (img.shape[1] - text_size[0] - x_offset, img.shape[0] - y_offset)
        elif text_pos == "top-left":
            position = (x_offset, text_size[1] + y_offset)
        elif text_pos == "top-right":
            position = (img.shape[1] - text_size[0] - x_offset, text_size[1] + y_offset)
        else:
            position = (x_offset, img.shape[0] - y_offset)
        cv2.putText(img, text_to_add, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return img.astype(np.float32) / 255.0


class Wiper:
    """Compute diagonal wipe masks over time."""

    def __init__(self, t_static: float, t_wipe: float, mid_point: float) -> None:
        self.t_static = t_static
        self.t_wipe = t_wipe
        self.mid_point = mid_point
        self._cached_w: Optional[int] = None
        self._cached_h: Optional[int] = None
        self._sum_grid: Optional[np.ndarray] = None

    def _cache_grid(self, w: int, h: int) -> None:
        """Generate and cache the coordinate grid when dimensions change."""

        if w != self._cached_w or h != self._cached_h:
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            self._sum_grid = xx + yy
            self._cached_w = w
            self._cached_h = h

    def get_wipe_progress(self, secs: float) -> Tuple[float, bool]:
        """Return progress in [0,1] and whether we are currently wiping."""

        high_point = 1.0 - self.mid_point
        total_cycle = 2 * self.t_static + 2 * self.t_wipe
        secs = secs % total_cycle

        if secs < self.t_static:
            return 0.0, False
        if secs < (self.t_static + self.t_wipe):
            rel_t = (secs - self.t_static) / self.t_wipe
            return (
                np.interp(rel_t, [0.0, 0.33, 0.66, 1.0], [0.0, high_point, self.mid_point, 1.0]),
                True,
            )
        if secs < (2 * self.t_static + self.t_wipe):
            return 1.0, False

        rel_t = (secs - (2 * self.t_static + self.t_wipe)) / self.t_wipe
        return (
            np.interp(rel_t, [0.0, 0.33, 0.66, 1.0], [1.0, self.mid_point, high_point, 0.0]),
            True,
        )

    def make_main_mask(self, secs: float, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build wipe mask and distance-to-edge map for the given time and size."""

        self._cache_grid(w, h)
        if self._sum_grid is None:
            raise RuntimeError("Grid cache failed to initialize.")

        progress, _ = self.get_wipe_progress(secs)
        threshold = progress * (w + h)
        mask = (self._sum_grid < threshold).astype(np.float32)
        dist = np.abs(self._sum_grid - threshold)

        return mask, dist


def load_image_paths_from_folder(
    folder: Path,
    text_to_add: Optional[str] = None,
    text_pos: str = "bottom-left",
    crop_x0y0x1y1: Optional[Tuple[int, int, int, int]] = None,
) -> List[ImagePathSpec]:
    """Collect image paths with annotation metadata for the provided folder."""

    if not folder.exists():
        print(f"Directory {folder} not found.")
        return []

    paths: List[Path] = list(sorted(folder.glob("*.png")))
    if len(paths) == 0:
        paths = list(sorted(folder.glob("*.jpg")))
    if len(paths) == 0:
        print(f"No images found in {folder}.")
        return []

    return [(p, text_to_add, text_pos, crop_x0y0x1y1) for p in paths]


def get_frame_idx_list(images_A: Sequence[ImagePathSpec], wiper: Wiper, fps: int = 24) -> List[int]:
    """Map output frame indices to source frames while respecting wipe pauses."""

    idx_list: List[int] = []
    pause: Callable[[float], bool] = lambda t: wiper.get_wipe_progress(t)[1]
    t = 0.0
    for i_frame, _ in enumerate(images_A):
        while pause(t) and i_frame < len(images_A) - 1:
            idx_list.append(i_frame)
            t += 1 / fps
        idx_list.append(i_frame)
        t += 1 / fps
    return idx_list


def postprocess_final_frame(
    frame: np.ndarray, target_w: int, target_h: int, pad_value: float = 0.0
) -> np.ndarray:
    """Resize the longer side to fit within target dims, then center pad the remainder."""

    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    if new_w == 0 or new_h == 0:
        raise ValueError("Invalid resize target computed from provided dimensions.")

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

    pad_x = max(0, target_w - new_w)
    pad_y = max(0, target_h - new_h)
    pad_left = pad_x // 2
    pad_right = pad_x - pad_left
    pad_top = pad_y // 2
    pad_bottom = pad_y - pad_top

    # Ensure final frame dims are divisible by 2 (ffmpeg yuv420p requirement).
    if (new_w + pad_left + pad_right) % 2 != 0:
        pad_right += 1
    if (new_h + pad_top + pad_bottom) % 2 != 0:
        pad_bottom += 1

    return np.pad(
        resized,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )


def create_and_write_frame(
    out_frame_idx: int,
    source_frames_idx: int,
    fps: int,
    w: int,
    h: int,
    output_w: int,
    output_h: int,
    temp_out_dir: Path,
    wiper: Wiper,
    image_paths_A: Sequence[ImagePathSpec],
    image_paths_B: Sequence[ImagePathSpec],
    image_paths_Source: Optional[Sequence[ImagePathSpec]],
    edge_width: float,
    line_color: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Compose a single frame and save it to disk, optionally prepending a source strip."""

    try:
        secs = out_frame_idx / fps
        os.makedirs(temp_out_dir, exist_ok=True)
        main_mask, dist_from_line = wiper.make_main_mask(secs, w, h)
        inv_mask = 1.0 - main_mask

        frame_A = _imread(*image_paths_A[source_frames_idx])
        frame_B = _imread(*image_paths_B[source_frames_idx])
        frame_S = (
            _imread(*image_paths_Source[source_frames_idx])
            if image_paths_Source is not None
            else None
        )
        if (
            frame_A is None
            or frame_B is None
            or (image_paths_Source is not None and frame_S is None)
        ):
            raise RuntimeError("One or more source frames failed to load.")

        combined_frame = frame_A * inv_mask[:, :, None] + frame_B * main_mask[:, :, None]
        is_near_edge = dist_from_line < edge_width
        combined_frame[is_near_edge] = np.array(line_color, dtype=np.float32)

        if frame_S is not None:
            frame_S_ = np.zeros((h, frame_S.shape[1], 3), dtype=np.float32)
            y0 = (h - frame_S.shape[0]) // 2
            y1 = y0 + frame_S.shape[0]
            frame_S_[y0:y1, :, :] = frame_S
            combined_frame = np.concatenate([frame_S_, combined_frame], axis=1)

        combined_frame = postprocess_final_frame(combined_frame, output_w, output_h)
        out_frame = (combined_frame * 255).astype(np.uint8)
        out_path = temp_out_dir / f"frame_{out_frame_idx:04d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"Error processing frame {out_frame_idx} with source index {source_frames_idx}: {e}")


def parse_args() -> Namespace:
    """Parse command-line arguments for the wipe generator."""

    parser = ArgumentParser(description="Create a diagonal wipe comparison video.")
    parser.add_argument(
        "--video-a-dir",
        type=Path,
        default=Path("./video_a_frames"),
        help="Folder with first comparison frames.",
    )
    parser.add_argument(
        "--video-b-dir",
        type=Path,
        default=Path("./video_b_frames"),
        help="Folder with second comparison frames.",
    )
    parser.add_argument(
        "--video-source-dir",
        type=Path,
        default=None,
        help="Optional folder with source frames to prepend.",
    )
    parser.add_argument(
        "--temp-out-dir",
        type=Path,
        default=Path("wipe_frames"),
        help="Temporary directory to store generated frames.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("wipe_output.mp4"), help="Output video path."
    )
    parser.add_argument(
        "--out-width",
        type=int,
        default=1920,
        help="Target output width after resizing and padding.",
    )
    parser.add_argument(
        "--out-height",
        type=int,
        default=1080,
        help="Target output height after resizing and padding.",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second for both reading and writing."
    )
    parser.add_argument(
        "--t-static", type=float, default=3.0, help="Seconds to hold before and after each wipe."
    )
    parser.add_argument("--t-wipe", type=float, default=3.0, help="Seconds duration of each wipe.")
    parser.add_argument(
        "--mid-point",
        type=float,
        default=0.0,
        help="Relative progress pivot used for easing the wipe.",
    )
    parser.add_argument(
        "--skip-seconds",
        type=float,
        default=2.0,
        help="Seconds to skip from the start of each sequence.",
    )
    parser.add_argument(
        "--repeat-final-frames",
        type=int,
        default=50,
        help="How many times to duplicate the final frame.",
    )
    parser.add_argument(
        "--processes", type=int, default=24, help="Worker processes for frame generation."
    )
    parser.add_argument(
        "--edge-width",
        type=float,
        default=5.0,
        help="Pixel width to paint white along the wipe edge.",
    )
    parser.add_argument("--text-a", type=str, default="OLD", help="Label for frames from video A.")
    parser.add_argument("--text-b", type=str, default="NEW", help="Label for frames from video B.")
    parser.add_argument(
        "--text-a-pos",
        type=str,
        default="bottom-left",
        choices=["bottom-left", "bottom-right", "top-left", "top-right"],
        help="Label position for video A.",
    )
    parser.add_argument(
        "--text-b-pos",
        type=str,
        default="top-right",
        choices=["bottom-left", "bottom-right", "top-left", "top-right"],
        help="Label position for video B.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the wipe pipeline based on parsed CLI arguments."""

    args = parse_args()

    image_paths_A = load_image_paths_from_folder(
        args.video_a_dir, text_to_add=args.text_a, text_pos=args.text_a_pos
    )
    image_paths_B = load_image_paths_from_folder(
        args.video_b_dir, text_to_add=args.text_b, text_pos=args.text_b_pos
    )
    image_paths_Source: Optional[List[ImagePathSpec]] = None
    if args.video_source_dir is not None:
        image_paths_Source = load_image_paths_from_folder(args.video_source_dir)

    if not image_paths_A or not image_paths_B:
        msg = "Not enough frames found before skipping; missing input folders or frames."
        msg += "run generate_fake_data.py first if you want to test with fake data"
        raise RuntimeError(msg)
    if args.video_source_dir is not None and not image_paths_Source:
        raise RuntimeError("Source frames were requested but none were found.")

    skip_frames = int(args.skip_seconds * args.fps)
    image_paths_A = image_paths_A[skip_frames:]
    image_paths_B = image_paths_B[skip_frames:]
    if image_paths_Source is not None:
        image_paths_Source = image_paths_Source[skip_frames:]

    min_len = min(len(image_paths_A), len(image_paths_B))
    if image_paths_Source is not None:
        min_len = min(min_len, len(image_paths_Source))
    if min_len == 0:
        raise RuntimeError("No frames available to process.")
    if len(image_paths_A) != len(image_paths_B) or (
        image_paths_Source is not None and len(image_paths_A) != len(image_paths_Source)
    ):
        print("Warning: sequences have different lengths; trimming to shortest.")
        image_paths_A = image_paths_A[:min_len]
        image_paths_B = image_paths_B[:min_len]
        if image_paths_Source is not None:
            image_paths_Source = image_paths_Source[:min_len]

    for _ in range(args.repeat_final_frames):
        image_paths_A.append(image_paths_A[-1])
        image_paths_B.append(image_paths_B[-1])
        if image_paths_Source is not None:
            image_paths_Source.append(image_paths_Source[-1])

    sample = cv2.imread(str(image_paths_A[0][0]))
    if sample is None:
        raise RuntimeError(f"Failed to load sample frame from {image_paths_A[0][0]}")
    h, w = sample.shape[:2]

    if args.temp_out_dir.exists():
        rmtree(args.temp_out_dir)

    wiper = Wiper(t_static=args.t_static, t_wipe=args.t_wipe, mid_point=args.mid_point)
    frame_idxs = get_frame_idx_list(image_paths_A, wiper, args.fps)

    func = partial(
        create_and_write_frame,
        fps=args.fps,
        w=w,
        h=h,
        output_w=args.out_width,
        output_h=args.out_height,
        temp_out_dir=args.temp_out_dir,
        wiper=wiper,
        image_paths_A=image_paths_A,
        image_paths_B=image_paths_B,
        image_paths_Source=image_paths_Source,
        edge_width=args.edge_width,
    )

    with Pool(args.processes) as pool:
        pool.starmap(func, [(i, src_idx) for i, src_idx in enumerate(frame_idxs)])

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(args.fps),
        "-i",
        str(args.temp_out_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(args.output),
    ]
    if args.output.exists():
        args.output.unlink()
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Wipe video saved to {args.output}.")
    rmtree(args.temp_out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
