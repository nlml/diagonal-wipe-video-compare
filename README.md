# diagonal-wipe-video-compare

![Example output](example.gif)

Generate a diagonal wipe comparison video from two image sequences (plus optional source strip).

## Quick start
- Install deps: `pip install -e .` (requires Python 3 + ffmpeg in PATH).
- Create test data: `python generate_fake_data.py` (writes sample frames to `video_*_frames/`).
- Run wipe: `python main.py --video-a-dir video_a_frames --video-b-dir video_b_frames --output wipe.mp4`.

## Example usage

For a quick test:

```shell
python generate_fake_data.py
python main.py
# or python main.py --video-source-dir video_source_frames --output with_source.mp4
```

Then see the outputs in `wipe_output.mp4`

```shell
uv run main.py --video-a-dir ./OldVideo --video-b-dir ./NewVideo --video-source-dir ./SourceVideo
```

## Common options
- `--video-source-dir PATH` prepend a source strip (must align length with A/B).
- `--t-static SECS` / `--t-wipe SECS` control hold and wipe durations.
- `--skip-seconds SECS` trim the start of each sequence before processing.
- `--
- `--repeat-final-frames N` freeze the final frame for N extra outputs.
- `--out-width/--out-height` final padded dimensions (ensures even sizes for ffmpeg).
- `--text-a/--text-b` labels; `--text-a-pos/--text-b-pos` positions.

Output: `wipe_output.mp4` by default, intermediate frames in `wipe_frames/` are cleaned after render.
