import cv2
import os
import re
from pathlib import Path

images_folder = r'/Users/johnmadrid/GitHub/pca-driving-behavior/results/biplots/event_one/'
video_name = "PCA_eye_head_hand_car_event_1_7fps.mp4"
fps = 7.0


def _extract_timestamp_from_biplot(filename):
    """
    Extract float timestamp from filenames like:
    'biplot_one_20.84_7_vars.png' or 'biplot_all_events_20.84_7_vars.png'.
    Returns a float for sorting. If not matched, returns +inf to push it to the end.
    """
    match = re.search(r"biplot_(?:.+?)_(\d+(?:\.\d+)?)_\d+_vars\.png$", filename)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return float("inf")


def make_video_and_store(
    image_folder,
    video_name,
    fps,
    max_width=3840,
    max_height=2160,
    codec_preference=None,
    allow_upscale=False,
):
    """
    Read PNG frames from image_folder, sort by timestamp in filename,
    resize to fit within max_width x max_height (keeping aspect), ensure even
    dimensions for codec compatibility, and save an MP4 under
    '<biplots>/videos/<video_name>'.

    - Uses high-quality interpolation: AREA for downscaling, LANCZOS4 for upscaling.
    - Tries a list of codecs (H.264 variants) with fallback to 'mp4v'.
    """
    image_folder_path = Path(image_folder)
    if not image_folder_path.exists():
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    images = [f for f in os.listdir(image_folder) if f.lower().endswith(".png")]
    if not images:
        raise FileNotFoundError(f"No PNG images found in {image_folder}")

    images.sort(key=_extract_timestamp_from_biplot)

    first_frame = cv2.imread(str(image_folder_path / images[0]))
    if first_frame is None:
        raise RuntimeError(f"Failed to read first image: {images[0]}")
    height, width, layers = first_frame.shape

    # Compute target size within constraints and ensure even dimensions
    scale_limit = min(max_width / width, max_height / height)
    scale = min(scale_limit, 1.0) if not allow_upscale else scale_limit
    target_width = int(round(width * scale))
    target_height = int(round(height * scale))
    # enforce even values
    if target_width % 2 != 0:
        target_width -= 1
    if target_height % 2 != 0:
        target_height -= 1
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Computed target video size is invalid. Check max_width/max_height.")

    if (target_width, target_height) != (width, height):
        interp_first = cv2.INTER_AREA if (target_width < width or target_height < height) else cv2.INTER_LANCZOS4
        first_frame = cv2.resize(first_frame, (target_width, target_height), interpolation=interp_first)

    # Create '<biplots>/videos' directory
    biplots_dir = image_folder_path.parent
    videos_dir = biplots_dir / 'videos'
    videos_dir.mkdir(parents=True, exist_ok=True)

    video_path = videos_dir / video_name

    # Try codec preferences in order
    if codec_preference is None:
        codec_preference = ['avc1', 'H264', 'X264', 'mp4v']

    writer = None
    chosen_codec = None
    for codec in codec_preference:
        tmp_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*codec), fps, (target_width, target_height))
        if tmp_writer.isOpened():
            writer = tmp_writer
            chosen_codec = codec
            break
        else:
            tmp_writer.release()
    if writer is None:
        raise RuntimeError("Failed to initialize VideoWriter with provided codecs: " + ", ".join(codec_preference))

    try:
        # Write the first frame (already resized as needed)
        writer.write(first_frame)

        for image_name in images[1:]:
            frame = cv2.imread(str(image_folder_path / image_name))
            if frame is None:
                continue
            if frame.shape[1] != target_width or frame.shape[0] != target_height:
                interp = cv2.INTER_AREA if (frame.shape[1] > target_width or frame.shape[0] > target_height) else cv2.INTER_LANCZOS4
                frame = cv2.resize(frame, (target_width, target_height), interpolation=interp)
            writer.write(frame)
    finally:
        writer.release()
        cv2.destroyAllWindows()

    return str(video_path)


if __name__ == "__main__":
    output_path = make_video_and_store(images_folder, video_name, fps)
    print(f"Saved video to: {output_path}")