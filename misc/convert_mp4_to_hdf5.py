import os
import glob
import cv2
import numpy as np
import subprocess
import tempfile
import sys
import argparse
import random
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward_extraction.data import H5PyTrajDset

def convert_video_to_readable_format(input_path, output_path):
    command = [
        'ffmpeg',
        '-y',
        '-i', input_path,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-an',
        '-loglevel', 'error',
        output_path
    ]
    subprocess.run(command, check=True)


def load_mp4_as_frames(video_path, resize=(224, 224)):
    frames = []

    with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_video:
        try:
            convert_video_to_readable_format(video_path, temp_video.name)

            cap = cv2.VideoCapture(temp_video.name)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if resize:
                    frame = cv2.resize(frame, resize)
                frame = np.transpose(frame, (2, 0, 1))
                frames.append(frame)
            cap.release()

        except subprocess.CalledProcessError as e:
            print(f"Error converting video {video_path}: {e}")
            return np.array([], dtype=np.uint8)
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return np.array([], dtype=np.uint8)

    return np.array(frames, dtype=np.uint8)


def create_dataset_from_files(mp4_files, output_path):
    if len(mp4_files) == 0:
        print(f"No mp4 files for {output_path}. Skipping.")
        return

    save_dir = os.path.dirname(output_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if os.path.exists(output_path):
        os.remove(output_path)

    dset = H5PyTrajDset(output_path, read_only_if_exists=False)

    print(f"Converting {len(mp4_files)} videos to {output_path}...")

    count = 0
    for video_file in tqdm(mp4_files):
        frames = load_mp4_as_frames(video_file)

        if len(frames) == 0:
            print(f"Warning: Could not read frames from {video_file}. Skipping.")
            continue

        states = np.expand_dims(frames, axis=0)
        traj_len = states.shape[1]

        actions = np.zeros((1, traj_len, 4), dtype=np.float32)
        rewards = np.zeros((1, traj_len), dtype=np.float32)

        final_frame = states[0, -1]
        goals = np.tile(final_frame[None, None, ...], (1, traj_len, 1, 1, 1))
        env_full_states = np.zeros((1, traj_len, 10), dtype=np.float32)

        dset.add_traj(states, actions, rewards, goals, env_full_states)
        count += 1

    print(f"Saved {count} trajectories to {output_path}.")


def split_files(mp4_files, train_ratio=0.8, shuffle=True, seed=42):
    files = list(mp4_files)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(files)

    split_idx = int(len(files) * train_ratio)
    split_idx = max(0, min(split_idx, len(files)))
    return files[:split_idx], files[split_idx:]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert MP4 videos from a single input path, split into train/test, and save as HDF5 datasets."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/videos/task_name",
        help="Path to input MP4 directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/hdf5/task_name",
        help="Output directory path (will be created if it doesn't exist)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of data to allocate for training (0.0-1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    parser.add_argument(
        "--max-trajs",
        type=int,
        default=None,
        help="Maximum number of files to convert (applied before train/test split)",
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomly shuffle data before split (default: enabled, disable with --no-shuffle)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.dataset_path):
        print(f"Error: dataset path not found: {args.dataset_path}")
        return

    if not (0.0 <= args.train_ratio <= 1.0):
        print("Error: --train-ratio must be between 0.0 and 1.0")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    mp4_files = sorted(glob.glob(os.path.join(args.dataset_path, "*.mp4")))
    if len(mp4_files) == 0:
        print(f"No mp4 files found in {args.dataset_path}")
        return

    if args.max_trajs is not None:
        mp4_files = mp4_files[:args.max_trajs]

    train_files, test_files = split_files(
        mp4_files,
        train_ratio=args.train_ratio,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    train_output_path = os.path.join(args.output_dir, "expert_data.hdf")
    test_output_path = os.path.join(args.output_dir, "expert_test_data.hdf")

    print(f"Total: {len(mp4_files)} | Train: {len(train_files)} | Test: {len(test_files)} | Shuffle: {args.shuffle}")

    create_dataset_from_files(train_files, train_output_path)
    create_dataset_from_files(test_files, test_output_path)


if __name__ == "__main__":
    main()