import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.gridspec import GridSpec
import cv2
import sys
import argparse
from tqdm import tqdm
from pathlib import Path
from unittest.mock import MagicMock

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(project_root)

r3m_repo_path = os.path.join(project_root, 'r3m')
if os.path.exists(r3m_repo_path):
    sys.path.append(r3m_repo_path)

sys.modules["mujoco_py"] = MagicMock()
sys.modules["metaworld"] = MagicMock()
sys.modules["metaworld.envs"] = MagicMock()
sys.modules["metaworld.envs.mujoco"] = MagicMock()
sys.modules["metaworld.envs.mujoco.env_dict"] = MagicMock()
sys.modules["metaworld.envs"].ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE = {}
sys.modules["metaworld.envs"].ALL_V2_ENVIRONMENTS_GOAL_HIDDEN = {}

mock_policies = MagicMock()
all_policy_names = [
    'SawyerAssemblyV1Policy', 'SawyerAssemblyV2Policy', 'SawyerBasketballV1Policy', 'SawyerBasketballV2Policy',
    'SawyerBinPickingV2Policy', 'SawyerBoxCloseV1Policy', 'SawyerBoxCloseV2Policy',
    'SawyerButtonPressTopdownV1Policy', 'SawyerButtonPressTopdownV2Policy',
    'SawyerButtonPressTopdownWallV1Policy', 'SawyerButtonPressTopdownWallV2Policy',
    'SawyerButtonPressV1Policy', 'SawyerButtonPressV2Policy', 'SawyerButtonPressWallV1Policy', 'SawyerButtonPressWallV2Policy',
    'SawyerCoffeeButtonV1Policy', 'SawyerCoffeeButtonV2Policy', 'SawyerCoffeePullV1Policy', 'SawyerCoffeePullV2Policy',
    'SawyerCoffeePushV1Policy', 'SawyerCoffeePushV2Policy', 'SawyerDialTurnV1Policy', 'SawyerDialTurnV2Policy',
    'SawyerDisassembleV1Policy', 'SawyerDisassembleV2Policy', 'SawyerDoorCloseV1Policy', 'SawyerDoorCloseV2Policy',
    'SawyerDoorLockV1Policy', 'SawyerDoorLockV2Policy', 'SawyerDoorOpenV1Policy', 'SawyerDoorOpenV2Policy',
    'SawyerDoorUnlockV1Policy', 'SawyerDoorUnlockV2Policy', 'SawyerDrawerCloseV1Policy', 'SawyerDrawerCloseV2Policy',
    'SawyerDrawerOpenV1Policy', 'SawyerDrawerOpenV2Policy', 'SawyerFaucetCloseV1Policy', 'SawyerFaucetCloseV2Policy',
    'SawyerFaucetOpenV1Policy', 'SawyerFaucetOpenV2Policy', 'SawyerHammerV1Policy', 'SawyerHammerV2Policy',
    'SawyerHandInsertV1Policy', 'SawyerHandInsertV2Policy', 'SawyerHandlePressSideV2Policy',
    'SawyerHandlePressV1Policy', 'SawyerHandlePressV2Policy', 'SawyerHandlePullV1Policy', 'SawyerHandlePullV2Policy',
    'SawyerHandlePullSideV1Policy', 'SawyerHandlePullSideV2Policy', 'SawyerPegInsertionSideV2Policy',
    'SawyerLeverPullV2Policy', 'SawyerPegUnplugSideV1Policy', 'SawyerPegUnplugSideV2Policy',
    'SawyerPickOutOfHoleV1Policy', 'SawyerPickOutOfHoleV2Policy', 'SawyerPickPlaceV2Policy', 'SawyerPickPlaceWallV2Policy',
    'SawyerPlateSlideBackSideV2Policy', 'SawyerPlateSlideBackV1Policy', 'SawyerPlateSlideBackV2Policy',
    'SawyerPlateSlideSideV1Policy', 'SawyerPlateSlideSideV2Policy', 'SawyerPlateSlideV1Policy', 'SawyerPlateSlideV2Policy',
    'SawyerReachV2Policy', 'SawyerReachWallV2Policy', 'SawyerPushBackV1Policy', 'SawyerPushBackV2Policy',
    'SawyerPushV2Policy', 'SawyerPushWallV2Policy', 'SawyerShelfPlaceV1Policy', 'SawyerShelfPlaceV2Policy',
    'SawyerSoccerV1Policy', 'SawyerSoccerV2Policy', 'SawyerStickPullV1Policy', 'SawyerStickPullV2Policy',
    'SawyerStickPushV1Policy', 'SawyerStickPushV2Policy', 'SawyerSweepIntoV1Policy', 'SawyerSweepIntoV2Policy',
    'SawyerSweepV1Policy', 'SawyerSweepV2Policy', 'SawyerWindowCloseV2Policy', 'SawyerWindowOpenV2Policy',
]
for name in all_policy_names:
    setattr(mock_policies, name, MagicMock())
mock_policies.__all__ = all_policy_names
sys.modules["metaworld.policies"] = mock_policies

from reward_extraction.reward_functions import LearnedImageRewardFunction

class EvalLearnedImageRewardFunction(LearnedImageRewardFunction):
    def init_ranking(self):
        pass

def create_video_with_dynamic_plot(model, video_index, output_path, on_train_data=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading data for trajectory index {video_index}...")
    dataset = model.expert_data if on_train_data else model.expert_test_data
    
    if video_index >= len(dataset):
        print(f"Error: Index {video_index} out of bounds (max {len(dataset)-1})")
        return

    traj = dataset[video_index]
    
    states_np = traj[0][:-1]
    goals_np = traj[3][:-1]
    
    print("Computing reward scores...")
    with torch.no_grad():
        states_tensor = torch.Tensor(states_np).float().to(device)
        goals_tensor = torch.Tensor(goals_np).float().to(device)
        
        states_proc = model._preprocess_images(states_tensor)
        goals_proc = model._preprocess_images(goals_tensor) if model.goal_is_image else goals_tensor

        if model.disable_ranking:
            progress = np.zeros(len(states_proc))
        else:
            progress_logits = model.ranking_network(states_proc, goals_proc)
            progress = torch.sigmoid(progress_logits).squeeze(-1).cpu().numpy()
        
        print(f"progress shape: {progress.shape}")

    frames_count = len(states_np)
    first_frame = states_np[0].transpose(1, 2, 0).astype(np.uint8)
    frame_h, frame_w = first_frame.shape[:2]
    graph_w = int(frame_w * 1.6)
    
    dpi = 110
    total_width_inches = (frame_w + graph_w) / dpi
    height_inches = frame_h / dpi
    
    figsize = (total_width_inches, height_inches)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.6], wspace=0.3)
    
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    print(f"Rendering video to {output_path}...")
    print(f"Video size: {width}x{height}")
    
    for i in tqdm(range(frames_count)):
        fig.clear()
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.6], wspace=0.3)
        
        ax_img = fig.add_subplot(gs[0, 0])
        current_img = states_np[i].transpose(1, 2, 0).astype(np.uint8)
        ax_img.imshow(current_img)
        ax_img.set_title(f"Frame: {i}/{frames_count-1}", fontsize=10)
        ax_img.axis('off')
        ax_img.set_aspect('equal')
        
        ax_plot = fig.add_subplot(gs[0, 1])
        ax_plot.set_xlim(0, frames_count - 1)
        ax_plot.set_ylim(0, 1.1)
        ax_plot.set_xlabel("Time Step", fontsize=10)
        ax_plot.set_ylabel("Score", fontsize=10)
        ax_plot.grid(True, alpha=0.3)
        ax_plot.tick_params(labelsize=8)
        
        x = np.arange(i + 1)
        ax_plot.plot(x, progress[:i+1], label='Progress', color='green', linewidth=2)
        
        if i > 0:
            ax_plot.plot(i, progress[i], 'go', markersize=4)
        
        if i == 0:
            ax_plot.legend(loc='upper left', fontsize=8)
        
        canvas.draw()
        img_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        img_array = img_array.reshape(height, width, 3)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        video_writer.write(img_bgr)

    video_writer.release()
    plt.close()
    print("Done.")

def main(args):
    class MockReplayBuffer:
        def __init__(self): pass
    rb = MockReplayBuffer()
    obs_shape = (3, 224, 224) 

    print("Loading Model...")
    lrf = EvalLearnedImageRewardFunction(
        obs_size=obs_shape,
        exp_dir=args.dataset_dir,
        replay_buffer=rb,
        train_classify_with_mixup=True,
        add_state_noise=True,
        goal_is_image=True, 
        do_film_layer=True
    )

    if os.path.exists(args.model_path):
        print(f"Loading weights from {args.model_path}...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(args.model_path, map_location=device)
        
        if 'ranking_network' in checkpoint:
            lrf.ranking_network.load_state_dict(checkpoint['ranking_network'])
            if 'same_traj_classifier' in checkpoint:
                lrf.same_traj_classifier.load_state_dict(checkpoint['same_traj_classifier'])
        else:
            print("Error: Could not load weights. Format mismatch.")
            return

        if not lrf.disable_ranking:
            lrf.ranking_network.to(device)
        if not lrf.disable_classifier:
            lrf.same_traj_classifier.to(device)
        lrf.eval_mode()
    else:
        print(f"Warning: Model path {args.model_path} not found. Using random weights.")

    if args.output_path:
        output_path = args.output_path
    else:
        output_path = f"outputs/video_plots/video_{args.traj_idx}_plot.mp4"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    create_video_with_dynamic_plot(
        lrf, 
        video_index=args.traj_idx, 
        output_path=output_path,
        on_train_data=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize trajectory with progress plot")
    parser.add_argument("--dataset-dir", type=str, default="datasets/hdf5/task_name",
                        help="Directory containing expert_data.hdf and expert_test_data.hdf")
    parser.add_argument("--traj-idx", type=int, default=0,
                        help="Index of the trajectory to visualize")
    parser.add_argument("--model-path", type=str, default="outputs/reward_model/my_reward_model.pt",
                        help="Path to trained model weights")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Path to save output video (default: outputs/video_plots/video_{traj_idx}_plot.mp4)")
    args = parser.parse_args()
    
    main(args)