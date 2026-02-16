import os
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
import sys
import argparse
from unittest.mock import MagicMock

# -------------------------------------------------------------------
# パス設定とモック化の共通処理
# -------------------------------------------------------------------
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(project_root)

r3m_repo_path = os.path.join(project_root, 'r3m')
if os.path.exists(r3m_repo_path):
    sys.path.append(r3m_repo_path)

# MuJoCo / Metaworld モック
sys.modules["mujoco_py"] = MagicMock()
sys.modules["metaworld"] = MagicMock()
sys.modules["metaworld.envs"] = MagicMock()
sys.modules["metaworld.envs.mujoco"] = MagicMock()
sys.modules["metaworld.envs.mujoco.env_dict"] = MagicMock()
sys.modules["metaworld.envs"].ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE = {}
sys.modules["metaworld.envs"].ALL_V2_ENVIRONMENTS_GOAL_HIDDEN = {}

# metaworld.policies モック
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

# -------------------------------------------------------------------

from reward_extraction.reward_functions import LearnedImageRewardFunction

def train(dataset_dir, model_path, num_steps=5000):
    if not os.path.exists(dataset_dir):
        print(f"Directory {dataset_dir} not found. Please run converter script first.")
        return

    class MockReplayBuffer:
        def __init__(self):
            pass
    rb = MockReplayBuffer()

    print("Initializing Reward Function and Starting Training...")
    
    obs_shape = (3, 224, 224)
    from reward_extraction.reward_functions import LearnedImageRewardFunction as BaseLRF
    from reward_extraction.reward_functions import mixup_data, mixup_criterion
    
    class CustomLearnedImageRewardFunction(BaseLRF):
        def __init__(self, *args, num_init_steps=5000, **kwargs):
            super().__init__(*args, **kwargs)
            self.num_init_steps = num_init_steps
        
        def init_ranking(self):
            pass
        
        def _train_ranking_step(self, do_inference=True):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            expert_idxs = np.random.randint(self.num_expert_trajs, size=(self.batch_size,))
            
            expert_t_idxs = np.zeros(self.batch_size, dtype=np.int64)
            expert_other_t_idxs = np.zeros(self.batch_size, dtype=np.int64)
            for i, traj_idx in enumerate(expert_idxs):
                traj_len = self.expert_data[traj_idx][0].shape[0]
                expert_t_idxs[i] = np.random.randint(0, max(1, traj_len))
                expert_other_t_idxs[i] = np.random.randint(0, max(1, traj_len))
            
            labels = np.zeros((self.batch_size,))
            first_before = np.where(expert_t_idxs < expert_other_t_idxs)[0]
            labels[first_before] = 1.0
            
            expert_states_t_np = np.concatenate([self.expert_data[traj_idx][0][t_idx][None] for (traj_idx, t_idx) in zip(expert_idxs, expert_t_idxs)])
            expert_states_other_t_np = np.concatenate([self.expert_data[traj_idx][0][t_idx][None] for (traj_idx, t_idx) in zip(expert_idxs, expert_other_t_idxs)])
            expert_goals_np = np.concatenate([self.expert_data[traj_idx][3][t_idx][None] for (traj_idx, t_idx) in zip(expert_idxs, expert_t_idxs)])
            
            expert_states_t = torch.Tensor(expert_states_t_np).float().to(device)
            expert_states_other_t = torch.Tensor(expert_states_other_t_np).float().to(device)
            expert_goals = torch.Tensor(expert_goals_np).float().to(device)
            
            expert_states_p_t = self._preprocess_images(expert_states_t)
            expert_states_other_p_t = self._preprocess_images(expert_states_other_t)
            if self.goal_is_image:
                expert_goals = self._preprocess_images(expert_goals)
            
            ranking_labels = F.one_hot(torch.Tensor(labels).long().to(device), 2).float()
            
            loss_monotonic = torch.Tensor([0.0])
            if do_inference:
                if not self.disable_ranking:
                    if self.train_classify_with_mixup:
                        rank_states = torch.cat([expert_states_p_t, expert_states_other_p_t], dim=0)
                        rank_goals = torch.cat([expert_goals, expert_goals], dim=0)
                        rank_labels = torch.cat([ranking_labels[:, 0], ranking_labels[:, 1]], dim=0).unsqueeze(1)
                        
                        mixed_rank_states, rank_labels_a, rank_labels_b, rank_lam, mixed_rank_goals = mixup_data(rank_states, rank_labels, goals=rank_goals)
                        mixed_rank_prediction_logits = self.ranking_network(mixed_rank_states, mixed_rank_goals)
                        loss_monotonic = mixup_criterion(
                            self.bce_with_logits_criterion, mixed_rank_prediction_logits, rank_labels_a, rank_labels_b, rank_lam
                        )
                    else:
                        expert_logits_t = self.ranking_network(expert_states_p_t, expert_goals)
                        expert_logits_other_t = self.ranking_network(expert_states_other_p_t, expert_goals)
                        expert_logits = torch.cat([expert_logits_t, expert_logits_other_t], dim=-1)
                        
                        loss_monotonic = self.bce_with_logits_criterion(expert_logits, ranking_labels)
            
            return loss_monotonic, expert_states_p_t, expert_goals
        
        def _train_step(self):
            # counterfactual sampling も修正
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.seen_on_policy_data = True
            
            loss_monotonic, expert_states_p_t, expert_goals = self._train_ranking_step(do_inference=False)
            
            half_batch_size = self.batch_size // 2
            expert_cf_idxs = np.random.randint(self.num_expert_trajs, size=(half_batch_size,))
            
            if self.train_classifier_with_goal_state_only:
                expert_cf_t_idxs = np.zeros(half_batch_size, dtype=np.int64)
                for i, traj_idx in enumerate(expert_cf_idxs):
                    traj_len = self.expert_data[traj_idx][0].shape[0]
                    expert_cf_t_idxs[i] = max(0, traj_len - 1)
            else:
                expert_cf_t_idxs = np.zeros(half_batch_size, dtype=np.int64)
                for i, traj_idx in enumerate(expert_cf_idxs):
                    traj_len = self.expert_data[traj_idx][0].shape[0]
                    expert_cf_t_idxs[i] = np.random.randint(0, max(1, traj_len))
            
            expert_cf_states = np.concatenate([self.expert_data[traj_idx][0][t_idx][None] for (traj_idx, t_idx) in zip(expert_cf_idxs, expert_cf_t_idxs)])
            
            if self.for_rlpd:
                next_batch = next(self.replay_buffer)
                rb_cf_states = next_batch['observations']['pixels']
                rb_cf_states = np.transpose(np.squeeze(rb_cf_states), (0, 3, 1, 2))
            else:
                rb_episode_trajs = [self.replay_buffer._sample_episode()[self.rb_buffer_obs_key][:-1] for _ in range(half_batch_size)]
                rb_cf_t_idxs = np.zeros(half_batch_size, dtype=np.int64)
                for i, traj in enumerate(rb_episode_trajs):
                    traj_len = traj.shape[0]
                    rb_cf_t_idxs[i] = np.random.randint(0, max(1, traj_len))
                rb_cf_states = np.concatenate([rb_episode_trajs[idx][t][None] for idx, t in enumerate(rb_cf_t_idxs)])
            
            cf_states_np = np.concatenate([expert_cf_states, rb_cf_states], axis=0)
            cf_states = torch.Tensor(cf_states_np).float().to(device)
            cf_states_p = self._preprocess_images(cf_states)
            
            classify_states = torch.cat([expert_states_p_t, cf_states_p], dim=0)
            classify_goals = torch.cat([expert_goals, expert_goals], dim=0)
            traj_labels = torch.cat([torch.ones((expert_states_p_t.size()[0], 1)), torch.zeros((cf_states_p.size()[0], 1))], dim=0).to(device)
            
            if self.train_classify_with_mixup:
                mixed_classify_states, traj_labels_a, traj_labels_b, lam, mixed_goals = mixup_data(classify_states, traj_labels, goals=classify_goals)
                mixed_traj_prediction_logits = self.same_traj_classifier(mixed_classify_states, mixed_goals)
                loss_same_traj = mixup_criterion(
                    self.bce_with_logits_criterion, mixed_traj_prediction_logits, traj_labels_a, traj_labels_b, lam
                )
            else:
                traj_prediction_logits = self.same_traj_classifier(classify_states, classify_goals)
                loss_same_traj = self.bce_with_logits_criterion(traj_prediction_logits, traj_labels)
            
            return {
                "ranking_loss": loss_monotonic,
                "same_traj_loss": loss_same_traj,
            }
        
        def train_ranking_with_steps(self):
            if self.disable_ranking:
                print(f"LRF not configured to use a ranking function. Not training!")
                return

            from tqdm import tqdm
            ranking_init_losses_mean = []
            ranking_init_losses_std = []
            ranking_init_losses_steps = []
            running_losses = []
            window_size = 10
            
            print(f"Training the ranking function for {self.num_init_steps} steps:")
            self.ranking_network.train()
            for i in tqdm(range(self.num_init_steps)):
                self.ranking_optimizer.zero_grad()
                ranking_loss, _, _ = self._train_ranking_step()
                ranking_loss.backward()
                self.ranking_optimizer.step()

                running_losses.append(ranking_loss.item())
                if i % window_size == 0:
                    ranking_init_losses_mean.append(np.mean(running_losses))
                    ranking_init_losses_std.append(np.std(running_losses))
                    ranking_init_losses_steps.append(i)
                    running_losses = []
            
            self.ranking_network.eval()
    
    lrf = CustomLearnedImageRewardFunction(
        obs_size=obs_shape,
        exp_dir=dataset_dir,
        replay_buffer=rb,
        train_classify_with_mixup=True,
        add_state_noise=True,
        goal_is_image=True, 
        do_film_layer=True,
        disable_classifier=True,
        num_init_steps=num_steps
    )
    
    max_traj_len = max(traj[0].shape[0] for traj in lrf.expert_data)
    print(f"Max trajectory length in dataset: {max_traj_len}")
    lrf.horizon = max_traj_len
    
    print("Training ranking function...")
    lrf.train_ranking_with_steps()
    
    print("Classifier disabled (only ranking function trained)")
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    save_dict = {
        'ranking_network': lrf.ranking_network.state_dict()
    }
    
    torch.save(save_dict, model_path)
    print(f"Model weights saved to: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ranking network for reward function")
    parser.add_argument("--dataset-dir", type=str, default="datasets/hdf5/dataset_name",
                        help="Directory containing expert_data.hdf and expert_test_data.hdf")
    parser.add_argument("--model-path", type=str, default="outputs/reward_model/my_reward_model.pt",
                        help="Path to save trained model weights")
    parser.add_argument("--num-steps", type=int, default=5000,
                        help="Number of training steps for ranking network")
    args = parser.parse_args()
    
    train(args.dataset_dir, args.model_path, args.num_steps)