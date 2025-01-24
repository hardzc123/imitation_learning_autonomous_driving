"""
bc_dagger_carracing.py

New: hybrid mode
    1) Human expert plays first => human_expert_carracing.npy
    2) Train "Human Expert" BC model => human_expert_carracing.pth + human_expert.gif
    3) Then train PPO => ppo_carracing_expert.zip + ppo_expert.gif
    4) During DAgger, use PPO as the initial policy, calculate action discrepancy with "Human Expert Model" and detect grass
        - Large action discrepancy or car enters grass => use "Human Expert Model" action
    5) Output the final DAgger model and record GIF (dagger_hybrid.gif)

Usage example:
  python bc_dagger_carracing.py --expert_source=hybrid --expert_episodes=5 --manual_maxsteps=5000

Or:
  python bc_dagger_carracing.py --expert_source=ppo --ppo_steps=100000 --expert_episodes=5 --manual_maxsteps=5000

Or:
  python bc_dagger_carracing.py --expert_source=manual --expert_episodes=5 --manual_maxsteps=5000
"""

import sys
import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import gymnasium as gym
import pygame
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

###############################################################################
# Parse command line arguments
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_source", type=str, default="hybrid",
                        choices=["ppo","manual","hybrid"],
                        help="Choose expert data source: 'ppo'/'manual'/'hybrid'")
    parser.add_argument("--ppo_steps", type=int, default=3000,
                        help="PPO training steps")
    parser.add_argument("--expert_episodes", type=int, default=2,
                        help="Number of expert data collection episodes (for manual/hybrid mode, refers to human episodes; for ppo mode, refers to AI episodes).")
    parser.add_argument("--manual_maxsteps", type=int, default=1000,
                        help="Maximum steps per episode in manual mode (to prevent infinite loop)")
    parser.add_argument("--bc_epochs", type=int, default=5,
                        help="Number of epochs for Behavior Cloning training")
    parser.add_argument("--dagger_iter", type=int, default=2,
                        help="Number of DAgger iterations")
    parser.add_argument("--dagger_rollout_eps", type=int, default=2,
                        help="Number of rollout episodes per iteration")
    parser.add_argument("--dagger_epochs", type=int, default=3,
                        help="Number of epochs for DAgger training")
    parser.add_argument("--out_file", type=str, default="expert_carracing.npy",
                        help="File name to save expert data in manual mode (when hybrid, save human data as human_expert_carracing.npy)")
    parser.add_argument("--n_steps", type=int, default=2048,
                        help="Number of steps collected per PPO iteration")
    parser.add_argument("--n_envs", type=int, default=1,
                        help="Number of parallel environments for PPO")
    parser.add_argument("--discrepancy_threshold", type=float, default=0.5,
                        help="Threshold for action discrepancy in DAgger")
    parser.add_argument("--grass_threshold", type=int, default=100,
                        help="Threshold for detecting grass pixels to consider the car entering grass")
    return parser.parse_args()

###############################################################################
# Helper function: create and wrap environment
###############################################################################
def create_env(render_mode, max_steps):
    env = gym.make("CarRacing-v3", render_mode=render_mode)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    return env

def create_vec_env(render_mode, max_steps, n_envs=1):
    def make_env():
        return create_env(render_mode, max_steps)
    if n_envs == 1:
        return DummyVecEnv([make_env])
    else:
        return SubprocVecEnv([make_env for _ in range(n_envs)])

###############################################################################
# 1) Train expert (ppo)
###############################################################################
def train_expert_ppo(total_steps=100000, max_steps=1000, n_steps=2048, n_envs=1, out_name="ppo_carracing_expert"):
    print(f"[ExpertPPO] total_steps={total_steps}, n_steps={n_steps}, n_envs={n_envs}")
    venv = create_vec_env(render_mode="rgb_array", max_steps=max_steps, n_envs=n_envs)
    model = PPO("CnnPolicy", venv, n_steps=n_steps, verbose=1)
    model.learn(total_timesteps=total_steps)
    model.save(out_name)
    venv.close()
    print(f"[ExpertPPO] Done, model saved => {out_name}.zip")

###############################################################################
# 2) Test PPO expert and record gif
###############################################################################
def test_ppo_expert_gif(model_path="ppo_carracing_expert", out_gif="ppo_expert.gif", ep_len=1000):
    if not os.path.exists(f"{model_path}.zip"):
        print(f"[TestPPO] No {model_path}.zip found.")
        return
    model = PPO.load(model_path)
    env = create_env("rgb_array", ep_len)
    frames=[]
    obs,_ = env.reset()
    done=False
    step=0
    total_reward=0.0
    while not done and step<ep_len:
        action, _ = model.predict(obs, deterministic=True)
        obs2, rew, done, truncated, info = env.step(action)
        total_reward += rew
        step += 1
        frame = env.render()
        frame = cv2.flip(frame, 1)  # Mirror flip
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Rotate 90 degrees counterclockwise
        frame = cv2.resize(frame, (800,700))  # Resize resolution
        frames.append(frame)
        obs = obs2
        if done or truncated:
            break
    env.close()
    imageio.mimsave(out_gif, frames, fps=30)
    print(f"[TestPPO] saved {out_gif}, steps={step}, total_reward={total_reward:.2f}")

###############################################################################
# 3) Data collection
###############################################################################
def collect_expert_data_manual(num_episodes=5, max_steps=1000, out_file="expert_carracing.npy"):
    pygame.init()
    # Create window
    screen = pygame.display.set_mode((800,800))
    pygame.display.set_caption("Manual CarRacing (PyGame)")
    
    # Initialize joystick
    pygame.joystick.init()
    joystick_count = pygame.joystick.get_count()
    if joystick_count > 0:
        stick = pygame.joystick.Joystick(0)
        stick.init()
        print(f"[ManualData] Joystick found: {stick.get_name()}")
    else:
        stick = None
        print("[ManualData] No joystick found, using keyboard controls.")
    
    # Create environment
    env = create_env("rgb_array", max_steps)
    clock = pygame.time.Clock()
    
    data = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        print(f"[ManualData] Episode {ep+1}/{num_episodes}, Press ESC to quit.")
        while not done and steps < max_steps:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("[ManualData] User closed the window. Exiting.")
                    env.close()
                    pygame.quit()
                    sys.exit(0)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    print("[ManualData] ESC pressed. Exiting.")
                    env.close()
                    pygame.quit()
                    sys.exit(0)
            
            # Get pressed keys
            keys = pygame.key.get_pressed()
            
            # Calculate action
            action = get_action_from_input(stick, keys)
            
            # Record (obs, action)
            data.append((obs, action))
            
            # Environment step
            obs2, rew, done_, truncated, info = env.step(action)
            obs = obs2
            steps += 1
            if done_ or truncated:
                print(f"[ManualData] Episode {ep+1} ended, steps={steps}")
                break
            
            # Render to screen
            frame = env.render()  # shape=(96,96,3), RGB
            frame_flipped = cv2.flip(frame, 1)  # Mirror flip
            frame_rotated = cv2.rotate(frame_flipped, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Rotate 90 degrees counterclockwise
            frame_resized = cv2.resize(frame_rotated, (800,700), interpolation=cv2.INTER_AREA)  # Resize resolution
            surf = pygame.surfarray.make_surface(frame_resized)
            screen.blit(surf, (0,0))
            pygame.display.flip()
            
            # Control frame rate
            clock.tick(30)
        
        print(f"[ManualData] Episode {ep+1} completed, steps={steps}")
    
    env.close()
    pygame.quit()
    np.save(out_file, np.array(data, dtype=object))
    print(f"[ManualData] Saved {len(data)} samples to {out_file}")

def collect_expert_data_ppo_run(model_path="ppo_carracing_expert", num_episodes=5, max_steps=1000, out_file="expert_carracing.npy"):
    if not os.path.exists(f"{model_path}.zip"):
        print(f"[AutoExpert] No {model_path}.zip found.")
        return
    model = PPO.load(model_path)
    env = create_env("rgb_array", max_steps)
    data = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            data.append((obs, action))
            obs2, rew, done, truncated, info = env.step(action)
            obs = obs2
            steps += 1
            if done or truncated:
                print(f"[AutoExpert] Episode {ep+1}/{num_episodes} ended, steps={steps}")
                break
    env.close()
    np.save(out_file, np.array(data, dtype=object))
    print(f"[AutoExpert] Saved {len(data)} samples to {out_file}")

###############################################################################
# 4) Action Input Handling
###############################################################################
def get_action_from_input(stick, keys_pressed):
    steer = 0.0
    gas = 0.0
    brake = 0.0
    
    if stick is not None and stick.get_init():
        # Joystick control
        steer_axis = stick.get_axis(0)
        steer = np.clip(steer_axis, -1, 1)
        
        # Read right trigger as gas
        if stick.get_numaxes() > 5:
            gas_val = stick.get_axis(5)
            gas = np.clip((gas_val + 1) / 2, 0, 1)  # Map [-1,1] to [0,1]
        else:
            gas = 0.0
        
        # Read left trigger as brake
        if stick.get_numaxes() > 4:
            brake_val = stick.get_axis(4)
            brake = np.clip((brake_val + 1) / 2, 0, 1)  # Map [-1,1] to [0,1]
        else:
            brake = 0.0
    else:
        # Keyboard control
        if keys_pressed[pygame.K_LEFT]:
            steer = -0.6
        if keys_pressed[pygame.K_RIGHT]:
            steer = 0.6
        if keys_pressed[pygame.K_UP]:
            gas = 1.0
        if keys_pressed[pygame.K_DOWN]:
            brake = 0.8
    
    return np.array([steer, gas, brake], dtype=np.float32)

###############################################################################
# 5) Behavior Cloning (BC) Model
###############################################################################
class BCCNN(nn.Module):
    """
    Simple CNN: CarRacing(96x96x3) -> (3)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,16,5,stride=2,padding=2), # =>16x48x48
            nn.ReLU(),
            nn.Conv2d(16,32,5,stride=2,padding=2), # =>32x24x24
            nn.ReLU(),
            nn.Conv2d(32,64,5,stride=2,padding=2), # =>64x12x12
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*12*12,256),
            nn.ReLU(),
            nn.Linear(256,3)
        )
    def forward(self,obs):
        # obs: [B,96,96,3], float->[0,1]
        x = obs.permute(0,3,1,2).float() / 255.0
        return self.net(x)

###############################################################################
# 6) Train Behavior Cloning (BC) Model
###############################################################################
def train_bc(epochs=5, lr=1e-4, batch_size=32, data_file="expert_carracing.npy", out_model="bc_policy_carracing.pth", out_gif="bc_agent.gif"):
    """
    Load data from expert_carracing.npy, supervised learning, => bc_policy_carracing.pth
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"{data_file} not found. Please collect expert data first.")
    
    print(f"[BC] Loading expert data from {data_file}...")
    data = np.load(data_file, allow_pickle=True)
    obs_list, act_list = [], []
    for (o, a) in data:
        obs_list.append(o)
        act_list.append(a)
    obs_arr = np.array(obs_list)
    act_arr = np.array(act_list)
    N = len(obs_arr)
    print(f"[BC] Dataset size: {N}, obs shape: {obs_arr.shape}, actions shape: {act_arr.shape}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = BCCNN().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Train/Test split
    idx = np.random.permutation(N)
    train_ratio = 0.9
    train_count = int(N * train_ratio)
    train_idx = idx[:train_count]
    test_idx = idx[train_count:]
    obs_train, act_train = obs_arr[train_idx], act_arr[train_idx]
    obs_test, act_test = obs_arr[test_idx], act_arr[test_idx]
    
    def get_batches(obs, act, bs):
        M = len(obs)
        perm = np.random.permutation(M)
        for i in range(0, M, bs):
            b_idx = perm[i:i+bs]
            yield obs[b_idx], act[b_idx]
    
    loss_history = []
    for ep in range(1, epochs+1):
        policy.train()
        ep_loss = 0
        batch_count = 0
        for ob_b, ac_b in get_batches(obs_train, act_train, batch_size):
            ob_t = torch.tensor(ob_b, device=device)
            ac_t = torch.tensor(ac_b, device=device, dtype=torch.float32)
            pred = policy(ob_t)
            loss = criterion(pred, ac_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            batch_count += 1
        avg_loss = ep_loss / batch_count
        loss_history.append(avg_loss)
    
        # Evaluation
        policy.eval()
        with torch.no_grad():
            ob_test_t = torch.tensor(obs_test, device=device)
            ac_test_t = torch.tensor(act_test, device=device, dtype=torch.float32)
            pred_test = policy(ob_test_t)
            test_loss = criterion(pred_test, ac_test_t).item()
        print(f"[BC] Epoch {ep}/{epochs}, Train Loss={avg_loss:.4f}, Test Loss={test_loss:.4f}")
    
    # Plot loss curve
    plt.figure()
    plt.plot(loss_history, label='Train Loss')
    plt.title("Behavior Cloning Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()
    
    # Save model
    torch.save(policy.state_dict(), out_model)
    print(f"[BC] Model saved as {out_model}")
    
    # Test BC model and record GIF
    test_and_gif(policy_file=out_model,
                out_gif=out_gif,
                ep_len=1000)

###############################################################################
# 10) Test BC model and record GIF
###############################################################################
def test_and_gif(policy_file="bc_policy_carracing.pth", out_gif="bc_agent.gif", ep_len=1000):
    """
    Run CarRacing-v3 with the specified policy and record -> out_gif
    Resolution 4:3 => 640x480
    """
    if not os.path.exists(policy_file):
        raise FileNotFoundError(f"{policy_file} not found. Please train the policy first.")
    print(f"[Test] Loading policy model from {policy_file}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = BCCNN().to(device)
    policy.load_state_dict(torch.load(policy_file, map_location=device))
    policy.eval()
    
    # Create environment
    env = create_env("rgb_array", ep_len)
    frames = []
    obs, _ = env.reset()
    done = False
    step_count = 0
    total_reward = 0.0
    print(f"[Test] Starting test run, recording GIF to {out_gif}...")
    while not done and step_count < ep_len:
        ob_tensor = torch.tensor(obs[None], device=device).float()
        action = policy(ob_tensor).detach().cpu().numpy()[0]
        obs2, rew, done, truncated, info = env.step(action)
        total_reward += rew
        step_count += 1
        frame = env.render()
        frame = cv2.flip(frame, 1)  # Mirror flip
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Rotate 90 degrees counterclockwise
        frame = cv2.resize(frame, (800,700))  # Resize resolution
        frames.append(frame)
        obs = obs2
        if done or truncated:
            print(f"[Test] Episode ended, steps={step_count}, total_reward={total_reward:.2f}")
            break
    env.close()
    imageio.mimsave(out_gif, frames, fps=30)
    print(f"[Test] GIF saved as {out_gif}, steps={step_count}, total_reward={total_reward:.2f}")

###############################################################################
# 11) Train PPO and test output
###############################################################################
def train_and_test_ppo(ppo_steps, max_steps, n_steps, n_envs, out_model="ppo_carracing_expert", out_gif="ppo_expert.gif"):
    train_expert_ppo(total_steps=ppo_steps, max_steps=max_steps, n_steps=n_steps, n_envs=n_envs, out_name=out_model)
    test_ppo_expert_gif(model_path=out_model, out_gif=out_gif, ep_len=max_steps)

###############################################################################
# 12) Main function
###############################################################################
def main():
    args = parse_args()
    
    if args.expert_source == "manual":
        # Manual mode
        # 1. Collect human expert data
        collect_expert_data_manual(num_episodes=args.expert_episodes, max_steps=args.manual_maxsteps, out_file=args.out_file)
        # 2. Train BC model
        train_bc(epochs=args.bc_epochs, lr=1e-4, batch_size=32,
                 data_file=args.out_file,
                 out_model="bc_policy_carracing.pth",
                 out_gif="bc_agent.gif")
        # 3. DAgger (optional, add as needed)
        # Note: In manual mode, there is no PPO model, DAgger may not be applicable or needs adjustment
        print("[Main] Manual mode completed.")
    
    elif args.expert_source == "ppo":
        # PPO mode
        # 1. Train PPO expert
        train_and_test_ppo(args.ppo_steps, args.manual_maxsteps, args.n_steps, args.n_envs,
                           out_model="ppo_carracing_expert",
                           out_gif="ppo_expert.gif")
        # 2. Collect PPO expert data
        collect_expert_data_ppo_run("ppo_carracing_expert", args.expert_episodes, args.manual_maxsteps, "expert_carracing.npy")
        # 3. Train BC model
        train_bc(epochs=args.bc_epochs, lr=1e-4, batch_size=32,
                 data_file="expert_carracing.npy",
                 out_model="bc_policy_carracing.pth",
                 out_gif="bc_agent.gif")
        # 4. DAgger training (based on action discrepancy)
        dagger_train(iterations=args.dagger_iter,
                     rollout_eps=args.dagger_rollout_eps,
                     dagger_epochs=args.dagger_epochs,
                     lr=1e-4,
                     batch_size=32,
                     max_steps=args.manual_maxsteps,
                     discrepancy_threshold=args.discrepancy_threshold)
        print("[Main] PPO mode completed.")
    
    elif args.expert_source == "hybrid":
        # Hybrid mode
        # 1. Collect human expert data
        human_data_file = "human_expert_carracing.npy"
        collect_expert_data_manual(num_episodes=args.expert_episodes, max_steps=args.manual_maxsteps, out_file=human_data_file)
        # 2. Train human expert BC model
        train_human_expert_bc(epochs=args.bc_epochs, lr=1e-4, batch_size=32,
                              data_file=human_data_file,
                              out_model="human_expert_carracing.pth",
                              out_gif="human_expert.gif")
        # 3. Train PPO expert
        train_and_test_ppo(args.ppo_steps, args.manual_maxsteps, args.n_steps, args.n_envs,
                           out_model="ppo_carracing_expert",
                           out_gif="ppo_expert.gif")
        # 4. DAgger training (hybrid mode)
        dagger_train_hybrid(iterations=args.dagger_iter,
                            rollout_eps=args.dagger_rollout_eps,
                            dagger_epochs=args.dagger_epochs,
                            lr=1e-4,
                            batch_size=32,
                            max_steps=args.manual_maxsteps,
                            discrepancy_threshold=args.discrepancy_threshold,
                            grass_threshold=args.grass_threshold,
                            ppo_model="ppo_carracing_expert",
                            human_model="human_expert_carracing.pth",
                            out_dagger="dagger_hybrid.pth")
        print("[Main] Hybrid mode completed.")
    
    else:
        print("Error: unknown expert_source", args.expert_source)
        return

if __name__=="__main__":
    main()