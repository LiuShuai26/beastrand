"""Evaluate a trained Tennis model with rendering.

Usage:
    python -m projects.tennis.eval --checkpoint projects/tennis/latest.pt
    python -m projects.tennis.eval --checkpoint projects/tennis/latest.pt --episodes 5
    python -m projects.tennis.eval --checkpoint projects/tennis/latest.pt --self-play
"""
import argparse
import time
from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from projects.atari.policy import AtariPolicy
from projects.tennis.make_env import make_env


def make_policy_cfg(env, args_ns):
    """Build a minimal cfg object that AtariPolicy expects."""
    obs_shape = env.observation_space.shape  # (4, 84, 84)
    act_n = env.action_space.n
    act_shape = (1,)
    return SimpleNamespace(
        obs_shape=obs_shape,
        act_shape=act_shape,
        act_n=act_n,
        args=args_ns,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate Tennis model")
    parser.add_argument("--checkpoint", type=str, default="projects/tennis/latest.pt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--self-play", action="store_true",
                        help="Use the same model for both agents (otherwise opponent is random)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic (greedy) actions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Delay between steps in seconds (for slower playback)")
    cli = parser.parse_args()

    # Minimal args namespace for make_env and policy
    args_ns = SimpleNamespace(frame_stack=4, normalize_input=True)

    # Create env with rgb_array rendering (pygame "human" hangs on macOS)
    env = make_env("tennis_v3", seed=cli.seed, render_mode="rgb_array", args=args_ns)

    # Build policy
    cfg = make_policy_cfg(env, args_ns)
    policy = AtariPolicy(cfg)

    # Load checkpoint
    ckpt = torch.load(cli.checkpoint, map_location="cpu", weights_only=True)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    print(f"Loaded checkpoint: {cli.checkpoint}")

    # Opponent policy (same model or random)
    opp_policy = None
    if cli.self_play:
        opp_policy = AtariPolicy(cfg)
        opp_policy.load_state_dict(ckpt["policy"])
        opp_policy.eval()
        print("Opponent: self-play (same model)")
    else:
        print("Opponent: random actions")

    for ep in range(cli.episodes):
        obs, info = env.reset(seed=cli.seed + ep)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            # Agent 0 action
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).unsqueeze(0)
                out = policy.act({"obs": obs_t}, deterministic=cli.deterministic)
                action = out["action"].squeeze().int().item()

            # Agent 1 action (opponent)
            if opp_policy is not None and env.agent_obs[1] is not None:
                with torch.no_grad():
                    opp_obs_t = torch.from_numpy(env.agent_obs[1]).unsqueeze(0)
                    opp_out = opp_policy.act({"obs": opp_obs_t}, deterministic=cli.deterministic)
                    env.agent_actions[1] = opp_out["action"].squeeze().int().item()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            # Render via OpenCV
            frame = env._env.render()
            if frame is not None:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_bgr = cv2.resize(frame_bgr, (640, 480), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Tennis", frame_bgr)
                key = cv2.waitKey(max(1, int(cli.delay * 1000)))
                if key == 27:  # ESC to quit
                    done = True

        print(f"Episode {ep + 1}: reward={total_reward:.1f}, steps={steps}")

    env.close()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
