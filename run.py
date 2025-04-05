# run_vlm_agent_demo.py
import os
import habitat
import cv2
import json
from config_unit import LAB_CONFIG


import argparse
import json
from collections import defaultdict
from typing import Dict

import habitat

from habitat.sims.habitat_simulator.actions import HabitatSimActions

# === 你自己的 VLM agent ===
# from agent.qwen_vl_agent import QwenVLAgent
# from qwen_dummy import load_qwen_vl

# 动作映射表
#   - actions:
#     - stop
#     - move_forward
#     - turn_left
#     - turn_right

action_map = {
    "move_forward": HabitatSimActions.move_forward,
    "turn_left": HabitatSimActions.turn_left,
    "turn_right": HabitatSimActions.turn_right,
    "stop": HabitatSimActions.stop,
}

def vlm_agent_benchmark(config, num_episodes=None):
    """
    用 Qwen2.5-VL 执行 R2R 导航任务并评估。
    """
    # model = load_qwen_vl()
    # agent = QwenVLAgent(model)

    with habitat.Env(config=config) as env:
        if num_episodes is None:
            num_episodes = len(env.episodes)

        all_metrics = []

        for ep in range(num_episodes):
            obs = env.reset()
            instruction = obs["instruction"]
            instruction_text = instruction["text"]
            done = False
            steps = 0
            print(f"Episode {ep} instruction: {instruction_text}")
            # while not done and steps < config.habitat.environment.max_episode_steps:
            #     rgb = obs["rgb"]
                
            #     action_str = agent.get_action(instruction, rgb)
            #     action = action_map.get(action_str, HabitatSimActions.stop)

            #     obs = env.step(action)
            #     done = obs["done"]
            #     steps += 1

            metrics = env.get_metrics()
            all_metrics.append(metrics)
            print(f"[Episode {ep}] Metrics: {metrics}")

    # 平均指标
    agg = defaultdict(float)
    for m in all_metrics:
        for k, v in m.items():
            if isinstance(v, (int, float)):
                agg[k] += v
    avg_metrics = {k: v / num_episodes for k, v in agg.items()}

    # 保存
    with open("vlm_agent_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    return avg_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="How many episodes to evaluate"
    )
    args = parser.parse_args()
  
    print(LAB_CONFIG)

    metrics = vlm_agent_benchmark(LAB_CONFIG, num_episodes=args.num_episodes)

    print("Benchmark for Qwen2.5-VL agent:")
    for k, v in metrics.items():
        print("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()