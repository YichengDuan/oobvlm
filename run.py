# run_vlm_agent_demo.py
import os
import habitat
import cv2
import json
from config_unit import LAB_CONFIG

import numpy as np
import argparse
import json
from collections import defaultdict
from typing import Dict

import habitat

from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
)
from habitat_sim.utils.common import d3_40_colors_rgb
from util import save_map,rgb_to_base64,depth_to_base64,draw_top_down_map

from agent.qwen_vl_agent import QwenVLAgent
# from qwen_dummy import load_qwen_vl

# 动作映射表
#   - actions:
#     - stop
#     - move_forward
#     - turn_left
#     - turn_right
#    forward_step_size: 0.25
#    turn_angle: 15
 
os.environ["MAGNUM_LOG"] = "quiet"
# os.environ["HABITAT_SIM_LOG"] = "quiet"
def make_semantic(semantic_obs):
    semantic_image = np.zeros((semantic_obs.shape[0],semantic_obs.shape[1],3),np.uint8)
    semantic_image = np.resize(d3_40_colors_rgb[semantic_obs.flatten()%40],semantic_image.shape)
    return semantic_image
action_map = {
    "move_forward": HabitatSimActions.move_forward,
    "turn_left": HabitatSimActions.turn_left,
    "turn_right": HabitatSimActions.turn_right,
    "stop_stop": HabitatSimActions.stop,
}

agent_specs = {
            "action_space": list(action_map.keys()),
            "forward_step_size": 0.25,
            "turn_angle": 15
        }
# -------------------------------
# Helper functions for ray-casting and overlay
# -------------------------------

def compute_possible_path(env:habitat.Env, num_rays=9, max_distance=5.0):
    """
    Uses ray-casting from the agent's current position to find candidate waypoints.
    Returns a list of 3D candidate endpoints in world coordinates.
    """
    agent_state = env.sim.get_agent_state()  # Get agent state from simulator
    origin = np.array(agent_state.position)   # Agent's current position
    print(agent_state)
    # Get the forward direction of the agent
    forward_vector = agent_state.rotation.transform_vector(np.array([0, 0, -1]))  # Forward direction in world coordinates
                    
    candidate_endpoints = []
    # Cast rays in a span from -45° to +45° relative to forward direction
    angles = np.linspace(-np.pi / 4, np.pi / 4, num_rays)
    for angle in angles:
        # Create a rotation about the Y axis (up)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_matrix = np.array([
            [cos_a, 0, -sin_a],
            [0,     1,  0],
            [sin_a, 0,  cos_a]
        ])
        direction = rot_matrix.dot(forward_vector)
        # Cast a ray: the API may differ; here we assume sim.cast_ray returns a hit object.
        hit = env.sim.cast_ray(origin, direction, max_distance)
        if hit.hit_fraction < 1.0:
            endpoint = origin + direction * (max_distance * hit.hit_fraction)
        else:
            endpoint = origin + direction * max_distance
        candidate_endpoints.append(endpoint)
    return candidate_endpoints

def project_to_image(point, camera_intrinsics, camera_pose):
    """
    Project a 3D world point into 2D image coordinates.
    This is a placeholder: implement using your camera model (intrinsics & extrinsics).
    For demonstration, we simply perform a scaled projection.
    """
    # For instance, assume a pinhole model and that the camera_pose is known.
    # Here we use a simplified dummy projection:
    x, y, z = point
    # Avoid division by zero:
    if z == 0:
        z = 1e-5
    u = int(500 * x / z + 320)  # fx=500, cx=320 as dummy values
    v = int(500 * y / z + 240)  # fy=500, cy=240 as dummy values
    return (u, v)

def overlay_possible_path(rgb, candidate_points, camera_intrinsics, camera_pose):
    """
    Draws the candidate path (list of 3D points) on the rgb image by projecting them to image space.
    """
    overlay_img = rgb.copy()
    # Project candidate endpoints to image plane
    pixel_points = [project_to_image(p, camera_intrinsics, camera_pose) for p in candidate_points]
    # Draw lines between consecutive candidate points
    for i in range(len(pixel_points) - 1):
        cv2.line(overlay_img, pixel_points[i], pixel_points[i+1], (0, 0, 255), thickness=2)
    # Optionally, draw circles at candidate points
    for pt in pixel_points:
        cv2.circle(overlay_img, pt, radius=3, color=(0, 255, 0), thickness=-1)
    return overlay_img

def vlm_agent_benchmark(config,agent:QwenVLAgent, num_episodes=None, save_video=False):
    """
    用 Qwen2.5-VL 执行 R2R 导航任务并评估。
    """
    results_dirname = "./results/"

    with habitat.Env(config=config) as env:
        if num_episodes is None:
            num_episodes = len(env.episodes)

        all_metrics = []
         # Dummy camera parameters for projection – replace with your actual values.
        camera_intrinsics = np.array([[500, 0, 320],
                                      [0, 500, 240],
                                      [0, 0, 1]])
        camera_pose = None  # Replace with proper camera extrinsics if available.
        for ep in range(num_episodes):
            obs = env.reset()
            images = []
            instruction = obs["instruction"]
            instruction_text = instruction["text"]
            done = False
            steps = 0
            episode_id = env.current_episode.episode_id
            print(f"Episode {episode_id} instruction: {instruction_text}")
            
            last_rgb_str = ""
            last_compose_str = ""
            while not done and steps < config.habitat.environment.max_episode_steps:
                current_rgb = obs["rgb"]

                # current_top_down_map = draw_top_down_map(env.get_metrics(), current_rgb.shape[0])
                # current_output_im = np.concatenate((current_rgb, current_top_down_map), axis=1)
                # current_compose_str = rgb_to_base64(current_output_im)

                current_rgb_str = rgb_to_base64(current_rgb)

                if len(last_rgb_str) == 0:
                    last_rgb_str = current_rgb_str
                # if len(last_compose_str) == 0:
                #     last_compose_str = current_compose_str


                # depth = obs["depth"]
                # depth_str = depth_to_base64(depth)


                
                # # semantic = obs["semantic"]
                # # --- Compute a possible path using ray-casting ---
                # candidate_path = compute_possible_path(env, num_rays=9, max_distance=5.0)

                # # Overlay the candidate path on the RGB image.
                # # (Make sure to substitute real camera parameters if available.)
                # rgb_with_path = overlay_possible_path(rgb, candidate_path, camera_intrinsics, camera_pose)

                # For visualization: show the rgb image with candidate path overlaid.
                # cv2.imshow("rgb_with_possible_path", rgb_with_path)
                # cv2.waitKey(1)
                # # visualization
                # cv2.imshow("rgb", current_rgb)
                # cv2.waitKey(1)
                # save image
                # cv2.imwrite(os.path.join(results_dirname, f"imgs/rgb_{episode_id}_step{steps}.jpg"), rgb)
                action_str = agent.get_action(img_str_list=[last_rgb_str,current_rgb_str],instruction=instruction)
                print(f"Episode {episode_id} action: {action_str}")
                if action_str == "stay":
                    continue
                action = action_map.get(action_str, HabitatSimActions.stop)
                # action = HabitatSimActions.stop
                obs = env.step(action)
                
                last_rgb_str = current_rgb_str
                # last_compose_str = current_compose_str

                if env.episode_over or action == HabitatSimActions.stop:
                    done = True
                
                if save_video:
                    save_map(obs,info=env.get_metrics(), images=images)

                steps += 1

            metrics = env.get_metrics()
            metrics.pop("top_down_map")
            all_metrics.append(metrics)
            print(f"Episode {episode_id} metrics: {metrics}")

            if save_video:
                print(f"Episode {episode_id} success!")
                images_to_video(images, results_dirname, str(episode_id))

    # average metrics
    agg = defaultdict(float)
    for m in all_metrics:
        for k, v in m.items():
            if isinstance(v, (int, float)):
                agg[k] += v
    avg_metrics = {k: v / num_episodes for k, v in agg.items()}

    # 保存
    with open("./results/vlm_agent_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    return avg_metrics

def load_reults(num_episodes):
    """
    Load the results from the JSON file.
    """
    with open("./results/vlm_agent_metrics.json", "r") as f:
        results = json.load(f)
    distance_to_goal = 0
    success = 0
    spl = 0
    for result in results:
        distance_to_goal += result["distance_to_goal"]
        success += result["success"]
        spl += result["spl"]
    avg_metrics = {
        "distance_to_goal": distance_to_goal / num_episodes,
        "success": success / num_episodes,
        "spl": spl / num_episodes,
    }
    print(avg_metrics)
    



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=20,
        help="How many episodes to evaluate"
    )
    args = parser.parse_args()
    agent = QwenVLAgent(model_path="./model/Qwen2.5-VL-7B-Instruct",agent_specs=agent_specs)
    metrics = vlm_agent_benchmark(LAB_CONFIG, agent=agent, num_episodes=args.num_episodes, save_video=False)

    print("Benchmark for Qwen2.5-VL agent:")
    for k, v in metrics.items():
        print("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
    # load_reults(20)