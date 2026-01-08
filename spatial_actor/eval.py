import os
import yaml
import csv
import argparse
import shutil
from copy import deepcopy
import numpy as np
from multiprocessing import Value
import torch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper
from yarr.utils.stat_accumulator import SimpleAccumulator
from yarr.agents.agent import VideoSummary

import spatial_actor.configs.config as default_cfg
from spatial_actor.models.agent import Agent
from spatial_actor.models.network import Network
from spatial_actor.datasets.demo_loading_utils import create_obs_config
from spatial_actor.envs.custom_rlbench_env import CustomMultiTaskRLBenchEnv
from spatial_actor.envs.rlbench_planning import (
    EndEffectorPoseViaPlanning2 as EndEffectorPoseViaPlanning,
)
from spatial_actor.envs.rollout_generator import RolloutGenerator
from spatial_actor.utils.constant import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
)
from spatial_actor.utils.agent_utils import (
    TensorboardManager,
    RLBENCH_TASKS,
    load_agent_state,
)


def load_agent(
    model_path=None,
    eval_log_dir="",
    device=0,
):
    device = f"cuda:{device}"

    model_folder = os.path.join(os.path.dirname(model_path))

    cfg = default_cfg.get_cfg_defaults()
    cfg.merge_from_file(os.path.join(model_folder, "cfg.yaml"))
    cfg.freeze()

    network = Network(
        renderer_device=device,
        **cfg.model,
    ).to(device)

    agent = Agent(
        network=network,
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        scene_bounds=SCENE_BOUNDS,
        cameras=CAMERAS,
        log_dir=f"{eval_log_dir}/eval_run",
        **cfg,
    )

    agent.build(training=False, device=device)
    load_agent_state(model_path, agent)
    agent.eval()

    print("Agent Information")
    print(agent)
    return agent


@torch.no_grad()
def eval(
    agent,
    tasks,
    eval_datafolder,
    start_episode=0,
    eval_episodes=25,
    episode_length=25,
    device=0,
    headless=True,
    logging=False,
    log_dir=None,
    verbose=True,
    save_video=False,
):
    agent.eval()

    CAMERAS = agent.cameras

    camera_resolution = [IMAGE_SIZE, IMAGE_SIZE]
    obs_config = create_obs_config(CAMERAS, camera_resolution, method_name="")

    gripper_mode = Discrete()
    arm_action_mode = EndEffectorPoseViaPlanning()
    action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)

    task_files = [
        t.replace(".py", "")
        for t in os.listdir(rlbench_task.TASKS_PATH)
        if t != "__init__.py" and t.endswith(".py")
    ]

    task_classes = []
    if tasks[0] == "all":
        tasks = RLBENCH_TASKS
        if verbose:
            print(f"evaluate on {len(tasks)} tasks: ", tasks)

    for task in tasks:
        if task not in task_files:
            raise ValueError("Task %s not recognised!." % task)
        task_classes.append(task_file_to_task_class(task))

    eval_env = CustomMultiTaskRLBenchEnv(
        task_classes=task_classes,
        observation_config=obs_config,
        action_mode=action_mode,
        dataset_root=eval_datafolder,
        episode_length=episode_length,
        headless=headless,
        swap_task_every=eval_episodes,
        include_lang_goal_in_obs=True,
        time_in_state=True,
        record_every_n=1 if save_video else -1,
    )

    eval_env.eval = True

    device = f"cuda:{device}"

    if logging:
        assert log_dir is not None

        # create metric saving writer
        csv_file = "eval_results.csv"
        if not os.path.exists(os.path.join(log_dir, csv_file)):
            with open(os.path.join(log_dir, csv_file), "w") as csv_fp:
                fieldnames = ["task", "success rate", "length", "total_transitions"]
                csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
                csv_writer.writeheader()

    # evaluate agent
    rollout_generator = RolloutGenerator(device)
    stats_accumulator = SimpleAccumulator(eval_video_fps=30)

    eval_env.launch()

    num_tasks = len(tasks)
    step_signal = Value("i", -1)

    scores = []
    for task_id in range(num_tasks):
        task_rewards = []
        task_lang_goals = []
        for ep in range(start_episode, start_episode + eval_episodes):
            episode_rollout = []
            generator = rollout_generator.generator(
                step_signal=step_signal,
                env=eval_env,
                agent=agent,
                episode_length=episode_length,
                timesteps=1,
                eval=True,
                eval_demo_seed=ep,
                record_enabled=False,
                replay_ground_truth=False,
            )

            try:
                for replay_transition in generator:
                    episode_rollout.append(replay_transition)
            except StopIteration as e:
                continue
            except Exception as e:
                eval_env.shutdown()
                raise e

            for transition in episode_rollout:
                stats_accumulator.step(transition, True)
                current_task_id = transition.info["active_task_id"]
                assert current_task_id == task_id

            task_name = tasks[task_id]
            reward = episode_rollout[-1].reward
            task_rewards.append(reward)
            lang_goal = eval_env._lang_goal
            task_lang_goals.append(lang_goal)
            if verbose:
                print(
                    f"Evaluating {task_name} | Episode {ep} | Score: {reward} | Episode Length: {len(episode_rollout)} | Lang Goal: {lang_goal}"
                )

        # report summaries
        summaries = []
        summaries.extend(stats_accumulator.pop())
        task_name = tasks[task_id]
        if logging:
            # writer csv first
            with open(os.path.join(log_dir, csv_file), "a") as csv_fp:
                fieldnames = ["task", "success rate", "length", "total_transitions"]
                csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
                csv_results = {"task": task_name}
                for s in summaries:
                    if s.name == "eval_envs/return":
                        csv_results["success rate"] = s.value
                    elif s.name == "eval_envs/length":
                        csv_results["length"] = s.value
                    elif s.name == "eval_envs/total_transitions":
                        csv_results["total_transitions"] = s.value
                    if "eval" in s.name:
                        s.name = "%s/%s" % (s.name, task_name)
                csv_writer.writerow(csv_results)
        else:
            for s in summaries:
                if "eval" in s.name:
                    s.name = "%s/%s" % (s.name, task_name)

        if len(summaries) > 0:
            task_score = [
                s.value for s in summaries if f"eval_envs/return/{task_name}" in s.name
            ][0]
        else:
            task_score = "unknown"

        print(f"[Evaluation] Finished {task_name} | Final Score: {task_score}\n")

        scores.append(task_score)

        if save_video:
            import cv2
            video_image_folder = "./tmp"
            record_fps = 25
            record_folder = os.path.join(log_dir, "videos")
            os.makedirs(record_folder, exist_ok=True)

            task_folder = os.path.join(record_folder, task_name)
            os.makedirs(task_folder, exist_ok=True)

            video_success_cnt = 0
            video_fail_cnt = 0
            video_cnt = 0

            for summary in summaries:
                if isinstance(summary, VideoSummary):
                    lang_goal = task_lang_goals[video_cnt]
                    video = deepcopy(summary.value)
                    video = np.transpose(video, (0, 2, 3, 1))
                    video = video[:, :, :, ::-1]

                    if task_rewards[video_cnt] > 99:
                        video_path = os.path.join(
                            task_folder,
                            f"{task_name}_success_{video_success_cnt}.mp4",
                        )
                        video_success_cnt += 1
                    else:
                        video_path = os.path.join(
                            task_folder, f"{task_name}_fail_{video_fail_cnt}.mp4"
                        )
                        video_fail_cnt += 1
                    video_cnt += 1
                    os.makedirs(video_image_folder, exist_ok=True)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.4
                    color = (255, 165, 0)
                    thickness = 1

                    # get the width and height of the text to be added
                    (text_width, text_height), _ = cv2.getTextSize(lang_goal, font, font_scale, thickness)
                    position = (video.shape[2] // 2 - text_width // 2, text_height + 10)

                    for idx in range(len(video) - 10):
                        frame = video[idx].astype(np.uint8)
                        # add 'lang_goal' centered at the top of each frame
                        cv2.putText(frame, lang_goal, position, font, font_scale, color, thickness, cv2.LINE_AA)
                        cv2.imwrite(os.path.join(video_image_folder, f"{idx}.png"), frame)

                    images_path = os.path.join(video_image_folder, r"%d.png")
                    os.system(
                        "ffmpeg -i {} -vf palettegen palette.png -hide_banner -loglevel error".format(
                            images_path
                        )
                    )
                    os.system(
                        "ffmpeg -framerate {} -i {} -i palette.png -lavfi paletteuse {} -hide_banner -loglevel error".format(
                            record_fps, images_path, video_path
                        )
                    )
                    os.remove("palette.png")
                    shutil.rmtree(video_image_folder)

    eval_env.shutdown()

    if logging:
        csv_fp.close()

    agent.train()

    return scores


def get_model_index(filename):
    """
    :param filenam: path of file of format /.../model_idx.pth
    :return: idx or None
    """
    if len(filename) >= 9 and filename[-4:] == ".pth":
        try:
            index = int(filename[:-4].split("_")[-1])
        except:
            index = None
    else:
        index = None
    return index


def _eval(args):
    tb = TensorboardManager(args.eval_log_dir)
    tasks_to_eval = deepcopy(args.tasks)

    model_idx = get_model_index(args.model_path)
    if model_idx is None:
        model_idx = 0

    agent = load_agent(
        model_path=args.model_path,
        eval_log_dir=args.eval_log_dir,
        device=args.device,
    )

    agent_eval_log_dir = os.path.join(
        args.eval_log_dir, os.path.basename(args.model_path).split(".")[0]
    )
    os.makedirs(agent_eval_log_dir, exist_ok=True)

    scores = eval(
        agent=agent,
        tasks=tasks_to_eval,
        eval_datafolder=args.eval_datafolder,
        start_episode=args.start_episode,
        eval_episodes=args.eval_episodes,
        episode_length=args.episode_length,
        device=args.device,
        headless=args.headless,
        logging=True,
        log_dir=agent_eval_log_dir,
        verbose=True,
        save_video=args.save_video,
    )

    print(f"model {args.model_path}, scores {scores}")

    task_scores = {}
    for i in range(len(tasks_to_eval)):
        task_scores[tasks_to_eval[i]] = scores[i]

    print("save ", task_scores)

    # caculate average success rate across all tasks
    valid_scores = [score for score in scores if isinstance(score, (int, float))]
    if valid_scores:
        avg_success_rate = np.mean(valid_scores)
        avg_success_rate_file = os.path.join(agent_eval_log_dir, f"{avg_success_rate:.2f}.txt")
        with open(avg_success_rate_file, "w") as f:
            f.write(f"Average Success Rate: {avg_success_rate:.2f}")
        print(f"Average Success Rate saved to {avg_success_rate_file}")
    else:
        print("No valid scores to calculate average success rate.")

    tb.update("eval", model_idx, task_scores)
    tb.writer.flush()
    tb.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval-datafolder", type=str, default="./data/val/")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--log-name", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=["insert_onto_square_peg"]
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="start to evaluate from which episode",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=25,
        help="how many episodes to be evaluated for each task",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=25,
        help="maximum control steps allowed for each episode",
    )

    args = parser.parse_args()

    assert args.model_path is not None

    if args.log_name is None:
        args.log_name = "none"

    args.eval_log_dir = os.path.join(os.path.dirname(args.model_path), "eval", args.log_name)
    os.makedirs(args.eval_log_dir, exist_ok=True)

    # save the arguments for future reference
    with open(os.path.join(args.eval_log_dir, "eval_cfg.yaml"), "w") as fp:
        yaml.dump(args.__dict__, fp)

    _eval(args)
