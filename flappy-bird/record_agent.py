import argparse
import time
from pathlib import Path
import sys
import torch
import os
import multiprocessing as mp

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from environ import FlappyBirdEnv
from record import RecorderEnvWrapper
from h5_to_video import create_annotated_video_ffmpeg, create_annotated_video_opencv, is_ffmpeg_available

def record_gameplay(run_name: str, duration_minutes: int, force_cpu: bool = False):
    """
    Records a single gameplay session of a trained agent.
    """
    model_dir = Path(__file__).parent / "agent_models"
    model_path = model_dir / f"agent_final_{run_name}.zip"

    if not model_path.exists():
        print(f"Warning: Final model not found at {model_path}")
        checkpoints = sorted(model_dir.glob(f"checkpoint_{run_name}_*.zip"), reverse=True)
        if checkpoints:
            model_path = checkpoints[0]
            print(f"Using the latest checkpoint instead: {model_path}")
        else:
            print(f"Error: No final model or checkpoints found for run '{run_name}' in {model_dir}.")
            sys.exit(1)

    # Save recordings into a folder named after the agent
    output_dir = Path(__file__).parent / "data" / "agent_recordings" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Recordings will be saved to: {output_dir}")

    # Give each recording a unique timestamped filename
    session_id = f"{run_name}_{int(time.time())}"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading agent from: {model_path} on device: {device}")
    model = PPO.load(model_path, device=device)

    def _init_env():
        env = FlappyBirdEnv(render_mode="rgb_array")
        env = RecorderEnvWrapper(env, output_dir=output_dir, worker_index=0, run=session_id, fps=60)
        env.enable_recording()
        return env

    env = DummyVecEnv([_init_env])
    env = VecFrameStack(env, n_stack=4)

    print(f"Starting recording for {duration_minutes} minute(s). Press Ctrl+C to stop early.")
    obs = env.reset()
    start_time = time.time()
    duration_seconds = duration_minutes * 60

    episode_count = 0
    steps_this_episode = 0
    base_deterministic_steps = 0
    step_increase_per_episode = 1

    try:
        while time.time() - start_time < duration_seconds:
            determinism_threshold = base_deterministic_steps + (episode_count * step_increase_per_episode)
            is_deterministic = steps_this_episode < determinism_threshold

            action, _ = model.predict(obs, deterministic=is_deterministic)
            obs, _, dones, info = env.step(action)
            steps_this_episode += 1

            if dones[0]:
                print(f"Episode {episode_count + 1} finished. Next episode will be deterministic for {determinism_threshold + step_increase_per_episode} steps.")
                steps_this_episode = 0
                episode_count += 1

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    finally:
        env.close() # This now waits for all save processes to finish
        print("Recording finished and data saved.")

    final_h5_path = output_dir / f"rollout_worker_0_run_{session_id}.h5"

    if not final_h5_path.exists() or final_h5_path.stat().st_size == 0:
        print("H5 file was not created or is empty. Aborting video conversion.")
        return

    video_output_path = final_h5_path.with_suffix('.mp4')
    print(f"Converting H5 to video: {video_output_path}")
    try:
        if not force_cpu and is_ffmpeg_available():
            create_annotated_video_ffmpeg(final_h5_path, video_output_path)
        else:
            create_annotated_video_opencv(final_h5_path, video_output_path)
        print(f"Video conversion successful!")
    except Exception as e:
        print(f"An error occurred during video conversion: {e}")

def main():
    parser = argparse.ArgumentParser(description="Record gameplay of a trained Flappy Bird agent.")
    parser.add_argument("run_name", type=str, help="The name of the wandb run for the agent to use.")
    parser.add_argument("minutes", type=int, help="The number of minutes to record the gameplay.")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU-based video encoding even if a GPU is available.")
    args = parser.parse_args()

    if __name__ == "__main__":
        mp.set_start_method('spawn', force=True)
        record_gameplay(args.run_name, args.minutes, args.force_cpu)

if __name__ == "__main__":
    main()
