import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
import os
from pathlib import Path
import wandb
from wandb.integration.sb3 import WandbCallback
import signal
import sys
import traceback
import time
import torch

from environ import FlappyBirdEnv
from record import RecorderEnvWrapper

model_dir = Path(__file__).parent / "agent_models"
os.makedirs(model_dir, exist_ok=True)

# Global variables for cleanup
model = None
env = None
run = None

def save_model_emergency(model, run_name, reason="crash"):
    """Emergency model save function"""
    try:
        if model is not None:
            timestamp = int(time.time())
            emergency_path = model_dir / f"emergency_save_{reason}_{run_name}_{timestamp}.zip"
            model.save(emergency_path)
            print(f"\n🚨 Emergency model saved to: {emergency_path}")
            return emergency_path
    except Exception as e:
        print(f"❌ Failed to save emergency model: {e}")
    return None

def cleanup_and_exit(reason="unknown"):
    """Clean up resources and exit gracefully"""
    global model, env, run
    
    print(f"\n🧹 Cleaning up due to: {reason}")
    
    # Save model if available
    if model and run:
        save_model_emergency(model, run.name, reason)
    
    # Close environment

    if env:
        try:
            env.close()
            print("✅ Environment closed")
        except Exception as e:
            print(f"⚠️ Error closing environment: {e}")
    
    # Finish wandb run
    if run:
        try:
            run.finish()
            print("✅ WandB run finished")
        except Exception as e:
            print(f"⚠️ Error finishing WandB run: {e}")
    
    print("🏁 Cleanup completed")

def signal_handler(signum, frame):
    """Handle interrupt signals (Ctrl+C, etc.)"""
    signal_name = signal.Signals(signum).name
    print(f"\n🛑 Received {signal_name} signal")
    cleanup_and_exit(f"signal_{signal_name}")
    sys.exit(0)

def setup_gpu_environment():
    """Configure GPU settings for optimal discrete GPU usage"""
    
    # Set environment variables to force dGPU usage for rendering
    os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
    os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
    
    # Force CUDA to use discrete GPU (usually GPU 0)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only GPU 0 (discrete GPU)
    
    # Set PyTorch to use CUDA if available
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)  # Set default device to GPU 0
        print(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"🔥 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return 'cuda:0'
    else:
        print("⚠️ CUDA not available, falling back to CPU")
        return 'cpu'

def make_env(run_name, rank, seed=0, output_dir=None, fps=60, is_recorder=False):
    """
    Utility function for multiprocessed env.
    
    :param run_name: (str) the name of the run
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    :param output_dir: (str) path to save recordings
    :param fps: (int) the target FPS for the recording
    :param is_recorder: (bool) whether this environment should record
    """
    def _init():
        # All environments must have the same render_mode to satisfy SubprocVecEnv.
        # The actual rendering will only be triggered by the RecorderEnvWrapper.
        render_mode = 'rgb_array'

        # Set environment variables for this process to use discrete GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        env = FlappyBirdEnv(render_mode=render_mode)
        env.reset(seed=seed + rank)
        
        # Wrap the environment with the recorder if it's the designated one
        if is_recorder and output_dir:
            print(f"Recording rollouts for worker {rank} to {output_dir} at {fps} FPS")
            env = RecorderEnvWrapper(env, output_dir=output_dir, worker_index=rank, run=run_name, fps=fps)
        
        return env
    return _init

class PeriodicSaveCallback(BaseCallback):
    """Custom callback to save model periodically during training"""
    def __init__(self, save_freq=10000, save_path=None, run=None, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.run = run
        self.last_save = 0
        
    def _on_step(self) -> bool:
        """Called by stable-baselines3 after each step"""
        try:
            # Get current timestep
            timestep = self.num_timesteps
            
            # Save every save_freq timesteps
            if timestep - self.last_save >= self.save_freq:
                checkpoint_path = self.save_path / f"checkpoint_{self.run}_{timestep}.zip"
                self.model.save(checkpoint_path)
                if self.verbose >= 1:
                    print(f"💾 Checkpoint saved at timestep {timestep}: {checkpoint_path}")
                self.last_save = timestep
        except Exception as e:
            print(f"⚠️ Error in periodic save callback: {e}")
        
        return True  # Continue training

class RecordingActivationCallback(BaseCallback):
    """Custom callback to activate recording after a certain number of steps."""
    def __init__(self, activation_step, verbose=0):
        super().__init__(verbose)
        self.activation_step = activation_step
        self.activated = False

    def _on_step(self) -> bool:
        """Called by stable-baselines3 after each step."""
        if not self.activated and self.num_timesteps >= self.activation_step:
            if self.verbose > 0:
                print(f"🎥 Activating recording at step {self.num_timesteps}")
            # Activate recording only on the first environment (the recorder)
            self.training_env.env_method("enable_recording", indices=[0])
            self.activated = True
        return True

if __name__ == '__main__':
    # Set up GPU environment first
    device = setup_gpu_environment()
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    try:
        # Initialize wandb
        run = wandb.init(
            entity="divyanshtut",
            project="FlappyBirDiT",
            sync_tensorboard=True,  # auto-sync sb3 logs
            monitor_gym=True,       # auto-upload videos
            save_code=True,         # save the main script

            notes=
            """
            1. Changed lr=1e-4, cliprange=0.1, entcoef=0.01 and vfcoef=0.5,
            2. didnt reward structure 
            3. normalized birdy, nextwindow_x. nextwindow_y
            4. added VecFramesStacking nframes=4
            """
        )

        output_path = Path(__file__).parent / "data" / "ppo_rollouts" / run.name
        os.makedirs(output_path, exist_ok=True)
        
        n_envs = 4  # Use 8 environments: 1 for recording, 7 for training
        n_frames = 4
        TOTAL_TIMESTEPS = 7_000_000 
        start_recording_at = 250_000

        
        # Create the vectorized environment.
        env_fns = [make_env(run.name, i, output_dir=output_path, fps=60, is_recorder=(i==0)) for i in range(n_envs)]
        env = SubprocVecEnv(env_fns)
        env = VecFrameStack(env, n_stack=n_frames)

        # Instantiate the PPO model with explicit device
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,  # Explicitly set device
            tensorboard_log=f"runs/{run.name}", # Log to a unique directory for each run

            learning_rate=1e-5,
            n_steps = 2048,
            batch_size=64,
            gamma=0.99,
            ent_coef=0.01,
            vf_coef=0.5,
            clip_range=0.1
        )

        # Create WandbCallback
        wandb_callback = WandbCallback(
            model_save_path=f"models/{run.name}",
            verbose=2,
        )
        
        # Create periodic save callback
        periodic_save_callback = PeriodicSaveCallback(
            save_freq=10000,  # Save every 10k timesteps
            save_path=model_dir,
            run=run.name,
            verbose=1
        )

        # Create recording activation callback
        recording_activation_callback = RecordingActivationCallback(
            activation_step=start_recording_at, # Start recording after 500k steps
            verbose=1
        )

        # Train the model with error handling
        print("Starting PPO training...")
        print(f"🔄 Automatic checkpoints will be saved every 10,000 timesteps to: {model_dir}")
        
        
        print(f"🎯 Training for {TOTAL_TIMESTEPS:,} timesteps")

        
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[wandb_callback, periodic_save_callback, recording_activation_callback]  # Multiple callbacks
        )
        
        print("✅ Training finished successfully!")

        # Save the final trained model
        final_model_path = model_dir / f"agent_final_{run.name}.zip"
        model.save(final_model_path)
        print(f"💾 Final model saved to: {final_model_path}")
        
        print(f"\n📁 Training rollouts have been saved in: {output_path}")

        # Clean up the environment
        cleanup_and_exit("normal_completion")

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        cleanup_and_exit("keyboard_interrupt")
        
    except OSError as e:
        if e.errno == 28:  # No space left on device
            print(f"\n💽 Disk space error: {e}")
            print("🧹 Consider cleaning up old rollout files:")
            print(f"   rm {output_path}/*.h5")
            cleanup_and_exit("disk_full")
        else:
            print(f"\n💥 OS Error: {e}")
            cleanup_and_exit("os_error")
            
    except Exception as e:
        print(f"\n💥 Unexpected error occurred: {e}")
        print("\n📋 Full traceback:")
        traceback.print_exc()
        cleanup_and_exit("unexpected_error")
        
    finally:
        # This ensures cleanup happens no matter what
        if 'model' in locals() or 'env' in locals() or 'run' in locals():
            print("\n🔒 Final cleanup in finally block...")
            # Don't call cleanup_and_exit here to avoid double cleanup
            # Just ensure critical resources are freed
            if 'env' in locals() and env:
                try:
                    env.close()
                except:
                    pass
            if 'run' in locals() and run:
                try:
                    run.finish()
                except:
                    pass