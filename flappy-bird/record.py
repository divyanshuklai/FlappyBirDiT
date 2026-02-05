import mss
import numpy as np
import h5py
from pynput import keyboard
from pathlib import Path
from multiprocessing import Process, Event
import time

class GameRecorder():
    """
    A class for recording game on the screen and its inputs and converting it into h5 dataset.
    """
    def __init__(self, name : str, window : tuple[int,int,int,int], input_map : list[str]):
        self.name = name
        self.window = window # (top, left, width, height)
        self.input_map = input_map
        self.pressed_keys = set()
        pass

    def record(self, save_path : Path, stop_event=None, fps = 24):
        frame_rate = 1 / fps
        frames = []
        inputs = []
        timestamps = []
        sct = mss.mss()


        start_time = time.time()
        next_frame_time = start_time
        frame_count = 0
        chunk_index = 0


        file_path = save_path / (self.name + ".h5")
        
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        recording = True
        while recording:
            current_time = time.time()
            if current_time >= next_frame_time:
                last_time = time.time()
                timestamps.append(last_time)
                frames.append(np.array(sct.grab(self.window)))
                input = {
                    key:key in self.pressed_keys for key in self.input_map
                }
                inputs.append(input)

                frame_count += 1
                next_frame_time = start_time + (frame_count * frame_rate)

                if len(frames) > 1000:
                    Process(target=save_frames, args=(chunk_index, file_path, frames, inputs, timestamps)).start()
                    frames = []
                    inputs = []  
                    timestamps = []
                    chunk_index += 1
            
            if stop_event and stop_event.is_set():
                recording = False
                break

            time.sleep(0.001)
        listener.stop()
        if frames:
            save_frames(chunk_index, file_path, frames, inputs, timestamps)

    def on_press(self, key):
        self.pressed_keys.add(self._key_to_string(key))

    def on_release(self, key):
        self.pressed_keys.discard(self._key_to_string(key))

    def _key_to_string(self, key):
        try:
            return key.char
        except AttributeError:
            return str(key).replace("Key.", "")

def save_frames(chunk_index, file_name, frames, inputs : list[dict[str,bool]], timestamps):
    
    frames = np.array(frames, dtype=np.uint8)
    timestamps = np.array(timestamps, dtype=np.float64)

    input_keys = list(inputs[0].keys())
    input_dtype = [(key,'bool') for key in input_keys]

    input_array = np.zeros(len(inputs), dtype=input_dtype)
    for i, input in enumerate(inputs):
        for key in input_keys:
            input_array[i][key] = input[key]

    with h5py.File(file_name, 'a') as f:
        chunk_name = f'chunk_{chunk_index:04d}'
        group = f.create_group(chunk_name)

        group.create_dataset('frames',data=frames)
        group.create_dataset('timestamps',data=timestamps)
        group.create_dataset('inputs',data=input_array)

import gymnasium as gym
from multiprocessing import Process

def save_chunk_to_h5(file_path, chunk_index, buffer, action_meanings):
    """Saves a buffer of gameplay data to a chunk in an H5 file."""
    if not buffer:
        return

    try:
        frames, actions, timestamps = zip(*buffer)

        frames = np.array(frames, dtype=np.uint8)
        timestamps = np.array(timestamps, dtype=np.float64)

        input_keys = [action_meanings.get(1, 'space')]
        input_dtype = [(key, 'bool') for key in input_keys]
        input_array = np.zeros(len(actions), dtype=input_dtype)
        
        for i, action in enumerate(actions):
            if action == 1:
                input_array[i][input_keys[0]] = True

        with h5py.File(file_path, 'a') as f:
            chunk_name = f'chunk_{chunk_index:04d}'
            group = f.create_group(chunk_name)
            group.create_dataset('frames', data=frames, compression="lzf")
            group.create_dataset('timestamps', data=timestamps)
            group.create_dataset('inputs', data=input_array)
    except Exception as e:
        print(f"Error in save_chunk_to_h5: {e}")


class RecorderEnvWrapper(gym.Wrapper):
    """
    A wrapper for a Gymnasium environment that records gameplay to an H5 file
    using a background process to prevent blocking.
    """
    def __init__(self, env, output_dir, worker_index, run, fps=60):
        super().__init__(env)
        self.output_dir = Path(output_dir)
        self.worker_index = worker_index
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.file_path = self.output_dir / f"rollout_worker_{self.worker_index}_run_{run}.h5"
        self.chunk_index = 0
        self.buffer = []
        self.chunk_size = 1000
        self.action_meanings = self.env.get_action_meanings()
        self.save_processes = []

        # FPS control
        self.target_fps = fps
        self.env_render_fps = self.env.metadata.get("render_fps", 120)
        self.record_every_n_steps = max(1, round(self.env_render_fps / self.target_fps))
        self.step_count = 0
        self.recording_enabled = False


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Always render the frame to keep the environment running at a consistent speed
        frame = self.env.render()

        if self.recording_enabled:
            # But only record the frame if it's a sampling step
            if self.step_count % self.record_every_n_steps == 0:
                timestamp = time.time()
                self.buffer.append((frame, action, timestamp))

        self.step_count += 1

        if len(self.buffer) >= self.chunk_size:
            self._save_chunk()

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.step_count = 0

        if self.recording_enabled:
            # Always record the first frame of an episode
            frame = self.env.render()
            timestamp = time.time()
            # Use action 0 (no-op) for the initial frame, as no action has been taken yet.
            self.buffer.append((frame, 0, timestamp))
        
        return result

    def close(self):
        if self.buffer:
            self._save_chunk()
        
        print(f"Waiting for {len(self.save_processes)} saving processes to finish...")
        for p in self.save_processes:
            p.join()
        print("All saving processes finished.")

        self.env.close()

    def _save_chunk(self):
        if not self.buffer:
            return

        p = Process(
            target=save_chunk_to_h5,
            args=(self.file_path, self.chunk_index, self.buffer, self.action_meanings)
        )
        p.start()
        self.save_processes.append(p)

        self.buffer = []
        self.chunk_index += 1

    def enable_recording(self):
        """Enable recording for this environment."""
        self.recording_enabled = True


