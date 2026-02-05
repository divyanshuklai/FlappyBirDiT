# FlappyBirDiT

FlappyBirDiT is a project aimed at creating a **Diffusion Transformer (DiT)** capable of generating Flappy Bird gameplay in real-time.

Currently, this repository contains the **Data Generation Pipeline**, which includes:
- A custom Gymnasium environment for Flappy Bird.
- A PPO (Proximal Policy Optimization) Reinforcement Learning agent to play the game effectively.
- Tools for recording gameplay (frames & actions) into H5 benchmarks.
- Utilities to convert recorded data into video for verification.

## 📂 Project Structure

```
flappy-bird/
├── environ.py          # Custom Gymnasium Environment for Flappy Bird
├── trainer2.py         # PPO Training script using Stable Baselines 3
├── record_agent.py     # Records trained agent gameplay to H5 files
├── record.py           # Core recording logic (using MSS & H5Py)
├── h5_to_video.py      # Utility to convert H5 recordings to MP4
├── play_game.py        # Script for human manual play
├── controller.py       # Multiprocessing controller for manual recording
├── flappy.py           # Core game implementation (Pygame)
├── agent_models/       # Saved RL models
└── data/               # Recorded gameplay data (.h5 files)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FlappyBirDiT.git
   cd FlappyBirDiT
   ```

2. Install dependencies:
   ```bash
   # Using uv
   uv sync
   
   # Or using pip
   pip install .
   ```

## 🛠 Usage

### 1. Train the Agent
Train a PPO agent to play Flappy Bird. This will save models to `flappy-bird/agent_models/`.

```bash
python flappy-bird/trainer2.py
```

### 2. Record Gameplay
Generate dataset by running the trained agent. This saves gameplay (frames and inputs) to `flappy-bird/data/`.

```bash
python flappy-bird/record_agent.py
```

### 3. Convert Data to Video
Verify the recorded H5 data by converting it to video.

```bash
python flappy-bird/h5_to_video.py --input flappy-bird/data/some_recording.h5 --output output_video.mp4
```

### 4. Play Manually
You can test the environment manually.
- **Space**: Flap
- **Esc**: Quit

```bash
python flappy-bird/play_game.py
```

## 📊 Data Format
The project uses H5 files to store gameplay data efficiently. Each H5 file contains:
- `frames`: Game screen captures (arrays).
- `inputs`: Boolean array of actions taken (e.g., Flap/No-Flap).
- `timestamps`: Timing data for each frame.

## 🤖 Future Work
- Implement the Diffusion Transformer (DiT) architecture.
- Train the DiT on the generated dataset.
- Real-time inference pipeline.
