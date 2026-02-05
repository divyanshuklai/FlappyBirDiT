import h5py
import numpy as np
import cv2
import argparse
from pathlib import Path
import subprocess
import shutil
import json

def is_ffmpeg_available():
    """Check if ffmpeg is in the system's PATH."""
    return shutil.which("ffmpeg") is not None

def get_gpu_info():
    """Check for NVIDIA GPU and NVENC support using nvidia-smi."""
    try:
        # Check for nvidia-smi tool
        if not shutil.which("nvidia-smi"):
            return None, "nvidia-smi not found"

        # Check for encoder support
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=encoder.support", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        encoder_support = result.stdout.strip().split('\n')

        # For simplicity, we'll just check if the first GPU supports encoding
        if "Yes" in encoder_support[0]:
            return "h264_nvenc", "NVIDIA GPU with NVENC support found."
        else:
            return None, "NVIDIA GPU found, but NVENC is not supported or enabled."

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return None, f"Error checking GPU: {e}"
    except Exception as e:
        return None, f"An unexpected error occurred while checking GPU: {e}"


def convert_frame_for_opencv(frame):
    """Convert frame to BGR format suitable for OpenCV."""
    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

def create_annotated_video_ffmpeg(h5_path, output_path, fps=None):
    """Create annotated video from H5 file using ffmpeg for hardware acceleration."""
    
    vcodec, gpu_message = get_gpu_info()
    if not is_ffmpeg_available():
        print("ffmpeg not found. Falling back to CPU-based OpenCV method.")
        return create_annotated_video_opencv(h5_path, output_path, fps)
    
    if vcodec is None:
        print(f"Warning: {gpu_message}. Using CPU encoding (libx264).")
        vcodec = 'libx264'
    else:
        print(f"Success: {gpu_message}. Using hardware encoder: {vcodec}")

    with h5py.File(h5_path, 'r') as f:
        chunk_names = sorted([name for name in f.keys() if name.startswith('chunk_')])
        if not chunk_names:
            print("No chunks found in the H5 file.")
            return

        first_chunk = f[chunk_names[0]]
        height, width, _ = first_chunk['frames'][0].shape

        if fps is None:
            timestamps = first_chunk['timestamps'][:]
            fps = 1.0 / np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 30
            print(f"Calculated FPS: {fps:.2f}")

        command = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',  # Input from stdin
            '-c:v', vcodec,
            '-preset', 'fast',
            '-pix_fmt', 'yuv420p',
            str(output_path)
        ]

        process = subprocess.Popen(command, stdin=subprocess.PIPE)

        total_frames_processed = 0
        try:
            for i, chunk_name in enumerate(chunk_names):
                chunk = f[chunk_name]
                frames = chunk['frames'][:]
                inputs = chunk['inputs'][:]
                timestamps = chunk['timestamps'][:]
                input_keys = list(inputs.dtype.names)

                print(f"Processing chunk {i+1}/{len(chunk_names)} ('{chunk_name}') with {len(frames)} frames...")

                for j in range(len(frames)):
                    frame = convert_frame_for_opencv(frames[j])
                    annotated_frame = frame.copy()

                    y_offset = 30
                    for key in input_keys:
                        is_pressed = inputs[j][key]
                        color = (0, 255, 0) if is_pressed else (0, 0, 255)
                        text = f"{key}: {'ON' if is_pressed else 'OFF'}"
                        
                        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(annotated_frame, (5, y_offset - text_height - 5), (15 + text_width, y_offset + baseline + 5), (0, 0, 0), -1)
                        cv2.putText(annotated_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        y_offset += 35

                    timestamp_text = f"Time: {timestamps[j]:.3f}s"
                    cv2.putText(annotated_frame, timestamp_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    process.stdin.write(annotated_frame.tobytes())
                
                total_frames_processed += len(frames)

        except (BrokenPipeError, IOError):
            print("ffmpeg process terminated early. There might be an error with the ffmpeg command or setup.")
        except Exception as e:
            print(f"Error during video creation: {e}")
            raise
        finally:
            process.stdin.close()
            process.wait()
            print(f"Video saved to: {output_path}")
            print(f"Total frames written: {total_frames_processed}")


def create_annotated_video_opencv(h5_path, output_path, fps=None):
    """Fallback function to create video using OpenCV's VideoWriter."""
    print("Using CPU-based OpenCV method for video creation.")
    with h5py.File(h5_path, 'r') as f:
        # ... (The original implementation of create_annotated_video_from_chunks)
        chunk_names = sorted([name for name in f.keys() if name.startswith('chunk_')])
        if not chunk_names:
            print("No chunks found in the H5 file.")
            return

        first_chunk = f[chunk_names[0]]
        first_frame_data = first_chunk['frames'][0]
        height, width, _ = first_frame_data.shape
        
        if fps is None:
            timestamps = first_chunk['timestamps'][:]
            if len(timestamps) > 1:
                fps = 1.0 / np.mean(np.diff(timestamps))
            else:
                fps = 30
            print(f"Calculated FPS: {fps:.2f}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not out.isOpened():
            raise RuntimeError("Could not open video writer")

        total_frames_processed = 0
        try:
            for i, chunk_name in enumerate(chunk_names):
                chunk = f[chunk_name]
                frames = chunk['frames'][:]
                inputs = chunk['inputs'][:]
                timestamps = chunk['timestamps'][:]
                input_keys = list(inputs.dtype.names)

                for j in range(len(frames)):
                    frame = convert_frame_for_opencv(frames[j])
                    annotated_frame = frame.copy()

                    y_offset = 30
                    for key in input_keys:
                        is_pressed = inputs[j][key]
                        color = (0, 255, 0) if is_pressed else (0, 0, 255)
                        text = f"{key}: {'ON' if is_pressed else 'OFF'}"
                        
                        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(annotated_frame, (5, y_offset - text_height - 5), (15 + text_width, y_offset + baseline + 5), (0, 0, 0), -1)
                        cv2.putText(annotated_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        y_offset += 35

                    timestamp_text = f"Time: {timestamps[j]:.3f}s"
                    cv2.putText(annotated_frame, timestamp_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    out.write(annotated_frame)
                
                total_frames_processed += len(frames)
        finally:
            out.release()
            print(f"Video saved to: {output_path}")
            print(f"Total frames written: {total_frames_processed}")


def main():
    parser = argparse.ArgumentParser(description='Convert H5 game recording to annotated video.')
    parser.add_argument('input', help='Input H5 file path')
    parser.add_argument('--output', '-o', help='Output video file path (default: input_filename.mp4)')
    parser.add_argument('--fps', type=float, help='Output video FPS (calculated from timestamps if not provided)')
    parser.add_argument('--force-cpu', action='store_true', help='Force CPU-based encoding even if GPU is available.')
    
    args = parser.parse_args()
    
    if args.output is None:
        input_path = Path(args.input)
        args.output = input_path.with_suffix('.mp4')
        print(f"Output will be saved to: {args.output}")

    try:
        if not args.force_cpu and is_ffmpeg_available():
            create_annotated_video_ffmpeg(args.input, args.output, args.fps)
        else:
            create_annotated_video_opencv(args.input, args.output, args.fps)
        print("Video creation completed successfully!")
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
