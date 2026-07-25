from datasets import load_dataset

def load_dataset():
    ds = load_dataset("ahmed-masry/ChartQA", split="train")
    return ds

def build_dataset():
    # divide workers across 100 workers
    videos_per_worker = 1800000/100 # 18000 videos each
    # Each worker processes its chunk independently
    for worker_id in range(100):
        # eg: 1 * 18000
        start_idx = worker_id * videos_per_worker
        # eg: 18000 + 18000 = 36000
        end_idx = start_idx + videos_per_worker
        #process_videos(videos[start_idx:end_idx], worker_id)

def is_video_static(video_file, threshold=0.4):
    """Check if video has static content using ffmpeg freezedetect."""

    # Get video duration
    result = subprocess.run([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", video_file
    ], capture_output=True, text=True)

    duration = float(result.stdout.strip())
    segments = math.ceil(duration / 60) # 60-second segments
    freeze_count = 0
    # Check each segment for freezes
    for start in range(0, int(duration), 60):
        result = subprocess.run([
            "ffmpeg", "-ss", str(start), "-i", video_file, "-t", "60",
            "-vf", "freezedetect=n=0.05:d=50", "-f", "null", "-"
        ],capture_output=True, text=True)

        if "freezedetect" in result.stderr:
            freeze_count += 1
    
    freeze_percentage = freeze_count / segments
    print(f"  Freeze percentage: {freeze_percentage:.1%}")
    return freeze_percentage >= threshold
