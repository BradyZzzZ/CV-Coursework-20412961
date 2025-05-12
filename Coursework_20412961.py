import os
import sys
import time
import types
import warnings
import subprocess

# Ignore the AMP Expiration Warning
warnings.filterwarnings(
    "ignore",
    message=".*torch\\.cuda\\.amp\\.autocast.*deprecated.*",
    category=FutureWarning,
)

# Hijack "torch._classes" to Avoid "Streamlit watcher" Errors
fake = types.ModuleType("torch._classes")
fake.classes = lambda *args, **kwargs: None
sys.modules["torch._classes"] = fake
sys.modules["torch.classes"]    = fake

import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
from deep_sort_realtime.deepsort_tracker import DeepSort

# Allow Repeated Loading of "OpenMP"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Cache the model loading to avoid repeated loading
@st.cache_resource
# Load the YOLOv5 Model
def load_model():
    """Load the YOLOv5 Model"""
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
    return model

# Convert MP4V to H.264 using "ffmpeg
def convert_to_h264(input_path, output_path_h264):
    """Convert MP4V to H.264 using "ffmpeg"""
    try: 
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-loglevel", "error",
            output_path_h264
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        st.error("Conversion failed, please check whether ffmpeg is available or the input file is damaged.")
        raise

# Process the Video
def process_video(input_path, output_path, confidence, iou_thresh, max_age, viz_mode):
    """Process the Video"""
    # Check the Device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running...")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    # Load the Model and Set the Threshold
    model = load_model().to(device).eval()
    model.conf = confidence  
    model.iou  = iou_thresh

    # Open the Video
    cap = cv2.VideoCapture(input_path)
    # Check Whether the Video can be Opened
    if not cap.isOpened():
        st.error("Unable to open the video file, please check the format and path of the input video.")
        return None, None 
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initializing DeepSORT
    tracker = DeepSort(max_age=max_age)
    trajectories = {}  
    colors = {}

    frame_count  = 0
    det_counts = []
    track_counts = []

    # Frame by Frame Processing
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img)

        # The Person Detection Box
        dets = []
        for *box, conf, cls in results.xyxy[0]:
            if int(cls) == 0:
                x1,y1,x2,y2 = map(int, box)
                w, h = x2-x1, y2-y1
                dets.append(([x1,y1,w,h], conf.item(), 'person'))

        # Update Detection Count and Frame Count
        frame_count += 1
        det_counts.append(len(dets))

        tracks = tracker.update_tracks(dets, frame=frame)

        # Update the Number of Active Tracks
        active = sum(1 for t in tracks if t.is_confirmed())
        track_counts.append(active)

        # Visualization
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            x1,y1,x2,y2 = map(int, track.to_ltrb())
            cX, cY = (x1+x2)//2, (y1+y2)//2

            # Save center point for trail
            trajectories.setdefault(tid, []).append((cX,cY))

            # Color Consistency
            if tid not in colors:
                seed = abs(hash(tid)) % (2**32)
                colors[tid] = tuple(int(v) for v in np.random.RandomState(seed).randint(0,255,3))
            color = colors[tid]

            # Frame
            # Bounding Box Mode
            if viz_mode == "Bounding Box":
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            else:
                # Outline Mode
                h, w = frame.shape[:2]
                x1c, x2c = np.clip([x1, x2], 0, w)
                y1c, y2c = np.clip([y1, y2], 0, h)
                if x2c > x1c and y2c > y1c:
                    roi = frame[y1c:y2c, x1c:x2c]
                    # Check ROI Firstly
                    if roi.size > 0 and roi.shape[0] > 1 and roi.shape[1] > 1:
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        edges = cv2.Canny(gray, 50, 150)
                        if edges is not None:
                            mask = edges > 0
                            roi[mask] = color
                        else:
                            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                    else:
                        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                else:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

            # ID
            cv2.putText(frame, f"ID {tid}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw the Gradient Trail for each person
            pts = trajectories[tid]
            n = len(pts)
            for i in range(1, n):
                alpha = i/(n-1) if n>1 else 1.0
                faded = tuple(int(color[j]*alpha) for j in range(3))
                cv2.line(frame, pts[i-1], pts[i], faded, 2)

        writer.write(frame)

    cap.release()
    writer.release()

    # Free all GPU Cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Calculate all metrics
    track_metrics = {}
    for tid, pts in trajectories.items():
        if len(pts)<2:
            track_metrics[tid] = (0.0,0.0)
        else:
            dists = [float(np.linalg.norm(np.array(pts[i]) - np.array(pts[i-1])))
                     for i in range(1,len(pts))]
            total = sum(dists)
            duration = len(pts)/fps
            track_metrics[tid] = (total, total/duration if duration>0 else 0.0)

    if det_counts:
        avg_det = round(np.mean(det_counts), 2)
        std_det = round(np.std(det_counts),  2)
    else:
        avg_det = std_det = 0.0

    if track_counts:
        avg_trk = round(np.mean(track_counts), 2)
        std_trk = round(np.std(track_counts),  2)
    else:
        avg_trk = std_trk = 0.0

    lengths = [len(pts) for pts in trajectories.values()]
    if lengths:
        avg_len = round(np.mean(lengths), 2)
        min_len = min(lengths)
        max_len = max(lengths)
    else:
        avg_len = min_len = max_len = 0

    eval_metrics = {
        # Detection level
        "Processed Frames":            frame_count,
        "Avg Detections / Frame":      avg_det,
        "Std  Detections / Frame":     std_det,
        # Tracking level
        "Avg Active Tracks / Frame":   avg_trk,
        "Std  Active Tracks / Frame":  std_trk,
        "Unique Track IDs":            len(trajectories),
        # Trajectory length
        "Avg Track Length (frames)":   avg_len,
        "Min Track Length (frames)":   min_len,
        "Max Track Length (frames)":   max_len,
    }

    return track_metrics, eval_metrics

def main():
    st.title("📹 Person Tracking Project")
    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

    st.markdown("###### Name: Bin Zhang")
    st.markdown("###### Student ID: 20412961")

    st.markdown("#### Upload Video File (MP4 Recommended)", unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Video File (MP4 Recommended)", type=['mp4','avi','mov','mkv'], label_visibility="collapsed")
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    st.markdown("##### YOLO Confidence Threshold")
    confidence = st.slider("YOLO Confidence Threshold", 0.0, 1.0, 0.55, step=0.05, label_visibility="collapsed")

    st.markdown("##### YOLO NMS IoU Threshold")
    iou_thresh = st.slider("YOLO NMS IOU Threshold", 0.0, 1.0, 0.40, step=0.05, label_visibility="collapsed")

    st.markdown("##### DeepSORT Maximum Lost Frames (max_age)")
    max_age = st.slider("DeepSORT Maximum Lost Frames (max_age)", 1, 100, 45, label_visibility="collapsed")

    st.markdown("##### Visualization Mode")
    viz_mode  = st.selectbox("Visualization Mode", ["Bounding Box","Outline"], label_visibility="collapsed")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    start_btn = st.button("Start to Process")


    if uploaded and start_btn:
        # Create "input" and "output" folders
        base_dir   = os.path.dirname(__file__)
        input_dir  = os.path.join(base_dir, "Input")
        output_dir = os.path.join(base_dir, "Output")
        h264_dir   = os.path.join(base_dir, "Output_H264")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(h264_dir, exist_ok=True)

        # Save "input" and "output" files
        video_name = uploaded.name
        in_path   = os.path.join(input_dir, uploaded.name)
        raw_out   = os.path.join(output_dir, f"output_for_{uploaded.name.split('.')[0]}.mp4")
        h264_out  = os.path.join(h264_dir, f"output_h264_for_{uploaded.name.split('.')[0]}.mp4")

        with open(in_path, "wb") as f:
            f.write(uploaded.getbuffer())

        # Automatically Display the Video Information
        cap_info = cv2.VideoCapture(in_path)
        fps    = cap_info.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_info.release()
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown(f"##### **Input Video:** `{video_name}` -- **Resolution:** {width}x{height}, **FPS:** {fps:.2f}")

        # Processing and Timing
        st.info("Processing, Please Wait...")
        start_time = time.time()
        track_metrics, eval_metrics = process_video(in_path, raw_out, confidence, iou_thresh, max_age, viz_mode)
        # Return Directly if can not Open the Video
        if track_metrics is None:
            return  
        proc_time = time.time() - start_time
        st.markdown(f"<h4 style='color:#FF4B4B;'>🔄 Processing Time: {proc_time:.1f} s</h4>", unsafe_allow_html=True)

        # Converting and Timing
        st.info("Converting to H.264 for Playback...")
        t0 = time.time()
        try:
            convert_to_h264(raw_out, h264_out)
        except subprocess.CalledProcessError:
            return 
        convert_time = time.time() - t0
        st.markdown(f"<h4 style='color:#4B8BFF;'>🎞️ H.264 Conversion Time: {convert_time:.1f} s</h4>", unsafe_allow_html=True)

        # The Download Botton
        st.success("✅ Done!")
        st.video(h264_out)
        with open(h264_out, "rb") as f:
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.download_button("📥 Download H.264 Output", f, file_name=os.path.basename(h264_out), mime="video/mp4")

        # Using DataFrame to Show all the Metrics
        # Statistical Metrics
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.subheader("🚦 Statistical Metrics")
        sta_df = pd.DataFrame.from_dict(
            track_metrics, orient="index",
            columns=["Cumulative Distance (px)", "Average Speed (px/s)"]
        )
        sta_df.index.name = "ID"
        st.dataframe(sta_df, use_container_width=True)

        # Extract Detection and Tracking Keys Separately
        det_keys = ["Processed Frames",
                    "Avg Detections / Frame",
                    "Std  Detections / Frame"]
        trk_keys = ["Avg Active Tracks / Frame",
                    "Std  Active Tracks / Frame",
                    "Unique Track IDs",
                    "Avg Track Length (frames)",
                    "Min Track Length (frames)",
                    "Max Track Length (frames)"]

        # Detection Metrics
        st.subheader("🎯 Detection Metrics")
        det_df = pd.DataFrame.from_dict(
            {k: eval_metrics[k] for k in det_keys},
            orient="index", columns=["Value"]
        )
        det_df.index.name = "Metric"
        st.dataframe(det_df, use_container_width=True)

        # Tracking Metrics 
        st.subheader("👣 Tracking Metrics")
        trk_df = pd.DataFrame.from_dict(
            {k: eval_metrics[k] for k in trk_keys},
            orient="index", columns=["Value"]
        )
        trk_df.index.name = "Metric"
        st.dataframe(trk_df, use_container_width=True)

# Run the Whole Program
if __name__ == "__main__":
    main()