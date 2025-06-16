#!/usr/bin/env python3

import sys
import argparse
import os
import time
import socket
import json
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log

# Demo metadata
DEMO_NAME = "detectnet"
DEMO_VERSION = "1.0"

# Path to the IoTConnect Unix socket
SOCKET_PATH = (
    "/var/snap/iotconnect/common/iotc.sock"
    if os.path.exists("/var/snap/iotconnect/common/iotc.sock")
    else os.path.expanduser("~/snap/iotconnect/common/iotc.sock")
)
# Minimum interval between telemetry sends (in seconds)
TELEMETRY_INTERVAL = 7.0

# Name of the model being used
MODEL_NAME = None
# Last time telemetry was sent
last_send_time = 0


def send_telemetry(payload):
    """
    Send a JSON payload over the IoTConnect Unix socket.
    """
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(1.0)
        s.connect(SOCKET_PATH)
        s.send((json.dumps(payload) + "\n").encode("utf-8"))
        s.close()
        print(f"[TELEMETRY] Sent: {payload}")
    except Exception as e:
        print(f"[SOCKET] Send failed: {e}")


def load_model_from_config(argv):
    """
    Read current-model.txt to determine which UFF model to load,
    set MODEL_NAME, and append the necessary flags to argv.
    """
    global MODEL_NAME
    base = "/var/snap/iotconnect/common/models/detectnet"
    model_name = "ssd_mobilenet_v2_coco.uff"
    try:
        with open(os.path.join(base, "current-model.txt"), 'r') as f:
            model_name = f.read().strip()
    except Exception as e:
        print(f"[MODEL] Error reading {base}/current-model.txt: {e}")

    MODEL_NAME = model_name
    model_path = os.path.join(base, MODEL_NAME)
    labels_path = os.path.join(base, "ssd_coco_labels.txt")

    # Specify UFF and correct layer names
    argv += [
        f"--model={model_path}",
        f"--labels={labels_path}",
        "--uff",
        "--input-blob=Input",
        "--output-cvg=NMS",
        "--output-bbox=NMS_1"
    ]


if __name__ == '__main__':
    # Build argument parser
    parser = argparse.ArgumentParser(
        description="Object detection with IoTConnect OTA support",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage()
    )
    parser.add_argument(
        "input", type=str, nargs='?', default="/dev/video0",
        help="URI of the input stream (e.g., /dev/video0)"
    )
    parser.add_argument(
        "output", type=str, nargs='?', default="display://",
        help="URI of the output stream (e.g., display://)"
    )
    parser.add_argument(
        "--network", type=str, default=None,
        help="Override OTA: name of built-in network to use"
    )
    args = parser.parse_known_args()[0]

    # Open I/O streams
    video_input = videoSource(args.input, argv=sys.argv)
    video_output = videoOutput(args.output, argv=sys.argv)
    font = cudaFont()

    # Load detection network
    if args.network:
        MODEL_NAME = args.network
        net = detectNet(args.network, sys.argv)
    else:
        load_model_from_config(sys.argv)
        net = detectNet("custom", sys.argv)

    # Main loop
    while True:
        img = video_input.Capture()
        if img is None:
            continue

        detections = net.Detect(img)
        print(f"[INFER] Detected {len(detections)} objects")
        for det in detections:
            print(f"  - {det.ClassID} ({net.GetClassDesc(det.ClassID)}) {det.Confidence*100:.2f}% at {det.Left},{det.Top},{det.Width},{det.Height}")

        video_output.Render(img)
        video_output.SetStatus(f"detectNet | Network {net.GetNetworkFPS():.0f} FPS")
        net.PrintProfilerTimes()

        # Telemetry
        current_time = time.time()
        if (current_time - last_send_time) >= TELEMETRY_INTERVAL:
            for det in detections:
                telemetry = {
                    "demo_name": DEMO_NAME,
                    "demo_version": DEMO_VERSION,
                    "model_name": MODEL_NAME,
                    "timestamp": int(current_time),
                    "class_id": det.ClassID,
                    "class_description": net.GetClassDesc(det.ClassID),
                    "confidence": round(det.Confidence, 5),
                    "bbox": [round(det.Left, 1), round(det.Top, 1), round(det.Width, 1), round(det.Height, 1)]
                }
                send_telemetry(telemetry)
            last_send_time = current_time

        if not video_input.IsStreaming() or not video_output.IsStreaming():
            break

