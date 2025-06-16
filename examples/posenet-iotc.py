#!/usr/bin/env python3

import sys
import argparse
import os
import time
import json
import socket
from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log

# Demo metadata
DEMO_NAME = "posenet"
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
    Read current-model.txt to determine which ONNX model to load,
    set MODEL_NAME, and append necessary flags to argv.
    """
    global MODEL_NAME
    base = "/var/snap/iotconnect/common/models/posenet"
    model_name = "posenet.onnx"
    try:
        with open(os.path.join(base, "current-model.txt"), 'r') as f:
            model_name = f.read().strip()
    except Exception as e:
        print(f"[MODEL] Error reading {base}/current-model.txt: {e}")

    MODEL_NAME = model_name
    model_path = os.path.join(base, MODEL_NAME)
    labels_path = os.path.join(base, "labels.txt")
    topo_path = os.path.join(base, "human_pose.json")
    colors_path = os.path.join(base, "colors.txt")

    argv += [
        f"--model={model_path}",
        f"--labels={labels_path}",
        f"--topology={topo_path}",
        f"--colormap={colors_path}"
    ]


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Pose estimation with IoTConnect OTA support",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=poseNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage()
    )
    parser.add_argument("input", type=str, nargs='?', default="/dev/video0",
                        help="URI of the input stream (e.g., /dev/video0)")
    parser.add_argument("output", type=str, nargs='?', default="display://",
                        help="URI of the output stream (e.g., display://)")
    parser.add_argument("--network", type=str, default=None,
                        help="Override OTA: name of built-in network to use")
    args = parser.parse_known_args()[0]

    # Open I/O streams
    input = videoSource(args.input, argv=sys.argv)
    output = videoOutput(args.output, argv=sys.argv)
    font = cudaFont()

    # Load model
    if args.network:
        MODEL_NAME = args.network
        net = poseNet(args.network, sys.argv)
    else:
        load_model_from_config(sys.argv)
        net = poseNet("custom", sys.argv)

    # Main loop
    while True:
        img = input.Capture()
        if img is None:
            continue

        poses = net.Process(img)
        output.Render(img)
        output.SetStatus(f"poseNet | Network {net.GetNetworkFPS():.0f} FPS")
        net.PrintProfilerTimes()

        # Telemetry
        current_time = time.time()
        if (current_time - last_send_time) >= TELEMETRY_INTERVAL:
            for pose in poses:
                keypoints = {net.GetKeypointName(p.ID): [round(p.x, 1), round(p.y, 1)]
                             for p in pose.Keypoints}
                telemetry = {
                    "demo_name": DEMO_NAME,
                    "demo_version": DEMO_VERSION,
                    "model_name": MODEL_NAME,
                    "timestamp": int(current_time),
                    "keypoints": keypoints
                }
                send_telemetry(telemetry)
            last_send_time = current_time

        if not input.IsStreaming() or not output.IsStreaming():
            break

