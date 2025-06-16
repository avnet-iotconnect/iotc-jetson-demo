#!/usr/bin/env python3

import sys
import argparse
import os
import time
import numpy as np
import socket
import json

from jetson_inference import depthNet
from jetson_utils import videoSource, videoOutput, cudaOverlay, cudaDeviceSynchronize, cudaToNumpy, Log

from depthnet_utils import depthBuffers

# Demo metadata
DEMO_NAME = "depthnet"
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
    Read current-model.txt to determine which model to load,
    set MODEL_NAME, and append the --model and related flags to argv.
    """
    global MODEL_NAME
    base = "/var/snap/iotconnect/common/models/depthnet"
    model_name = "fcn-mobilenet.onnx"
    try:
        with open(os.path.join(base, "current-model.txt"), 'r') as f:
            model_name = f.read().strip()
    except Exception as e:
        print(f"[MODEL] Error reading {base}/current-model.txt: {e}")

    MODEL_NAME = model_name
    model_path = os.path.join(base, MODEL_NAME)
    labels_path = os.path.join(base, "labels.txt")

    argv += [
        f"--model={model_path}",
        f"--labels={labels_path}"
    ]


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Mono depth estimation on a video/image stream using depthNet DNN.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=depthNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage()
    )
    parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
    parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
    parser.add_argument("--network", type=str, default=None,
                        help="Override OTA: pre-trained model to load (fcn-mobilenet, etc.)")
    parser.add_argument("--visualize", type=str, default="input,depth",
                        help="visualization options: input, depth, or input,depth")
    parser.add_argument("--depth-size", type=float, default=1.0,
                        help="scales depth map visualization as percentage of input size")
    parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"],
                        help="filtering mode for visualization: point or linear")
    parser.add_argument("--colormap", type=str, default="viridis-inverted",
                        choices=["inferno", "inferno-inverted", "magma", "magma-inverted", 
                                 "parula", "parula-inverted", "plasma", "plasma-inverted", 
                                 "turbo", "turbo-inverted", "viridis", "viridis-inverted"],
                        help="colormap for visualization")
    args = parser.parse_known_args()[0]

    # Load model (OTA or override)
    if args.network:
        MODEL_NAME = args.network
        net = depthNet(args.network, sys.argv)
    else:
        load_model_from_config(sys.argv)
        net = depthNet("custom", sys.argv)

    buffers = depthBuffers(args)
    input = videoSource(args.input, argv=sys.argv)
    output = videoOutput(args.output, argv=sys.argv)

    # Main loop
    while True:
        img_input = input.Capture()
        if img_input is None:
            continue

        buffers.Alloc(img_input.shape, img_input.format)
        net.Process(img_input, buffers.depth, args.colormap, args.filter_mode)

        if buffers.use_input:
            cudaOverlay(img_input, buffers.composite, 0, 0)
        if buffers.use_depth:
            x = img_input.width if buffers.use_input else 0
            cudaOverlay(buffers.depth, buffers.composite, x, 0)

        output.Render(buffers.composite)
        output.SetStatus(f"{MODEL_NAME} | depthNet {net.GetNetworkName()} | {net.GetNetworkFPS():.0f} FPS")

        cudaDeviceSynchronize()
        #net.PrintProfilerTimes()

        # Telemetry
        current_time = time.time()
        if (current_time - last_send_time) >= TELEMETRY_INTERVAL:
            depth_np = cudaToNumpy(buffers.depth)
            if depth_np.size > 0:
                telemetry = {
                    "timestamp": int(current_time),
                    "demo_name": DEMO_NAME,
                    "demo_version": DEMO_VERSION,
                    "model_name": MODEL_NAME,
                    "average_depth_m": round(float(np.mean(depth_np)), 3),
                    "min_depth_m": round(float(np.min(depth_np)), 3),
                    "max_depth_m": round(float(np.max(depth_np)), 3)
                }
                send_telemetry(telemetry)
            last_send_time = current_time

        if not input.IsStreaming() or not output.IsStreaming():
            break

