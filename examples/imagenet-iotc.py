#!/usr/bin/env python3

import sys
import argparse
import os
import time
import socket
import json
from jetson_inference import imageNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log

# Demo metadata
DEMO_NAME = "imageNet"
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
    Read the Caffe model filename from imagenet/current-model.txt,
    set MODEL_NAME, then append the --model, --prototxt, and --labels flags to argv.
    """
    global MODEL_NAME
    base = "/var/snap/iotconnect/common/models/imagenet"
    model_name = "bvlc_googlenet.caffemodel"
    try:
        with open(os.path.join(base, "current-model.txt"), 'r') as f:
            model_name = f.read().strip()
    except Exception as e:
        print(f"[MODEL] Error reading {base}/current-model.txt: {e}")

    MODEL_NAME = model_name
    model_path = os.path.join(base, model_name)

    # Determine matching prototxt
    proto_candidates = [model_name.replace('.caffemodel', '.prototxt'),
                        'googlenet.prototxt', 'googlenet_noprob.prototxt']
    proto_path = None
    for proto in proto_candidates:
        candidate = os.path.join(base, proto)
        if os.path.isfile(candidate):
            proto_path = candidate
            break
    if not proto_path:
        raise FileNotFoundError(f"No .prototxt found for {model_name} in {base}")

    labels_path = "/var/snap/iotconnect/common/models/labels.txt"

    argv += [
        f"--model={model_path}",
        f"--prototxt={proto_path}",
        f"--labels={labels_path}"
    ]

if __name__ == '__main__':
    # Build argument parser
    parser = argparse.ArgumentParser(
        description="Image classification with IoTConnect OTA support",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=imageNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage()
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

    # Load the DNN model
    if args.network:
        MODEL_NAME = args.network
        net = imageNet(args.network, sys.argv)
    else:
        load_model_from_config(sys.argv)
        net = imageNet("custom", sys.argv)

    # Open I/O streams
    input = videoSource(args.input, argv=sys.argv)
    output = videoOutput(args.output, argv=sys.argv)
    font = cudaFont()

    # Main processing loop
    while True:
        img = input.Capture()
        if img is None:
            continue

        # Perform inference
        class_id, confidence = net.Classify(img)
        class_desc = net.GetClassDesc(class_id)
        print(f"[INFER] {confidence * 100:.2f}% class #{class_id} ({class_desc})")

        # Overlay result on image
        font.OverlayText(
            img, img.width, img.height,
            f"{confidence * 100:.2f}% {class_desc}", 5, 5,
            font.White, font.Gray40
        )

        # Render and status
        output.Render(img)
        output.SetStatus(f"imageNet | Network {net.GetNetworkFPS():.0f} FPS")
        net.PrintProfilerTimes()

        # Send telemetry at intervals
        current_time = time.time()
        if (current_time - last_send_time) >= TELEMETRY_INTERVAL:
            telemetry = {
                "timestamp": int(current_time),
                "demo_name": DEMO_NAME,
                "demo_version": DEMO_VERSION,
                "model_name": MODEL_NAME,
                "class_id": class_id,
                "class_description": class_desc,
                "confidence": round(confidence, 5)
            }
            send_telemetry(telemetry)
            last_send_time = current_time

        # Exit when streams close
        if not input.IsStreaming() or not output.IsStreaming():
            break
