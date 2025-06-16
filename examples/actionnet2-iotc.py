#!/usr/bin/env python3

import sys
import argparse
import socket
import json
import time
import os

from jetson_inference import actionNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log

# --- Configurable OTA-compatible model path ---
MODEL_DIR = "/var/snap/iotconnect/common/models"
MODEL_LINK = os.path.join(MODEL_DIR, "current-model.txt")
DEFAULT_MODEL = "resnet-18"

# --- Socket Path ---
SOCKET_PATH = (
    "/var/snap/iotconnect/common/iotc.sock"
    if os.path.exists("/var/snap/iotconnect/common/iotc.sock")
    else os.path.expanduser("~/snap/iotconnect/common/iotc.sock")
)
TELEMETRY_INTERVAL = 7.0  # Send every 7 seconds

# --- Load model name from OTA-updated file ---
def load_model_from_config():
    MODEL_DIR = "/var/snap/iotconnect/common/models"
    model_name = "resnet-18-kinetics-moments.onnx"
    try:
        with open(os.path.join(MODEL_DIR, "current-model.txt")) as f:
            model_name = f.read().strip()
    except Exception as e:
        print(f"[MODEL] Error reading current-model.txt: {e}")

    model_path = os.path.join(MODEL_DIR, model_name)
    labels_path = os.path.join(MODEL_DIR, "labels.txt")

    print(f"[DEBUG] model path: {model_path}")
    print(f"[DEBUG] labels path: {labels_path}")
    print(f"[DEBUG] File exists? {os.path.isfile(model_path)}")

    # Inject model and labels into sys.argv for Jetson-inference backend
    sys.argv += ["--model=" + model_path, "--labels=" + labels_path]

    return actionNet("custom", sys.argv)



# --- Send telemetry via UNIX socket ---
def send_telemetry(class_id, class_desc, confidence):
    telemetry = {
        "timestamp": int(time.time()),
        "class_id": class_id,
        "class_description": class_desc,
        "confidence": round(confidence, 5)
    }
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(1.0)
        s.connect(SOCKET_PATH)
        s.send((json.dumps(telemetry) + "\n").encode("utf-8"))
        s.close()
        print(f"[TELEMETRY] Sent: {telemetry}")
    except Exception as e:
        print(f"[SOCKET] Send failed: {e}")

# --- Argument parsing (matches original script style) ---
parser = argparse.ArgumentParser(description="Classify the action/activity of an image sequence.",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog=actionNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default=None, help="Override model name manually (optional)")

try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# --- Load model (from OTA-configured file unless overridden) ---
net = actionNet(args.network, sys.argv) if args.network else load_model_from_config()

# --- Create video input/output ---
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)
font = cudaFont()

# --- Main processing loop ---
while True:
    img = input.Capture()
    if img is None:
        continue

    # Classify the image
    class_id, confidence = net.Classify(img)
    class_desc = net.GetClassDesc(class_id)

    print(f"[INFER] {confidence * 100:.2f}% class #{class_id} ({class_desc})")

    # Overlay result
    font.OverlayText(img, img.width, img.height,
                     "{:05.2f}% {:s}".format(confidence * 100, class_desc),
                     5, 5, font.White, font.Gray40)

    output.Render(img)
    output.SetStatus("actionNet {:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))
    net.PrintProfilerTimes()

    # Telemetry
    current_time = time.time()
    if (current_time - last_send_time) >= TELEMETRY_INTERVAL:
        send_telemetry(class_id, class_desc, confidence)
        last_send_time = current_time

    if not input.IsStreaming() or not output.IsStreaming():
        break

