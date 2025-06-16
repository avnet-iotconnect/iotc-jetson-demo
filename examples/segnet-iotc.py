
#!/usr/bin/env python3

import jetson.inference
import jetson.utils
import argparse
import time
import socket
import json
import os
import numpy as np
import threading

parser = argparse.ArgumentParser(description="Run SegNet and send telemetry to IoTConnect.")
parser.add_argument("input", type=str, help="Camera input (e.g., /dev/video0)")
parser.add_argument("--network", type=str, default="fcn-resnet18-cityscapes-512x256", help="SegNet model")
parser.add_argument("--stats", action="store_true", help="Display profiling stats")
args = parser.parse_args()

telemetry_interval = 7.0
telemetry_enabled = True
watched_class = None

socket_path = f"/home/{os.environ.get('USER', 'mlamp')}/snap/iotconnect/common/iotc.sock"
if not os.path.exists(socket_path):
    socket_path = "/var/snap/iotconnect/common/iotc.sock"

net = jetson.inference.segNet(args.network)
input_stream = jetson.utils.videoSource(args.input)
output_stream = jetson.utils.videoOutput("display://0")

def compute_class_coverage(mask_img, num_classes):
    mask_array = jetson.utils.cudaToNumpy(mask_img)
    total_pixels = mask_array.size
    coverage = {}
    for i in range(num_classes):
        class_pixels = np.count_nonzero(mask_array == i)
        if class_pixels > 0:
            coverage[i] = round((class_pixels / total_pixels) * 100, 2)
    return coverage

def send_telemetry(coverage):
    telemetry = {
        "demo_name": "segnet",
        "demo_version": "1.0",
        "model_name": "fcn_resnet18.onnx",
        "timestamp": int(time.time()),
        "frequency": telemetry_interval,
        "class_coverage": coverage
    }
    try:
        print(f"[DEBUG] Sending telemetry: {telemetry}")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(socket_path)
        sock.sendall(json.dumps({ "d": [ { "d": telemetry } ] }).encode("utf-8"))
        sock.shutdown(socket.SHUT_WR)
        sock.close()
        print("[TEL] Sent telemetry cleanly.")
    except Exception as e:
        print(f"[TEL] Error sending telemetry: {e}")

def listen_for_commands():
    global telemetry_interval, telemetry_enabled, watched_class
    if os.path.exists(socket_path):
        try:
            cmd_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            cmd_sock.connect(socket_path)
            print("[CMD] Command listener connected to socket")
            while True:
                data = cmd_sock.recv(1024)
                if data:
                    try:
                        cmd = json.loads(data.decode())
                        if cmd.get("command") == "set_frequency":
                            telemetry_interval = float(cmd.get("value", 7.0))
                            print(f"[CMD] Telemetry frequency updated to {telemetry_interval}s")
                        elif cmd.get("command") == "pause_telemetry":
                            telemetry_enabled = False
                            print("[CMD] Telemetry paused.")
                        elif cmd.get("command") == "resume_telemetry":
                            telemetry_enabled = True
                            print("[CMD] Telemetry resumed.")
                        elif cmd.get("command") == "watch_class":
                            watched_class = cmd.get("value")
                            print(f"[CMD] Watching for class: {watched_class}")
                    except Exception as e:
                        print(f"[CMD] Failed to process command: {e}")
        except Exception as e:
            print(f"[CMD] Command listener error: {e}")

threading.Thread(target=listen_for_commands, daemon=True).start()

last_telemetry = time.time()
mask_output = None

print("[INFO] Running SegNet demo. Press Ctrl+C to exit.")

while output_stream.IsStreaming():
    img = input_stream.Capture()
    if img is None:
        continue

    if mask_output is None:
        mask_output = jetson.utils.cudaAllocMapped(width=img.width, height=img.height, format="gray8")

    net.Process(img)
    net.Overlay(img)
    output_stream.Render(img)

    if args.stats:
        net.PrintProfilerTimes()

    now = time.time()
    if telemetry_enabled and (now - last_telemetry >= telemetry_interval):
        last_telemetry = now
        net.Mask(mask_output)
        coverage_raw = compute_class_coverage(mask_output, net.GetNumClasses())
        sorted_coverage = sorted(coverage_raw.items(), key=lambda x: x[1], reverse=True)
        top_coverage = {net.GetClassDesc(i): v for i, v in sorted_coverage[:5]}
        if watched_class and watched_class in top_coverage:
            print(f"[ALERT] Watched class '{watched_class}' detected with {top_coverage[watched_class]}% coverage.")
        send_telemetry(top_coverage)

print("[INFO] Exited SegNet demo.")
