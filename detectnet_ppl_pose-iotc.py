#!/usr/bin/env python3
import sys
import argparse
import os
import time
import socket
import json
import threading
from jetson_inference import detectNet, poseNet
from jetson_utils import videoSource, videoOutput, cudaDrawRect, cudaFont

SOCKET_PATH = "/var/snap/iotconnect/common/iotc.sock"
CMD_SOCKET_PATH = "/var/snap/iotconnect/common/iotc_cmd.sock"
TELEMETRY_INTERVAL = 7.0
PERSON_CLASS_ID = 1

RETRY_INTERVAL = 0.5  # seconds
MAX_RETRIES = 5

WRIST_BOX = [200, 200, 400, 400]  # default [x,y,w,h]

# KPI Counters
people_counter = set()
interaction_counter = 0
minute_start_time = time.time()

def send_telemetry(payload):
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(SOCKET_PATH)
            sock.send((json.dumps(payload) + "\n").encode("utf-8"))
        print(f"[TELEMETRY] Sent: {payload}")
    except Exception as e:
        print(f"[SOCKET] Send failed: {e}")

def wrist_in_box(keypoints, box):
    x_min, y_min, w, h = box
    x_max, y_max = x_min + w, y_min + h
    for k, v in keypoints.items():
        if 'wrist' in k.lower():
            x, y = v
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return True
    return False

def connect_command_socket():
    while not os.path.exists(CMD_SOCKET_PATH):
        print("[CMD] Waiting for command socket...")
        time.sleep(0.5)
    while True:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(CMD_SOCKET_PATH)
            print("[CMD] Connected to command socket.")
            return sock
        except Exception as e:
            print(f"[CMD] Connection error: {e}, retrying in 1 sec...")
            time.sleep(1)

def command_listener():
    global WRIST_BOX
    sock = connect_command_socket()
    buffer = b""
    while True:
        try:
            data = sock.recv(4096)
            if not data:
                print("[CMD] Disconnected by server, reconnecting...")
                sock.close()
                sock = connect_command_socket()
                buffer = b""
                continue

            buffer += data
            try:
                cmd_json = json.loads(buffer.decode("utf-8"))
                buffer = b""
                cmd = cmd_json.get("cmd") or cmd_json.get("name", "")
                args = cmd_json.get("args", [])
                print(f"[CMD] Received: {cmd}, Args: {args}")
                if cmd.startswith("set_box"):
                    parts = cmd.split()
                    if len(parts) == 5:
                        WRIST_BOX = [int(x) for x in parts[1:]]
                    elif len(args) == 4:
                        WRIST_BOX = [int(x) for x in args]
                    print(f"[CMD] Box updated: {WRIST_BOX}")
            except json.JSONDecodeError:
                continue  # partial JSON, keep receiving

        except Exception as e:
            print(f"[CMD] Listener socket error: {e}, reconnecting...")
            sock.close()
            sock = connect_command_socket()
            buffer = b""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, default="/dev/video4", nargs="?")
    parser.add_argument("output", type=str, default="display://", nargs="?")
    args = parser.parse_known_args()[0]

    #video_input = videoSource(args.input, argv=["--input-flip=rotate-180", "--input-flip=horizontal"])
    video_input = videoSource(args.input)


    video_output = videoOutput(args.output)

    detect_net = detectNet("ssd-mobilenet-v2", threshold=0.5)
    pose_net = poseNet("resnet18-body")

    font = cudaFont()

    threading.Thread(target=command_listener, daemon=True).start()

    last_send_time = 0

    # KPI counters
    people_counter = set()
    interaction_counter = 0
    minute_start_time = time.time()

while True:
    retries = 0
    img = None

    # Attempt to capture with retries
    while retries < MAX_RETRIES and img is None:
        try:
            img = video_input.Capture()
        except Exception as e:
            print(f"[CAMERA ERROR] Capture failed: {e}. Retrying ({retries+1}/{MAX_RETRIES})")
            img = None
        if img is None:
            retries += 1
            time.sleep(RETRY_INTERVAL)

    # Check after retries
    if img is None:
        print("[CAMERA ERROR] Maximum retries reached, skipping frame.")
        continue

    detections = detect_net.Detect(img)
    current_occupancy = sum(1 for det in detections if det.ClassID == PERSON_CLASS_ID)

    # Check interactions explicitly
    poses = pose_net.Process(img)
    interaction_active = any(
        wrist_in_box({pose_net.GetKeypointName(p.ID): [p.x, p.y] for p in pose.Keypoints}, WRIST_BOX)
        for pose in poses
    )

    # Categorize occupancy clearly
    if current_occupancy <= 1:
        occupancy_level = "Low"
    elif current_occupancy <= 3:
        occupancy_level = "Medium"
    else:
        occupancy_level = "High"

    # Visualization explicitly clear
    x, y, w, h = WRIST_BOX
    cudaDrawRect(img, (x, y, x + w, y + h), (255, 0, 0, 150))

    font.OverlayText(img, img.width, img.height,
                     f"Occupancy: {current_occupancy} ({occupancy_level}), Interaction: {'Yes' if interaction_active else 'No'}",
                     10, 10, font.White, font.Gray40)

    video_output.Render(img)

    # Regular periodic telemetry (every TELEMETRY_INTERVAL seconds)
    current_time = time.time()
    if current_time - last_send_time >= TELEMETRY_INTERVAL:
        telemetry = {
            "timestamp": int(current_time),
            "current_occupancy": current_occupancy,
            "occupancy_level": occupancy_level,
            "interaction_active": "yes" if interaction_active else "no",
            "box_coordinates": WRIST_BOX
        }
        send_telemetry(telemetry)
        last_send_time = current_time

    if not video_input.IsStreaming() or not video_output.IsStreaming():
        break

