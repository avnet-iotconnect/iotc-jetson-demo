#!/usr/bin/env python3
import os
import json
import time
import socket
import signal
import subprocess
import threading
import psutil
import re

from jtop import jtop

SOCKET_PATH = "/var/snap/iotconnect/common/iotc.sock"
CMD_SOCKET_PATH = "/var/snap/iotconnect/common/iotc_cmd.sock"
TELEMETRY_INTERVAL = 7
DEMO_PROCESS = None
CURRENT_SCRIPT = "notme"
SUFFIX = "-iotc.py"

def set_socket_permissions(path):
    try:
        os.chmod(path, 0o666)
    except Exception as e:
        print(f"[SOCKET] Could not set permissions on {path}: {e}")

def send_telemetry(data):
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(SOCKET_PATH)
            sock.sendall(json.dumps(data).encode("utf-8"))
        print(f"[TELEMETRY] Sent: {data}")
    except Exception as e:
        print(f"[TELEMETRY] Failed to send: {e}")

def get_gpu_stats_fallback():
    try:
        output = subprocess.check_output(["tegrastats", "--interval", "1000", "--count", "1"], stderr=subprocess.DEVNULL)
        output = output.decode()
        gpu_match = re.search(r"GR3D_FREQ (\d+)%", output)
        emc_match = re.search(r"EMC_FREQ (\d+)%", output)
        return {
            "gpu": int(gpu_match.group(1)) if gpu_match else -1,
            "emc_freq": int(emc_match.group(1)) if emc_match else -1,
        }
    except Exception as e:
        print(f"[TEGRAS] Fallback failed: {e}")
        return {"gpu": -1, "emc_freq": -1}

def get_system_stats():
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    stats = {"cpu": cpu, "mem": mem, "gpu": -1, "gpu_freq": -1, "emc_freq": -1}

    try:
        with jtop() as jetson:
            jetson_stats = jetson.stats
            stats["gpu"] = jetson_stats.get("GPU", -1)
            stats["gpu_freq"] = jetson_stats.get("GR3D_FREQ", -1)
            stats["emc_freq"] = jetson_stats.get("EMC_FREQ", -1)
    except Exception as e:
        print(f"[JTOP] Failed: {e}, using fallback")
        fallback = get_gpu_stats_fallback()
        stats["gpu"] = fallback.get("gpu", -1)
        stats["emc_freq"] = fallback.get("emc_freq", -1)

    return stats

def telemetry_loop():
    while True:
        stats = get_system_stats()
        if CURRENT_SCRIPT.endswith(SUFFIX):
            active_script = CURRENT_SCRIPT[:-len(SUFFIX)]
        else:
            active_script = CURRENT_SCRIPT

        stats.update({
            "timestamp": int(time.time()),
            "launcher": "iotc-launcher",
            "active_script": active_script
        })

        send_telemetry(stats)
        time.sleep(TELEMETRY_INTERVAL)


def stop_current_script():
    global DEMO_PROCESS, CURRENT_SCRIPT
    if DEMO_PROCESS and DEMO_PROCESS.poll() is None:
        print(f"[PROCESS] Terminating {CURRENT_SCRIPT}...")
        DEMO_PROCESS.terminate()
        try:
            DEMO_PROCESS.wait(timeout=3)
        except subprocess.TimeoutExpired:
            DEMO_PROCESS.kill()
    DEMO_PROCESS = None
    CURRENT_SCRIPT = "notme"
    print("[PROCESS] Demo stopped.")

def launch_script(script_name):
    global DEMO_PROCESS, CURRENT_SCRIPT
    stop_current_script()
    full_path = os.path.join(os.getcwd(), script_name)
    if not os.path.exists(full_path):
        print(f"[LAUNCH] Script not found: {full_path}")
        return
    DEMO_PROCESS = subprocess.Popen(["python3", full_path, "/dev/video0"])
    CURRENT_SCRIPT = script_name
    print(f"[LAUNCH] Started {script_name} on /dev/video0")

def connect_command_socket():
    while not os.path.exists(CMD_SOCKET_PATH):
        time.sleep(0.2)
    while True:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(CMD_SOCKET_PATH)
            return sock
        except:
            time.sleep(1)

def command_loop():
    sock = connect_command_socket()
    buffer = b""
    while True:
        try:
            data = sock.recv(4096)
            if not data:
                sock.close()
                sock = connect_command_socket()
                continue
            buffer += data
            try:
                cmd_json = json.loads(buffer.decode("utf-8"))
                buffer = b""
            except json.JSONDecodeError:
                continue
            cmd = cmd_json.get("cmd") or cmd_json.get("name", "")
            args = cmd_json.get("args", [])
            print(f"[COMMAND] {cmd} {args}")
            if cmd == "launch" and args:
                script = args[0]
                if script.endswith("-iotc.py") and os.path.exists(script):
                    launch_script(script)
                else:
                    print(f"[COMMAND] Not a valid demo: {script}")
            elif cmd == "set_frequency" and args:
                global TELEMETRY_INTERVAL
                try:
                    TELEMETRY_INTERVAL = int(args[0])
                    print(f"[COMMAND] Telemetry frequency set to {TELEMETRY_INTERVAL}s")
                except:
                    print("[COMMAND] Invalid frequency arg")
            elif cmd == "stop_demo":
                stop_current_script()
        except Exception as e:
            print(f"[COMMAND] Error: {e}")
            sock.close()
            sock = connect_command_socket()

if __name__ == "__main__":
    threading.Thread(target=telemetry_loop, daemon=True).start()
    threading.Thread(target=command_loop, daemon=True).start()
    print("[LAUNCHER] IoTC demo launcher running...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_current_script()
        print("\n[LAUNCHER] Exiting...")
