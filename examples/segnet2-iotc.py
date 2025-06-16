#!/usr/bin/env python3

import sys
import argparse
import os
import time
import socket
import json
import threading
import stat
import numpy as np
from jetson_inference import segNet
from jetson_utils import videoSource, videoOutput, cudaOverlay, cudaDeviceSynchronize, Log

from segnet_utils import segmentationBuffers

DEMO_NAME = "segnet"
DEMO_VERSION = "1.1"

BASE_ROOT = "/var/snap/iotconnect/common"
SOCKET_PATH = os.path.join(BASE_ROOT, "iotc.sock")
CMD_SOCKET_PATH = os.path.join(BASE_ROOT, "iotc_cmd.sock")

TELEMETRY_INTERVAL_DEFAULT = 7
TELEMETRY_INTERVAL = TELEMETRY_INTERVAL_DEFAULT
MODEL_NAME = None

last_send_time = None


def send_telemetry(data):
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(SOCKET_PATH)
            sock.sendall((json.dumps(data)).encode("utf-8"))
        print(f"[TELEMETRY] Sent: {data}")
    except Exception as e:
        print(f"[TELEMETRY] Send failed: {e}")


def telemetry_loop():
    global last_send_time, TELEMETRY_INTERVAL
    last_send_time = time.time()
    while True:
        now = time.time()
        delta = now - last_send_time
        if delta >= TELEMETRY_INTERVAL:
            # Calculate histogram (pixel counts per class)
            unique_classes, class_counts = np.unique(mask_np, return_counts=True)
            total_pixels = mask_np.size

            # Calculate coverage percentages per class
            coverage_percentages = {}
            num_classes = net.GetNumClasses()

            for c, count in zip(unique_classes, class_counts):
                if 0 <= c < num_classes:
                    label = net.GetClassLabel(c)
                else:
                    label = "unknown"

                coverage_percentages[label] = (count / total_pixels) * 100

            # Determine dominant class safely
            valid_indices = unique_classes[unique_classes < num_classes]
            if valid_indices.size > 0:
                dominant_class_idx = valid_indices[class_counts[unique_classes < num_classes].argmax()]
                dominant_class_label = net.GetClassLabel(dominant_class_idx)
                dominant_class_coverage = coverage_percentages[dominant_class_label]
            else:
                dominant_class_label = "unknown"
                dominant_class_coverage = 0.0

            payload = {
                "timestamp": int(time.time()),
                "demo_name": DEMO_NAME,
                "demo_version": DEMO_VERSION,
                "model_name": MODEL_NAME,
                "frequency": TELEMETRY_INTERVAL,
                "segnet_dominant_class": dominant_class_label,
                "segnet_dominant_class_coverage": dominant_class_coverage,
                "segnet_coverage_percentages": coverage_percentages
            }

            send_telemetry(payload)
            last_send_time = now
        time.sleep(0.1)

def load_model_from_config():
    global MODEL_NAME
    base = "/var/snap/iotconnect/common/models/segnet"
    model_name = "fcn_resnet18.onnx"
    try:
        with open(os.path.join(base, "current-model.txt"), 'r') as f:
            model_name = f.read().strip()
    except Exception as e:
        print(f"[MODEL] Error reading {base}/current-model.txt: {e}")
    MODEL_NAME = model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Semantic segmentation with IoTConnect telemetry & command sockets",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=segNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage()
    )
    parser.add_argument("input", type=str, nargs="?", default="", help="URI of the input stream")
    parser.add_argument("output", type=str, nargs="?", default="", help="URI of the output stream")
    parser.add_argument("--network", type=str, default="fcn-resnet18-voc")
    parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"])
    parser.add_argument("--visualize", type=str, default="overlay,mask")
    parser.add_argument("--ignore-class", type=str, default="void")
    parser.add_argument("--alpha", type=float, default=150.0)
    parser.add_argument("--stats", action="store_true")

    args = parser.parse_known_args()[0]

    load_model_from_config()
    if args.network:
        MODEL_NAME = args.network

    net = segNet(args.network, sys.argv)
    net.SetOverlayAlpha(args.alpha)
    input_stream = videoSource(args.input, argv=sys.argv)
    output_stream = videoOutput(args.output, argv=sys.argv)
    buffers = segmentationBuffers(net, args)

    threading.Thread(target=telemetry_loop, daemon=True).start()

    while True:
        img_input = input_stream.Capture()
        if img_input is None:
            continue

        buffers.Alloc(img_input.shape, img_input.format)
        net.Process(img_input, ignore_class=args.ignore_class)
        mask = buffers.mask  # assuming mask is an image buffer with class IDs

        # Convert CUDA image buffer to numpy array
        mask_np = np.asarray(mask)

        if buffers.overlay:
            net.Overlay(buffers.overlay, filter_mode=args.filter_mode)
        if buffers.mask:
            net.Mask(buffers.mask, filter_mode=args.filter_mode)
        if buffers.composite:
            cudaOverlay(buffers.overlay, buffers.composite, 0, 0)
            cudaOverlay(buffers.mask, buffers.composite, buffers.overlay.width, 0)

        output_stream.Render(buffers.output)
        output_stream.SetStatus(f"{MODEL_NAME} | Network {net.GetNetworkFPS():.0f} FPS")
        cudaDeviceSynchronize()

        if not input_stream.IsStreaming() or not output_stream.IsStreaming():
            break

