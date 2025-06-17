## IoTConnect SDK Snap with Canonical: Edge AI + GenAI Model Upgrade Demo

This detailed guide provides instructions for recreating the IoTConnect SDK Snap demo, showcasing Edge AI with NVIDIA Jetson and AWS Bedrock integration for generative AI insights.

### Demo Overview

The demo utilizes a Seeed reComputer J2021 (Jetson Xavier NX) executing six AI models:

- **DepthNet**
- **DetectNet**
- **PoseNet**
- **SegNet**
- **ImageNet**
- **ActionNet**

Each model is managed through Canonical Snapcraft packages and controlled remotely via the IoTConnect dashboard. The setup includes:

- Real-time CPU, GPU, and memory telemetry visualization.
- An overhead Logitech BRIO 4K camera monitoring booth engagement through PoseNet and DetectNet.
- Telemetry data streaming to AWS S3.
- Conversational analysis using Needle.
- OTA (Over-The-Air) model update capability.

### Physical Setup

#### Hardware

- **Seeed reComputer J2021 (Jetson Xavier NX)** — [Purchase from Newark](https://www.newark.com/seeed-studio/110061381/nvidia-jetson-xavier-nx-8gb-16gb/dp/62AK4319)
- **Two Logitech BRIO 4K USB cameras**:
  - **Camera 1**: Eye-level for AI model demonstrations
  - **Camera 2**: Overhead for engagement detection
- **Two monitors**:
  - Dashboard visualization
  - Live AI model video output

### Software Requirements

- [NVIDIA JetPack 5.1.5 (Ubuntu 20.04)](https://developer.nvidia.com/embedded/jetpack-sdk-512)
- IoTConnect SDK Snap
- Modified Jetson Inference Python scripts
- OTA update mechanism scripts
- **jtop (Jetson statistics monitoring tool)**

### Required Files (GitHub)

- [Device Template for IoTConnect](https://github.com/avnet-iotconnect/avnet-iotconnect-iotc-jetson-demo/blob/master/templates/NVIDIAdemo_template.json)
- [Dashboard Export File](https://github.com/avnet-iotconnect/avnet-iotconnect-iotc-jetson-demo/blob/master/dashboards/AWS_CXO-Canonical_NVIDIA_Jetson_dashboard_export.json)
- [Scripts & Models for Jetson Inference](https://github.com/avnet-iotconnect/avnet-iotconnect-iotc-jetson-demo)

### Step-by-Step Installation

#### 1. Install NVIDIA JetPack 5.1.5

Follow NVIDIA's installation guide:

[NVIDIA JetPack SDK 5.1.2 Download](https://developer.nvidia.com/embedded/jetpack-sdk-512)

#### 2. Install IoTConnect SDK Snap

```bash
sudo snap install iotconnect
snap services iotconnect
```

#### 3. Clone and Set Up Jetson Inference Examples

```bash
git clone https://github.com/avnet-iotconnect/avnet-iotconnect-iotc-jetson-demo.git
cd avnet-iotconnect-iotc-jetson-demo

sudo cp examples/*.py ~/jetson-inference/python/examples/
sudo mkdir -p /var/snap/iotconnect/common/models/
sudo cp -r models/* /var/snap/iotconnect/common/models/
```

Set default model:

```bash
echo "segnet2" | sudo tee /var/snap/iotconnect/common/models/current-model.txt
```

#### 4. Launching the Demo

Use three terminals:

**Terminal 1 (Start Snap Socket)**:

```bash
sudo snap start iotconnect.socket
```

**Terminal 2 (IoTConnect Model Launcher)**:

```bash
cd ~/jetson-inference/python/examples
python3 iotc-launcher.py
```

**Terminal 3 (Engagement Monitoring)**:

```bash
python3 detectnet_ppl_pose-iotc.py /dev/video4
```

#### 4.5. Optional Direct Script Execution

Run individual demos manually:

```bash
python3 imagenet-iotc.py /dev/video0
python3 detectnet-iotc.py /dev/video0
python3 depthnet-iotc.py /dev/video0
python3 segnet2-iotc.py /dev/video0 --stats

python3 posenet-iotc.py /dev/video4
python3 detectnet_ppl_pose-iotc.py /dev/video4
```

#### 5. Dashboard Configuration

Import dashboard JSON:

[Dashboard Import JSON](https://github.com/avnet-iotconnect/avnet-iotconnect-iotc-jetson-demo/blob/master/dashboards/AWS_CXO-Canonical_NVIDIA_Jetson_dashboard_export.json)

Features:

- Launch/stop AI demos
- Adjust telemetry frequency
- Interaction box settings
- Real-time telemetry

### OTA Model Updates

Package and upload new models:

```bash
cd ~/jetson-inference/python/examples/models
tar -czvf ~/jetson-inference/python/examples/ota/model_update.tar.gz model.onnx labels.txt config.json
```

Upload to AWS S3 and send OTA commands via IoTConnect dashboard.

### Troubleshooting

- View snap logs:

```bash
snap logs iotconnect
```

- Verify camera paths:

```bash
ls /dev/video*
```

- Restart Jetson statistics:

```bash
sudo systemctl restart jtop.service
```

- Check resource usage:

```bash
sudo jtop
```

### Enhanced Telemetry & Interaction Detection

The demo includes enhanced capabilities such as detecting wrist interactions near defined monitor regions and reporting real-time occupancy and engagement levels.

### Additional Resources

- [IoTConnect Platform](https://www.iotconnect.io)
- [Jetson Inference GitHub](https://github.com/dusty-nv/jetson-inference)

### License

Licensed under the MIT License.
---

### Why Three Terminals?

To ensure proper function of the demo, use three terminals:

- **Terminal 1: Start IoTConnect socket**
  - Runs the socket listener to enable telemetry and command exchange.
  ```bash
  sudo snap start iotconnect.socket
  ```

- **Terminal 2: Launch IoTConnect AI launcher**
  - Enables IoTConnect dashboard to launch and stop demos via commands.
  ```bash
  python3 iotc-launcher.py
  ```

- **Terminal 3: Run a demo manually (optional)**
  - Used for engagement detection or a second AI script like PoseNet.
  ```bash
  python3 detectnet_ppl_pose-iotc.py /dev/video4
  ```

---

### Socket Permission Tip

If you encounter a permission denied error:
```bash
sudo chmod 666 /var/snap/iotconnect/common/iotc.sock
```

---

### Sample Telemetry Output

This is an example of telemetry successfully sent to IoTConnect:
```json
{
  "timestamp": 1749074557,
  "demo_name": "posenet",
  "model_name": "pose_resnet18_body.onnx",
  "keypoints": {
    "left_wrist": [833.7, 486.2],
    "right_wrist": [677.7, 552.9]
  }
}
```

---

### OTA Command Example

To launch a script from the dashboard or API:
```json
{
  "cmd": "launch",
  "args": ["depthnet-iotc.py"]
}
```

To stop:
```json
{
  "cmd": "stop_demo"
}
```

---

### Fallback for Jetson Stats

If `jtop` fails or is not installed, the launcher uses `tegrastats` for telemetry fallback automatically.

---

### Default Camera Notes

By default:
- `iotc-launcher.py` assumes `/dev/video0`
- `detectnet_ppl_pose-iotc.py` assumes `/dev/video4`

Check camera mapping with:
```bash
v4l2-ctl --list-devices
```

---

### Known Issues

- **Camera capture failure**: Error like `videoSource failed to capture image` often means the camera is busy or not recognized. Try `/dev/video2`, `/dev/video4`, etc.
- **Socket send fails**: Ensure the socket is running and permissions are correct.
  ```bash
  snap logs iotconnect
  ```

---

