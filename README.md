# IoTConnect SDK Snap with Canonical: Edge AI + GenAI Model Upgrade Demo

This detailed guide provides explicit instructions for recreating the IoTConnect SDK Snap demo, showcasing Edge AI with NVIDIA Jetson and AWS Bedrock integration for generative AI insights.

## Demo Overview

The demo utilizes a Seeed reComputer J2021 (Jetson Xavier NX) executing six AI models: DepthNet, DetectNet, PoseNet, SegNet, ImageNet, and ActionNet. Each model is managed through Canonical Snapcraft packages and controlled remotely via the IOTCONNECT dashboard. The setup includes:

- Real-time CPU, GPU, and memory telemetry visualization.
- An overhead Logitech BRIO 4K camera monitoring booth engagement through customized PoseNet and DetectNet scripts.
- Telemetry data streaming to AWS S3.
- Interactive conversational insights using Needle.

## Physical Setup

### Hardware

- Seeed reComputer J2021 (Jetson Xavier NX) — [Purchase from Newark](https://www.newark.com/seeed-studio/110061381/nvidia-jetson-xavier-nx-8gb-16gb/dp/62AK4319)
- Two Logitech BRIO 4K USB cameras:
  - **Camera 1**: Positioned at eye level, front-facing for AI model demonstrations.
  - **Camera 2**: Positioned overhead, aimed downward for engagement monitoring.
- Two monitors:
  - Dashboard visualization.
  - Live AI model video output.

### Software Requirements

- [NVIDIA JetPack 5.1.5 (Ubuntu 20.04)](https://developer.nvidia.com/embedded/jetpack-sdk-512)
- IoTConnect SDK Snap
- Modified jetson-inference Python scripts (from this repo)
- OTA update mechanism (provided scripts)

## Required Files (GitHub)

- [Device Template for IOTCONNECT](https://github.com/avnet-iotconnect/avnet-iotconnect-iotc-jetson-demo/blob/master/templates/NVIDIAdemo_template.json)
- [Dashboard Configuration Import](https://github.com/avnet-iotconnect/avnet-iotconnect-iotc-jetson-demo/blob/master/dashboards/AWS_CXO-Canonical_NVIDIA_Jetson_dashboard_export.json)
- [This README & Scripts](https://github.com/avnet-iotconnect/avnet-iotconnect-iotc-jetson-demo)

## Step-by-Step Installation

### 1. Install NVIDIA JetPack 5.1.5

Follow NVIDIA’s JetPack installation guide for version 5.1.5:

[NVIDIA JetPack SDK 5.1.2 Download](https://developer.nvidia.com/embedded/jetpack-sdk-512)

### 2. Install IoTConnect SDK Snap

Open terminal and run:

```bash
sudo snap install iotconnect
```

Verify Snap services:

```bash
snap services iotconnect
```

### 3. Clone and Set Up Modified Jetson Inference Examples

Clone the repository:

```bash
git clone https://github.com/avnet-iotconnect/avnet-iotconnect-iotc-jetson-demo.git
cd avnet-iotconnect-iotc-jetson-demo
```

Then copy the modified AI scripts and models into the correct location:

```bash
sudo cp examples/*.py ~/jetson-inference/python/examples/
sudo mkdir -p /var/snap/iotconnect/common/models/
sudo cp -r models/* /var/snap/iotconnect/common/models/
```

Set the active model:

```bash
echo "segnet2" | sudo tee /var/snap/iotconnect/common/models/current-model.txt
```

### 4. Demo Execution

Open **three separate terminals**:

- **Terminal 1: Start Snap Socket**

```bash
sudo snap start iotconnect.socket
```

- **Terminal 2: Launch IoTC Model Launcher**

```bash
cd ~/jetson-inference/python/examples
python3 iotc-launcher.py
```

- **Terminal 3: Start Engagement Monitoring**

```bash
python3 detectnet_ppl_pose-iotc.py /dev/video4
```

The IOTCONNECT dashboard commands control model execution.

## Dashboard Configuration

Import the provided dashboard:

[Dashboard Import JSON](https://github.com/avnet-iotconnect/avnet-iotconnect-iotc-jetson-demo/blob/master/dashboards/AWS_CXO-Canonical_NVIDIA_Jetson_dashboard_export.json)

Key features:

- Launch and stop AI demos.
- Telemetry frequency control.
- Interaction box (x, y, width, height) definition.
- Real-time metrics (CPU, GPU, Memory).

## OTA Model Updates

Package models:

```bash
cd ~/jetson-inference/python/examples/models
tar -czvf ~/jetson-inference/python/examples/ota/model_update.tar.gz model.onnx labels.txt config.json
```

Upload to S3 and trigger an OTA command from the IOTCONNECT dashboard.

## Troubleshooting

- Check camera device paths (`/dev/video0`, `/dev/video4`).
- Run `snap logs iotconnect` to debug.
- Ensure device template and commands match the provided JSON.

## Additional Resources

- [IoTConnect Platform](https://www.iotconnect.io)
- [Jetson Inference GitHub](https://github.com/dusty-nv/jetson-inference)

## License

Licensed under the MIT License.

