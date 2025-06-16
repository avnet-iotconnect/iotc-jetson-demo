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

- Seeed reComputer J2021 (Jetson Xavier NX)
- Two Logitech BRIO 4K USB cameras:
  - **Camera 1**: Positioned at eye level, front-facing for AI model demonstrations.
  - **Camera 2**: Positioned overhead, aimed downward for engagement monitoring.
- Two monitors:
  - Dashboard visualization.
  - Live AI model video output.

### Software Requirements

- NVIDIA JetPack 5.1.5 (Ubuntu 20.04)
- IoTConnect SDK Snap
- Modified jetson-inference Python scripts (provided via GitHub)
- OTA update mechanism (provided scripts)

## Required Files (GitHub)

- [IoTConnect Device Template](https://github.com/\[YOUR_REPO]/NVIDIAdemo_template.JSON)
- [Dashboard Configuration Import](https://github.com/\[YOUR_REPO]/AWS_CXO-Canonical_NVIDIA_Jetson_dashboard_export.json)

## Step-by-Step Installation

### 1. Install NVIDIA JetPack 5.1.5

Follow NVIDIA’s JetPack installation guide for version 5.1.5:

[NVIDIA JetPack SDK Installation](https://developer.nvidia.com/embedded/jetpack)

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

Clone the modified jetson-inference repository into the correct directory:

```bash
git clone [YOUR_REPO_URL] ~/jetson-inference
cd ~/jetson-inference/python/examples
```

Copy provided AI model files and labels to Snap common directory:

```bash
sudo cp -r models/* /var/snap/iotconnect/common/models/
```

Set the active model via:

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

The IoTConnect dashboard commands control the execution of the models remotely.

## Dashboard Configuration

Import the provided IoTConnect dashboard configuration:

[Dashboard Template](https://github.com/\[YOUR_REPO]/AWS_CXO-Canonical_NVIDIA_Jetson_dashboard_export.json)

Dashboard capabilities include:

- Launching and stopping AI models via commands.
- Adjusting telemetry frequency.
- Defining interaction zones via box coordinates.
- Real-time telemetry monitoring (CPU, GPU, memory usage).

## OTA Model Updates

Prepare OTA packages:

```bash
cd ~/jetson-inference/python/examples/models
tar -czvf ~/jetson-inference/python/examples/ota/model_update.tar.gz model.onnx labels.txt config.json
```

Upload OTA packages to AWS S3 and trigger updates through the IoTConnect dashboard.

## Troubleshooting

- Confirm the camera paths (`/dev/video0`, `/dev/video4`) match your setup.
- Check Snap logs:
  ```bash
  snap logs iotconnect
  ```
- Ensure dashboard commands and telemetry attributes match provided templates.

## Additional Resources

- [Official IoTConnect Documentation](https://www.iotconnect.io)
- [NVIDIA Jetson Inference Documentation](https://github.com/dusty-nv/jetson-inference)

## License

Licensed under the MIT License.

