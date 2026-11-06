# GPU PC setup (Ubuntu + Docker) for Model B + E

Files: `infra/ml/Dockerfile`, `infra/ml/requirements-gpu.txt`, `infra/ml/compose.gpu.yaml`,
`infra/ml/setup_third_party.sh`, `infra/ml/get_checkpoint.sh`.

Status: written on the laptop, **not yet built or run** (no Docker daemon or GPU here). Step 6 is the first real test.
Treat any build error as expected feedback and paste it back.

## What lives where
- **Host (the PC):** NVIDIA driver, Docker, NVIDIA Container Toolkit, the git checkout, `data/`.
- **Container:** Python 3.11, PyTorch 2.1.0 (CUDA 12.1 wheels), all pip packages. Repo is mounted at `/workspace`.
- You do NOT install CUDA toolkit or Python on the host. Only the driver.

## 1. Find the GPU and install the driver
```bash
lspci | grep -i nvidia          # on this PC: "AD104GL [RTX 4000 Ada Generation]"
sudo apt update
sudo apt install -y ubuntu-drivers-common
ubuntu-drivers devices          # lists drivers; one is marked "recommended"
```
Install the **proprietary NVIDIA driver** that `ubuntu-drivers devices` marks recommended. The CUDA 12.1 wheels in the image need driver >= 525.60,
and the GPU must be a GTX 900 series or newer (Maxwell+). If your GPU is older than that, tell Claude, since the image would need different wheels.
Easiest: `sudo ubuntu-drivers install` (installs the recommended one). Or by name, e.g. `sudo apt install -y nvidia-driver-570`
(use the number `ubuntu-drivers devices` offers; "-server" and "-open" variants are not needed).
```bash
sudo reboot
nvidia-smi                      # must now print a table with the GPU, driver version, CUDA version
```
If a blue "MOK" screen appears on reboot, Secure Boot is on: choose Enroll MOK, enter the password you were asked to set.
Paste the `lspci` line and the `nvidia-smi` table to Claude before continuing.

## 2. Docker + GPU access
```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi   # must print the same GPU table
```

## 3. Get the code
On the laptop first: commit and push your branch (`feat/ml-tracking-e2`), including `infra/ml/` and this file.
```bash
sudo apt install -y git
git clone <YOUR REPO URL> sentinel && cd sentinel
git checkout feat/ml-tracking-e2
bash infra/ml/setup_third_party.sh      # clones uav-dot + uav-dot-track(TrackEval) at the laptop's commits
```
The script prints TrackEval's commit; it should be `5477c170...`.

## 4. Get DroneCrowd (about 11 GB)
Download from the dataset's own page: https://github.com/VisDrone/DroneCrowd (links in its README) and unzip into
`data/dronecrowd/` with the **same layout as on the laptop** (see `data/README.md`):
`train_data-002/train_data/{images,ground_truth}`, `test_data-001/test_data/...`, `val_data/val_data/...`,
`annotations/annotations/`, `trainlist.txt`, `testlist.txt`.
`data/manifests/` already comes with git (the frozen split). If the download link is a Google Drive file, tell me the
link and I will write the exact `gdown` command.

## 5. Build the image
```bash
export UID GID=$(id -g)
docker compose -f infra/ml/compose.gpu.yaml build        # 10-20 min, downloads ~3 GB
docker compose -f infra/ml/compose.gpu.yaml run --rm ml bash     # you are now in /workspace inside the container
bash infra/ml/get_checkpoint.sh                          # DOT+PD checkpoint (~1 file)
```

## 6. Verify (inside the container)
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pytest ml/tests -q
ruff check .
python -m sentinel_ml.localization.infer --config ml/configs/infer_b.yaml --split val --max-clips 1 --max-frames 5 --out-dir data/outputs/smoke
```
Expect `True` and the name of whatever GPU the PC really has. Paste me the outputs (and any error).

## 7. Re-create the outputs (they are git-ignored)
`data/outputs/` did not travel. Rebuild what you need, e.g.
```bash
python -m sentinel_ml.localization.infer --config ml/configs/infer_b.yaml --split val
python -m sentinel_ml.localization.infer --config ml/configs/infer_b.yaml --split test
```
Announce GPU jobs to Mahid and Ahmar first (shared GPU, one heavy job at a time). Check `nvidia-smi` before you start.

## Notes
- Deviations from the authors' pins (`uav-dot/requirements.txt`): `opencv-python-headless` instead of `opencv-python==4.5.5.64`,
  `scikit-learn>=1.3` instead of `1.0.2` (no Python 3.11 wheels). Neither is used by our inference path.
- Torch 2.1.0 loads the released checkpoint without the `weights_only` problem seen on Kaggle (that came from torch >= 2.6).
  The checkpoint still comes from the authors' Google Drive link and is pickle-based; same trust caveat as in decision 0003.
- First model start downloads ImageNet encoder weights (mit_b2), so the container needs internet once; they are cached in a docker volume.
- Not covered: `apps/` (frontend/backend) and Mahid's/Ahmar's modules; use the existing `infra/compose.yaml` for those.
