Rem Install pip library
python -m ensurepip --default-pip
Rem Install onnxruntime for inference
pip install onnxruntime
Rem Install PyTorch without GPU
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
Rem Install TorchVision for image processing
pip install torchvision