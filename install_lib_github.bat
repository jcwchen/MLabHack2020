Rem Install pip library
python -m ensurepip --default-pip
Rem Install prerequirement for load_onnx_from_github.py
pip install requests
pip install protobuf==3.11.3
pip install numpy
pip install onnx
Rem Install PyTorch without GPU
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html