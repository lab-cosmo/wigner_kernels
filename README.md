# tmp_pytorch

## To get the data for testing

```
wget "https://archive.materialscloud.org/record/file?file_id=b612d8e3-58af-4374-96ba-b3551ac5d2f4&filename=methane.extxyz.gz&record_id=528" -O methane.extxyz.gz
gunzip -k methane.extxyz.gz
rm methane.extxyz.gz
mv methane.extxyz structures/methane.extxyz
```

# Install torch2trt

```
pip install nvidia-pyindex
pip install nvidia-tensorrt

git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install
```