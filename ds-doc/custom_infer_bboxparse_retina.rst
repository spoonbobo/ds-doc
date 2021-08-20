Custom Model - Custom Parser - RetinaNet
========================================

sudo docker run --gpus all -it --rm -v $(pwd)/custom_parser_retina:/opt/nvidia/deepstream/deepstream-5.1/sources/retina -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=:1 -w /opt/nvidia/deepstream/deepstream-5.1 nvcr.io/nvidia/deepstream:5.1-21.02-triton

Output Layer 1: 5 Dims: 720x60x80
Output Layer 2: 6 Dims: 720x30x40
Output Layer 3: 7 Dims: 720x15x20
Output Layer 4: 8 Dims: 720x8x10
Output Layer 5: 9 Dims: 720x4x5
Output Layer 6: 0 Dims: 36x60x80
Output Layer 7: 1 Dims: 36x30x40
Output Layer 8: 2 Dims: 36x15x20
Output Layer 9: 3 Dims: 36x8x10
Output Layer 10: 4 Dims: 36x4x5
