docker run -it --rm --privileged --gpus all -v /dev:/dev \
    -v /home/daniel/projects/idp/dataset/scannet:/data \
    -v /home/daniel/projects/idp/dataset/model:/model \
    otoc python3 test.py --config config/pointgroup_run1_scannet.yaml --pretrain /model/pointgroup_run1_scannet-000001250.pth
