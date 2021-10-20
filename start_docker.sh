ROOT_DIR=/home/daniel/projects/idp
DATASET_DIR=$ROOT_DIR/dataset/
CODE_DIR=$ROOT_DIR/One-Thing-One-Click

docker build -t derkaczda/one-thing-one-click .

docker run -it --rm --privileged --gpus all -v /dev:/dev \
    -v $DATASET_DIR/scannet:/data \
    -v $DATASET_DIR/model:/model \
    -v $CODE_DIR/3D-U-Net/config/pointgroup_run1_scannet.yaml:/otoc/3D-U-Net/config/pointgroup_run1_scannet.yaml \
    -v $CODE_DIR/experiments:/otoc/3D-U-Net/exp \
    derkaczda/one-thing-one-click python3 train.py --config config/pointgroup_run1_scannet.yaml #--pretrain /model/pointgroup_run1_scannet-000000000.pth
