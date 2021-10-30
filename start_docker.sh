U_NET_TEST="cd /otoc/3D-U-Net; python3 test.py \
    --config config/pointgroup_run1_scannet.yaml \
    --pretrain /model/pointgroup_run1_scannet-000001250.pth"
RELATION_TEST="cd /otoc/relation; python3 test.py \
    --config config/pointgroup_run1_scannet.yaml \
    --pretrain /model/pointgroup_run1_scannet-000002891_weight.pth"
MERGE_TEST="cd /otoc/merge; python3 test.py \
    --config config/pointgroup_run1_scannet.yaml"

CONTAINER_NAME="derkaczda/one-thing-one-click"
CODE_ROOT=/home/daniel/projects/idp/One-Thing-One-Click

docker run -it --rm --privileged --gpus all -v /dev:/dev \
    -v scannet_dataset:/data \
    -v otoc_pretrain_model:/model \
    -v otoc_result:/result \
    -v $CODE_ROOT/3D-U-Net/config/pointgroup_run1_scannet.yaml:/otoc/3D-U-Net/config/pointgroup_run1_scannet.yaml \
    -v $CODE_ROOT/relation/config/pointgroup_run1_scannet.yaml:/otoc/relation/config/pointgroup_run1_scannet.yaml \
    -v $CODE_ROOT/merge/config/pointgroup_run1_scannet.yaml:/otoc/merge/config/pointgroup_run1_scannet.yaml \
    $CONTAINER_NAME bash -c "$U_NET_TEST"