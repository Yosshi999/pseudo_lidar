KITTI=$HOME/KITTI
python finetune_v2.py --maxdisp 192 \
	--model stackhourglass \
	--datapath $KITTI/object/training \
	--split_file $KITTI/object/train.txt \
	--epochs 300 \
	--lr_scale 50 \
	--loadmodel ./pretrained_sceneflow.tar \
	--savemodel ./kitti_3d/ \
	--btrain 2
