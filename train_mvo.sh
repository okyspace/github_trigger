# training with flow (GT); only train pose and lstm

# MODE ={test, local, local-cluster, aip-cluster}
PLATFORM=local-cluster #[local, local-cluster, aip-cluster]
MODE=test #[test, actual]

case $MODE in
    "test")
      DATA_FILE=data/rel1test/tartan_train_list.txt
      VAL_FILE=data/rel1test/tartan_test_list.txt
      ;;
    "actual")
      DATA_FILE=data/tartan_train_list.txt
      VAL_FILE=data/tartan_test_list.txt
      ;;
esac

# create these folders for tmp data files
mkdir tartanair-rel1
mkdir tartanair-rel1-test

# run training
python3 train_mvo_wf.py \
    --exp-prefix ky_gtflow_ \
    --use-int-plotter \
    --print-interval 10 \
    --batch-size 32 \
    --worker-num 0 \
    --train-step 20 \
    --snapshot 20 \
    --test-interval 20 \
    --lr 0.0001 \
    --lr-decay \
    --train-data-type tartan \
    --data-file $DATA_FILE \
    --val-file  $VAL_FILE \
    --platform $PLATFORM \
    --linear-norm-trans-loss \
    --image-height 448 \
    --image-width 640 \
    --normalize-output 0.05 \
    --resvo-config 1 \
    --network 0 \
    --multi-gpu 1 \
    --no-data-augment \
    --downscale-flow \
    --intrinsic-layer \
    --random-intrinsic 600 \
    --load-flow-model \
    --flow-model pwc_net.pth.tar \
    --model-name tartanvo_1914.pkl \
    --train-vo \
    --fix-flow \
    --vo-gt-flow 1.0 \
    --min-num-frames 5 \
    --len-of-pose 6 \
    --nwk-lstm-hidden-layers 16 \
    --nwk-lstm-num-layers 2 \
    --nwk-lstm-dropout 0.1 \
    --torch-tensorboard-folder logTensorboard