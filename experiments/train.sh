DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
#python train.py --train ../ouster_data/train --test ../ouster_data/test --image_format ria --comment BASE --dir ${DIR}
#python train.py --train ../ouster_data/train_no_int --test ../ouster_data/test --image_format ria --comment NO-INT --dir ${DIR}
#python train.py --train ../ouster_data/train_vahi --test ../ouster_data/test --image_format ria --comment SINGLE-TRACK --dir ${DIR}
python train.py --train ../ouster_data/train --test ../ouster_data/test --image_format r --comment RNG --dir ${DIR}
python train.py --train ../ouster_data/train --test ../ouster_data/test --image_format i --comment INT --dir ${DIR}
python train.py --train ../ouster_data/train --test ../ouster_data/test --image_format a --comment AMB --dir ${DIR}
python train.py --train ../ouster_data/train --test ../ouster_data/test --image_format ra --comment RNG+AMB --dir ${DIR}
python train.py --train ../ouster_data/train --test ../ouster_data/test --image_format ri --comment RNG+INT --dir ${DIR}
python train.py --train ../ouster_data/train --test ../ouster_data/test --image_format ia --comment INT+AMB --dir ${DIR}
python train.py --train ../ouster_data/train --test ../ouster_data/test --image_format ria --comment FUT --num_steers 3 --dir ${DIR}
python train.py --train ../ouster_data/train --test ../ouster_data/test --image_format ria --comment DIFF --num_steers 1 --use_diff --dir ${DIR}
python train.py --train ../ouster_data/train --test ../ouster_data/test --image_format ria --comment FUT+DIFF --num_steers 3 --use_diff --dir ${DIR}
#python -u train.py --train ../ouster_data/all --test ../ouster_data/all --q2 0.5 --curve_thresh 0.005 --batch_size 512 --lr 0.002 --epochs 250 --dir ${DIR}
#python -u train.py --train ../ouster_data/all --test ../ouster_data/all --q2 0.5 --curve_thresh 0.005 --batch_size 512 --lr 0.002 --epochs 250 --dir ${DIR}
#python -u train.py --train ../ouster_data/all --test ../ouster_data/all --q2 0.5 --curve_thresh 0.005 --batch_size 256 --lr 0.1 --epochs 250 --dir ${DIR}
