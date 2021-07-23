DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
python train.py --train ../ouster_data/train_sulaoja_all --test ../ouster_data/test_sulaoja --image_format ria --comment BASE_SULAOJA_ALL --dir ${DIR}
#python train.py --train ../ouster_data/train_no_int --test ../ouster_data/test --image_format ria --comment NO-INT --dir ${DIR}
#python train.py --train ../ouster_data/train_vahi --test ../ouster_data/test --image_format ria --comment SINGLE-TRACK --dir ${DIR}
#python -u train.py --train ../ouster_data/all --test ../ouster_data/all --q2 0.5 --curve_thresh 0.005 --batch_size 512 --lr 0.002 --epochs 250 --dir ${DIR}
#python -u train.py --train ../ouster_data/all --test ../ouster_data/all --q2 0.5 --curve_thresh 0.005 --batch_size 512 --lr 0.002 --epochs 250 --dir ${DIR}
#python -u train.py --train ../ouster_data/all --test ../ouster_data/all --q2 0.5 --curve_thresh 0.005 --batch_size 256 --lr 0.1 --epochs 250 --dir ${DIR}
