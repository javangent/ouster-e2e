import torch
from data import *
import argparse
import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser(description='Evaluate closed-loop metrics offline')
parser.add_argument('models', type=str, nargs='+', help='path of model file (TorchScript)')
parser.add_argument('--dataset', type=str, help='path of dataset used for evaluation', default='../ouster_data/test')
parser.add_argument('--training_mean', type=float, help='mean value of training data seering angles (degree)', default='0.0')
args = parser.parse_args()

model_to_format = defaultdict(lambda: 'ria')
model_to_format['RNG'] = 'r'
model_to_format['AMB'] = 'a'
model_to_format['INT'] = 'i'
model_to_format['RNG+INT'] = 'ri'
model_to_format['RNG+AMB'] = 'ra'
model_to_format['INT+AMB'] = 'ia'

def get_whiteness_from_df(df):
    a = df['steer_angle_0'] / np.pi * 180
    W = np.sqrt(((a.diff()*10)**2).mean())
    print('look down')
    print(((a.diff()*10)**2).sum())
    print(a.diff().count())
    return W


MAEs = []
RMSEs = []
W_scores = []
model_names = []
max_offsets = []

with torch.no_grad():
    for m in args.models:
        model_name = m.split('/')[-2].split('_')[-1]
        model_names.append(model_name)
        print(f'{model_name=}')

        delta_steers = []
        prev_steer = None
        errors = []
        outs = []
        steers = []
        image_format = model_to_format[model_name]
        frame_dist = 1 if 'DIFF' in model_name else 0
        ds = OusterConcatDataset(args.dataset, frame_dist=frame_dist, only_curves=False, future_steer_dist=10, num_steers=1, image_format=image_format)

        model = torch.jit.load(m)
        model.eval()
        for data in ds:
            img = data['image'].unsqueeze(0).cuda()
            steer = data['steer'].item()
            steer = steer/np.pi * 180
            out = model(img)[0,0].cpu().item()
            out = out/np.pi * 180
            if prev_steer is not None:
                delta_steers.append(out - prev_steer)
            prev_steer = out
            errors.append(abs(steer - out))
            outs.append(out)
            steers.append(steer)
        delta_steers = np.array(delta_steers)
        outs = np.array(outs)
        steers = np.array(steers)
        W_model = np.sqrt(((delta_steers*10)**2).mean()) # 10 Hz
        W_scores.append(W_model)
        print(f'{W_model=}')
        errors = np.array(errors)
        MAE = errors.mean()
        MAEs.append(MAE)
        print(f'{MAE=}')
        RMSE = np.sqrt((errors**2).mean())
        RMSEs.append(RMSE)
        print(f'{RMSE=}')
        max_offset = errors.max()
        max_offsets.append(max_offset)
        print(f'{max_offset=}')
        print()
    W_expert = get_whiteness_from_df(ds.df)
    model_names.append('Human driver')
    W_scores.append(W_expert)
    MAEs.append(None)
    RMSEs.append(None)
    max_offsets.append(None)
    print(f'{W_expert=}')

    # Zero predictor
    model_names.append('Zero predictor')
    W_scores.append(0)
    MAEs.append(abs(steers).mean())
    RMSEs.append(np.sqrt((steers**2).mean()))
    max_offsets.append(abs(steers).max())

    # Mean predictor
    model_names.append('Mean predictor')
    W_scores.append(0)
    MAEs.append(abs(steers - args.training_mean).mean())
    RMSEs.append(np.sqrt(((steers-args.training_mean)**2).mean()))
    max_offsets.append(abs(steers-args.training_mean).max())

    models_s = pd.Series(model_names, name='Model')
    W_scores_s = pd.Series(W_scores, name='Whiteness')
    MAEs_s = pd.Series(MAEs, name='MAE')
    RMSEs_s = pd.Series(RMSEs, name='RMSE')
    max_offsets_s = pd.Series(max_offsets, name='Maximum error')

    table = pd.concat([models_s, RMSEs_s, MAEs_s, W_scores_s, max_offsets_s ], axis=1)
    table.sort_values('MAE')
    caption = "Open-loop evaluation metrics on test data collected from the training track."
    latex = table.to_latex(index=False, bold_rows=True, longtable=True, caption=caption, label='tab:ol-test', float_format='%.2f', na_rep='N/A')
    print(latex)
