import pandas as pd
import numpy as np
import sys
import os
import argparse
from sklearn.neighbors import BallTree

def get_steers(f, only_autonomy=False):
    df = pd.read_csv(os.path.join(f, 'steers.csv'), index_col=0)
    steers = df['steer_angle'] / np.pi * 180
    if only_autonomy:
        steers = steers[df.autonomy==True]
        steers = steers.reindex(range(steers.index[-1]+1)) # fill in-between values with NaN
    l = steers.count()
    sq_sum = ((steers.diff()*10)**2).sum() # Multiply diffs by 10 to get deg/sec
    return sq_sum, l

def get_traj_df(f, only_autonomy=False):
    df = pd.read_csv(os.path.join(f, 'traj.csv'), index_col=0)
    traj = df[['X', 'Y']]
    ints = df.autonomy.astype(int).diff().eq(-1).sum()
    if only_autonomy:
        traj = traj[df.autonomy == True].reset_index(drop=True)
    return traj, ints

def lat_errors(T1, T2, thres=2):
    tree = BallTree(T1.values) 
    inds, dists = tree.query_radius(T2.values, r=thres, sort_results=True, return_distance=True)
    closest_l = []
    for i, ind in enumerate(inds):
        if len(ind) >= 2:
            closest = pd.DataFrame({'X1': [T1.iloc[ind[0]].X], 'Y1': [T1.iloc[ind[0]].Y], 'X2': [T1.iloc[ind[1]].X], 'Y2': [T1.iloc[ind[1]].Y]},
                    index=[i])
            closest_l.append(closest)
    closest_df = pd.concat(closest_l)
    f = T2.join(closest_df)
    lat_errors = abs((f.X2-f.X1)*(f.Y1-f.Y) - (f.X1-f.X)*(f.Y2-f.Y1))/np.sqrt((f.X2-f.X1)**2+(f.Y2-f.Y1)**2)
    return lat_errors

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate closed loop metrics on steers.scv and traj.csv data')
    parser.add_argument('folders', type=str, nargs='+',
                        help='model results from forward drive folders')
    parser.add_argument('--expert', type=str, help='human driver folder')
    parser.add_argument('--fr_thresh', type=float, help='failure rate threshold (meters) (default 1.0)', default=1.0)

    args = parser.parse_args()

    # Get expert data
    ef = os.path.join(args.expert, '')

    # create human driven traj dfs
    df_exp_traj, _ = get_traj_df(ef)

    Ws = []
    MAEs = []
    RMSEs = []
    FRs = []
    names = []
    interventions = []

    for f in args.folders:
        print(f)
        f = os.path.join(f, '')

        # Calculate Whiteness for model 
        sq_sum, l = get_steers(f, True)
        W = np.sqrt((sq_sum)/l)
        Ws.append(W)

        # get trajectory dfs including only autonomous driving
        df_traj, ints = get_traj_df(f, True)

        # calucale lat errors for models, given expert traj
        lat_errs_total = lat_errors(df_exp_traj, df_traj)
        #print(f'{lat_errs.max()=}')

        # calculate metrics
        mae = lat_errs_total.mean()
        #print(f'{mae=}')
        MAEs.append(mae)
        rmse = np.sqrt((lat_errs_total**2).mean())
        #print(f'{rmse=}')
        RMSEs.append(rmse)
        fr = len(lat_errs_total[lat_errs_total > args.fr_thresh])/float(len(lat_errs_total))*100
        #print(f'{fr=}')
        FRs.append(fr)
        name = '\_'.join(f.split('/')[-2].split('_')[3:])
        #print(f'{name=}')
        names.append(name)
        interventions.append(ints)

    # human driver metrics
    sq_sum, l = get_steers(ef)
    W = np.sqrt((sq_sum)/(l))

    Ws.append(W)
    names.append('Human driver')
    MAEs.append(None)
    RMSEs.append(None)
    FRs.append(None)
    interventions.append(None)

    table = pd.DataFrame(
        {'Model': names,
        'Interventions': interventions,
        'MAE': MAEs,
        'RMSE': RMSEs,
        'Whiteness': Ws,
        'Failure Rate %': FRs})
    caption = "Closed-loop evaluation metrics on test data collected from the training track."
    table.sort_values('MAE', inplace=True)
    latex = table.to_latex(index=False, bold_rows=True, longtable=True, caption=caption, label='tab:cl-test', float_format='%.2f', na_rep='N/A', column_format='lcrrrr')
    print(latex)
