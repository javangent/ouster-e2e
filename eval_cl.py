import pandas as pd
import numpy as np
import sys
import os
import argparse
from sklearn.neighbors import BallTree

def get_steers(f, only_autonomy=False):
    df = pd.read_csv(os.path.join(f, 'steers.csv'), index_col=0)
    steers = df['steer_angle'] / np.pi * 180
    print(f'{steers.count()=}')
    if only_autonomy:
        steers = steers[df.autonomy==True]
        steers = steers.reindex(range(steers.index[-1]+1)) # fill in-between values with NaN
    l = steers.count()
    sq_sum = ((steers.diff()*10)**2).sum() # Multiply diffs by 10 to get deg/sec
    return sq_sum, l

def get_traj_df(f, only_autonomy=False):
    df = pd.read_csv(os.path.join(f, 'traj.csv'), index_col=0)
    traj = df[['X', 'Y']]
    if only_autonomy:
        traj = traj[df.autonomy == True].reset_index(drop=True)
    return traj

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

    # Get expert forward and backward data
    ef = os.path.join(args.expert, '')
    ef2 = os.path.dirname(ef)[:-1]+'2'
    ef2 = os.path.join(ef2, '')

    # create human drived traj dfs
    df_exp_traj = get_traj_df(ef)
    df_exp_traj2 = get_traj_df(ef2)

    Ws = []
    MAEs = []
    RMSEs = []
    FRs = []
    names = []
    interventions = []

    name_to_int = {'AMB': 4,
            'RNG': 0,
            'INT': 1,
            'BASE': 1,
            'NO-INT': 2,
            'SINGLE-TRACK': 6,
            'RNG+INT': 0,
            'RNG+AMB': 3,
            'INT+AMB': 1,
            'TEST': 0,
            'FUT': 0}

    for f in args.folders:
        print(f)
        # Get forward and backward data
        f = os.path.join(f, '')
        f2 = os.path.dirname(f)[:-1]+'2'
        f2 = os.path.join(f2, '')

        # Calculate Whiteness for model (forward and backward)
        sq_sum, l = get_steers(f, True)
        sq_sum2, l2 = get_steers(f2, True)
        W = np.sqrt((sq_sum + sq_sum2)/(l + l2))
        #print(f'{W=}')
        Ws.append(W)

        # get trajectory dfs including only autonomous driving
        df_traj = get_traj_df(f, True)
        df_traj2 = get_traj_df(f2, True)

        # calucale lat errors for models, given expert traj
        lat_errs = lat_errors(df_exp_traj, df_traj)
        print(f'{lat_errs.max()=}')
        lat_errs2 = lat_errors(df_exp_traj2, df_traj2)
        print(f'{lat_errs2.max()=}')
        lat_errs_total = lat_errs.append(lat_errs2).reset_index(drop=True)
        print(f'{lat_errs_total.max()=}')

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
        name = f.split('/')[-2].split('_')[0]
        #print(f'{name=}')
        names.append(name)
        interventions.append(name_to_int[name])

    # human driver metrics
    sq_sum, l = get_steers(ef)
    sq_sum2, l2 = get_steers(ef2)
    W = np.sqrt((sq_sum + sq_sum2)/(l + l2))

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
