import torch
from torchvision import io
from pathlib import Path
import pandas as pd
import numpy as np
from  scipy import stats

class OusterDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, curve_thresh=0, frame_dist=0, future_steer_dist=1, num_steers=1, only_curves=False, image_format='ria'):
        assert frame_dist >= 0
        assert future_steer_dist >= 1
        assert num_steers >= 1

        ccodes = {'r': 0, 'i':1, 'a':2}
        c_inds = [ccodes[c] for c in image_format]
        self.c_inds = torch.tensor(c_inds)

        df = pd.read_csv(root_dir / 'steers.csv')

        images_frame = df["file_name"]
        # Create diff image
        if frame_dist > 0:
            prev_images_frame = images_frame.rename('prev_file_name')
            prev_images_frame.index += frame_dist
            images_frame = pd.concat(
                    [
                        images_frame,
                        prev_images_frame
                    ], axis=1
                ).dropna()

        # Future steer angles
        steer_dfs = []
        for i in range(num_steers):
            steers_frame = df['steer_angle'].rename(f'steer_angle_{i}')
            steers_frame.index -= future_steer_dist * i
            steer_dfs.append(steers_frame)
        steers_frame = pd.concat(steer_dfs, axis=1)

        self.df = pd.concat([images_frame, steers_frame], axis=1).dropna().reset_index(drop=True)

        self.root_dir = root_dir
        self.curve_thresh = curve_thresh
        self.frame_dist = frame_dist

        if only_curves:
            mask = self.df['steer_angle_0'].abs() >= self.curve_thresh
            self.df = self.df[mask].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def get_curve_inds(self):
        mask = self.df['steer_angle_0'].abs() >= self.curve_thresh
        return self.df.index[mask].tolist()
    
    def get_steer_sum(self):
        return self.df['steer_angle_0'].sum()

    def __getitem__(self, idx):
        """
        __getitem__ returns a dictionary with the following possible keys:

        idx - index of sample
        steer - current steering angle
        steer_change - diff between future_steer and current_steer
        image - either a diff image or current image
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'idx': idx}

        dp = self.df.loc[idx]

        img_name = self.root_dir / dp["file_name"]
        image = io.read_image(str(img_name))
        image = torch.index_select(image, 0, self.c_inds)

        sample['steer'] = torch.Tensor(dp["steer_angle_0":])

        if self.frame_dist > 0:
            prev_img_name = self.root_dir / dp["prev_file_name"]
            prev_image = io.read_image(str(prev_img_name))
            diff_image = image - prev_image
            diff_image = (diff_image + 255) / (2*255)
            sample['image'] = diff_image
        else:
            sample['image'] = image / 255

        return sample

class OusterConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, root, curve_thresh=0, frame_dist=0, future_steer_dist=1, num_steers=1, only_curves=False, image_format='ria'):
        root = Path(root)
        data_folders = [f for f in root.iterdir() if f.is_dir()]
        self.curve_inds = []
        l = 0
        steers_total = 0.0
        datasets = []
        dfs = []
        for folder in data_folders:
            ds = OusterDataset(folder,
                    curve_thresh=curve_thresh,
                    future_steer_dist=future_steer_dist,
                    frame_dist=frame_dist,
                    num_steers=num_steers,
                    only_curves=only_curves,
                    image_format=image_format)
            datasets.append(ds)
            inds = [i+l for i in ds.get_curve_inds()]
            l += len(ds)
            steers_total += ds.get_steer_sum()
            self.curve_inds.append(inds)
            dfs.append(ds.df)
        self.df = pd.concat(dfs).reset_index(drop=True)

        super(type(self), self).__init__(datasets)
        self.curve_inds = sum(self.curve_inds, [])
        self.y_mean = steers_total/len(self)

    def get_curve_inds(self):
        return self.curve_inds

    def get_steer_mean(self):
        return self.y_mean

    def get_std(self):
        return self.df["steer_angle_0"].std()

    def get_threshold_weights(self, q2):
        q1 = len(self.curve_inds)/len(self)
        scale = 1 if q1 == 1.0 else (q2 * (1-q1))/(q1 * (1-q2))
        weights = [1 if i not in self.curve_inds else scale for i in range(len(self))]
        return weights

    def get_binning_weights(self, bins=10):
        steers = self.df['steer_angle_0']
        bin_counts, edges = np.histogram(steers, bins=bins, range=(steers.min()-0.00001, steers.max()+0.00001))
        bin_weights = 1/bin_counts
        bin_inds = np.digitize(steers, edges)
        weights = bin_weights[bin_inds-1]
        return weights

    def get_density_weights(self, bw=None):
        steers = self.df['steer_angle_0']
        weights = 1/stats.gaussian_kde(steers, bw)(steers)
        return weights


    def get_numpy_sample(self, idx):
        sample = self[idx]
        img = sample["image"].permute(1,2,0).numpy()
        img = np.ascontiguousarray(img)
        steer = sample["steer"].tolist()
        return img, steer, idx
