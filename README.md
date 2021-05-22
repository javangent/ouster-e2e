# Ouster-E2E
Repository for the thesis project "Using LiDAR as a Camera for End-to-End Driving". 
This readme acts as a guide to running the scripts to train and evaluate networks as outlined in the thesis.
The thesis itself can be found [here](https://comserv.cs.ut.ee/ati_thesis/datasheet.php?id=72431&year=2021).

## Dataset extraction
To extract the dataset from a recorded ROSBAG, run the following command
```
python extract_ouster_images.py path/to/rosbags/*.bag
```
The extracted data will be in the same place as the ROSBAGS. The feolders of each ROSBAG must be put into a common folder, which is considered a dataset.
## Validating the dataset
To visually validade the dataset, run the following command
```
python vis.py path/to/dataset
```
This will cfeate an OpenCV image viewer where you can view the next and previous image with the 'k' and 'j' keys respectively.
A vertical line overlayed on the image will show the ground truth steering output.
## Training the network
To train the network, run the following script
```
python train.py --train path/to/training/set --test path/to/validation/set --comment my_comment --dir path/to/store/resuts --gpus 1
```
This will create a new folder `path/to/store/results/runs/[datetime]_my_comment/` which will inclide TensorBoard logs and the saved models in different formats.
Note that an NVIDIA GPU is expected to run the training script. The folder `final_models` provides pretrained models used in the thesis.
## Measuring open-loop performace
Open-loop performance of multiple models on a particular datset can be measured with the following command
```
python eval_ol.py path/to/models/*ts --dataset path/to/dataset --training_mean 0.0
```
The `--training_mean` parameter is optional and is used to measure the performance of the train set mean predictor as a reference.
Note that the model argument here expects TorchScript files of the trained models to simplify comparing different architectures.
The output is code for a Latex table showing the results of each model.
## Measuring closed-loop performance
To measure the closed-loop performance, a different type of dataset is needed. In particular, a dataset containing only autonomous driving
 from the evaluated model and its GNSS trajectories are needed. In addition, a similar dataset must be created from a human driven ROSBAG recording.
 This is known as the 'expert' dataset.
Run the following command to extract closed-loop data from both model driven and expert driven recordings:
```
python extract_steers_and_traj.py path/to/rosbags/*.bag 
```
After the closed-loop data has been extracted, run the following command to calculate the closed-loop metrics:
```
python eval_cl.py path/to/extracted/model/data --expert path/to/expert/data --fr_thresh 1.0
```
the `--fr_thesh` controls the theshold (in meters) of the failure rate metric. 
The metric calculates the percentage of time the model spent driving too far away from the expert trajectory. The output is code for a Latex table showing the 
results of each model.
## Visualizing the CNN
Visualization of the trained CNN uses the VisualBackProp method created by Nvidia. The reference implementation used can be found [here]. 
To see the results of VisualBackProp for the trained model run the following command
```
python visual_backprop.py path/to/model.pt path/to/dataset
```
This will launch an OpenCV image viewer window. You can move forwards and backward through the dataset images by pressing 'k' and 'j' respectively.
[image]
