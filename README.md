## qMTNet: Accelerated Quantitative Magnetization Transfer Imaging with Artificial Neural Networks

This is the accompanying codes and data for the paper **qMTNet: Accelerated Quantitative Magnetization Transfer Imaging with Artificial Neural Networks**.
### Overview
![Overview](https://user-images.githubusercontent.com/33406397/84028425-a7639d00-a9cb-11ea-98a1-c1a2aa4c95dd.png)
**_Figure 1_**
![Overview2](https://user-images.githubusercontent.com/33406397/84029069-c3b40980-a9cc-11ea-8d7e-6fab3815de02.png)
**_Figure 2_**  
qMTNet is a family of neural networks that seek to accelerate qMT acquisition and fitting. **qMTNet-1** (Figure 2c) is a single network that directly maps undersampled MT images to qMT parameters. **qMTNet-2** composes of 2 networks, **qMTNet-acq** (Figure 2a) that produces 8 MT images from 4 MT images and **qMTNet-fit** (Figure 2b) that generate qMT parameters from 12 MT images. Both networks can produce high quality quantitative maps with a fraction of the processing time compared to conventional fitting methods.

### Requirements
The codes in this repository has been developed and tested in Ubuntu 16.04.6 and Windows 10 with Anaconda. 
Versions of notable packages are as followed.
```
Python				3.5.2
Tensorflow			1.15.0
Keras				2.3.1
Numpy				1.18.1
```
### Data
Samples of the data used in this study can be found in the `data` folder. This data was acquired with a Siemens Tim Trio 3T scanner. 

* **sample_train_data.mat**: 
	* Contains input, output data for pre-saturation and inter-slice MT data
	* Converted from 2D slice to pixelwise data
	* Data format: [Number of pixels] x [14 features]. The 14 features are T1, T2, and 12 MT intensities of the pixel.
	* Non-valid pixels (background) has been removed.

* **sample_test_data.mat**:
	* Same as sample_train_data.mat but non-valid pixels are kept so you can use reshape to get original slice. 


### Model
Sampled of the trained models can be found in the `models` folder. The model name denotes the type of model (qMTNet-fit, qMTNet-acq, or qMTNet-1) as well as the type of data that the model was trained on (conventional or inter-slice). The models included here has not been trained on the data found in the `data` folder.

### Codes
Source codes for training and testing the models are described briefly below. You can find more details by looking through the comments inside the codes.

* **model.py**: define various models that was used in this study. You can define your own model here.
* **utils.py**: contain utility function to visualize the output of the network and the ground truth.
* **train.py**: train the model
* **test.py** : test the model

To train the network with the sampled data, you can use this command:

```
python train.py --data_dir ./data/sample_train_data.mat --data_mode conv --name example_exp --gpu_ids 0 --checkpoints_dir ./checkpoints --model_type qMTNet_fit
```
To test the included model:
```
python test.py --data_dir ./data/sample_test_data.mat --data_mode conv --name example_exp --gpu_ids 0 --checkpoints_dir ./checkpoints --model_type qMTNet_fit --model_dir ./models/qMTNet_fit_conv.h5
```

### Reference
To be added

