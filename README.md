## Gradient Multi-Foci Networks for 3D skeleton-based Human Motion Prediction
This is the code for the paper

Junyu Shi, Jianqi Zhong, Zhiquan He, Wenming Cao. 
"Gradient Multi-Foci Networks for 3D skeleton-based Human Motion Prediction." 

### Dependencies

* cuda 12.1
* Python 3.10.0
* [Pytorch](https://github.com/pytorch/pytorch) =2.1.0

### Get the data

[Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).

Directory structure: 
```shell script
H3.6m
|-- S1
|-- S5
|-- S6
|-- ...
`-- S11
```

### Training
Training on Human3.6M
```bash
python main_h36m_3d.py --kernel_size 10 --dct_n 20 --input_n 50 --output_n 10 --skip_rate 1 --batch_size 32 --test_batch_size 256 --in_features 66 --lr 0.001 --dev cuda:0 --data_dir [PATH TO DATA]
```

### Evaluation
Testing on Human3.6M
```bash
python main_h36m_3d_eval.py --is_eval --kernel_size 10 --dct_n 20 --input_n 50 --output_n 10 --skip_rate 1 --test_batch_size 256 --in_features 66 --ckpt [PATH TO CKPT] --dev cuda:0 --data_dir [PATH TO DATA]
```

### To do
- [x] Release the training code on Human3.6M dataswet
- [ ] Release the training code on amass dataset
- [ ] Release the training code on CMU dataset

### Acknowledgments
The overall code framework (dataloading, training, testing etc.) is based on [HisRepItself](https://github.com/wei-mao-2019/HisRepItself). 

### Licence
MIT
