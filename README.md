# EffiNet

A PyTorch implementation of the 2020 paper

## 1.Installation
-	Install PyTorch by selecting your environment on the website and running the appropriate command.
-	install numpy 
-	install torchstat 

## 2.Datasets
For our breast cancer diagnosis work , use Camelyon16 benchmark. The training data consists of 270 WSIs (160 normal and 110 tumor) with pixel-level annotations. The test data consists of 129 WSIs. 22 and 27 of them contain macro and micro tumors, respectively. 
You can see https://camelyon16.grand-challenge.org/ for download.

## 3.Training 
For the patch-based classification model generation stage, we directly extract massive image patches according to the coordinates released by NCRF[1], which is a public resource associated with Camelyon16. It contains the coordinates of over 400k representative level-0 patches. We further employed different data augmentation methods like rotation, flipping, and color jittering. SGD with a momentum of 0.9 is used to optimize the network and the learning rate is fixed to 0.001. For method without the detection model, you can use the Otsu algorithm to to exclude the background regions of each given slide firstly, and then a sliding-window breast cancer diagnosis method can be performed on all the foreground pixels.
We have provided our best.ckpt in the checkpoint forlder. Except that the model used is different, we have implemented the same diagnostic process as Wang et al.[2], and improved the AUC score from the reported 0.925 to 0.9342. 
You can download it for further experiments.

## 4.Evaluation
To evaluate the model performace:
'Python mixblaze_1_6_3.py'

You will see the memory, MAdd, Flops, MemR+W of each layer and total.

## 5.Performance
Total params: 65,498
.

Total memory: 13.89MB

Total MAdd: 102.03MMAdd

Total Flops: 52.56MFlops

Total MemR+W: 27.74MB

## Reference
[1] Li, Y., Ping, W.: Cancer metastasis detection with neural conditional random field.
arXiv preprint arXiv:1806.07064 (2018)

[2] Wang, D., Khosla, A., Gargeya, R., Irshad, H., Beck, A.H.: Deep learning for
identifying metastatic breast cancer. arXiv preprint arXiv:1606.05718 (2016)

