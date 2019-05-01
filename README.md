# FR Loss on Mnist
Implement diffrent Face Recognition Loss with Pytorch. Testing and visualizing on Mnist. It's just a toy example but very useful to understand and compare those loss function.

## Example
### Raw 
![epoch=79](https://i.loli.net/2019/05/01/5cc9a6708ab78.jpg)
2D Embedded Feature
![cos_epoch=79](https://i.loli.net/2019/05/01/5cc9a64be699e.jpg)
Normalized Feature
### CenterLoss 
![epoch=79](https://i.loli.net/2019/05/01/5cc9aae8d3bf0.jpg)
![cos_epoch=79](https://i.loli.net/2019/05/01/5cc9aacec1233.jpg)
### SphereFace
![epoch=79](https://i.loli.net/2019/05/01/5cc9ad3b57fc1.jpg)
![cos_epoch=79](https://i.loli.net/2019/05/01/5cc9ad54262e6.jpg)


## Quick Start
### Dependencies
- Pytorch 1.0 (0.4 maybe work either)
- tensorboardX 1.4

I highly recommend you to use Anaconda.
### How to run
- the net.py include the network and loss functional
- the other python scripts using different loss to train the Mnist. Simply use python3 xx.py and you can use tensorboard to see the learning curve and feature visualization. Also, the images are saved on the work directory.

## About the project
- I try to use the same structure to implement different loss. If you have any questions or you find any mistakes, please submmit an issue. Thanks a lot!

