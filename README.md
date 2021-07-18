change to pruning branch to see src PLZ
This project is based on https://github.com/AlexeyAB and developend to prune Darknet weights file as your will.

Steps to use it 

1 build this project as the original Darknet for windows from  https://github.com/AlexeyAB;
2 in the cmd line window type in:
  darknet.exe xx.cfg xx.weights m n
xx.cfg is the cfg file for the model you want to prune
xx.weights is the weights file for the model you want to prune
m is the mth layer you want to prune
n is the percentage of the filters you want to prune off.

for example  darknet.exe yolov2.cfg yolov2.weights 23, 0.5 which means you will cut off 50% of the filters in the 23th Conv layer, the number of original filters in 23th layer is 1024, now the you will get 512 more important filters.

The pruning algorithm is simple, referring to this "Pruning Filters for Efficient ConvNets"
