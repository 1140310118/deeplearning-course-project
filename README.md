# deeplearning-course-project
We are trying to learn a good metric of distance between images using deep convolutional neural netowrks, 
which can be then applied to image classification based on the classsic knn method.

## 数据部分
orls_faces文件夹是ORL face database 人脸识别数据集 
the Olivetti Research Laboratory in Cambridge, UK
40个类别，每个类别10个样本
PGM格式，92x112的8位灰度图。

## 参考
 F. Samaria and A. Harter 
  "Parameterisation of a stochastic model for human face identification"
  2nd IEEE Workshop on Applications of Computer Vision
  December 1994, Sarasota (Florida).

## 链接
http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html

## load_data.py
load_all_data() 返回所有的数据，返回数据是list，包含40个list元素，每个list包含同一个类别的10个np.matrix(112x92,dtype=uint8)
[[np.matrix(112x92),...],...]
