# ISEF2022

##Introduction

###RDN

The number of RDB, learning rate, and number of kernels for each layer used for the AAPM is 5, 0.001, and 64. However, the number of RDB used for the TCIA data set is 10. Since the model is built in a subclass method, the number RDB and number of kernels for each layer can not be changed. Since all of the data used for trainning is gray images, the input channle and output channel is always 1. 

###Sharp RDN

The number of RDB, learning rate, and number of kernels for each layer used for the AAPM is 5, 0.001, and 64. However, the number of RDB used for the TCIA data set is 10. The only difference between the RDN and Sharp RDN is that there is a Sharpen Layer added to the RDN.
### Sharp U-net 

All of the program is ran in Google Colab.
