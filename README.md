This repo contains my university coursework. It is a convolutional neural network (CNN) built to be able to identify specific pet breeds from the Oxford Pet dataset. 
Due to the nature of the coursework, 30 epochs is the maximum amount of training the model is allowed, and this model currently achieves around 70% accuracy on the official non augmented test dataset.
The model uses trimaps and various other data augmentation techniques, and the model is built similarly to a Resnet style architecture, with a residual block seperating layers. 
Image size used is 224*224, and therefore needs GPU usage to be able to train quickly.
Enjoy!
