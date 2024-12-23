


# YOLOX
[Source-1](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/) [source-2](https://www.datacamp.com/blog/yolo-object-detection-explained?utm_source=google&utm_medium=paid_search&utm_campaignid=19589720830&utm_adgroupid=152984015734&utm_device=c&utm_keyword=&utm_matchtype=&utm_network=g&utm_adpostion=&utm_creative=724847714833&utm_targetid=aud-1685385913382:dsa-2222697811358&utm_loc_interest_ms=&utm_loc_physical_ms=9197829&utm_content=DSA~blog~Data-Science&utm_campaign=230119_1-sea~dsa~tofu_2-b2c_3-us_4-prc_5-na_6-na_7-le_8-pdsh-go_9-nb-e_10-na_11-na-dec24&gad_source=1&gclid=CjwKCAiAjp-7BhBZEiwAmh9rBXffQ738y7053wLqw3z1H2dPpelwheSQk2R7-YQ060P4GN_dVLbz6xoCqD0QAvD_BwE)\
Building over and above [YOlO - AndrewNG Notes](./DeepLearning.ai-Summary-master/4-%20Convolutional%20Neural%20Networks/Readme.md#object-detection-1)

Yolox is Fully convolution network (FCN) no FC (fully connected) layer hence it is very fast. It utilized identity-convolution to flatten the last layer output.

it divides the image in a grid and gives out prediction for each block of grid using the sliding windows algorithm and as discussed in the Andrew notes, it is much more efficient.

For each block it outputs one-out-block for each anchor. one-out-block (cx, cy, wx, wy, c1 c2 c3..)

in object labelling object is assigned to the anchor with higher IoU so anchor kind of save the spatial information about the general shape (aspect ratio) of object being categorized.

And that's it, that's yolo for you. using 
- Sliding-Windows
- FCN
- Anchor boxes
- NMS
- IoU filtering 

in NMS we are removing same object detected which is not max.
Later in post-processing we have to perform NMS and IoU filtering which is drawback.

# R-CNN, Fast-RCNN, Faster RCNN
Note: R here stands for Region proposal CNN and not `recurrent`

In general any object detection requires

1. **Region proposal network**: a network that tells us if any class is present in the given area of image or not, this is checked using FCN that at the end layer tells which blocks contains objects. Detailed are like YOLOX where the last layer only has one output for one proposal layer. From all the proposal where we have detected the object the object is inside it not sure where exactly so the later network take these proposal and tell where exactly are the object of interest located in the original image. In a gist, this network aim is to convolute through the image (generally in a fixed shape convolution) and tell us the region in which there is an object in simple yes or no style. This can be implemented in RCNN, Fast-RCNN and Faster-RCNN
2. 
We will discuss in details only the faster-RCNN 

**RCNN** - post first layer RCNN feed all the proposal one by one to the classification network (SVM for k class) and BBOX regressor and hence is very slow in this step (45 second)
**Fast-RCNN** - This saved computation by using the ROI pooling layer

This is Heavily engineered method from 
- region Proposal 
- ROI pooling layer - in itself a separate paper :/
- FC layer for classification and object location
[Faster-RCNN Paper](https://arxiv.org/pdf/1506.01497v3) [RCNN](https://www.youtube.com/watch?v=5DvljLV4S1E)
[Salute to this video Level Explanations] (https://www.youtube.com/watch?v=itjQT-gFQBY)
Introduced RPN Region proposal network, FPN - Feature Pyramid Network
These all belong to the same family of model, using the same Res-net Backbone for classification but implemented different strategy for object detection problem



Very tough understand all the details skipping but the main things is to understand the three underscore network and their work.
The same has been proposed in the efficentDet for bi-FPN. 

![img.png](../Assets/fasterRCNN_arch.png)


### RCNN
Layer-1 
![img_1.png](../Assets/RCNN-2.png)
Layer- 2 & 3
![img.png](../Assets/RCNN-1.png)

### Fast-RCNN: [source](https://www.youtube.com/watch?v=pCkxu9958bU)

Rather than passing individual region, we passed the entire image to CNN for feature extraction.

![img.png](../Assets/fastRCNN/fastRCNN-1.png)

From the entire feature of whole image, figure out the feature corresponding to proposed region using scaling.
Require boundary case handling but this is it in a gist
![img_1.png](../Assets/fastRCNN/fastRCNN-2.png)

Finally take the feature map and flatten them (ROI pooling comes here) and feed them to classification and bbox regressor. Here we note that regressor has to be run on all proposed boxes again and again, which we cannot avoid.
![img_2.png](../Assets/fastRCNN/fastRCNN-3.png)

not exactly possible because FC layer require fixed input size
![img_3.png](../Assets/fastRCNN/fastRCNN-4.png)

here comes the ROI polling layer to take the feature map of proposed region and convert it to fixed size
![img_4.png](../Assets/fastRCNN/fastRCNN-5.png)

ROI pooling layering essentially divides the given shape of proposed region in required output format and take max out of it. 
![img_5.png](../Assets/fastRCNN/fastRCNN-6.png)
Some details are skipped because converting the proposed region to feature map has float value so we have to shift our proposed region here and there.
![img_6.png](../Assets/fastRCNN/fastRCNN-7.png)
Proposal feature 
![img.png](../Assets/fastRCNN/fastRCNN-9.png)

Which alter were feed to classifier individually i.e one by one 
![img.png](../Assets/fastRCNN/fastRCNN-8.png)

And thats it RPN + ROI pooling layer + classification layer makes the Fast-RCNN, the difference being pooling layers and feature Extraction layer allows to feed iamge only once

### Faster-RCNN: [source](https://www.youtube.com/watch?v=Qq1yfWDdj5Y)
Why Faster-RCNN: still very slow in region proposal

Till now we were using selective search proposal
![img.png](../Assets/fasterRCNN/img.png)

In faster-RCNN we train a separate network for region proposal (trivially also called RPN) to propose region which were then feed to the ROI pooling layer as in the fastRCNN

![img_1.png](../Assets/fasterRCNN/img_1.png)

Region Proposal Network:

This is a difficult task because size of proposal can be different, aspect ratio can be different and proposing all the combination of above is computationally expensive
![img_2.png](../Assets/fasterRCNN/img_2.png)

Instead we pass the whole image to CNN and take the feature map output (of reduced size) and run our sliding window (with different aspect ratios as defined by anchors) region proposal on this.
![img_3.png](../Assets/fasterRCNN/img_3.png)


here is exactly what we do
We run our 3X3 convolution kernel on the image and ask if the object is present in anchor of different size and aspect ratio that is centred around the centre of the given kernel. all these prediction are saved in the output in the depth of kernel
![img_4.png](../Assets/fasterRCNN/img_4.png)

our 1X1Xc dimensional feature is used to predict if there is an object in different aspect ratio of bounding box or not
![img_5.png](../Assets/fasterRCNN/img_5.png)

All anchor box output is computed, so we run convolution only once but for all different type of achor boxes hence avoiding computation again. Labelling of this is again tricky because anchor which has IoU > threshold will only be considered as conatining object and IoU <thresh2 will be considered as background (negative) rest is ignored.
![img_6.png](../Assets/fasterRCNN/img_6.png)
The assumption here is we can use a small part of the object to tell if there is a object in the proposed region or not, and anyway we only need a decent guess and later we will exactly figure out where is the object.
![img_7.png](../Assets/fasterRCNN/img_7.png)

we later uses a convolution layer for these proposed regions with output for each anchor boxes. we have 9 anchor boxes for each location so output is huge and training and labelling has to be handled by the code. 2k for k anchor box, one for object and second for background, though one would have worked. 4k in bounding box 4 for each anchor box.
![img_8.png](../Assets/fasterRCNN/img_8.png)

in later layers some filtering is done on these proposed region which have proposal lying outside the iamges and then feed to the classification part of the model ie ROI pooling. Also Note ROI pooling also uses the same backbone CNN so it improve the accuracy
![img.png](img.png)



### EfficientDet
Uses Bi-Feature Pyramid Network based on the intution that previous layer of the CNN also stores critical information useful for the region
![img_1.png](img_1.png)