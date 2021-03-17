# SALIENCE

An Unsupervised User Adaptation Model for Multiple  Wearable Sensors Based Human Activity Recognition

Unsupervised user adaptation aligns the feature distributions of the data from training users and the new user, so a well trained wearable human activity recognition (WHAR) model can be well adapted to the new user. With the development of wearable sensors, multiple wearable sensors based human activity recognition is gaining more and more attention. The use of multiple wearable sensors supports recognizing more diverse and complex activities and brings more accurate recognition results, but also poses challenges for unsupervised user adaptation: First, different wearable sensors show  different data distributions, and it is difficult to cope with multiple sensor data at the same time and achieve fine-grained alignment; Second, there may be some sensors whose data are difficult or even impossible to align, resulting in poor model generalization. To address these problems, we propose SALIENCE model. It combines **local and global alignments** for multiple sensor data, which aligns the data of each sensor separately to achieve fine-grained alignment, while uniformly aligning the data of all sensors to ensure global consistency of alignment. Furthermore, **attention mechanism** is introduced to focus WHAR model on the sensors with strong feature discrimination and well distribution alignment to improve the performance and generalization. Experiments are conducted on two public WHAR datasets, and the experimental results show that our model can yield a competitive performance.

## Framework
![图片](https://user-images.githubusercontent.com/50646282/111409380-cee15600-8711-11eb-933d-8826ed78fa67.png)


## Without Adaptation 
![SALIENCE_base_S_T_PAMAP2(test user2)](https://user-images.githubusercontent.com/50646282/111409005-16b3ad80-8711-11eb-80b6-5e99084c5e97.png)


## After Adaptation
![SALIENCE_S_T_PAMAP2(test user2)](https://user-images.githubusercontent.com/50646282/111409077-3c40b700-8711-11eb-9da6-1cebfa479fda.png)
 

## Attention Weight of walking
 ![user2_walking](https://user-images.githubusercontent.com/50646282/111409214-7742ea80-8711-11eb-861f-c4584c2be978.png)


## Attention Weight of rope jumping
![user2_rope jumping](https://user-images.githubusercontent.com/50646282/111409266-917cc880-8711-11eb-8f89-cbfa7386f224.png)


## Attention Weight of ironing
![user2_ironing](https://user-images.githubusercontent.com/50646282/111409532-1667e200-8712-11eb-8ecd-9f7ee993b909.png)
