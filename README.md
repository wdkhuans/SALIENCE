# SALIENCE

An Unsupervised User Adaptation Model for Multiple  Wearable Sensors Based Human Activity Recognition

Unsupervised user adaptation aligns the feature distributions of the data from training users and the new user, so a well-trained wearable human activity recognition (WHAR) model can be well adapted to the new user. With the development of wearable sensors, multiple wearable sensors based WHAR is gaining more and more attention. In order to address the challenge that the transferabilities of different sensors are different, we propose SALIENCE (unsupervised user adaptation model for multiple wearable sensors based human activity recognition) model. It aligns the data of each sensor separately to achieve local alignment, while uniformly aligning the data of all sensors to ensure global alignment. In addition, an attention mechanism is proposed to focus the activity classifier of SALIENCE on the sensors with strong feature discrimination and well distribution alignment. Experiments are conducted on two public WHAR datasets, and the experimental results show that our model can yield a competitive performance.

## The framework of SALIENCE
<img src="https://user-images.githubusercontent.com/50646282/128142470-8e833ad6-9b7d-44b0-b1bf-96702ca4f186.jpg" style="zoom:25%;" />
