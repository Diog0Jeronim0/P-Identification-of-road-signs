# Identification-of-road-signs
 Identification of road signs using Python + Roboflow with yolov8

 In this project I made a model in roboflow with more than 1000 images of identified European road signs in which I exported the model in yolov8, then trained it on my machine, then validated the model by obtaining representative graphics and finally tested the model in a video which I recorded from street road signs.
To stop this challenge I had to install yolov8 and ultralytics in the computer's cmd

This project contains several steps to complete successfully. The first thing we have to do is get several images of different signs on the street with different approaches and angles, around 20 different images per sign, so that the model is more accurate and there are no errors.
The second step we have to complete is to upload all the images to roboflow and annotate them with the corresponding caption, and then create a version of the model and export it to the machine. In the third step we have to train the model on the machine, using the file {data.yaml}, for this we use code 1 to do this, to allow the model to recognize specific road signs. In the fourth step we have to validate the model using code 2 on cmd, to ensure that it generalizes well to new data and not just to the training data set. This is a very important step in model development as it allows you to evaluate the model performance objectively and also adjust the parameters to improve effectiveness. In the fifth step, we have to save the model in the original folder, giving it a name of our choice, using code 3. Finally, we test the model on the video of road signs that we recorded on the street, putting in the code only the numbers corresponding to each video signal, using code 4.

NOTE: The captions of the signals are in portuguese because the signals are from mainland Portugal.
