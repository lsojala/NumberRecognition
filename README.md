# NumberRecognition

Simple AI to recognize numbers drawn by the user.  
AI output is interpreted and translated into a short sentence:

![Screenshot1](/screenshots/AI-recg-1.png) 

AI uses two layer neural network (392 nodes and 10 nodes) to categorize images supplied by the user into one of ten categories.
AI is pre-trained using the canonical MNIST dataset of handwritten digits. The dataset contain images for all ten digits 0-9, scaled to same size and centered. This is reflected in AI's capability to recognize drawn number. If the number is drawn large, small or off-centered, it is most likely to be unrecognized or identified wrong.

When checkbox "Show reply details" is checked, the details of the AI output can be seen. The ten numbers on top row is the AI output of "score" of each category. For example, during training AI is taugth that all number 7s correspond to vector [0,0,0,0,0,0,0,100,0,0] (zero is the first one). Output where the highest category scores close to 100, is deemed certain. As the score achieved by the top most category decreases the AI is deemed more and more uncertain, until no category reaches 25 score, at which point no assignation to a number is made.

| ![Screenshot2](/screenshots/AI-recg-2.png) | ![Screenshot3](/screenshots/AI-recg-3.png)               |
|---------------------------------------------- | ----------------------------------------------------- |
|AI output is scored high on specific digit     | AI output with lower scores on a digit                |  


### Requirements:
Python 3.x  
Pillow==8.1.0
tensorflow-cpu==2.4.1
numpy==1.19.5
mnist==0.2.2




