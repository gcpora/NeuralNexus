# Build a model that identifies damaged parts from car photos and predicts the estimated repair cost based on damage severity and car model metadata

**Project work as part of IIT Delhi ML and DL course - Batch 7**

Dataset : https://www.kaggle.com/datasets/nasimetemadi/car-damage-detection

Input - 
  Car Images - car photo
	Tabular data - Car model metadata

Output - Estimated repair cost

Hint : Use a pre-trained ResNet50 or VGG16 (Transfer Learning) to extract image features, then concatenate these with tabular car data (year, make, model) to feed into an XGBoost regressor. 

**Models**

**ResNet50 (Residual Network)**
ResNet is a convolutional neural network that democratized the concepts of residual learning and skip connections. 
residual learning framework 
50 refers to number of layers - 48 convolutional layers, 1 MaxPool, 1 Fully connected.
Composed of one initial convolution layer, followed by a series of Residual Blocks
ResNet model pre-trained on ImageNet-1k at resolution 224x224. This means input size will be 224x224 pixels.
Skip connection
Helps vanishing gradients


Why this model?
Transfer learning - ResNet50 was trained on over a million images (ImageNet), it already "understands" what a car looks like. You don't need to train it from scratch; you only need to "fine-tune" the last few layers to recognize specific types of damage.
Because it is a residual network, it is good in detecting anomalies - things that should not be there, like a dent, without losing track of - what a normal car looks like. Ex: the model knows what a car door looks like, if the photo has a weird shape, it identifies it as dent.

Why not other convolutional models?

Why not VGG16?
Has 16 layers but 138 million parameters, while ResNet50 has 25.6 million parameters.
Slower to learn, slower to execute and takes more memory.

**XGBoost - eXtreme Gradient Boosting**
Boosting - In boosting we train several small and simple models (tree).
Machine learning algorithm to process tabular data.
Based on decision tree

