**Project work as part of IIT Delhi ML and DL course - Batch 7**

# Build a model to identify damaged parts from car photos and predicts the estimated repair cost based on damage severity and car model metadata

**Dataset Used: **https://www.kaggle.com/datasets/nasimetemadi/car-damage-detection

The dataset consists of labeled images of various car damage scenarios, intended to train machine learning models for automatic detection and assessment of car damage. Dataset includes a diverse range of car types and damage patterns, making it suitable for this scenario.

Input - 
  Car Images - 
      COCO Directory - Coco is an annotation format. It pairs images with a JSON file describing:
	  	1. What objects are in the image (categories: dent, scratch etc.)
		2. Where they are (bounding boxes or polygon outlines)
		3. Severity levels
		In summary the hwlp is finding out "What is damaged and how bad?"
	    COCO/
			images/
			      0001.jpg
				  0002.jpg
			annotations.json  <--- Annotation file tells us about the damage. Ex: dent is near the bumper.

	  SOD Directory - SOD stands for Salient Object Detection. Instead of a JSON file, it gives us a binary mask per image - a black/white  
	  image where white stands for damaged area while black stands for background. This is spatial guide. It answers "where in the image is 
	  the damage?
	      SOD/
		      images/
			         0001.jpg
					 0002.jpg
			  masks/
			         0001.jpg
					 0002.jpg
	  
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

