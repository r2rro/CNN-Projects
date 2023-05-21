# Neural-Style-Transfer
The Neural Style Transfer algorithm created by [Gatys et al. (2015)](https://arxiv.org/abs/1508.06576).
For most of algorithm such face recognition we optimize a cost function to get a set of parameter values. With Neural Style Transfer, we'll get to optimize a cost function to get pixel values.

## 1- Problem Statment

Neural Style Transfer (NST) is one of the most fun and interesting optimization techniques in deep learning. It merges two images, namely: a "content" image (C) and a "style" image (S), to create a "generated" image (G). The generated image G combines the "content" of the image C with the "style" of image S.

In this code, we are going to combine a photo of the Science World in Vancouver,BC (content image C) with the style of Van Gogh's Starry Night Over the Rhône painting (content image S) to generate the following image:

<p align="center">
  <img width="800" src="https://github.com/r2rro/CNN-Projects/blob/main/Neural%20Style%20Transfer/images/generated.png" alt="NST-science">
</p>

Let's get started!

## 2- Transfer Learning

Neural Style Transfer (NST) uses a previously trained convolutional network, and builds on top of that. The idea of using a network trained on a different task and applying it to a new task is called transfer learning.

We will be using the the epynomously named VGG network from the [original NST paper](https://arxiv.org/abs/1508.06576) published by the Visual Geometry Group at University of Oxford in 2014. Specifically, we'll use VGG-19, a 19-layer version of the VGG network. This model has already been trained on the very large ImageNet database, and has learned to recognize a variety of low level features (at the shallower layers) and high level features (at the deeper layers).

## 3- Neural Style Transfer (NST)
Next, we will be building the Neural Style Transfer (NST) algorithm in three steps:
* First, we will build the content cost function <img src="https://render.githubusercontent.com/render/math?math=J_{content}(C,G)">
* Second, we will build the style cost function <img src="https://render.githubusercontent.com/render/math?math=J_{style}(S,G)">
* Finally, we'll put it all together to get <img src="https://render.githubusercontent.com/render/math?math=\alpha J_{content}(C,G) + \beta J_{style}(S,G) ">

#### 3.1- Computing the Content Cost

#### 3.1.1- Make Generated Image G Match the Content of Image C 
One goal we should aim for when performing NST is for the content in generated image G to match the content of image C. To do so, we'll need an understanding of shallow versus deep layers :
 * The shallower layers of a ConvNet tend to detect lower-level features such as edges and simple textures.
 * The deeper layers tend to detect higher-level features such as more complex textures and object classes.
 
 **To choose a "middle" activation layer <img src="https://render.githubusercontent.com/render/math?math=a^{[l]}"> :**
 
We need the "generated" image <img src="https://render.githubusercontent.com/render/math?math=G">
  
  to have similar content as the input image C. Suppose we have chosen some layer's activations to represent the content of an image.

**Note:** In practice, we'll get the most visually pleasing results if we choose a layer in the middle of the network--neither too shallow nor too deep. This ensures that the network detects both higher-level and lower-level features.

**To forward propagate image "C:"**
* Set the image <img src="https://render.githubusercontent.com/render/math?math=C"> as the input to the pre-trained VGG network, and run forward propagation.
* Let  <img src="https://render.githubusercontent.com/render/math?math=a^{(C)}"> be the hidden layer activations in the layer you had chosen. This will be an <img src="https://render.githubusercontent.com/render/math?math=n_H \times n_W \times n_C"> tensor.

**To forward propagate image "G":**
* Repeat this process with the image <img src="https://render.githubusercontent.com/render/math?math=G">: Set <img src="https://render.githubusercontent.com/render/math?math=G"> as the input, and run forward progation.
* Let  <img src="https://render.githubusercontent.com/render/math?math=a^{(G)}">  be the corresponding hidden layer activation.

#### 3.1.2- Content Cost Function <img src="https://render.githubusercontent.com/render/math?math=J_{content}(C,G)">
One goal we should aim for when performing NST is for the content in generated image G to match the content of image <img src="https://render.githubusercontent.com/render/math?math=C">. A method to achieve this is to calculate the content cost function, which will be defined as:
<p align = "center">
<img src="https://render.githubusercontent.com/render/math?math=J_{content}(C,G) = \frac{1}{4\times n_H \times n_W \times n_C}\sum_{\text{all entries}}(a^{(C)}-a^{(G)})^2">
</p>

* Here,  <img src="https://render.githubusercontent.com/render/math?math=n_H,n_W"> and <img src="https://render.githubusercontent.com/render/math?math=n_C">  are the height, width and number of channels of the hidden layer we have chosen, and appear in a normalization term in the cost.
* For clarity, note that  <img src="https://render.githubusercontent.com/render/math?math=a^{(C)}">   and  <img src="https://render.githubusercontent.com/render/math?math=a^{(G)}">   are the 3D volumes corresponding to a hidden layer's activations.
* In order to compute the cost  <img src="https://render.githubusercontent.com/render/math?math=J_{content}(C,G)"> , it might also be convenient to unroll these 3D volumes into a 2D matrix, as shown below.
* Technically this unrolling step isn't needed to compute  <img src="https://render.githubusercontent.com/render/math?math=J_{content}"> , but it will be good practice for when you do need to carry out a similar operation later for computing the style cost  <img src="https://render.githubusercontent.com/render/math?math=J_{style}"> .

<p align="center">
  <img width="700" src="https://github.com/r2rro/CNN-Projects/blob/main/Neural%20Style%20Transfer/images/NST_LOSS.png" alt="unrolled version">
</p>

#### 3.2- Computing the Style Cost
Let's see how you can now define a "style" cost function <img src="https://render.githubusercontent.com/render/math?math=J_{style}(S,G)">!

#### 3.2.1- Style Matrix
**Gram matrix**
* The style matrix is also called a "Gram matrix."
* In linear algebra, the Gram matrix <img src="https://render.githubusercontent.com/render/math?math=G"> of a set of vectors <img src="https://render.githubusercontent.com/render/math?math=(\nu_1, \dots, \nu_n)"> is the matrix of dot products, whose entries are  
<img src="https://render.githubusercontent.com/render/math?math=G_{ij}= \nu_{i}^{T}\nu_{j} = np.dot(\nu_{i},\nu_{j})">.

* In other words,  <img src="https://render.githubusercontent.com/render/math?math=G_{ij}">  compares how similar  <img src="https://render.githubusercontent.com/render/math?math=\nu_{i}">  is to  <img src="https://render.githubusercontent.com/render/math?math=\nu_{j}"> : If they are highly similar, you would expect them to have a large dot product, and thus for  <img src="https://render.githubusercontent.com/render/math?math=G_{ij}">  to be large.

**Compute Gram matrix <img src="https://render.githubusercontent.com/render/math?math=G_{gram}">**
We will compute the Style matrix by multiplying the "unrolled" filter matrix with its transpose:

<p align="center">
  <img width="700" src="https://github.com/r2rro/CNN-Projects/blob/main/Neural%20Style%20Transfer/images/NST_GM.png" alt="Gram matrix">
</p>

<img src="https://render.githubusercontent.com/render/math?math=G_{gram}= A_{unrolled}A_{unrolled}^{T}">

**<img src="https://render.githubusercontent.com/render/math?math=G_{(gram)ij}">: correlation**

The result is a matrix of dimension <img src="https://render.githubusercontent.com/render/math?math=(n_{C},n_{C})"> where  <img src="https://render.githubusercontent.com/render/math?math=n_{C}"> is the number of filters (channels). The value <img src="https://render.githubusercontent.com/render/math?math=G_{(gram)ij}"> measures how similar the activations of filter <img src="https://render.githubusercontent.com/render/math?math=i"> are to the activations of filter <img src="https://render.githubusercontent.com/render/math?math=j">.


<img src="https://render.githubusercontent.com/render/math?math=G_{(gram)ij}">: **prevalence of patterns or textures**

* The diagonal elements  <img src="https://render.githubusercontent.com/render/math?math=G_{(gram)ii}"> measure how "active" a filter  <img src="https://render.githubusercontent.com/render/math?math=i">  is.
* For example support filter <img src="https://render.githubusercontent.com/render/math?math=i"> is detecting vertical textures in the image. Then <img src="https://render.githubusercontent.com/render/math?math=G_{(gram)ii}"> measures how common vertical textures are in the image as a whole.
* If <img src="https://render.githubusercontent.com/render/math?math=G_{(gram)ii}"> is large, this means that the image has a lot of vertical texture.

By capturing the prevalence of different types of features (<img src="https://render.githubusercontent.com/render/math?math=G_{(gram)ii}">), as well as how much different features occur together (<img src="https://render.githubusercontent.com/render/math?math=G_{(gram)ij}">), the style matrix <img src="https://render.githubusercontent.com/render/math?math=G_{gram}"> measures the style of an image.

#### 3.2.2- Style Cost
Our next goal will be to minimize the distance between the Gram matrix of the "style" image <img src="https://render.githubusercontent.com/render/math?math=S"> and the Gram matrix of the "generated" image <img src="https://render.githubusercontent.com/render/math?math=G">.
* For now, we will use only a single hidden layer <img src="https://render.githubusercontent.com/render/math?math=a^{[l]}">.
* The corresponding style cost for this layer is defined as:

<p align = "center">
<img src="https://render.githubusercontent.com/render/math?math=J_{style}^{[l]}(S,G) = \frac{1}{4\times n_{C}^{2} (n_H \times n_W)^{2}}\sum_{i=1}^{n_{C}}\sum_{j=1}^{n_C}(G^{(S)}_{(gram)i,j}-G^{(G)}_{(gram)i,j})^2">
</p>
* <img src="https://render.githubusercontent.com/render/math?math=G^{(S)}_{gram}"> Gram matrix of the "style" image.
* <img src="https://render.githubusercontent.com/render/math?math=G_{gram}^{(G)}"> Gram matrix of the "generated" image.
* Make sure you remember that this cost is computed using the hidden layer activations for a particular hidden layer in the network <img src="https://render.githubusercontent.com/render/math?math=a^{[l]}">.

####3.3 - Defining the Total Cost to Optimize

Finally, we will create a cost function that minimizes both the style and the content cost. The formula is:

<p align = "center">
<img src="https://render.githubusercontent.com/render/math?math=J(G)=\alpha J_{content}(C,G)\dotplus \beta J_{style}(S,G)">
 </p>
 
 ## 4- Solving the Optimization Problem
 Finally, we get to put everything together to implement Neural Style Transfer!
 Here's what our program be able to do:
 
 1. Load the content image
2. Load the style image
3. Randomly initialize the image to be generated
4. Load the VGG19 model
5. Compute the content cost
6. Compute the style cost
7. Compute the total cost
8. Define the optimizer and learning rate
