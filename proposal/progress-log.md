# Thesis Progress Log

## Week 8 (10/29)

- I've gotten the differentiable renderer setup to work on my setup and am now ready to include it in my pipeline to get texture applications to work properly. In the current setup, I have a virtual environment called `dray` that I have the differentiable renderer installed on. This puts me at two important dev environments in my CSAIL afs locker: `dray` and `thesis`. Moving forward I'll lean towards using dray in most occasions and eventually phase out the `thesis` environment out.
- I had to change some PYTHONPATH and CC env variables in order to get everything to work out correctly. If any related issues arise later on, I should refer to this note to see how to fix this.

## Week 7 (10/22)

### Training Notes (Abridged)

- I've been struggling to get the Inpainting task to perform correctly as there seems to be some bug in my system preventing me from getting the optimization to actually work correctly.
- I've had a meeting with Fredo and it seems that this inpainting task isn't as crucial to my work as I've been understanding and that I should focus in on putting the differentiable renderer into my system to be the priority.
- I've fixed some bugs from the Inpainting model but there still seems to be some issues with the model so I've relegated this task to be more low priority. Will address this later on


## Week 6 (10/15)

### Training Notes (Abridged)

- After trying most combinations of changes I can make to the SGAN model while also trying to follow some of the hyperparameters of the SGAN paper, I've fully given up on this line of thinking
- Instead I will attempt to change the DCGAN setup step by step to get it to work on latent tensors that are larger than 1x1 in its spatial dimensions
- I've successfully gotten the DCGAN model to morph into the SGAN model where I can feed it spatially large (9x9) latent tensors and get realistic looking textures. However after 15k ish iterations there seems to be some issues with the training that makes it regress to a worse state for some reason. More investigation on this for later
- I've asked Prafull on tips on next steps. The new task is to implement an inpainting model based on the paper by Raymond Yeh, 2018 (see reading directory for link to paper).

## Week 5 (10/08)

### Training Notes (Abridged)

- I've managed to get the basic DCGAN model to converge to a realistic looking texture that's 64x64 in size
- The next step would be to figure out what the problem with the SGAN model seems to be and see if I can get the SGAN model to work.

## Week 4 (10/01)

### Training Notes

- Unable to replicate good results from the SGAN model. Likely due to dubious results of the images they produced in the paper
- Before I can proceed forward with any model, I need to figure out the most simple way to generate textures.
- This got relegated to me trying to implement the DCGAN paper instead.
- One way to check if a GAN network architecture is working on a base level from an architectural standpoint is to flip the generator and discriminator's input output pairs and turn the GAN into an AutoEncoder. This allows us to assess the input latent space's ability to generate realistic samples from within the latent space.

## Week 3 (09/24)

### Training Notes

- The SGAN paper uses 'same' padding (meaning `filter_size // 2`). Don't let the paper fool you with their "zero" padding. That just means they pad their inputs with 0s.
- Theano/Lasagne's Transpose Convolutional layer output padding is not configurable, hence the SGAN authors had to finnagle with their code to make the input output texture sizes to match up correctly
- Transpose Convolutional layers end up having weird rounding errors in their output size depening on whether the output padding is odd or even. The `output_padding` argument in the `Conv2dTranspose` model in pytorch fixes this issue. Make sure to align the convolution layer's spatial sizes before going into any trainining
- The `Print(nn.Model)` model will be very useful for debugging the state of the neural net. Moved this into its own library called `debug.py`
- zero-padding has proven to be bad for training. Not having padding / reflective padding makes training slightly better. But regardless the base implementation of the SGAN paper so far has been very lackluster with no good results
- The spatial extent for the images seem to be too large to learn maybe. Next step would be to try with smaller patch size

### Data collection

[**Textures.com**](https://www.textures.com/)
- This website has a lot of 3D scanned textures I can use for training

[**Poliigon.com**](https://www.poliigon.com/)
- This is where Nish got his road textures from

[**Describable Textures Dataset: M. Cimpoi, 2014**](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
- This is a more academically prevalent dataset for texture learning tasks

### Papers Skimmed

[**Deconvolution and Checkerboard Artifacts: Augustus Odena, 2016**](https://distill.pub/2016/deconv-checkerboard/)
- Discussion on how Transposed Convolutional layers work and its limitations

[**Non-parametric Sampling: Alexei Efros, 1999**]( https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/papers/efros-iccv99.pdf)
- Really I just need this for my bibliography probably
- Very old paper that doesn't use any learning for texture generation

[**Patch based Quilting: Alexei Efros, 2001**](https://people.eecs.berkeley.edu/~efros/research/quilting/quilting.pdf)
- Same as above

[**Learning Texture Manifolds with the Periodic Spatial GAN, Bergmann, 2017**](https://arxiv.org/pdf/1705.06566.pdf)
- This paper is an improvement on the architecture proposed by SGAN
- Implements spatial non-ergodicity features to the input noise tensor to improve spatial translation changes to the image

## Week 2 (09/17)

### Papers Skimmed

**Revisited the SGAN paper**
- This paper is now quite very straightforward to understand
- This paper implements the DCGAN with a grid sized noise input instead of one 100-dim vector. This gives the result translation invariance as well as ergodicity

**Revisited the DCGAN paper**
- I think I understand most of the contributions by Radford.
- The broad gist is that there really isn't much new nor rigorous. They just eventually finagled with the architecture enough to learn that:
    - Strided Convolutions and Transposed Convolutions are very goo
    - Batch Normalization is also very good
    - Remove Fully Connected Layers
    - ReLU and Leaky ReLU for activations

[**TextureNet Hotfix: Ulyanov, 2016**](https://arxiv.org/pdf/1607.08022.pdf)
- Lazy paper that just changes a batch normalization layer to a instance normalization one
- Good resource to see how `InstanceNorm` works

[**TextureNet: Ulyanov, 2016**](http://proceedings.mlr.press/v48/ulyanov16.pdf)
- This method simply trains a new network based on the loss function proposed by Gatys

[**Style Transfer using CNN: Gatys, 2016**](https://arxiv.org/pdf/1505.07376.pdf)
- Same method for texture extraction
- A very similar loss function applied for extracting the content represenation of an image from VGG

[**Texture synthesis using CNN: Gatys, 2015**](https://arxiv.org/pdf/1505.07376.pdf)
- Uses VGG trained filters to extract features from textures
- These extracted features are used to optimize a newly generated feature

[**Unet: Olaf Ronneberger, 2015**](https://arxiv.org/pdf/1505.04597.pdf)
- Didn't read much of this
- Describes the Unet architecture used in the pix2pix paper


[**LAPGAN: Emily Denton, 2015**](https://arxiv.org/pdf/1506.05751.pdf)
- Earlier mention of the iterative resolution improvement of images using GANs is presented here
- Results are not very impressive

[**A Note on Inception Score**](https://arxiv.org/pdf/1801.01973.pdf)
- Did not really read this in detail. Felt like a lot of theory on GAN convergence that's not really relevant to my research

[**Wasserstein GAN**](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html)
- Very theoretical. I don't think I really get it :(

## Week 1 (09/10)

### Pytorch notes

[**GAN tutorial using pytorch on MNIST**](https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f)

- Mostly faithful implementation of the Original Paper on GAN
- NOTE: Has a lot of links for further reading as required
- Prafull gave me a link which has a similar tutorial using Keras. Dubious whether this is more useful than the post above [link](https://skymind.ai/wiki/generative-adversarial-network-gan)

More notes on specific pytorch functions:

`ReflectionPad2d` -- padding function that uses reflection on the edges
`InstanceNorm2d` -- applies batch normalization on an input
`ConvTranspose2d` -- applies a fractional strided convolution on an input

**Spatial GAN Pytorch implementation notes**

- this seems actually quite straightforward to implement things
- once i implement a couple more things within existing codebases, i'll make sure to start from scratch and develop a very good robust pipeline for myself

### CycleGAN Deep Dive

[**Cycle GANs paper**](https://arxiv.org/pdf/1703.10593.pdf)

The CycleGAN paper's GAN architecture uses the Generator structure referenced in [Perceptual Losses for Real-Time Style Transfer
and Super-Resolution](https://arxiv.org/pdf/1603.08155.pdf).

Notes on this paper:
- Reading this paper in order to understand the ResNet structure
- Note about not using pooling layers and using strided and fractionally strided convolutions
- NOTE: Only notes related to residual networks were skimmed

From here I was referred to [Training and investigating Residual Networks](http://torch.ch/blog/2016/02/04/resnets.html).

- Residual connections simply mean that in a residual block, the input gets a shortcut connection to the output of the block (adding it to the convolution layer's output)
- The paper discusses about a "batch normalization" layer. More on this described [here](https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c). The gist is that it normalizes the outputs of any layer for faster training.

Going back to the CycleGAN paper we see that the Discriminator structure is an implementation of the PatchGAN's discriminator as referenced by the pix2pix paper. The general gist of the discriminator is the fact that we only consider small NxN patches of an image to determine whether it is real or fake. Once we collect votes from across the whole image convolutionally, we take the majority vote to determine whether the image is real or fake. The paper then describes how this is in a way a form of a Markov Random Field since every decision is made from patches that are N pixel away.

The CycleGAN pytorch implementation looks pretty legit. I understand most of the details that are implemented in the code.

### Papers skimmed

[**Temporal GAN**](https://github.com/pfnet-research/tgan)
- Skimmed the first part relating to network architecture
- Got a little lost as I was trying to understand their adjustments to the Wasserstein GAN model

[**pix2pixhd**](https://arxiv.org/pdf/1711.11585.pdf)
- [Github](https://github.com/NVIDIA/pix2pixHD)
- This paper adds a LOT of improvements to the original pix2pix architecture
- Very good paper to look at the source code for

[**Non-Stationary Texture Synthesis by Adversarial Expansion**](http://vcc.szu.edu.cn/research/2018/TexSyn)
- This paper proposes a really OP method of synthesizing textures and enlarging an image

[**Texture Synthesis with Spatial Generative Adversarial Networks**](https://arxiv.org/pdf/1611.08207.pdf)
- The texture synthesis paper describes how for the most part, the structure of the network is very similar to that of DCGAN. The big difference is between the size of the input.
- It has a proof of how the output texture is able to be applied on an output of arbitrary size
- Implemented the training script in Pytorch

[**DiscoGAN paper**](https://arxiv.org/pdf/1703.05192.pdf)
- Both these papers have very similar motivation and implementation on broad strokes. Details on the differences can be found [here](https://www.quora.com/What-is-the-difference-between-CycleGAN-and-DiscoGAN-They-both-seem-to-be-the-same-thing#)

[**pix2pix paper**](https://arxiv.org/pdf/1611.07004.pdf)
- This paper proposes a conditioned generator where a generated output is created when provided an input image
- A lot of these papers refer to "PatchGAN"

[**Cycle GANs paper**](https://arxiv.org/pdf/1703.10593.pdf)
- This paper's related work section has quite a lot of good resources to go through
- This could be used to train our network when the mapping of different latent spaces isn't paired

[**Progressive growing of GANs for improved quality, stability, and variation**](https://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of/karras2018iclr-paper.pdf)

## Week 0 (09/05)

### Papers Skimmed

[**Deep Convolutional GANs**](https://arxiv.org/pdf/1511.06434.pdf)

- CNNs applied for GANs were not very successful
- The DCGAN has found some techniques that make it successful
    - Replace pooling layers with strided convolutions
    - Use batchnorm on input
    - Remove fully connected hidden layers
    - Use ReLU for all layers except output where we use Tanh
    - Use Leaky ReLU activation in the disciminator for all layers
- [Pytorch code demo](https://github.com/pytorch/examples/tree/master/dcgan)

[**Original Paper on GAN**](https://arxiv.org/pdf/1406.2661.pdf)

- The adverserial network architecture has a lot of good pragmatic uses and is very computationally favorable over other methods
- Training the $D$ and the $G$ close together is very important.
    - This usually entails in training $D$ for $k$ steps followed by 1 step of training $G$