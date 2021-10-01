# Reading Directory

## GAN Architectures
[*GAN: Ian Goodfellow, 2014*](https://arxiv.org/pdf/1406.2661.pdf) -- Canon Paper
[*LAPGAN: Emily Denton, 2015*](https://arxiv.org/pdf/1506.05751.pdf) -- Laplacian Pyramid Idea
[*DCGAN: Alec Radford, 2016*](https://arxiv.org/pdf/1511.06434.pdf) -- CNNs in GAN
[*TGAN: Masaki Saito, 2016*](https://arxiv.org/pdf/1611.06624.pdf) -- Add temporal dimension
[*pix2pix: Philip Isola, 2016*](https://arxiv.org/pdf/1611.07004.pdf) -- Conditional GAN OG
[*DiscoGAN: Takesoo Kim, 2017*](https://arxiv.org/pdf/1703.05192.pdf) -- Improvement on pix2pix
[*CycleGAN: Jun-Yan Zhu, 2017*](https://arxiv.org/pdf/1703.10593.pdf) -- Really big improvement on pix2pix
[*pix2pixhd: Ting-Chung Wang, 2017*](https://arxiv.org/pdf/1711.11585.pdf) -- Upgrade on top of pix2pix
[*Progressive GAN: Tero Karras, 2018*](https://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of/karras2018iclr-paper.pdf) -- More on Iterative resolution enhancement

## Theoretical Discussion related to GAN training
[*Deconvolution and Checkerboard Artifacts: Augustus Odena, 2016*](https://distill.pub/2016/deconv-checkerboard/) -- Checkerboard artifacts
[*Making GANs easier to train: Tim Salimans, 2016*](https://arxiv.org/pdf/1606.03498.pdf) -- Tips for GAN performance improvements
[*From GAN to WGAN (Blog): Lillian Weng, 2017*](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html) -- Theoretical basis on why to WGAN
[*A Note on the Inception Score: Shane Barratt, 2018*](https://arxiv.org/pdf/1801.01973.pdf) -- Evaluating GAN models

## Texture Synthesis Specific Models
[*Non-parametric Sampling: Alexei Efros, 1999*]( https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/papers/efros-iccv99.pdf) -- Original idea for MRF
[*Patch based Quilting: Alexei Efros, 2001*](https://people.eecs.berkeley.edu/~efros/research/quilting/quilting.pdf) -- Non-parametric Patch Based Quilting
[*Unet: Olaf Ronneberger, 2015*](https://arxiv.org/pdf/1505.04597.pdf) -- Encoder Decoder structure with residuals piped to decoder
[*Texture CNN: Leon Gatys, 2015*](https://arxiv.org/pdf/1505.07376.pdf) -- Gram matrices to remove spatial context
[*Style Transfer CNN: Leon Gatys, 2016*](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) -- Use texture CNN for style transfer
[*TextureNet: Dmitry Ulyanov, 2016*](http://proceedings.mlr.press/v48/ulyanov16.pdf) -- Gatys + train a new network
[*TextureNet Hotfix: Dmitry Ulyanov, 2016*](https://arxiv.org/pdf/1607.08022.pdf) -- Lazy paper about a hotfix to TextureNet
[*Realtime Style Transfer: Justin Johnson, 2016*](https://arxiv.org/pdf/1603.08155.pdf) -- Gatys + (DCGAN + residual)
[*SGAN: Nikolay Jetchev, 2016*](https://arxiv.org/pdf/1611.08207.pdf) -- Exacly the problem I'm trying to solve
[*MRFs + CNNs for Style Transfer: Chuan Li, 2016*](https://arxiv.org/pdf/1601.04589.pdf) -- Method utilizes MRFs
[*MRFs + GAN for Style Transfer: Chuan Li, 2016*](https://arxiv.org/pdf/1604.04382.pdf) -- Extends MRF model to train a GAN
[*PSGAN: Urs Bergmann, 2017*](https://arxiv.org/pdf/1705.06566.pdf) -- Improvement on SGANs
[*Deep Corellations: Omry Sendrik, 2017*](https://www.cs.tau.ac.il/~dcor/articles/2017/Deep.Correlations.pdf) -- Extension and improvement upon Gatys method
[*Non-stationary textures: Yang Zhou, 2018*](https://arxiv.org/pdf/1805.04487.pdf) -- Best paper on texture synthesis so far
[*Semantic Image Inpainting: Raymond Yeh, 2018*](https://arxiv.org/pdf/1607.07539.pdf) -- Optimization based. [Repo](https://github.com/moodoki/semantic_image_inpainting)
*Differentiable Rendering: Tzu-Mao Li, 2018* -- A differentiable renderer

## Datasets
[*Describable Textures Dataset: M. Cimpoi, 2014*](https://www.robots.ox.ac.uk/~vgg/data/dtd/) -- Oxford DTD dataset
[*Textures.com*](https://www.textures.com/) -- 3D Scanned images
[*Poliigon.com*](https://www.poliigon.com/) -- Nish's road textures source