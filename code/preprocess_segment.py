import os
import tarfile

import numpy as np
from skimage.transform import resize

import tensorflow as tf

from util.image_util import imread, imwrite
from util.transform_util import Resize


# Model Downloads at:
# _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
# _MODEL_URLS = {
#     'mobilenetv2_coco_voctrainaug': 'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
#     'mobilenetv2_coco_voctrainval': 'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
#     'xception_coco_voctrainaug': 'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
#     'xception_coco_voctrainval': 'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
# }


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""
  
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'
  
    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
    
        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break
    
        tar_file.close()
    
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')
    
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
    
        self.sess = tf.Session(graph=self.graph)
  
    def run(self, input):
      """Runs inference on a single image.
  
      Args:
          input: A numpy array, raw input image (assumes square)
  
      Returns:
          seg_map: Segmentation map of `input`.
      """
      #assert(input.shape[0] == input.shape[1])
      loadSize = input.shape[0]
      ToLoadSize = Resize(loadSize, order=0)
      ToFineSize = Resize(self.INPUT_SIZE)
      resized_input = ToFineSize(input)
      batch_seg_map = self.sess.run(
          self.OUTPUT_TENSOR_NAME,
          feed_dict={self.INPUT_TENSOR_NAME: [resized_input]})
      output = batch_seg_map[0]
      output[output != 7] = 0
      # Need to cast to float32 before resizing
      output = ToLoadSize(output.astype(np.float32))
      return output

model_path = ('./datasets/models/deeplab_model.tar.gz')
model = DeepLabModel(model_path)

dataroot = './datasets/tiam/'
resolution = 512
raw_dir = os.path.join(dataroot, 'raw')
img_dir = os.path.join(dataroot, 'img')
mask_dir = os.path.join(dataroot, 'mask')

if not os.path.exists(img_dir):
    os.makedirs(img_dir)
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

resize = Resize(resolution)
fn_list = os.listdir(raw_dir)
for fn in fn_list:
    # Fetch file metadata
    fn = fn.split('.')[0]
    print('Processing: {}'.format(fn), end='\t')
    raw_path = os.path.join(raw_dir, '{}.JPG'.format(fn))
    img_path = os.path.join(img_dir, '{}.png'.format(fn))
    mask_path = os.path.join(mask_dir, '{}.png'.format(fn))

    img = resize(imread(raw_path)) * 255.0
    img = np.rot90(img, -1)
    mask = model.run(img)

    print('Saving mask')
    imwrite(img / 255.0, img_path)
    imwrite(mask, mask_path)
