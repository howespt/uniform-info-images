import numpy as np

class ImageObfuscator(object):
  """docstring for ImageObfuscator"""
  def __init__(self, image, window_size=16, stride=None):
    super(ImageObfuscator, self).__init__()
    self.image = image
    self.window_size = window_size
    self.stride = window_size

  def create_images(self):
    """assume image has shape (3,x,y)"""
    image = self.image
    block = 0*np.ones([3, self.window_size, self.window_size])
    _, height, width = image.shape
    ims = []
    for r in range(0,height,self.stride):
      for c in range(0,width,self.stride):
        im_copy = image.copy()
        im_copy[:, r:r+self.window_size, c:c+self.window_size] = block
        ims.append(im_copy)
    self.images = np.array(ims)
    