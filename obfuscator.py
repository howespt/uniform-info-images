
class ImageObfuscator(object):
  """docstring for ImageObfuscator"""
  def __init__(self, image, window_size=5, stride=5):
    super(ImageObfuscator, self).__init__()
    self.image = image
    self.window_size = window_size

  def create_images(self):
    """assume image has shape (x,y,3)"""
    image = self.image
    block = 255*np.ones([self.window_size, self.window_size, 3])
    height, width, _ = image.shape
    ims = []
    for r in range(0,height-self.window_size,self.stride):
      for c in range(0,width - self.window_size,self.stride):
        im_copy = image.copy()
        im_copy[r:r+self.window_size, c:c+self.window_size] = block
        ims.append(im_copy)
    