"""Morphs the image"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import transform as tf
from skimage.transform import ProjectiveTransform
from coordinate_solver import CoordinateSolver
import matplotlib.path as path

# image = data.text()

class ImageUniformer(object):
  """docstring for ImageUniformer"""
  def __init__(self, image, resolution=16, density=None):
    super(ImageUniformer, self).__init__()
    self.image = image
    self.coord_solver = CoordinateSolver(self.image, resolution, density)
    print "converging image..."
    self.coord_solver.converge()
    self.new_image = None

  def prep_image(self):
    """Takes the solved coordinate system and makes a piecewise \
    transform on the origin image to the target image"""
    transform = ProjectiveTransform()
    self.coord_solver.coordinates = self.coord_solver.min_coords.copy()
    self.new_image = np.zeros(self.coord_solver.image.shape)
    coords = np.array([self.coord_solver.coordinates[x:x+2, y:y+2, :].reshape([4, 2]) for x in \
      range(self.coord_solver.coordinates.shape[0]) for y in range(self.coord_solver.coordinates.shape[1]) \
      if (self.coord_solver.coordinates[x:x+2, y:y+2, :].shape == (2, 2, 2))])
    canonical_coords = np.indices((self.coord_solver.width, self.coord_solver.height)).T.astype('float32')
    flattened_canonical = np.array([canonical_coords[x:x+2, y:y+2, :].reshape([4, 2]) for x in \
      range(canonical_coords.shape[0]-1) for y in range(canonical_coords.shape[1]-1)])
    mesh_size = self.coord_solver.mesh_factor
    print "needs %s calcs" % coords.shape[0]
    coord_grid = np.indices(self.coord_solver.image.shape[:-1]).T.astype('float32').reshape(-1,2)
    for k in range(coords.shape[0]):
      des = mesh_size*coords[k, :, :]
      canon_coord = mesh_size*flattened_canonical[k, :, :]
      src = mesh_size*flattened_canonical[0, :, :]
      if not transform.estimate(des, canon_coord):
        raise Exception("estimate failed at %s" % str(k))
      area_in_question_x = canon_coord[0, 0].astype(int)
      area_in_question_y = canon_coord[0, 1].astype(int)
      scaled_area = tf.warp(self.coord_solver.image, transform)
      area_path = path.Path([des[0],des[1],des[3],des[2],des[0]])
      points_in_area = area_path.contains_points(coord_grid,radius=0.00001).reshape(self.coord_solver.image.shape[:-1])
      self.new_image += scaled_area*points_in_area[:,:,np.newaxis]

  def plot(self):
    """Plots the new image"""
    plt.imshow(self.new_image)
    plt.show()