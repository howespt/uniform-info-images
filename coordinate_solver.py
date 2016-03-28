"""Something"""

import numpy as np
from scipy.stats import norm

def calculate_areas(coordinate_array):
  """Will calculate the area as given by the norm of the cross product of the \
  diagonal vectors of a quadrilateral, does it for a grid of quads"""
  area = 0.5*((coordinate_array[:-1, :-1, 0] - coordinate_array[1:, 1:, 0])\
          *(coordinate_array[:-1, 1:, 1] - coordinate_array[1:, :-1, 1])\
          - (coordinate_array[:-1, :-1, 1] - coordinate_array[1:, 1:, 1])\
          *(coordinate_array[:-1, 1:, 0] - coordinate_array[1:, :-1, 0]))
  return np.abs(area)

class CoordinateSolver(object):
  """docstring for CoordinateSolver"""
  def __init__(self, image=None, mesh_factor=14, density_distribution=None):
    super(CoordinateSolver, self).__init__()
    self.image = image
    self.height, self.width, _ = self.image.shape
    self.mesh_factor = mesh_factor
    self.height /= self.mesh_factor
    self.width /= self.mesh_factor
    self.image = self.image[:self.mesh_factor*self.height, :self.mesh_factor*self.width]
    if type(density_distribution) == np.ndarray:
      restricted_density = density_distribution[:self.mesh_factor*self.height, :self.mesh_factor*self.width]
      target_areas = restricted_density
      target_areas = target_areas[:-1, :-1]
    else:
      target_areas = np.indices((self.width-1, self.height-1)).T.astype('float32')
      target_areas = norm.pdf(target_areas[:, :, 0], self.width/2, self.width/5)\
                    *norm.pdf(target_areas[:, :, 1], self.height/2, self.height/5)
    target_areas /= sum(sum(target_areas))
    
    normalisation_factor = (self.height-1)*(self.width-1)
    target_areas_normalised = target_areas*normalisation_factor
    self.padded_targets = np.zeros([self.height+1, self.width+1])
    self.padded_targets[1:-1, 1:-1] = target_areas_normalised
    self.coordinates = np.indices((self.width, self.height)).T.astype('float32')
    self.total_error = (self.height-1)*(self.width-1)
    
    self.min_coords = self.coordinates.copy()
    self.areas = calculate_areas(self.coordinates)
    self.errors = np.zeros(self.padded_targets.shape)
    self.x_weights = np.ones([self.height*self.width, self.height + 1, self.width + 1])
    self.y_weights = np.ones([self.height*self.width, self.height + 1, self.width + 1])
    self.make_weights()

  def make_weights(self):
    www = np.indices((self.width+2, self.height+2)).T.astype('float32') - 1
    for ii in range(self.height):
      for jj in range(self.width):
        qq = [jj,ii] - www
        dist = np.abs(qq[:,:,0])*np.abs(qq[:,:,1])
        xx = np.sign(qq[:,:,0])*dist*dist
        yy = np.sign(qq[:,:,1])*dist*dist
        xw = 1/xx[xx!=0].reshape(-1, xx.shape[1] - 1)
        positive_xw = xw[xw>0].sum()
        negative_xw = -xw[xw<0].sum()
        xw = (xw>0)*xw*negative_xw + (xw<0)*xw*positive_xw
        yw = 1/yy[yy!=0].reshape(-1, yy.shape[0] - 1)
        positive_yw = yw[yw>0].sum()
        negative_yw = -yw[yw<0].sum()
        yw = (yw>0)*yw*negative_yw + (yw<0)*yw*positive_xw
        self.x_weights[ii*self.width + jj] = xw
        self.y_weights[ii*self.width + jj] = yw

  def converge(self):
    """Converge the mesh onto the target grid of areas by \
    minimising squared diffs in area

    updating mechanism is:
      - calc delta in area and target area
      - if delta is +, quad has outward pressure uniformly
      - if delta is -, quad has inward uniform pressure
      - pressures are summed for each vertex (corner of quad)
      - vertices move at epsilon*(net pressure)
      - if vertices become unordered or move outside \
        of boundary -> reposition iteratively until not the case

    *edge cases: sides can't expand outside grid -> pressure is doubled in
                 opposite direction.

    this seems to work in finding good minima.

    Interrupt when you want it to stop or if it fell out of the minima.
    """
    k = 0
    mse = 10**10
    min_total_error = 10**10
    base_eps = 0.0001
    try:
      while mse > 0.01:
        old_error = self.total_error
        eps = min(base_eps, min(0.01, 0.00001*self.total_error))
        self.areas = calculate_areas(self.coordinates)
        padded_areas = np.zeros([self.height+1, self.width+1])
        padded_areas[1:-1, 1:-1] = self.areas.copy()
        self.errors = self.padded_targets - padded_areas
        self.errors = self.errors*self.errors*np.sign(self.errors)
        self.total_error = sum(sum(np.abs(self.errors)))
        mse = self.total_error/(self.height-1)/(self.width-1)
        if self.total_error > min_total_error:
          self.total_error = min_total_error
          print "reached local min!"
          print "total squared error: ", self.total_error
          print "mse: ", min_total_error/(self.height-1)/(self.width-1)
          break

        if self.total_error < min_total_error:
          min_total_error = self.total_error
          self.min_coords = self.coordinates.copy()
        else:
          base_eps *=0.5

        if k % 200 == 0:
          print "total squared error: ", self.total_error
          print "mse: ", min_total_error/(self.height-1)/(self.width-1)

        if self.total_error == old_error:
          print "error is same, attempting to get unstuck!"
          self.coordinates += 0.1*np.random.normal(0, 0.1, [self.width, self.height, 2])

        self.update_x(eps)
        self.update_y(eps)

        k += 1
    except KeyboardInterrupt:
      self.update_x(eps)
      self.update_y(eps)
      

  def update_x(self, eps):
    """Updates x for vertices"""
    x_errors = self.errors*self.x_weights
    x_adjustments = eps*x_errors.sum(axis=1).sum(axis=1).reshape(self.height, self.width)
    self.coordinates[:, :, 0] += x_adjustments[:, :]
    x_lattice_differences = self.coordinates[:, 1:, 0] - self.coordinates[:, :-1, 0]
    min_xld = x_lattice_differences.min()
    forward_or_back = np.random.choice([-1, 1])
    while min_xld < 0:
      if forward_or_back == 1:
        self.coordinates[:, 1:, 0] -= np.minimum(x_lattice_differences[:, :], 0)
      else:
        self.coordinates[:, :-1, 0] += np.minimum(x_lattice_differences[:, :], 0)
      x_lattice_differences = self.coordinates[:, 1:, 0] - self.coordinates[:, :-1, 0]
      min_xld = x_lattice_differences.min()
    self.coordinates[:, 0, 0] = 0
    self.coordinates[:, -1, 0] = self.width-1
    self.coordinates[:, :, 0] = (self.coordinates[:, :, 0] >= 0)*self.coordinates[:, :, 0]
    self.coordinates[:, :, 0] = (self.coordinates[:, :, 0] <= self.width-1)*self.coordinates[:, :, 0] \
      + (self.width-1)*np.ones([self.height, self.width])*(self.coordinates[:, :, 0] > self.width-1)

  def update_y(self, eps):
    """Updates y for vertices"""
    y_errors = self.errors*self.y_weights
    y_adjustments = eps*y_errors.sum(axis=1).sum(axis=1).reshape(self.height, self.width)
    self.coordinates[:, :, 1] += y_adjustments[:, :]
    y_lattice_differences = self.coordinates[1:, :, 1] - self.coordinates[:-1, :, 1]
    min_yld = y_lattice_differences.min()
    forward_or_back = np.random.choice([-1, 1])
    while min_yld < 0:
      if forward_or_back == 1:
        self.coordinates[1:, :, 1] -= np.minimum(y_lattice_differences[:, :], 0)
      else:
        self.coordinates[:-1, :, 1] += np.minimum(y_lattice_differences[:, :], 0)
      y_lattice_differences = self.coordinates[1:, :, 1] - self.coordinates[:-1, :, 1]
      min_yld = y_lattice_differences.min()
    self.coordinates[0, :, 1] = 0
    self.coordinates[-1, :, 1] = self.height-1
    # set everything within bounds
    self.coordinates[:, :, 1] = (self.coordinates[:, :, 1] >= 0)*self.coordinates[:, :, 1] \
      + np.zeros([self.height, self.width])
    self.coordinates[:, :, 1] = (self.coordinates[:, :, 1] <= self.height-1)*self.coordinates[:, :, 1] \
      + (self.height-1)*np.ones([self.height, self.width])*(self.coordinates[:, :, 1] > self.height-1)

