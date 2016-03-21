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
  def __init__(self, image=None, mesh_factor=10, density_distribution=None):
    super(CoordinateSolver, self).__init__()
    self.image = image
    self.height, self.width, = self.image.shape
    self.mesh_factor = mesh_factor
    self.height /= self.mesh_factor
    self.width /= self.mesh_factor
    self.image = self.image[:self.mesh_factor*self.height, :self.mesh_factor*self.width]
    if type(density_distribution) == np.ndarray:
      restricted_density = density_distribution[:self.mesh_factor*self.height, :self.mesh_factor*self.width]
      target_areas = sum([restricted_density[offset::self.mesh_factor, offset2::self.mesh_factor] \
        for offset in range(self.mesh_factor) for offset2 in range(self.mesh_factor)])/(self.mesh_factor**2)
      target_areas = target_areas[:-1, :-1]
    else:
      target_areas = np.indices((self.width-1, self.height-1)).T.astype('float32')
      target_areas = norm.pdf(target_areas[:, :, 0], self.width/2, self.width/10)\
                    *norm.pdf(target_areas[:, :, 1], self.height/2, self.height/10)
    target_areas /= sum(sum(target_areas))
    
    normalisation_factor = (self.height-1)*(self.width-1)
    target_areas_normalised = target_areas*normalisation_factor
    self.padded_targets = np.zeros([self.height+1, self.width+1])
    self.padded_targets[1:-1, 1:-1] = target_areas_normalised
    self.coordinates = np.indices((self.width, self.height)).T.astype('float32')
    self.total_error = (self.height-1)*(self.width-1)
    self.errors = np.zeros(self.padded_targets.shape)
    self.min_coords = self.coordinates.copy()
    self.areas = calculate_areas(self.coordinates)

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
    base_eps = 0.00005
    try:
      while mse > 0.001:
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
          self.coordinates += np.random.normal(0, 0.5, [self.width, self.height, 2])

        self.update_x(eps)
        self.update_y(eps)

        k += 1
    except KeyboardInterrupt:
      self.update_x(eps)
      self.update_y(eps)
      return

  def update_x(self, eps):
    """Updates x for vertices"""
    x_errors = self.errors.copy()
    x_errors[:, 0] += x_errors[:, 0]
    x_errors[:, -1] += x_errors[:, -1]
    x_adjustments = eps*(x_errors[:-1, :-1] + x_errors[1:, :-1] - x_errors[:-1, 1:] - x_errors[1:, 1:])
    self.coordinates[:, 1:-1, 0] += x_adjustments[:, 1:-1]
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
    self.coordinates[:, :, 0] = (self.coordinates[:, :, 0] >= 0)*self.coordinates[:, :, 0]
    self.coordinates[:, :, 0] = (self.coordinates[:, :, 0] <= self.width-1)*self.coordinates[:, :, 0] \
      + (self.width-1)*np.ones([self.height, self.width])*(self.coordinates[:, :, 0] > self.width-1)

  def update_y(self, eps):
    """Updates x for vertices"""
    y_errors = self.errors.copy()
    y_errors[0, :] += y_errors[0, :]
    y_errors[-1, :] += y_errors[-1, :]
    y_adjustments = eps*(y_errors[:-1, :-1] + y_errors[:-1, 1:] - y_errors[1:, :-1] - y_errors[1:, 1:])
    self.coordinates[1:-1, :, 1] += y_adjustments[1:-1, :]
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
    # set everything within bounds
    self.coordinates[:, :, 1] = (self.coordinates[:, :, 1] >= 0)*self.coordinates[:, :, 1] \
      + np.zeros([self.height, self.width])
    self.coordinates[:, :, 1] = (self.coordinates[:, :, 1] <= self.height-1)*self.coordinates[:, :, 1] \
      + (self.height-1)*np.ones([self.height, self.width])*(self.coordinates[:, :, 1] > self.height-1)

