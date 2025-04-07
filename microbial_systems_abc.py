"""
Alireza Adli
_ The corresponding notation of parameters are mentioned in property decorated
methods.
The naming has been done in order to make the code more readable. For example,
ideal_gas is more informative than r. Not to mention, variables should be
defined with more than a letter.
- All the 'constant', and 'rate' have been removed from the naming to make
the names shorter. For example, following parameters' names consist a constant:
    faraday, ideal_gas, mediator_half, andophilic_half
    andophilic_biofilm_retention, methanogenic_biofilm_retention
- andophilic_limitation is in fact: Andophilic biofilm space limitation
- Some data members of different classes have been defined with the same name.
- faraday constant and ideal gas are suffixed with ec in both Electrolysis Cell
systems. Because they have different values in the same systems.
"""

from abc import ABC, abstractmethod
import math
import numpy as np
from scipy.integrate import solve_ivp


class MicrobialSystem(ABC):
  """This class cannot be instantiated and is used to define
  common behaviors and attributes of a microbial system.

  All new child classes should consist a mass_balance(t, values) method
  with the mentioned parameters. Otherwise, they cannot be instantiated.
  This is the interface (abstract method) of this Abstract Base Class.
  This rule is not applied to DualChamberElectrolysisCell class because
  it inherits from the SingleChamberElectrolysisCell class."""
  def __init__(self):
    self._faraday = 96485
    self._ideal_gas = 0.08205
    self._ideal_gas_ec = 8.314
    self._electrons_per_mole = 2
    self._mediator_half = 0.01
    self._andophilic_decay = 0.04
    self._andophilic_half = 20
    self._methanogenic_half = 80
    self._andophilic_limitation = 512.5
    self._andophilic_max_growth = 1.97
    self._mediator_molar_mass = 663400
    self._andophilic_biofilm_retention = 0.5410
    self._methanogenic_biofilm_retention = 0.4894
    self._anode_surface_area = 0.01
    self._andophilic_reaction_max = None
    self._methanogenic_reaction_max = None
    self._methanogenic_max_growth = None
    self._resistance_min = None
    self._resistance_max = None
    self._curve_slope = None
    self._mediator_fraction = None

  @property
  def faraday(self):
    return self._faraday

  @property
  def ideal_gas(self):
    return self._ideal_gas

  @property
  def ideal_gas_ec(self):
    return self._ideal_gas_ec

  @property
  def electrons_per_mole(self):
    return self._electrons_per_mole

  @property
  def mediator_half(self):
    return self._mediator_half

  @property
  def andophilic_decay(self):
    return self._andophilic_decay

  @property
  def andophilic_half(self):
    return self._andophilic_half

  @property
  def methanogenic_half(self):
    return self._methanogenic_half

  @property
  def andophilic_limitation(self):
    return self._andophilic_limitation

  @property
  def andophilic_max_growth(self):
    return self._andophilic_max_growth

  @property
  def mediator_molar_mass(self):
    return self._mediator_molar_mass

  @property
  def andophilic_biofilm_retention(self):
    return self._andophilic_biofilm_retention

  @property
  def methanogenic_biofilm_retention(self):
    return self._methanogenic_biofilm_retention

  @property
  def anode_surface_area(self):
    return self._anode_surface_area

  @property
  def andophilic_reaction_max(self):
    return self._andophilic_reaction_max

  @property
  def methanogenic_reaction_max(self):
    return self._methanogenic_reaction_max

  @property
  def methanogenic_max_growth(self):
    return self._methanogenic_max_growth

  @property
  def resistance_min(self):
    return self._resistance_min

  @property
  def resistance_max(self):
    return self._resistance_max

  @property
  def curve_slope(self):
    return self._curve_slope

  @abstractmethod
  def mass_balance(self, t, masses):
    pass

  def andophilic_reaction(self, acetate, oxidized_mediator):
    return self.andophilic_reaction_max * (acetate /
                                           (self.andophilic_half + acetate)) * \
           (oxidized_mediator / (self.mediator_half + oxidized_mediator))

  def methanogenic_reaction(self, acetate):
    return self.methanogenic_reaction_max * \
           (acetate / (self.methanogenic_half + acetate))

  def andophilic_growth(self, acetate, oxidized_mediator):
    return self.andophilic_max_growth * (acetate /
                                         (self.andophilic_half + acetate)) * \
           (oxidized_mediator / (self.mediator_half + oxidized_mediator))

  def methanogenic_growth(self, acetate):
    return self.methanogenic_max_growth * \
           (acetate / (self.methanogenic_half + acetate))

  def ordinary_differential_equation(self, time_interval, y_0):
    times = list(range(time_interval[0], time_interval[1]))
    return solve_ivp(self.mass_balance, time_interval,
                     y_0, method='Radau', t_eval=times)

  def reduction_mediator(self, oxidized_mediator):
    """oxidized_mediator index for SingleChamberMicrobialSystem is
    ordinary_differential_equation.y[4]

    And for DualChamberElectrolysisCell is
    ordinary_differential_equationy[3]"""
    return np.subtract(self._mediator_fraction, oxidized_mediator)

  def internal_resistance(self, andophilic_population):
    """Index of andophilic_population for current three systems is
    ordinary_differential_equation.y[1]"""
    return np.add(self.resistance_min, np.multiply(self.resistance_max
                  - self.resistance_min, np.power(math.e, np.multiply(
                   -self.curve_slope, andophilic_population))))