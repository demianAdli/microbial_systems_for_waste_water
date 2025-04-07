from microbial_systems_abc import MicrobialSystem


class FuelCell(MicrobialSystem):
  def __init__(self):
    super().__init__()
    self._mediator_fraction = 0.05
    self._e_ocv_min = 0.01
    self._e_ocv_max = 0.66
    self._methanogenic_decay = 0.006
    self._steepness = 0.04
    self._curve_slope = 0.006
    self._andophilic_reaction_max = 8.48
    self._methanogenic_reaction_max = 8.20
    self._biofilm_space_limitation = 525
    self._methanogenic_max_growth = 0.1
    self._resistance_min = 1
    self._resistance_max = 10
    self._mediator_yield = 22.75
    self._methane_yield = 0.3
    self._reactor_volume = 1
    self._exchange_current_density = 1
    self._acetate_initial = 2500
    self._andophilic_population_initial = 550
    self._methanogenic_population_initial = 50
    self._initial_current_density = 0.005
    self._oxidized_mediator_initial = 50
    self._initial_reduction_mediator = 0.8
    self._influent_flow_initial = 3.5
    self._temperature = 298.15

  @property
  def mediator_fraction(self):
    return self._mediator_fraction

  @property
  def e_ocv_min(self):
    return self._e_ocv_min

  @property
  def e_ocv_max(self):
    return self._e_ocv_max

  @property
  def methanogenic_decay(self):
    return self._methanogenic_decay

  @property
  def steepness(self):
    return self._steepness

  @property
  def curve_slope(self):
    return self._curve_slope

  @property
  def andophilic_reaction_max(self):
    return self._andophilic_reaction_max

  @property
  def methanogenic_reaction_max(self):
    return self._methanogenic_reaction_max

  @property
  def biofilm_space_limitation(self):
    return self._biofilm_space_limitation

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
  def mediator_yield(self):
    return self._mediator_yield

  @property
  def methane_yield(self):
    return self._methane_yield

  @property
  def reactor_volume(self):
    return self._reactor_volume

  @property
  def exchange_current_density(self):
    return self._exchange_current_density

  @property
  def acetate_initial(self):
    return self._acetate_initial

  @property
  def andophilic_population_initial(self):
    return self._andophilic_population_initial

  @property
  def methanogenic_population_initial(self):
    return self._methanogenic_population_initial

  @property
  def initial_current_density(self):
    return self._initial_current_density

  @property
  def oxidized_mediator_initial(self):
    return self._oxidized_mediator_initial

  @property
  def initial_reduction_mediator(self):
    return self._initial_reduction_mediator

  @property
  def influent_flow_initial(self):
    return self._influent_flow_initial

  @property
  def temperature(self):
    return self._temperature

  def mass_balance(self, t, masses):
    """eq is short for equation"""
    acetate, andophilic_population, methanogenic_population,\
        oxidized_mediator = masses

    acetate_eq = (self.influent_flow_initial / self.reactor_volume) * \
                 (self.acetate_initial - acetate) - \
        self.andophilic_reaction(acetate, oxidized_mediator) * \
        andophilic_population - self.methanogenic_reaction(acetate) * \
        methanogenic_population

    andophilic_population_eq = \
        self.andophilic_growth(acetate, oxidized_mediator) * \
        andophilic_population - self.andophilic_decay * \
        andophilic_population - self.andophilic_biofilm_retention * \
        (self.influent_flow_initial / self.reactor_volume) * andophilic_population

    methanogenic_population_eq = \
        self.methanogenic_growth(acetate) * methanogenic_population - \
        self.methanogenic_decay * methanogenic_population - \
        self.methanogenic_biofilm_retention * \
        (self.influent_flow_initial / self.reactor_volume) * methanogenic_population

    oxidized_mediator_eq = -self.mediator_yield *\
        self.andophilic_reaction(acetate, oxidized_mediator) +\
        self.mediator_molar_mass * (self.initial_current_density /
                                    (self.electrons_per_mole * self.faraday))\
        * (1 / (self.reactor_volume * andophilic_population))

    return acetate_eq, andophilic_population_eq, methanogenic_population_eq, \
        oxidized_mediator_eq

  def methane(self, methanogenic_population, methanogenic_reaction_rate):
    """methanogenic_population index for FuelCell MicrobialSystem is
       ordinary_differential_equation[1][:, 2]

       methanogenic_reaction_rate index for FuelCell MicrobialSystem is
       ordinary_differential_equation[1][:, 5] """
    return np.divide(np.multiply(
      np.multiply(self.methane_yield, methanogenic_reaction_rate),
      np.multiply(methanogenic_population, self.reactor_volume)), 100)

  def internal_resistance_ocv(self, andophilic_population):
    """Index of andophilic_population for current three systems is
       ordinary_differential_equation[1][:, 1]"""
    return np.add(self.e_ocv_min,
                  np.multiply(self._e_ocv_max - self.e_ocv_min,
                              np.power(math.e, np.divide(-1, (
                                np.multiply(self.curve_slope,
                                            andophilic_population))))))

  def concentration_losses(self, reduction_mediator):
    """reduction_mediator index for FuelCell MicrobialSystem is
       ordinary_differential_equation[1][:, 4]"""
    return np.multiply((self.ideal_gas_ec * self.temperature) /
                       (self.electrons_per_mole * self.faraday),
                       np.log(np.absolute(np.divide(self.mediator_fraction,
                                                    reduction_mediator))))

  @staticmethod
  def andophilic_current_density(internal_resistance_ocv,
                                 concentration_losses, internal_resistance,
                                 external_resistance):
    """internal_resistance_ocv index for FuelCell MicrobialSystem is
       ordinary_differential_equation[1][:, 8]

       concentration_losses index for FuelCell MicrobialSystem is
       ordinary_differential_equation[1][:, 9]

       internal_resistance index for FuelCell MicrobialSystem is
       ordinary_differential_equation[1][:, 7]

       external_resistance will be computed by adding 50 to the
       internal_resistance"""
    return np.divide(np.subtract(internal_resistance_ocv,
                                 concentration_losses),
                     np.add(external_resistance, internal_resistance))

  @staticmethod
  def voltage(andophilic_current_density, external_resistance):
    """andophilic_current_density index for FuelCell MicrobialSystem is
       ordinary_differential_equation[1][:, 10]

       external_resistance will be computed by adding 50 to the
       internal_resistance"""
    return np.multiply(external_resistance, andophilic_current_density)
