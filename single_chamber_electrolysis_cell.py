from microbial_systems_abc import MicrobialSystem
import numpy as np


class SingleChamberElectrolysisCell(MicrobialSystem):
  def __init__(self):
    super().__init__()
    self._mediator_fraction = 1000
    self._methanogenic_decay = 0.01
    self._faraday_ec = 1.1167
    self._hydrogen_2_saturation = 1
    self._applied_potential = 0.6
    self._counter_electromotive_force = 50
    self._hydrogen_2_dissolved = 1.5
    self._hydrogenotrophic_half = 0.001
    self._curve_slope = 0.024
    self._hydrogenotrophic_decay = 0.01
    self._andophilic_reaction_max = 13.14
    self._methanogenic_reaction_max = 14.12
    self._resistance_min = 2
    self._resistance_max = 200
    self._max_biomass = 1215
    self._reactor_volume = 1
    self._hydrogen_yield_methanogenic = 0.05
    self._hydrogen_yield = 0.9
    self._mediator_yield = 3.3
    self._oxidation_coefficient = 0.5
    self._hydrogenotrophic_max_growth = 0.5
    self._methanogenic_max_growth = 0.3
    self._oxidized_mediator_initial = 800
    self._acetate_initial = 3000
    self._andophilic_population_initial = 400
    self._hydrogenotrophic_population_initial = 10
    self._methanogenic_population_initial = 50
    self._initial_current_density = 0.001
    self._influent_flow_initial = 3.5
    self._current_density_0 = 0.1

  @property
  def mediator_fraction(self):
    return self._mediator_fraction

  @property
  def methanogenic_decay(self):
    return self._methanogenic_decay

  @property
  def faraday_ec(self):
    return self._faraday_ec

  @property
  def hydrogen_2_saturation(self):
    return self._hydrogen_2_saturation

  @property
  def applied_potential(self):
    return self._applied_potential

  @property
  def counter_electromotive_force(self):
    return self._counter_electromotive_force

  @property
  def hydrogen_2_dissolved(self):
    return self._hydrogen_2_dissolved

  @property
  def hydrogenotrophic_half(self):
    return self._hydrogenotrophic_half

  @property
  def curve_slope(self):
    return self._curve_slope

  @property
  def hydrogenotrophic_decay(self):
    return self._hydrogenotrophic_decay

  @property
  def andophilic_reaction_max(self):
    return self._andophilic_reaction_max

  @property
  def methanogenic_reaction_max(self):
    return self._methanogenic_reaction_max

  @property
  def resistance_min(self):
    return self._resistance_min

  @property
  def resistance_max(self):
    return self._resistance_max

  @property
  def max_biomass(self):
    return self._max_biomass

  @property
  def reactor_volume(self):
    return self._reactor_volume

  @property
  def hydrogen_yield_methanogenic(self):
    return self._hydrogen_yield_methanogenic

  @property
  def hydrogen_yield(self):
    return self._hydrogen_yield

  @property
  def mediator_yield(self):
    return self._mediator_yield

  @property
  def oxidation_coefficient(self):
    return self._oxidation_coefficient

  @property
  def hydrogenotrophic_max_growth(self):
    return self._hydrogenotrophic_max_growth

  @property
  def methanogenic_max_growth(self):
    return self._methanogenic_max_growth

  @property
  def oxidized_mediator_initial(self):
    return self._oxidized_mediator_initial

  @property
  def acetate_initial(self):
    return self._acetate_initial

  @property
  def andophilic_population_initial(self):
    return self._andophilic_population_initial

  @property
  def hydrogenotrophic_population_initial(self):
    return self._hydrogenotrophic_population_initial

  @property
  def methanogenic_population_initial(self):
    return self._methanogenic_population_initial

  @property
  def initial_current_density(self):
    return self._initial_current_density

  @property
  def influent_flow_initial(self):
    return self._influent_flow_initial

  @property
  def current_density_0(self):
    return self._current_density_0

  def hydrogenotrophic_growth(self):
    return self.hydrogenotrophic_max_growth *\
           (self.hydrogen_2_dissolved /
            (self.hydrogenotrophic_half + self.hydrogen_2_dissolved))

  def mass_balance(self, t, masses):
    """eq is short for equation"""
    acetate, andophilic_population, methanogenic_population,\
        hydrogenotrophic_population, oxidized_mediator = masses

    acetate_eq = (self.influent_flow_initial / self.reactor_volume) *\
                 (self.acetate_initial - acetate) -\
        self.andophilic_reaction(acetate, oxidized_mediator) *\
        andophilic_population -\
        self.methanogenic_reaction(acetate) *\
        methanogenic_population

    andophilic_population_eq =\
        self.andophilic_growth(acetate, oxidized_mediator) *\
        andophilic_population - self.andophilic_decay *\
        andophilic_population - self.andophilic_biofilm_retention *\
        (self.influent_flow_initial / self.reactor_volume) * andophilic_population

    methanogenic_population_eq =\
        self.methanogenic_growth(acetate) * methanogenic_population -\
        self.methanogenic_decay * methanogenic_population -\
        self.methanogenic_biofilm_retention *\
        (self.influent_flow_initial / self.reactor_volume) * methanogenic_population

    hydrogenotrophic_population_eq =\
        self.hydrogenotrophic_growth() * methanogenic_population -\
        self.hydrogenotrophic_decay * hydrogenotrophic_population -\
        self.methanogenic_biofilm_retention *\
        (self.influent_flow_initial / self.reactor_volume) * hydrogenotrophic_population

    oxidized_mediator_eq =\
        (self.mediator_molar_mass /
         (self.reactor_volume * andophilic_population)) *\
        (self.initial_current_density /
         (self.electrons_per_mole * self.faraday_ec)) -\
        self.mediator_yield *\
        self.andophilic_reaction(acetate, oxidized_mediator)

    return acetate_eq, andophilic_population_eq, methanogenic_population_eq,\
        hydrogenotrophic_population_eq, oxidized_mediator_eq

  def hydrogen_single(self, current_density_val, hydrogenotrophic_growth,
                      hydrogenotrophic_population, t=298.15, p=1):
    """Values for hydrogenotrophic_growth and hydrogenotrophic_population
       can be retrived from different indexes of
       microbial_system_workflow.py.MicrobialSystemWorkflow.single_chamber_output()

       current_density_val can be retrieved from cerrent_density() method
       of the MicrobialSystemWorkflow class."""

    return np.multiply(np.subtract(self.hydrogen_yield *
                                   (np.absolute(current_density_val) /
                                    (self.electrons_per_mole * self.faraday_ec)) *
                                   ((self.ideal_gas * t) / p), np.multiply
                                   (self.hydrogen_yield_methanogenic *
                                    hydrogenotrophic_growth * self.reactor_volume,
                                    hydrogenotrophic_population)), 100)
