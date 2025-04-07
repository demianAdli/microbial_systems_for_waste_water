from  single_chamber_electrolysis_cell import SingleChamberElectrolysisCell


class DualChamberElectrolysisCell(SingleChamberElectrolysisCell):
  def mass_balance(self, t, masses):
    """eq is short for equation"""
    acetate, andophilic_population, methanogenic_population, oxidized_mediator = masses

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

    oxidized_mediator_eq =\
        (self.mediator_molar_mass /
         (self.reactor_volume * andophilic_population)) *\
        (self.initial_current_density /
         (self.electrons_per_mole * self.faraday_ec)) -\
        self.mediator_yield *\
        self.andophilic_reaction(acetate, oxidized_mediator)

    return acetate_eq, andophilic_population_eq, methanogenic_population_eq, oxidized_mediator_eq

  def hydrogen_dual(self, current_density_val, t=298.15, p=1):
    return np.multiply(self.hydrogen_yield * (np.absolute(current_density_val)
                       / (self.electrons_per_mole * self.faraday_ec)) *
                       ((self.ideal_gas * t) / p), 100)
