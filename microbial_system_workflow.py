import math

import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt

from single_chamber_electrolysis_cell import SingleChamberElectrolysisCell
from dual_chamber_electrolysis_cell import DualChamberElectrolysisCell
from fuel_cell import FuelCell


class MicrobialSystemWorkflow:
  def __init__(self, system, time_interval, number_of_inhabitants=1, water_consumption=1):
    self.system = system
    self.time_interval = time_interval
    self.number_of_inhabitants = number_of_inhabitants
    self.water_consumption = water_consumption

    if str(type(self.system)).lstrip('<class ') == \
            '\'microbial_systems.SingleChamberElectrolysisCell\'>':
      self.system_type = 'single'
      self.ode_solution = self.electrolysis_cell_ode()
    elif str(type(self.system)).lstrip('<class ') == \
            '\'microbial_systems.DualChamberElectrolysisCell\'>':
      self.system_type = 'dual'
      self.ode_solution = self.dual_cell_ode()
    elif str(type(self.system)).lstrip('<class ') == \
            '\'microbial_systems.FuelCell\'>':
      self.system_type = 'fuel'
      self.ode_solution = self.fuel_cell_ode()
    else:
      print('Wrong input')

  def electrolysis_cell_ode(self):
    y_0 = 1500, self.system.andophilic_population_initial, \
          self.system.methanogenic_population_initial, \
          self.system.hydrogenotrophic_population_initial, \
          self.system.oxidized_mediator_initial
    return self.system.ordinary_differential_equation(self.time_interval, y_0)

  def dual_cell_ode(self):
    y_0 = 1500, self.system.andophilic_population_initial, \
          self.system.methanogenic_population_initial, \
          self.system.oxidized_mediator_initial
    return self.system.ordinary_differential_equation(self.time_interval, y_0)

  def fuel_cell_ode(self):
    y_0 = 1700, self.system.andophilic_population_initial, \
          self.system.methanogenic_population_initial, \
          self.system.oxidized_mediator_initial
    return self.system.ordinary_differential_equation(self.time_interval, y_0)

  def system_output_csv(self):
    if self.system_type == 'single':
      self.single_chamber_dataframe().to_csv('single.csv', index=False)
    elif self.system_type == 'dual':
      self.dual_chamber_dataframe().to_csv('dual.csv', index=False)
    else:
      self.fuel_cell_dataframe().to_csv('fuel.csv', index=False)

  def electrolysis_cell_output(self):
    time = self.ode_solution.t
    acetate = self.ode_solution.y[0]
    andophilic_population = self.ode_solution.y[1]
    methanogenic_popoulation = self.ode_solution.y[2]
    hydrogenotrophic_population = self.ode_solution.y[3]
    oxidized_mediator = self.ode_solution.y[4]
    reduction_mediator = self.system.reduction_mediator(self.ode_solution.y[4])
    internal_resistance = self.system.internal_resistance(self.ode_solution.y[1])
    hydrogenotrophic_growth = self.system.hydrogenotrophic_growth()

    return time, acetate, andophilic_population, methanogenic_popoulation,\
        hydrogenotrophic_population, oxidized_mediator, reduction_mediator,\
        internal_resistance, hydrogenotrophic_growth

  def single_chamber_dataframe(self):
    output_1 = self.electrolysis_cell_output()
    current_density = [self.current_density_values()] * len(output_1[0])
    hydrogen = self.system.hydrogen_single(current_density, output_1[8],
                                           output_1[4])

    output_dictionary = {'Time': output_1[0], 'Acetate (mg_s/L)': output_1[1],
                         'Andophilic Population (mg_x/L)': output_1[2],
                         'Methanogenic Popoulation (1/d)': output_1[3],
                         'Hydrogenotrophic Population (mg/L)': output_1[4],
                         'Oxidized Mediator (mgM_O/mgX)': output_1[5],
                         'Reduction Mediator (mgM_R/mgX)': output_1[6],
                         'Internal Resistance (ohm)': output_1[7],
                         'Current Density (A)': current_density,
                         'Hydrogenotrophic Growth (1/d)': output_1[8],
                         'Hydrogen (L_h2/d)': hydrogen}
    return pd.DataFrame(output_dictionary)

  def dual_cell_output(self):
    time = self.ode_solution.t
    acetate = self.ode_solution.y[0]
    andophilic_population = self.ode_solution.y[1]
    methanogenic_popoulation = self.ode_solution.y[2]
    oxidized_mediator = self.ode_solution.y[3]
    reduction_mediator = self.system.reduction_mediator(self.ode_solution.y[3])
    internal_resistance = self.system.internal_resistance(self.ode_solution.y[1])

    return time, acetate, andophilic_population, methanogenic_popoulation,\
        oxidized_mediator, reduction_mediator, internal_resistance

  def dual_chamber_dataframe(self):
    output_1 = self.dual_cell_output()
    current_density = [self.current_density_values()] * len(output_1[0])
    hydrogen = self.system.hydrogen_dual(current_density)
    output_dictionary = {'Time': output_1[0], 'Acetate (mg_s/L)': output_1[1],
                         'Andophilic Population (mg/L)': output_1[2],
                         'Methanogenic Popoulation (1/d)': output_1[3],
                         'Oxidized Mediator (mgM_O/mgX)': output_1[4],
                         'Reduction Mediator (mgM_R/mgX)': output_1[5],
                         'Internal Resistance (ohm)': output_1[6],
                         'Current Density (A)': current_density,
                         'Hydrogen (L_h2/d)': hydrogen}
    return pd.DataFrame(output_dictionary)

  def fuel_cell_output(self):
    time = self.ode_solution.t
    acetate = self.ode_solution.y[0]
    andophilic_population = self.ode_solution.y[1]
    methanogenic_popoulation = self.ode_solution.y[2]
    oxidized_mediator = self.ode_solution.y[3]
    reduction_mediator = self.system.reduction_mediator(oxidized_mediator)
    methanogenic_reaction =\
        self.system.methanogenic_reaction(self.ode_solution.y[0])
    methane =\
        self.system.methane(self.ode_solution.y[2], methanogenic_reaction)
    internal_resistance =\
        self.system.internal_resistance(self.ode_solution.y[1])
    resistance_ocv = \
        self.system.internal_resistance_ocv(self.ode_solution.y[1])
    concentration_loss = self.system.concentration_losses(reduction_mediator)
    current_density = \
        self.system.andophilic_current_density(resistance_ocv,
                                               concentration_loss,
                                               internal_resistance,
                                               internal_resistance + 50)
    voltage = np.divide(self.system.voltage(current_density, internal_resistance + 50), 2)
    power = self.power(current_density, internal_resistance + 50)
    final_power = self.final_power(power, self.influent_flow())
    energy = self.energy(final_power)

    return time, acetate, andophilic_population, methanogenic_popoulation,\
        oxidized_mediator, reduction_mediator, methanogenic_reaction, methane,\
        internal_resistance, resistance_ocv, concentration_loss,\
        current_density, voltage, final_power, energy

  def fuel_cell_dataframe(self):
    output_fuel = self.fuel_cell_output()
    output_dictionary = {'Time': output_fuel[0], 'Acetate (mg_s/L)': output_fuel[1],
                         'Andophilic Population (mg_x/L)': output_fuel[2],
                         'Methanogenic Popoulation (1/d)': output_fuel[3],
                         'Oxidized Mediator (mgM_O/mgX)': output_fuel[4],
                         'Reduction Mediator (mgM_R/mgX)': output_fuel[5],
                         'Methanogenic Reaction (mgS/mgXd)': output_fuel[6],
                         'Methane (L_ch4/d)': output_fuel[7],
                         'Internal Resistance (ohm)': output_fuel[8],
                         'Resistance OCV (ohm)': output_fuel[9],
                         'Concentration Loss': output_fuel[10],
                         'Current Density (A)': output_fuel[11],
                         'Voltage (V)': output_fuel[12],
                         'Power (W)': output_fuel[13],
                         'Energy (J)': output_fuel[14]}
    return pd.DataFrame(output_dictionary)

  def _current_density_function(self, current_density, t=298.15):
    """This function will be an input for scipy.optimize.newton which
    \"Find a zero of a real or complex function using
    the Newton-Raphson (or secant or Halley’s) method\"

    Values for reduction mediator and internal resistance are needed
    to compute the current density. So, these two functions are parts of
    a run of a system (a workflow).

    To make it work for all the three systems, the if clause assign
    the correct values to reduction_mediator and internal_density.

    cd_eq: current density equation
    """
    if self.system_type == 'single':
      reduction_mediator = self.electrolysis_cell_output()[6]
      internal_density = self.electrolysis_cell_output()[7]
    elif self.system_type == 'dual':
      reduction_mediator = self.dual_cell_output()[5]
      internal_density = self.dual_cell_output()[6]
    else:
      reduction_mediator = self.fuel_cell_output()[5]
      internal_density = self.fuel_cell_output()[8]

    numerator_1 = np.log(np.absolute(np.divide(self.system.mediator_fraction,
                                               reduction_mediator)))
    numerator_2 = (1 / self.system.oxidation_coefficient) * \
        math.asinh(current_density /
                   (self.system.anode_surface_area * self.system.current_density_0))
    numerator = np.subtract(self.system.counter_electromotive_force +
                            self.system.applied_potential,
                            np.multiply(self.system.ideal_gas_ec * t,
                                        np.add(numerator_1, numerator_2)))
    cd_eq = np.divide(numerator, internal_density)
    for value in cd_eq:
      return current_density - value

  def current_density_values(self):
    return scipy.optimize.newton(self._current_density_function,
                                 x0=self.system.initial_current_density,
                                 rtol=0.0001)

  def influent_flow(self):
    return self.number_of_inhabitants * self.water_consumption

  @staticmethod
  def power(current_density, external_resistance):
    return np.multiply(external_resistance, np.square(current_density))

  @staticmethod
  def final_power(power, influent_flow):
    return np.divide(np.multiply(power, influent_flow), 3.8)

  @staticmethod
  def energy(final_pow):
    return np.multiply(np.divide(final_pow, 1000), 24)

  @staticmethod
  def lookup_parameters():
    """This method makes it easier for user to choose a parameter
       to plot, using three first letters only."""
    return {'Ace': 'Acetate (mg_s/L)',
            'And': 'Andophilic Population (mg_x/L)',
            'Met': 'Methanogenic Popoulation (1/d)',
            'Oxi': 'Oxidized Mediator (mgM_O/mgX)',
            'Hyd': 'Hydrogenotrophic Population (mg/L)'}

  def plotting(self, save_plot=False):
    system = pd.read_csv(f'{self.system_type}.csv')
    parameters = self.lookup_parameters()
    print(str(parameters.values()).lstrip('dict_values'))
    while True:
      try:
        parameter = input('Enter the first three charaters of the parameter\'s'
                          ' name, from above list, to see the plot:\n'
                          '(Hydrogenotrophic Population is only specific to single'
                          ' chamber microbial system)\n').capitalize()

        plt.plot(system['Time'], system[parameters[parameter]],
                 label='[X]')
      except KeyError:
        print('An incorrect value has been entered for the parameter.\n'
              'Remember, only first three characters of the parameter.')
      else:
        break
    plot_param = plt.gcf()
    plt.xlabel('time (Day)')
    plt.ylabel(parameters[parameter])
    plt.show()
    if save_plot:
      plot_param.savefig(f'{parameters[parameter]}.png',
                         dpi=100)


if __name__ == '__main__':
  wf_1 = MicrobialSystemWorkflow(SingleChamberElectrolysisCell(), (0, 152))
  # wf_1.system_output_csv()
  # wf_1.plotting(save_plot=True)
  # wf_2 = MicrobialSystemWorkflow(DualChamberElectrolysisCell(), (0, 151))
  # wf_2.system_output_csv()
  # wf_3 = MicrobialSystemWorkflow(FuelCell(), (0, 62))
  # wf_3.system_output_csv()
  # print(wf_3.fuel_cell_ode().nfev)

