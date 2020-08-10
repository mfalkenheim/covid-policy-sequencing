"""
Contains the SIR_Node class and related classes
    """
from functions import *
from functions import calculate_deaths
from functions import find_next_point
from functions import find_continuation_cost
from functions import calculate_time
import numpy as np
from math import exp
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import pandas as pd

class SIR_Point:
    """
    """
    def __init__(self, immune, infected):
        """
        This object contains the coordinates and other attributes of any point in the state space
        """
        self.infected = infected
        self.immune = immune
        self.deaths_if_vaccine = []
        
    def find_deaths_if_vaccine(self, fatality_rate, healthcare_capacity, overcapacity_fatality_rate):
        """
        Returns
        -------
        None.

        """
        self.deaths_if_vaccine = calculate_deaths(self.infected, 0, [], [], fatality_rate, 
                                                  healthcare_capacity, overcapacity_fatality_rate)
    
class SIR_Policy:
    """
    A policy is a measure that can be taken at any point in the state space, defined by its
    daily cost and effect on the initial transmission rate
    """
    def __init__(self, name, r0, daily_cost):
        self.name = name
        self.r0 = r0
        self.daily_cost = daily_cost
        self.herd_immunity_threshold = []
        
    def make_herd_immunity_threshold():
        self.herd_immunity_threshold = max(1 - 1 / self.r0, 0)

class SIR_Action:
    """
    An action is a policy adopted at a specific point in the state space, resulting in 
    a local trajectory of the epidemic, and a cost.
    """
    def __init__(self, policy, start_point):
        self.policy = policy
        self.start_point = start_point
        self.end_point = []
        self.time = []
        self.continuation_cost = []
        self.action_cost = []
        self.total_cost = []
    
    def fill(self, minimum_infected, last_level, serial_interval,
                            herd_immunity_thresholds, fatality_rate, healthcare_capacity, 
                            overcapacity_fatality_rate, time_to_vaccine):
        """

        Returns
        -------
        None.

        """
        slope_infected_time = self.start_point.infected * (self.policy.r0 * (1 - self.start_point.immune - self.start_point.infected) \
            - 1) / serial_interval
        slope_immune_time = self.start_point.infected / serial_interval
        slope = slope_infected_time / slope_immune_time
        if last_level == []:
            level = 1
        else:
            level = last_level.level
        end_immune, end_infected = find_next_point(self.start_point.immune, self.start_point.infected, 
                                                                 slope, level)
        self.end_point = SIR_Point(end_immune, end_infected)
        self.end_point.find_deaths_if_vaccine(fatality_rate, healthcare_capacity, overcapacity_fatality_rate)

  
        action_deaths = calculate_deaths(self.start_point.infected, self.end_point.infected, 
                                            self.start_point.immune, self.end_point.immune, fatality_rate, 
                                            healthcare_capacity, overcapacity_fatality_rate) 
        self.time = calculate_time(self.start_point.infected, self.end_point.infected, 
                                   self.start_point.immune, self.end_point.immune,
                                   serial_interval)
        if level < 1:
            self.continuation_cost = find_continuation_cost(self.end_point.immune, self.end_point.infected, last_level.cost, 
                                                            self.end_point.deaths_if_vaccine, self.time,
                                                            time_to_vaccine, herd_immunity_thresholds,
                                                            minimum_infected)
        else:
            self.continuation_cost = self.end_point.deaths_if_vaccine  
        self.action_cost = action_deaths + self.time * self.policy.daily_cost
        self.total_cost = self.action_cost + self.continuation_cost

class SIR_Node:
        
    """
    A node is a point in the space space where the model calculates the cost of each policy and then
    assigns the cost minimizing policy as the action taken at that node. 
    """
    
    def __init__(self, immune, infected):

        """
        """
        self.point = SIR_Point(immune, infected)
        self.minimum_infected = []
        self.actions = []
        self.best_action_name = []
        self.best_action_index = []
        self.total_cost = []
        
    def fill(self, minimum_infected, last_level, policies, serial_interval, 
             herd_immunity_thresholds, fatality_rate, healthcare_capacity, 
             overcapacity_fatality_rate, time_to_vaccine):
        self.point.find_deaths_if_vaccine(fatality_rate, healthcare_capacity, overcapacity_fatality_rate)
        minimum_total_cost = []
        best_action = []
        for i, policy in enumerate(policies):
            self.actions.append(SIR_Action(policy, self.point))
            self.actions[i].fill(minimum_infected, last_level, serial_interval,
                            herd_immunity_thresholds, fatality_rate, healthcare_capacity, 
                            overcapacity_fatality_rate, time_to_vaccine)
            if minimum_total_cost == []:
                minimum_total_cost = self.actions[i].total_cost
                best_action_name = policy.name
                best_action_index = i
            elif self.actions[i].total_cost < minimum_total_cost:
                minimum_total_cost = self.actions[i].total_cost
                best_action_name = policy.name
                best_action_index = i
        self.best_action_name = best_action_name
        self.best_action_index = best_action_index
        self.total_cost = minimum_total_cost
        
            
class SIR_Levels:
    """
    A level is a set of nodes arrayed along a diagonal in the state space where the 
    immune and infected shares of the population add to a constant level
    """
    def __init__(self, level):
        """
        """
        self.level = level
        self.nodes = []
        self.cost = []
        
    def make_cost(self):
        """
        """
        infected = []
        cost = []
        for node in self.nodes:
            infected.append(node.point.infected)
            cost.append(node.total_cost)
        self.cost = interp1d(infected, cost, fill_value = 'extrapolate')
        

class SIR_Grid:
    """
    A grid is a set of levels which in turn contain a set of nodes. 
    """
    def __init__(self, log_minimum, n, k):
    #def __init__(self, n):
        """
        """
        self.SIR_Levels = []
        self.log_minimum = log_minimum
        self.k = k
        self.n = n
        self.r0 = []
        self.serial_interval = []
        self.policies = []
        self.best_policy_function = []
        self.best_policy_indexes = []
                
    def fill(self, r0, serial_interval, policies, 
            fatality_rate, healthcare_capacity, 
            overcapacity_fatality_rate, time_to_vaccine):
        shares = make_shares(self.n)
        infected_levels = make_infected_levels(self.log_minimum, log(1/r0), self.k)
        #shares = make_shares(self.n)
        last_level = []
        count_policies = [0]
        self.r0 = r0
        self.serial_interval = serial_interval
        herd_immunity_thresholds = [[max(1 - 1/r0, 0)], [0]]
        available_policies = [SIR_Policy('none', r0, 0)]
        data = []
        self.best_policy_indexes = []
        for policy in policies:
            available_policies.append(SIR_Policy(policy[0], policy[1], policy[2]))
            herd_immunity_thresholds[0].append(max(1 - 1/policy[1], 0))
            herd_immunity_thresholds[1].append(policy[2] * time_to_vaccine)
            count_policies.append(0)
        self.policies = available_policies
        for i, share in enumerate(reversed(shares)):
            self.SIR_Levels.append(SIR_Levels(share))
            current_level = self.SIR_Levels[i]
            minimum_infected = infected_levels[0]
            for j, infected in enumerate(infected_levels):
                if infected <= share:
                    immune = share - infected
                    current_level.nodes.append(SIR_Node(immune, infected))
                    current_node = current_level.nodes[j]
                    current_node.fill(minimum_infected, last_level, available_policies, serial_interval, 
                                      herd_immunity_thresholds, fatality_rate, healthcare_capacity, 
                                      overcapacity_fatality_rate, time_to_vaccine)
                    count_policies[current_node.best_action_index] =count_policies[current_node.best_action_index] + 1
                    data.append([immune, infected])
                    self.best_policy_indexes.append(current_node.best_action_index)
            if i + 1 < self.n:
                current_level.make_cost()
                last_level = current_level
        self.best_policy_function = KDTree(data)
        print(count_policies)
            
    def find_best_policy(self, immune, infected):
        """
        This function uses the KDTree object populated in the 'fill' method to approximate the cost minimizing
        policy at the point (immune, infected). That policy will be based on the nearest node.
        """
        distance, index = self.best_policy_function.query([immune, infected])
        return(self.best_policy_indexes[index])
    
    def plot(self,policies,colors):
        """
        this function creates a plot of the optimal decision at each node in the state space using a color coded
        arrow. The arrow indicates the start and end points of the epidemic under the action.
        """
        ax = plt.axes(xlim = (0,0.5), ylim=(0,0.025))
        ax.set_xlabel('share of population immune')
        ax.set_ylabel('share of population infected')
        ax.text(0.05,0.0255,"none", color = 'g')
        ax.text(0.15,0.0255,"sustainable", color = 'b')
        ax.text(0.25,0.0255,"unsustainable", color = 'r')
        for SIR_Level in self.SIR_Levels:
            for node in SIR_Level.nodes:
                action = node.best_action_index
                start_arrow_x = node.actions[action].start_point.immune
                start_arrow_y = node.actions[action].start_point.infected
                length_arrow_x = node.actions[action].end_point.immune - start_arrow_x
                length_arrow_y = node.actions[action].end_point.infected - start_arrow_y
                ax.arrow(start_arrow_x, start_arrow_y, length_arrow_x, length_arrow_y, width=0.0005, head_width = 0.001, color = colors[action])
        plt.show()


    def plot_with_projection(self,policies,colors, s0, st, it, date_range):
        ax = plt.axes(xlim = (0,0.5), ylim=(0,0.025))
        ax.set_xlabel('share of population immune')
        ax.set_ylabel('share of population infected')
        ax.text(0.05,0.0255,"none", color = 'g')
        ax.text(0.15,0.0255,"sustainable", color = 'b')
        ax.text(0.25,0.0255,"unsustainable", color = 'r')
        for SIR_Level in self.SIR_Levels:
            for node in SIR_Level.nodes:
                action = node.best_action_index
                start_arrow_x = node.actions[action].start_point.immune
                start_arrow_y = node.actions[action].start_point.infected
                length_arrow_x = node.actions[action].end_point.immune - start_arrow_x
                length_arrow_y = node.actions[action].end_point.infected - start_arrow_y
                ax.arrow(start_arrow_x, start_arrow_y, length_arrow_x, length_arrow_y, width=0.0005, head_width = 0.001, color = colors[action])
        projection = self.project_epidemic(1, st/s0, it/s0, date_range)
        ax.plot(projection['immune'], projection['infected'], lw = 5, color = 'k')
        plt.show()

            
    def project_epidemic(self, s0, st, it, date_range):
        """
        This function projects the path of the epidemic through the state space
        """
        projected = pd.DataFrame(columns = ['date', 'policy','immune', 'infected','new infections'])
        infected = it / s0
        immune =  1 - st / s0
        for date in date_range:
            best_policy_index = self.find_best_policy(immune, infected)
            policy_r0 = self.policies[best_policy_index].r0
            policy = self.policies[best_policy_index].name
            immune_after, infected_after = SIR_step(immune, infected, policy_r0, self.serial_interval)
            new_infections = infected_after - infected + immune_after - immune
            immune = immune_after
            infected = infected_after
            projected = projected.append({'date': date, 'policy': policy, 'immune': immune*s0, 'infected': infected*s0, 'new infections': new_infections*s0}, ignore_index=True)
        return(projected)
                
        








