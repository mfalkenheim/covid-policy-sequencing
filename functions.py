"""
Module contains the functions of the SIR recursion model
"""

import numpy as np
from math import exp
from math import log
import pandas as pd


def make_shares(n):
    x = np.linspace(0, (n - 1) / n, n)
    return(x)

def make_infected_levels(log_minimum, log_maximum, k):
    y = np.linspace(log_minimum, log_maximum, k)
    return(np.exp(y) / (np.exp(y) + 1))

def calculate_deaths(infections_start, infections_end, immune_start, immune_end, fatality_rate,
                    healthcare_capacity, overcapacity_fatality_rate):
    if immune_start == []:
        deaths_normal = infections_start * fatality_rate
        deaths_overcapacity = infections_start * (overcapacity_fatality_rate - fatality_rate) * \
            ((infections_start - healthcare_capacity) / infections_start)**2 \
                if infections_start > healthcare_capacity else 0
        return(deaths_normal + deaths_overcapacity)
    recoveries_deaths = immune_end - immune_start
    max_infections = max(infections_start, infections_end)
    min_infections = min(infections_start, infections_end)
    if max_infections >= healthcare_capacity:
        if min_infections >= healthcare_capacity:
            average_infections = (infections_start + infections_end) / 2
            weighted_fatality_rate = fatality_rate + (overcapacity_fatality_rate - fatality_rate) \
                * (average_infections - healthcare_capacity) / average_infections
            return(weighted_fatality_rate * recoveries_deaths)
        else:
            trapezoid = (min_infections + max_infections) / 2
            triangle = (max_infections - healthcare_capacity) ** 2 / (max_infections - min_infections)
            share_overcapacity = triangle / trapezoid 
            weighted_fatality_rate = fatality_rate + share_overcapacity * \
                (overcapacity_fatality_rate - fatality_rate) 
            return(weighted_fatality_rate * recoveries_deaths)
    else:
        return(recoveries_deaths * fatality_rate)

def calculate_time(infections_start, infections_end, immune_start, immune_end, serial_interval):
    recoveries_deaths = immune_end - immune_start
    if infections_start == infections_end:
        return(recoveries_deaths / infections_start * serial_interval)
    elif infections_end == 0:
        return(2 * recoveries_deaths / infections_start * serial_interval)
    else:
        return(log(infections_end / infections_start) * recoveries_deaths \
               / (infections_end - infections_start) * serial_interval)

def find_next_point(immune, infected, slope, level_after):
    slope_threshold = -infected / (level_after - immune)
    if slope > slope_threshold:
        immune_after = (level_after - infected + slope * immune) / (1 + slope)
        infected_after = level_after - immune_after
        return(immune_after, infected_after)
    else:
        x_intercept = immune - infected / slope
        return(x_intercept, 0)

        
def find_continuation_cost(immune, infected, cost_function, deaths_if_vaccine, time_lapse, time_to_vaccine, herd_immunity_thresholds, minimum_infected):
    if infected == 0:
        continuation_cost = 1
        for i, threshold in enumerate(herd_immunity_thresholds[0]):
            if threshold <= immune:
                continuation_cost = min(herd_immunity_thresholds[1][i], continuation_cost)
        return(continuation_cost)
    else:
        cost_no_vaccine = cost_function(infected)
        probability_vaccine = 1 - ((time_to_vaccine - 1) / time_to_vaccine) ** time_lapse
        weighted_cost = probability_vaccine * deaths_if_vaccine \
            + (1 - probability_vaccine) * cost_no_vaccine
        return(weighted_cost)
    
def SIR_step(immune, infected, r0, serial_interval):            
    growth_infected = ((1 - immune - infected) * r0 - 1) / serial_interval
    new_immune = max(0, min(1, immune + infected / serial_interval)) 
    new_infected = max(0, min(1, infected * (1 + growth_infected)))
    return(new_immune, new_infected)

def SIR_projection(s0, st, it, r0, serial_interval, date_range):
    projected = pd.DataFrame(columns = ['date', 'immune', 'infected','new infections'])
    infected = it / s0
    immune =  1 - st / s0 - it / s0
    for date in date_range:
        immune_after, infected_after = SIR_step(immune, infected, r0, serial_interval)
        new_infections = infected_after - infected + immune_after - immune
        immune = immune_after
        infected = infected_after
        projected = projected.append({'date': date, 'immune': immune*s0, 'infected': infected*s0, 'new infections': new_infections*s0}, ignore_index=True)
    return(projected)

def phased_projection(s0, st, it, r0s, serial_interval, date_ranges):
    projected = pd.DataFrame(columns = ['date', 'immune', 'infected','new infections'])
    for index, r0 in enumerate(r0s):
        date_range = date_ranges[index]
        projected = projected.append(SIR_projection(s0, st, it, r0, serial_interval, date_range))
        st = s0 - projected.iloc[-1]['immune'] - projected.iloc[-1]['infected']        
        it = projected.iloc[-1]['infected']
    return(projected)
        