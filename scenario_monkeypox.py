
import os

import random
import sys

random.seed(4475772444933854010)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import argparse
# For time-series modeling
from seir_monkeypox import SEIR
from helpers_monkeypox import *
np.set_printoptions(threshold=np.inf)

output_path = "./"    
region = 'nyc'



def calc_cost(sim_results, percent_diagnosed, actual_cases):
    #The following line was used when we were updating cases in real time and 
    # live case counts were still being updated
    #case_df = actual_cases.loc[actual_cases['incomplete'] == '-',:].copy()
    case_df = actual_cases.copy().set_index("diagnosis_date")
    sim_results_sc = (sim_results * percent_diagnosed).rename("sim_counts")
    case_df = case_df.join(sim_results_sc)
    case_df['err'] = case_df["count"] - case_df['sim_counts']
    case_df['se'] = case_df['err'] * case_df['err']
    return case_df.loc[:,"se"].dropna().mean()

def save_output_to_csv (new_infections, output_combined, append=''):
    output_dir = os.path.join(output_path,"epidemic_thresholds/")
    os.makedirs(os.path.join(output_path,"epidemic_thresholds/"), exist_ok=True)
        
    new_infections.to_csv(os.path.join(output_dir, "nyc_new_infections_%s" % append)) #Number of new daily infections only 
    output_combined.to_csv(os.path.join(output_dir, "nyc_combined_%s" % append)) #Number for all compartments
    
def samples_2_df ( samples, time):
    outputS = samples[:, 0, 1:]
    outputE = samples[:, 1, 1:] 
    outputI = samples[:, 2, 1:] 
    outputR = samples[:, 3, 1:] 
    output_vacc_actual = samples[:, 4, 1:]
    output_newinfections = samples[:, 5, 1:]
    
    # Convert to dataframe for clarity
    output_newinfections = pd.DataFrame(output_newinfections.T,
                            columns=["s{}".format(i) for i in range(output_newinfections.shape[0])],
                            index=time[1:]).median(axis=1)

    outputS = pd.DataFrame(outputS.T,
                            columns=["s{}".format(i) for i in range(outputS.shape[0])],
                            index=time[1:]).median(axis=1)                                           

    outputE = pd.DataFrame(outputE.T,
                            columns=["s{}".format(i) for i in range(outputE.shape[0])],
                            index=time[1:]).median(axis=1)                                              

    outputI = pd.DataFrame(outputI.T,
                            columns=["s{}".format(i) for i in range(outputI.shape[0])],
                            index=time[1:]).median(axis=1)  

    outputR = pd.DataFrame(outputR.T,
                            columns=["s{}".format(i) for i in range(outputR.shape[0])],
                            index=time[1:]).median(axis=1)  
    
    output_vacc_actual = pd.DataFrame(output_vacc_actual.T,
                                columns=["s{}".format(i) for i in range(output_vacc_actual.shape[0])],
                                index=time[1:]).median(axis=1)
    
    output_combined = pd.concat([outputS, outputE, outputI, outputR, output_vacc_actual, output_newinfections], axis=1).reindex(outputS.index)
    return output_newinfections, output_combined

def sim_monkeypox(immune_num_effects, 
                    social_distance_start_date = "2022-06-15", 
                    social_distance_effect =  0.95, 
                    D_e = 10, 
                    D_i = 21, 
                    monkeypox_base_R0 = 2.0, 
                    population = 70_180, 
                    g_i = 3, 
                    write_output_to_csv=False,
                    time_start="2022-05-01",
                    time_end="2022-09-27"):

    # Reindex the scenarios to a particular time
    # horizon.
    time = pd.date_range(start=time_start, end=time_end, freq="d")

    #Assumes some social distancing / reducing number of partners to reduce contact numbers that affects R0 reduction 
    #This assumption needs to be calibrated better
    dates2= pd.date_range(start = social_distance_start_date,periods=30, freq='d')
    values2 = sigmoid_interpolation(start=1, end = social_distance_effect, num=len(dates2))         

    #Basically it updates the values of R0 on the corresponding dates   
    social_distancing_dates_effects = dict(zip(dates2, values2))
    
    dataset = pd.DataFrame({"date":[], "importations":[], "cases":[] })
    dataset = dataset.set_index('date').fillna(0)
    
    scenarios = dataset.reindex(time).fillna(0)
    
    #Initial importations
    scenarios["hypothetical"] = scenarios["importations"].copy()
    scenarios.loc[scenarios.index.min(), "hypothetical"] = g_i
    
    
    # Set up the importation scenarios
    # Initialize model for hypothetical scenario
    hypothetical_scenario = SEIR(S0=population,
                                        D_e=D_e,
                                        D_i=D_i,
                                        z_t=scenarios["hypothetical"].values,
                                        add_z_to_inf=True)                                                                                 
    hypothetical_scenario.R0 = monkeypox_base_R0

    immune_num_t = pd.Series(0 * np.ones((len(time),)), index=time, name="immune_s1")
    immune_num_t = apply_intervention(immune_num_t , immune_num_effects)
    
    #Set up the attrack rate 
    beta_t = pd.Series(((monkeypox_base_R0 * (1/hypothetical_scenario.D_i))/hypothetical_scenario.S0) 
                    * np.ones((len(time),)),
                    index=time, name="beta_s1")
    
    # Do you want some social distancing? Modify attack rate based on function
    beta_t = apply_social_distancing_to_beta(beta_t, social_distancing_dates_effects)
    
    # Run replicates of scenarios
    population_samples = hypothetical_scenario.sample_scenario(beta_t.values, immune_num_t.values)

    # Save output
    append = "nyc_monkeypox_R0_%0.2f.csv" % (monkeypox_base_R0) 
    ts, ts_all = samples_2_df(population_samples, time=time)
    if write_output_to_csv:
        save_output_to_csv(ts, ts_all, append)
    return ts



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python scenario_monkeypox.py")
    parser.add_argument("--vaccine_file",
                        default=None,
                        help="""File with vaccine dose and date pairs. Should be CSV with columns:
                        date,vx_doses""")
    parser.add_argument("--vaccine_effectiveness",
                        default=0.77,
                        help="""effectiveness of a single vaccine does [0,1]""",
                        type=float)
    parser.add_argument("--social_distance_start_date",
                        default="2022-07-16",
                        help="""Start date at which attack rate starts to slow.
                        Format: YYYY-MM-DD""")
    parser.add_argument("--social_distance_effect",
                        default=0.56,
                        help="""Multiplier to attack rate due to social distancing""",
                        type=float)
    parser.add_argument("--incubation_period",
                        default=7.5,
                        help="""Duration of incubation period in days.""",
                        type=float)
    parser.add_argument("--infectious_period",
                        default=14,
                        help="""Duration of infectious period.""",
                        type=int)
    parser.add_argument("--r0",
                        default=3.8,
                        type=float)
    parser.add_argument("--population",
                        default=70_161,
                        help="""Susceptible population""",
                        type=int)
    parser.add_argument("--initial_exposed",
                        default=19,
                        help="Number exposed at t=0",
                        type=int)
    parser.add_argument("--date_start",
                        default="2022-05-01",
                        help="""Start date of simulation. Format YYYY-MM-DD""")
    parser.add_argument("--date_end",                        
                        default="2022-09-27",
                        help="""End date of simulation. Format YYYY-MM-DD""")
    args = parser.parse_args(sys.argv[1:])
    vaccine_trend =  pd.read_csv(parser["vaccine_file"], parse_dates=["date"]) if args.vaccine_file else pd.DataFrame({"date":[],"vx_doses":[]})
    vaccine= np.array(vaccine_trend['vx_doses'])
    vaccine_date = vaccine_trend.date
    vaccine_date_list = list(vaccine_date)
    immune_date_list = list(vaccine_date + timedelta(days=5))
    immune_date = [date.strftime('%Y-%m-%d') for date in immune_date_list]
    
    #Zip as a dictionary for vaccindation dates and the corresponding daily dose
    immune = dict(zip(immune_date, vaccine*args.vaccine_effectiveness))

    ts = sim_monkeypox(immune, 
                       monkeypox_base_R0=args.r0, 
                       g_i=args.initial_exposed, 
                       social_distance_effect=args.social_distance_effect, 
                       social_distance_start_date=args.social_distance_start_date,
                       D_e = args.incubation_period, 
                       D_i = args.infectious_period, 
                       population = args.population, 
                       write_output_to_csv=True,
                       time_start=args.date_start,
                       time_end=args.date_end)

