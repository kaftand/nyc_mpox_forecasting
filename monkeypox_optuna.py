import sys
import optuna
import scenario_monkeypox
import datetime as dt
import pandas as pd
import numpy as np
import argparse
from functools import partial

def objective(trial : optuna.trial.BaseTrial, immune_date, vaccine, actual_cases, population):
    """
    Objective function for optuna. First, the parameters are set. Then, the simulation is run.
    Finally, the mean squared error is calculated and returned.
    """
    monkeypox_base_R0 = trial.suggest_float('monkeypox_base_R0', 0.5, 4)
    g_i = trial.suggest_int("g_i",1, 20)
    social_distance_effect = trial.suggest_float('social_distance_effect', 0.5, 1. )
    percent_diagnosed = trial.suggest_float("percent_diagnosed", 0.05, 0.5)
    social_distance_date_offset = trial.suggest_int("social_distance_date_offset", -15, 15)
    vaccine_efficacy = trial.suggest_float("vaccine_efficacy", 0.55, 0.95)
    incubation_period = trial.suggest_float("incubation_period", low=6.6, high=10.9) #https://doi.org/10.2807/1560-7917.ES.2022.27.24.2200448
    infectious_period = trial.suggest_float("infectious_period", low=14, high=28)
    ts = ts_wrapper(vaccine_efficacy,infectious_period=infectious_period,
                    incubation_period=incubation_period, 
                    monkeypox_base_R0=monkeypox_base_R0, g_i=g_i, 
                    social_distance_effect=social_distance_effect, 
                    social_distance_date_offset=social_distance_date_offset, 
                    immune_date=immune_date, vaccine=vaccine, population=population, 
                    time_start=actual_cases.diagnosis_date.min(), time_end=actual_cases.diagnosis_date.max())
    return scenario_monkeypox.calc_cost(ts, percent_diagnosed, actual_cases)

def ts_wrapper(vaccine_efficacy, infectious_period, incubation_period, monkeypox_base_R0, g_i, social_distance_effect, social_distance_date_offset, immune_date, vaccine, population=70_180, **kwargs):
    """
    Wrapper around the simulation for easy calling by the objective function.
    """
    social_distance_date  = dt.date(2022, 7, 15) + dt.timedelta(days=social_distance_date_offset)
    immune = dict(zip(immune_date, vaccine*vaccine_efficacy))
    ts = scenario_monkeypox.sim_monkeypox(immune, monkeypox_base_R0=monkeypox_base_R0, g_i=g_i, 
                                            social_distance_effect=social_distance_effect, social_distance_start_date=social_distance_date, 
                                            D_e = incubation_period, D_i=infectious_period, population=population, **kwargs)
    return ts


def best_fit(params: dict, immune_date, vaccine, output_file="simulated_results.csv"):
    """
    Plot a set of params and save the results to a csv
    """
    percent_diagnosed = params.pop('percent_diagnosed', 1)
    ts = ts_wrapper(**params, immune_date=immune_date, vaccine=vaccine, write_output_to_csv=True)
    (ts * percent_diagnosed).to_csv(output_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="python monkeypox_optuna.py")
    parser.add_argument("--vaccine_file",
                        default=None,
                        help="""File with vaccine dose and date pairs. Should be CSV with columns:
                        date,vx_doses""")
    parser.add_argument("--cases_file",
                        required=True,
                        help="""File with daily case counts over time. Should be CSV with columns:
                        diagnosis_date,count
                        \n
                        Simulation and calibration will occur only for the dates in this file""")
    parser.add_argument("--population",
                        default=70_180,
                        help="""Susceptible population""",
                        type=int)
    parser.add_argument("--ntrials",
                        default=10_000,
                        help="""Number of trials to calibrate""",
                        type=int)
    args = parser.parse_args(sys.argv[1:])
    actuals = pd.read_csv(args.cases_file, parse_dates=["diagnosis_date"])
    vaccine_trend =  pd.read_csv(args.vaccine_file, parse_dates=["date"]) \
                        if args.vaccine_file \
                        else pd.DataFrame({"date":actuals.diagnosis_date.values,"vx_doses":0*actuals['count'].values})
    vaccine= np.array(vaccine_trend['vx_doses'])
    vaccine_date = vaccine_trend.date
    vaccine_date_list = list(vaccine_date)
    immune_date_list = list(vaccine_date + dt.timedelta(days=5))
    immune_date = [date.strftime('%Y-%m-%d') for date in immune_date_list]

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=10))
    study_objective = partial(objective, population=args.population, immune_date=immune_date, vaccine=vaccine, actual_cases=actuals)
    study.optimize(study_objective, n_trials=args.ntrials)
    print(study.best_params)
    ts = best_fit( study.best_params, immune_date, vaccine)

