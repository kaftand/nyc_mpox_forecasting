## Environment Setup

Using python 3.9.18 on windows

    python -m venv venv
    .venv/Scripts/activate
    pip install -r requirements.txt

## Running a calibration

running monkeypox_optuna.py will calibrate 

    usage: python monkeypox_optuna.py [-h] [--vaccine_file VACCINE_FILE]
                                    --cases_file CASES_FILE
                                    [--population POPULATION]
                                    [--ntrials NTRIALS]

    optional arguments:
    -h, --help            show this help message and exit
    --vaccine_file VACCINE_FILE
                            File with vaccine dose and date pairs. Should be CSV
                            with columns: date,vx_doses
    --cases_file CASES_FILE
                            File with daily case counts over time. Should be CSV
                            with columns: diagnosis_date,count Simulation and
                            calibration will occur only for the dates in this file
    --population POPULATION
                            Susceptible population
    --ntrials NTRIALS     Number of trials to calibrate

## Running a simulation

    usage: python scenario_monkeypox.py [-h] [--vaccine_file VACCINE_FILE]
                                        [--vaccine_effectiveness VACCINE_EFFECTIVENESS]
                                        [--social_distance_start_date SOCIAL_DISTANCE_START_DATE]
                                        [--social_distance_effect SOCIAL_DISTANCE_EFFECT]
                                        [--incubation_period INCUBATION_PERIOD]
                                        [--infectious_period INFECTIOUS_PERIOD]
                                        [--r0 R0] [--population POPULATION]
                                        [--initial_exposed INITIAL_EXPOSED]
                                        [--date_start DATE_START]
                                        [--date_end DATE_END]

    optional arguments:
    -h, --help            show this help message and exit
    --vaccine_file VACCINE_FILE
                            File with vaccine dose and date pairs. Should be CSV
                            with columns: date,vx_doses
    --vaccine_effectiveness VACCINE_EFFECTIVENESS
                            effectiveness of a single vaccine does [0,1]
    --social_distance_start_date SOCIAL_DISTANCE_START_DATE
                            Start date at which attack rate starts to slow.
                            Format: YYYY-MM-DD
    --social_distance_effect SOCIAL_DISTANCE_EFFECT
                            Multiplier to attack rate due to social distancing
    --incubation_period INCUBATION_PERIOD
                            Duration of incubation period in days.
    --infectious_period INFECTIOUS_PERIOD
                            Duration of infectious period.
    --r0 R0
    --population POPULATION
                            Susceptible population
    --initial_exposed INITIAL_EXPOSED
                            Number exposed at t=0
    --date_start DATE_START
                            Start date of simulation. Format YYYY-MM-DD
    --date_end DATE_END   End date of simulation. Format YYYY-MM-DD
