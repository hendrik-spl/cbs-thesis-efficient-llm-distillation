import pandas as pd

def load_emissions_csv(filename="results/metrics/emissions/emissions.csv"):
    return pd.read_csv(filename)

def get_emissions_data(experiment_id):
    df = load_emissions_csv()
    return df[df["experiment_id"] == experiment_id]

def get_duration(experiment_id):
    df = get_emissions_data(experiment_id)
    return df["duration"].values[0]