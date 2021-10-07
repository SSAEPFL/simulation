import datetime
import Database_Simulation as ds
import pandas as pd
import os
import json

# process the TLE/3LE
filename = '3le_all.txt'
path = 'D:/COURS 2020-2021/Projet IC/Simulation/Data/satellites.json'

num_satellites = 5  # Number maximum of satellites to simulate
duration = 1  # Number of hours in the simulation
sec_step = 5  # Number of seconds per time step
start = datetime.datetime.now()  # Start of simulation

#data = ds.readsDataframe('Data/satellites.json')
data = ds.initializeDatabase(ds.process3LE(filename)[:num_satellites], start, duration=duration, sec_step=sec_step)
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(data)
ds.saveDataframe(data, path)
