import pandas as pd
import numpy as np
import datetime
import Simulation_Tools as st
import os


def saveDataframe(data, path):
    """
    Saves the given dataframe in the specified path

    Parameters
    ----------
    data: The dataframe to save
    path: The path to save to, can be .csv .json or .excel; For instance 'Data/satellites.json'

    Returns
    ----------
    df: Dataframe containing every unique object with its two corresponding lines
    """

    format = path.split('.')[-1]
    folder = '/'.join(path.split('/')[:-1])

    print("\n--------\n")
    if os.path.exists(folder):
        if format == 'csv':
            data.to_csv(path, index=False)
            print("Saved the file as a csv.")
        elif format == 'json':
            data.to_json(path, orient="records")
            print("Saved the file as a json.")
        elif format == 'excel':
            data.to_excel(path, index=False)
            print("Saved the file as excel.")
        else:
            print("Wrong extension, didn't manage to save the file ! Accepted extension : csv, json or excel")
    else:
        print("Specified folder doesn't exist !")
    print()


def readsDataframe(path):
    """
    Reads the given dataframe in the specified path

    Parameters
    ----------
    path: The path to read from, can be .csv .json or .excel; For instance 'Data/satellites.json'

    Returns
    ----------
    df: Dataframe containing every unique object with its two corresponding lines
    """

    format = path.split('.')[-1]
    folder = '/'.join(path.split('/')[:-1])
    data = pd.DataFrame({'A': [np.nan]})

    print("\n--------\n")
    if os.path.exists(folder):
        if format == 'csv':
            data = pd.read_csv(path, index=False)
            print("Read the file as a csv.")
        elif format == 'json':
            data = pd.read_json(path, orient="records")
            print("Read the file as a json.")
        elif format == 'excel':
            data = pd.read_excel(path, index=False)
            print("Read the file as excel.")
        else:
            print("Wrong extension, didn't manage to read the file ! Accepted extension : csv, json or excel")
    else:
        print("Specified folder doesn't exist !")
    print()

    return data


def process3LE(tlefilename):
    """
    Process a .txt 3LE file, to get distinct satellites in a 2D dataframe

    Parameters
    ----------
    tlefilename: .txt file of 3LE data

    Returns
    ----------
    df: Dataframe containing every unique object with its two corresponding lines
    """

    list_of_satellites = st.getSatelliteFromTLE(tlefilename)

    print('\n--------\nLoaded', len(list_of_satellites), 'TLEs\n')

    # removes duplicates from a 3le file. The satellites need to be ordered by norad id number
    processed_list = []
    for i in range(len(list_of_satellites)):
        if list_of_satellites[i][0] != 'TBA - TO BE ASSIGNED':
            if i + 1 < len(list_of_satellites) and list_of_satellites[i + 1][0] == list_of_satellites[i][0]:
                continue
            else:
                processed_list.append(list_of_satellites[i])

    print('Removed duplicates,', len(processed_list), 'objects remaining')

    return pd.DataFrame(processed_list, columns=['Satellite Name', 'Line 1', 'Line 2'])


def initializeDatabase(satellites, start, duration=48, sec_step=5):
    """
    Computes the complete space objects' database from scratch

    Parameters
    ----------
    satellites: Dataframe of every unique objects, containing its name and its TLE
    start: Datetime at when to start the search
    duration: Optional - The number of hours in the simulation
    sec_step: Optional - The number of seconds per time step (can be a fraction)

    Returns
    ----------
    df: The complete database
    """

    columns = ["Index", "Name", "Line 1", "Line 2", "Times", "Positions", "Velocities", "Nearest", "NNTimes",
               "Real Nearest", "Real NNTimes", "Collisions"]
    data = []

    print("\n--------\n Initialization of the database...\n")
    # Do a first pass to fill the database, with the information computable from each object individually
    for i in range(len(satellites)):
        sat = satellites.iloc[i]
        data.append([])

        data[i].append(i)  # Index
        data[i].append(sat[0])  # Name
        data[i].append(sat[1])  # Line 1
        data[i].append(sat[2])  # Line 2
        data[i].append((start.isoformat(), sec_step))  # Times
        p, v = st.computeTrajectory(sat, start, duration, sec_step)
        data[i].append(p)  # Positions
        data[i].append(v)  # Velocities
    df1 = pd.DataFrame(data, columns=columns[:7])

    # Do a second pass, to add information related to the nearest neighbors
    for i in range(len(satellites)):
        print("Satellite %s/%s" % (i + 1, len(satellites)))
        nn, t, c, r_nn, r_t = st.allNNofObject(df1, i, 0, start, sec_step)

        data[i].append(nn)  # Nearest Approching Neighbor
        data[i].append(t)  # NNTimes
        data[i].append(r_nn)  # Real Names for NN
        data[i].append(r_t)  # Real date for NNTimes
        data[i].append(c)  # Collisions

    df = pd.DataFrame(data, columns=columns)
    df = df[df['Positions'].map(len) != 0]
    print("\nInitialization of the database complete.")
    return df


def insert(tle, data):
    """
    Checks if the given object is already inside the database (by name, or if the given object is similar in positions to another one)
    If not, inserts one object in the database after computing every needed information

    Parameters
    ----------
    tle: string format 3LE of the object to add
    data: Dataset of all objects and their related information

    Returns
    ----------
    new_data: The new complete database
    """

    start = data['Times'][0][0]
    sec_step = data['Times'][0][1]
    duration = len(data['Positions'][0])
    hour_duration = duration * sec_step / 3600

    sat = st.getSatelliteFromTLE(tle)[0]
    start = datetime.datetime.strptime(start, '%Y-%m-%dT%H:%M:%S.%f')

    print("\n--------\nComputation of object %s...\n" % sat[0])

    p, v = st.computeTrajectory(sat, start, hour_duration, sec_step)
    closest, _, _ = st.nearestApprochingNeighbor(data, -1, 0, positions=p)

    if len(searchByName(sat[0][0], data)) > 0 or st.isClosestSameObject(p, data['Positions'][closest]):
        print("The given object is %s, already in the database ! No change made.\n" % data['Name'][closest])
        return data

    print("Insertion of object %s..." % sat[0])
    nn, t, c, r_nn, r_t = st.allNNofObject(data, -1, 0, start, sec_step, positions=p)

    # The new row to add to the database
    row = pd.Series([len(data), sat[0], sat[1], sat[2], (start, sec_step), p, v, nn, t, r_nn, r_t, c],
                    index=["Index", "Name", "Line 1", "Line 2", "Times", "Positions", "Velocities", "Nearest",
                           "NNTimes", "Real Nearest", "Real NNTimes", "Collisions"])

    print("New object correctly inserted in the database.\n")

    return data.append(row, ignore_index=True)


def delete(satellites, data):
    """
    Deletes given objects from the database

    Parameters
    ----------
    satellites: Array of all names of objects to delete
    data: Dataset of all objects and their related information

    Returns
    ----------
    new_data: The new complete database
    """

    print("\n--------\nDeletion of object(s)...\n")

    args = []
    # Finds the indexes of given objects
    for sat in satellites:
        arg = np.where(data['Name'] == sat)[0]
        if len(arg) == 0:
            print("Object %s doesn't exist in the database !" % sat)
        else:
            args.append(arg[0])
            print("Deleted object %s from the database." % sat)

    if len(args) == 0:
        print("\nNo objects removed !")
        new_data = data
    else:
        new_data = data.drop(args)
        print("\nNew dataset complete.")
    return new_data


def searchByName(name, data):
    """
    Returns the row of the given object, depending on its name

    Parameters
    ----------
    data: Dataset of all objects and their related information
    name: String format of the name of the object to search for

    Returns
    ----------
    row: The complete row of the given object, as a dataframe of 1 line
    """

    return data.loc[data['Name'] == name]


def updateDatabase(data, seconds_to_add):
    """
    Updates the database to add a given number of seconds of simulation

    Parameters
    ----------
    data: Dataset of all objects and their related information
    seconds_to_add: Number of seconds to add to the simulation

    Returns
    ----------
    new_data: The new complete database
    """

    print("\n--------\nUpdate of database...\n")

    sat = data['Name'].tolist()
    l1 = data['Line 1'].tolist()
    l2 = data['Line 2'].tolist()

    satellites = pd.DataFrame(list(zip(sat, l1, l2)), columns=['Satellite Name', 'Line 1', 'Line 2'])
    new_data = st.computeUpdateDatabase(data, satellites, seconds_to_add)

    print("Database correctly updated.\n")

    return new_data
