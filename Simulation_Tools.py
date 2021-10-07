from sgp4.api import Satrec, jday, days2mdhms, WGS72
import pandas as pd
import numpy as np
import datetime


def getSatelliteFromTLE(tleFileName):
    """
    Process a .txt 3LE file, to get all satellites

    Parameters
    ----------
    tleFileName: .txt file of 3LE data

    Returns
    ----------
    arr: 3xN Array containing all objects with its two corresponding lines
    """

    file = open(tleFileName, 'r')
    list_of_satellites = [[]]

    i = 0
    try:
        for line in file:
            # Satellite Name
            if line[0] == str(0):
                list_of_satellites[i].append(line[2:-1])  # Removes the number of the line and the '\n' character
            # Line 1
            if line[0] == str(1):
                list_of_satellites[i].append(line[:-1])  # Removes the '\n' character
            # Line 2
            if line[0] == str(2):
                list_of_satellites[i].append(line[:-1])  # Removes the '\n' character
                list_of_satellites.append([])
                i = i + 1
    except (AttributeError, TypeError):
        raise AssertionError('Input variables should be strings')

    list_of_satellites.pop(len(list_of_satellites) - 1)

    return list_of_satellites


def computeTrajectory(sat_lines, start, duration, sec_step):
    """
    Computes the trajectory of a specific satellite, given a duration and step time

    Parameters
    ----------
    sat_lines: 3LE of the given object in an array of size 3
    start: Start of the simulation, in datetime format
    duration: Number of hours in the simulation
    sec_step: Number of seconds per time step (can be a fraction)

    Returns
    ----------
    arr: Array containing the complete state vectors (ie. positions and velocities) per time step
    """

    trajectory = [[], []]

    num_iterations = duration * 3600 / sec_step  # Number of iterations
    jday_step = 1 / 24 / 3600 * sec_step  # Step as a fraction of a day

    # Get julian date from datetime
    jd, fr = jday(start.year, start.month, start.day, start.hour, start.minute, start.second)

    # Create a satellite object, given the TLE
    satellite = Satrec.twoline2rv(sat_lines[1], sat_lines[2])

    for i in range(int(num_iterations)):
        e, r, v = satellite.sgp4(jd, fr + i * jday_step)

        if e == 0:
            trajectory[0].append(r)  # True Equator Mean Equinox position (km)
            trajectory[1].append(v)  # True Equator Mean Equinox velocity (km/s)
        # else:
        # print("error %s while computing trajectory of satellite "%e, sat_line[0])

    return np.array(trajectory)


def computeDerivatives(velocities, timeArg):
    """
    Computes the derivative of the velocity at a given time.

    Parameters
    ----------
    velocities: Array of velocity vectors per time step
    timeArg: Index at where to derivate the velocity

    Returns
    ----------
    V_derivative: Derivative of the velocity
    """

    return 0


def percentageOfCollision(data, actual, nn, time):
    """
    Computes the percentage of collision between two satellites at a given time

    Parameters
    ----------
    data: Array of velocity vectors per time step
    actual: Actual object
    nn: Nearest neighbor to check the collision risk with
    time: Index at when to compute the collision risk

    Returns
    ----------
    collision: Collision risk between both objects
    """

    return 0.0


def timeForCollision(data_a, data_nn, time):
    """
    Computes the time where the two given objects intersect.

    Parameters
    ----------
    data_a: Array of position of the actual object
    data_nn: Array of position of the nearest neighbor object
    time: Index at when to start the search

    Returns
    ----------
    closest_time: Index at where the two objects intersect
    """

    closest_time = time
    final_time = len(data_a)

    # For every possible time step, see when the two objects intersect
    for t in range(time, final_time - 1):
        d1 = np.linalg.norm(data_nn[t] - data_a[t])  # Distance at actual position
        d2 = np.linalg.norm(data_nn[t + 1] - data_a[t + 1])  # Distance after 1 time step

        # If the distance after 1 time step is greater than before, then the two objects have intersected
        if d2 > d1:
            closest_time = t
            break
        # If the time reached the end, return Infinite
        elif t == final_time - 1:
            closest_time = float('inf')

    return closest_time


def nearestApprochingNeighbor(data, actual, time, positions=None):
    """
    Do a Nearest Neighbor search for a given object, over the complete dataset, keeping only the approaching neighbors

    Parameters
    ----------
    data: Dataset of all objects and their related information
    actual: Actual object from which to do the NN search - put -1 if not in database
    time: Index at when to start the search
    positions: Optional - Array of positions of the actual object, if not in database already

    Returns
    ----------
    closest[0]: Number (index from data) of the NN
    t: Index at when the two objects intersect
    collision: Collision probability
    """

    closest = [-1, float('inf')]  # Set a variable to keep track of the NN, with (index of object in data, distance)
    t = float('inf')  # Set a variable to keep track of time steps
    collision = -1  # Collision probability with NN

    if positions is None:
        positions = data['Positions'][actual]

    # Ends if it is the last argTime of the Positions
    if time + 1 >= len(positions):
        return closest[0], t, collision

    position1 = np.array(positions[time])  # Actual position
    position2 = np.array(positions[time + 1])  # Position after 1 time step

    # Check the distance for every other object than the actual and takes the smallest
    for sat in range(len(data)):
        if sat != actual:
            other_positions = data.iloc[sat]["Positions"]

            # If we get to the last time_step, check the next satellite
            if time + 1 >= len(other_positions):
                continue

            p1 = np.array(other_positions[time])
            p2 = np.array(other_positions[time + 1])

            d1 = np.linalg.norm(p1 - position1)  # Norm of the distance at the actual position
            d2 = np.linalg.norm(p2 - position2)  # Norm of the distance after 1 time step

            # If the distance between them gets closer after 1 time step and this distance is lower as compared
            # to the previous objects, then save the object in question
            if closest[1] > d1 > d2:
                closest[0] = sat
                closest[1] = d1

    # If a NN is found, then computes its time for intersection and collision probability
    if closest[0] != -1:
        t = timeForCollision(positions, data.iloc[closest[0]]["Positions"], time)
        collision = percentageOfCollision(data, actual, closest[0], t)

    return closest[0], t, collision


def allNNofObject(data, actual, fromT, start, sec_step, positions=None):
    """
    Computes all of the Nearest approaching Neighbors of a given object, from a given time to the last possible

    Parameters
    ----------
    data: Dataset of all objects and their related information
    actual: Actual object from which to do the NN search - put -1 if not in database
    fromT: Index at when to start the search
    start: Datetime at when to start the search
    sec_step: Number of seconds per time step (can be a fraction)
    positions: Optional - Array of positions of the actual object, if not in database already

    Returns
    ----------
    nn: Array of all nearest neighbors through time
    times: Array of all nearest neighbors intersections index-time
    collisions: Array of all nearest neighbors collision probability
    real_nn: Array of all nearest neighbors real names through time
    real_times: Array of all nearest neighbors intersections datetimes
    """

    nn = []  # Array with every nearest neighbor
    real_nn = []  # Array with names of nn
    times = []  # Array with times of collision
    real_times = []  # Array with datetime format of collisions
    collisions = []  # Array with percentages of collisions

    t = fromT
    final_time = len(data['Positions'][actual]) if actual != -1 else len(data['Positions'][0])
    # For every possible times, compute the nearest approaching neighbor
    while t < final_time:
        print("Time %s/%s" % (t+1, final_time), end='\r')
        n, t2, c = nearestApprochingNeighbor(data, actual, t, positions)  # Get the index for the nn

        if t2 != float('inf'):
            # Append the different wanted values
            nn.append(n)
            real_nn.append(data.iloc[n]["Name"] if t2 != float('inf') else 'None')
            times.append(t2)
            new_date = (start + datetime.timedelta(seconds=t2 * sec_step)).isoformat() if t2 != float('inf') else None
            real_times.append(new_date)
            collisions.append(c)

        t = t2 + 1 if t2 != float('inf') else t+1

    print()
    return nn[:-1], times[:-1], collisions[:-1], real_nn[:-1], real_times[:-1]


def isClosestSameObject(actual_pos, closest_pos, epsilon=50, nb_iter_to_check=10):
    """
    Checks if the nearest neighbor of an object is the actual object, given their positions

    Parameters
    ----------
    actual_pos: Array of positions of the actual
    closest_pos: Array of positions of the NN
    epsilon: Optional - Maximum distance allowed between both objects
    nb_iter_to_check: Number of iterations to check between both objects

    Returns
    ----------
    isClosest: Boolean, True if they are considered the same object, False otherwise
    """

    for i in range(nb_iter_to_check):
        # Checks x, y, z of each position
        if abs(actual_pos[i][0] - closest_pos[i][0]) < epsilon:
            if abs(actual_pos[i][1] - closest_pos[i][1]) < epsilon:
                if abs(actual_pos[i][2] - closest_pos[i][2]) < epsilon:
                    return True

    return False


def addUpdate(data, sat, start, nn, t, r_nn, r_t, c):
    """
    Updates every column information of one object in the given dataset (without positions and velocities who
    need to be computed before doing the NN search

    Parameters
    ----------
    data: Dataset of all objects and their related information
    sat: Index in dataset of the object to update
    start: New start from which to compute the update of the given object
    nn: Array of all nearest neighbors through time
    t: Array of all nearest neighbors intersections index-time
    r_nn: Array of all nearest neighbors real names through time
    r_t: Array of all nearest neighbors intersections datetimes
    c: Array of all nearest neighbors collision probability

    Returns
    ----------
    data: The new complete database
    """

    data.at[sat, 'Nearest'] = np.concatenate((data.loc[sat]['Nearest'][:start], nn))  # Nearest Approaching Neighbor
    data.at[sat, 'NNTimes'] = np.concatenate((data.loc[sat]['NNTimes'][:start], t))  # NNTimes
    data.at[sat, 'Real Nearest'] = np.concatenate((data.loc[sat]['Real Nearest'][:start], r_nn))  # Real Names for NN
    data.at[sat, 'Real NNTimes'] = np.concatenate((data.loc[sat]['Real NNTimes'][:start], r_t))  # Real date for NNTimes
    data.at[sat, 'Collisions'] = np.concatenate((data.loc[sat]['Collisions'][:start], c))  # Collisions

    return data


def removeUpdate(data, sat, duration):
    """
    Removes the start of every column information of one object in the given dataset, to match an update

    Parameters
    ----------
    data: Dataset of all objects and their related information
    sat: Index in dataset of the object to update
    duration: Number of steps to remove

    Returns
    ----------
    data: The new database
    """

    data.at[sat, 'Positions'] = np.delete(data['Positions'][sat], np.s_[:duration], 0)
    data.at[sat, 'Velocities'] = np.delete(data['Velocities'][sat], np.s_[:duration], 0)

    nn_t = np.array(data['NNTimes'][sat])
    to_remove = np.where(nn_t[nn_t < duration])

    data.at[sat, 'Nearest'] = np.delete(data['Nearest'][sat], to_remove)
    data.at[sat, 'NNTimes'] = np.delete(data['NNTimes'][sat], to_remove)
    data.at[sat, 'Real Nearest'] = np.delete(data['Real Nearest'][sat], to_remove)
    data.at[sat, 'Real NNTimes'] = np.delete(data['Real NNTimes'][sat], to_remove)
    data.at[sat, 'Collisions'] = np.delete(data['Collisions'][sat], to_remove)

    return data


def computeUpdateDatabase(data, satellites, seconds_to_add):
    """
    Computes the update of informations inside the database from a given start point

    Parameters
    ----------
    data: Dataset of all objects and their related information
    satellites: Dataframe of every unique objects, containing its name and its TLE
    seconds_to_add: The number of seconds we want to add to the simulation

    Returns
    ----------
    data: The new updated database
    """

    # Do a first pass for the trajectories
    for i in range(len(satellites)):
        if i in data.Index:
            sec_step = data['Times'][i][1]
            nb_hours = seconds_to_add / 3600

            real_start = datetime.datetime.strptime(data['Times'][i][0], '%Y-%m-%dT%H:%M:%S.%f')
            new_start = real_start + datetime.timedelta(seconds=sec_step * len(data['Positions'][i]))

            p, v = computeTrajectory(satellites.loc[i], real_start, nb_hours, sec_step)

            # Adds positions and velocities, to be able to compute NN on the new positions
            data.at[i, 'Positions'] = np.concatenate((data.loc[i]['Positions'], p))  # Positions
            data.at[i, 'Velocities'] = np.concatenate((data.loc[i]['Velocities'], v))  # Velocities

    # Do a second pass for the NN search
    for i in range(len(satellites)):
        if i in data.Index:
            print("Satellite %s/%s" % (i + 1, len(satellites)))

            sec_step = data['Times'][i][1]

            new_start = data['NNTimes'][i][-1] + sec_step
            last_time = datetime.datetime.strptime(data['Real NNTimes'][i][-1], '%Y-%m-%dT%H:%M:%S.%f')
            real_start = last_time + datetime.timedelta(seconds=sec_step)

            nn, t, c, r_nn, r_t = allNNofObject(data, i, new_start, real_start, sec_step)

            duration = int(seconds_to_add / sec_step)
            data = removeUpdate(data, i, duration)
            data = addUpdate(data, i, new_start, nn, t, r_nn, r_t, c)

    return data
