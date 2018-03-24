import os
import csv
import json
import math

import cv2
import numpy as np

import config


def write_csv(trajectories):
    '''
    Writing Trajectoies to Tracking.csv
    :param trajectories: list of Trajectory objects.
    :return:
    '''
    longest_course = max([len(trajectory.course) for trajectory in trajectories])
    header = ['ID', 'Length', 'Sinuosity']
    header.extend(['x', 'y', 'area', 'distance', 'angle'] *  longest_course)

    with open('Tracking.csv', 'w') as f:
        writer = csv.writer(f)

        writer.writerow(header)

        id = 1
        for trajectory in trajectories:
            row = [str(id), str(len(trajectory.course)), '']
            for dot in trajectory.course:
                row.extend([str(dot.x), str(dot.y), str(dot.size), str(dot.distance), str(dot.angle)])

            writer.writerow(row)

def plot_dot(trajectories):
    '''
    Generating Tracking.jpg that show Trajectories with Opencv
    :param trajectories: list of Trajectoy objects.
    :return:
    '''
    vidcap = cv2.VideoCapture(input_video)
    success, image = vidcap.read()

    # for i in range(0, image.shape[0]):
    #     image[i, 100] = [255,255,255]
    #
    # cv2.imwrite('Tracking.jpg', image)

    for trajectory in trajectories:
        for i in range(len(trajectory.course)-1):
            # image[dot.x, dot.y] = [255,255,255]
            start_dot = trajectory.course[i]
            end_dot = trajectory.course[i+1]
            cv2.line(image,(start_dot.x,start_dot.y),(end_dot.x,end_dot.y),(255,255,255), 1)

    cv2.imwrite('Tracking.jpg', image)

class Dot():
    '''
    An object to be element of a trajectory
    '''
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        self.distance = ''
        self.angle = ''

    def __str__(self):
        return str([self.x, self.y, self.size, self.distance, self.angle])

    def add_distance_angle(self, distance, angle):
        self.distance = distance
        self.angle = angle



class Trajectory():
    '''
    It has 2 properties, course and wating_step.
    Course is consists of the dots. And wating_step is the count of the frame since last dot was add to the course.
    '''
    def __init__(self, dot):
        self.course = [dot]
        self.waiting_step = 0

    def __str__(self):
        return str([len(self.course), self.waiting_step])

    def add_dot(self, dot):
        self.course.append(dot)

def find_dots(locations):
    '''
    From list of pixels, we get the dots by combinging the near pixels
    :param locations: 2d list. ex: [[192, 35], [36, 37]]
    :return:
    '''
    locations = np.asarray(locations)
    dots = []
    while len(locations) > 0:
        t_location = locations[0]
        d_array = np.sqrt(np.sum((locations-t_location) ** 2, axis=1)) # calculate distances between all dots and a dot.

        # Get dots that be around the dot.
        index_of_dot = np.argwhere(d_array < config.max_dot_size)
        index_of_dot = np.reshape(index_of_dot, len(index_of_dot))

        dots.append((t_location, len(index_of_dot)))

        locations = np.delete(locations, index_of_dot, 0)

    return dots

def closest_node(node, nodes):
    '''
    Bewteen node and each node of the nodes, it calculate the distances and return the minimum distance and the index.
    :param node: 2d value ex: [10, 20]
    :param nodes:  2d list ex: [[20, 30], [30, 40]]
    :return: min_value and min_index
    '''
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return [np.min(dist_2), np.argmin(dist_2)]

def add_dot(tras, dots):
    '''
    For each trajectory, find the closes dot and add the dot to the course of the trajectory.
    :param tras: list of the Trajectory objects
    :param dots: list of the Dot objects
    :return new tras: updated list of the Trajectory objects.
    '''
    if len(tras) < 1:
        new_tras = [Trajectory(dot) for dot in dots]
    else:
        new_tras = []
        while len(dots) and len(tras):
            print ('Count of Tras: %d' %len(tras))
            print ('Count of Dots: %d' %len(dots))

            tras_last_locations = [[tra.course[-1].y, tra.course[-1].x] for tra in tras] # the locations of the last dots of the trajectories.
            dots_locations = [[dot.y, dot.x] for dot in dots] # the locations of the dot objects

            # for trajecoty, find the closest dot and closest distance.
            closest_point = np.array([closest_node(location, dots_locations) for location in tras_last_locations]) #

            # find trajecoty and dot that has closest distance
            target_tras_index = np.argmin([closest_point[:, 0]])
            dot_index = closest_point[target_tras_index][1]
            distance = np.sqrt(closest_point[target_tras_index][0]) # get distance

            target_tra = tras[target_tras_index]
            target_tra_last_dot = target_tra.course[-1]
            target_dot = dots[dot_index]

            if distance/np.sqrt(target_tra_last_dot.size) < 15:
                dots.remove(target_dot)
                tras.remove(target_tra)

                # calculate angle
                delta_x = target_tra_last_dot.x - target_dot.x
                delta_y = target_tra_last_dot.y - target_dot.y
                angle = (math.atan2(delta_y, delta_x))*180/math.pi
                if angle < 0:
                    angle = 360 +  angle

                target_dot.add_distance_angle(distance, angle)

                target_tra.add_dot(target_dot)
                target_tra.wating_step = 0

                new_tras.append(target_tra)

            else:
                break

        for tra in tras:
            tra.waiting_step += 1

        new_tras.extend(tras)
        new_tras.extend([Trajectory(dot) for dot in dots])

        new_tras.extend([Trajectory(dot) for dot in dots])

    return new_tras

def main():

    trajectories = []
    tracking_tras= []
    vidcap = cv2.VideoCapture(input_video)

    # Read frame until the end of the vidoe.
    i = 0
    while True:
        # Read a Frame
        success, image = vidcap.read()
        if success:
        # if i < 5:
            # Convert RGB to Grey and find the whilt pixels.
            image =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            white_locations = np.argwhere(image>=200)

            if len(white_locations) > 0:
                # Define dots among above while pixels using size of the dot.
                dots = find_dots(white_locations)
                print ("Step %d -> Count of dots: %d" %(i, len(dots)))
                print (dots)
                dots = [Dot(dot[0][1], dot[0][0], dot[1]) for dot in dots]

                # For each dot, find suitable trajecoty and add the dot to the trajecoty.
                # For each trajecoty, check that max_wating_step is larger than a certain value, the trajectory is completed one.
                if len(dots) > 0:
                    tracking_tras = add_dot(tracking_tras, dots)
                    ended_tras = [tra for tra in tracking_tras if tra.waiting_step > config.max_wating_step]
                    tracking_tras = [tra for tra in tracking_tras if tra.waiting_step <= config.max_wating_step]

                    trajectories.extend(ended_tras)
        else:
            trajectories.extend(tracking_tras)
            break

        i += 1

    print ("\n\n")
    print ("Count of Trajectories: %d" %len(trajectories))

    return trajectories
if __name__ == "__main__":
    input_video = "for_tracking.avi"

    trajectories = main() # find trajecoties
    write_csv(trajectories) # write the trajectoies to the csv
    plot_dot(trajectories) # plot the trajectoris.
