"""
James Simmons
Fall 2021
Machine Learning
Homework 4
"""

import numpy as np
#from pprint import pprint
#from pprint import pformat

NUM_ITERATIONS = 500
DEBUG = True

def debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def manhattan_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.abs(x1 - x2) + np.abs(y1 - y2)


def euclidean_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def k_means(points, centroids, distance_metric, return_first_iteration=False):    
    first_iteration = None
    
    for iteration in range(NUM_ITERATIONS):
        
        clusters = [[] for _ in range(len(centroids))]
    
        for point in points:
            best_distance = None
            closest_centroid = None
            
            for i, centroid in enumerate(centroids):
                distance = distance_metric(point, centroid)
                
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    closest_centroid = i
                    
            clusters[closest_centroid].append(point)
            
            
        centroids_unchanged = True

        for i, centroid in enumerate(centroids):
            avg_x = np.mean([point[0] for point in clusters[i]])
            avg_y = np.mean([point[1] for point in clusters[i]])
            
            if centroid != (avg_x, avg_y):
                centroids[i] = (avg_x, avg_y)
                centroids_unchanged = False
            
        if first_iteration is None:
            first_iteration = [centroid for centroid in centroids]
            
        if centroids_unchanged:
            debug(f'breaking after {iteration} iterations')
            break
        
        if iteration == NUM_ITERATIONS - 1:
            debug(f'halting after {NUM_ITERATIONS} iterations')
          
            
    if return_first_iteration:
        return clusters, centroids, first_iteration
    
    return  clusters, centroids
            
    
    
if __name__ == "__main__":
    pass
