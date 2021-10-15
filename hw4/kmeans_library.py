"""
James Simmons
Fall 2021
Machine Learning
Homework 4
"""


import secrets
import numpy as np


NUM_ITERATIONS = 500
DEBUG = False

def debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# Old distance metrics; only support 2D cartesisan points
def manhattan_distance_simple(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.abs(x1 - x2) + np.abs(y1 - y2)
    

def euclidean_distance_simple(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


# Better, generalized distance metrics; support any number of features
# Use lists of tuples
# Used in tasks 1 and 3
def manhattan_distance_tuples(p1, p2):
    total = 0
    for x1, x2 in zip(p1, p2):
        total += np.abs(x1 - x2)
    return total


def euclidean_distance_tuples(p1, p2):
    total = 0
    for x1, x2 in zip(p1, p2):
        total += (x1 - x2) ** 2
    return np.sqrt(total)


# Even better; now it uses np arrays
# Do NOT work with task 1 or 3
def manhattan_distance(p1, p2):
    return np.sum(np.abs(p1 - p2))


def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def calculate_center_of_cluster(cluster):
    return cluster.mean(axis=0)
        

def sum_of_squared_errors(points, cluster_labels, k, distance_metric):
    cluster_errors = []
    for cluster in range(k):
        cluster_points = points[cluster_labels == cluster]
        center = calculate_center_of_cluster(cluster_points)
        center = center[np.newaxis,:]
        sse = np.sum(distance_metric(cluster_points, center) ** 2)
        cluster_errors.append(sse)
        
    return np.round(np.sum(np.array(cluster_errors)), 3)


# I completely broke this to the point of no return, and I am now disgusted by it
# Before breaking it, I used this for task 1
# It (mostly) is the same as k_means(), but it uses lists of tuples rather than numpy arrays
def k_means_original_kinda(points, centroids, distance_metric, return_first_iteration=False):    
    first_iteration = None
    
    # Initialize random centers if not specified already
    if isinstance(centroids, int):
        k = centroids
        centroids = []
        for _ in range(k):
            centroid = secrets.choice(points)
            while centroid in centroids:
                centroid = secrets.choice(points)
            centroids.append(centroid)
            
        centroids = np.array(centroids)
        
    
    for iteration in range(NUM_ITERATIONS):
        debug(f'Starting iteration {iteration}')
        
        clusters = [np.array([]) for _ in range(len(centroids))]
        #clusters = np.array(clusters)
    
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
            #avg_x = np.mean([point[0] for point in clusters[i]]) # Outdated, only supports 2D
            #avg_y = np.mean([point[1] for point in clusters[i]])
            
            new_centroid = calculate_center_of_cluster(clusters[i])
            
            if centroid != new_centroid: #(avg_x, avg_y):
                centroids[i] = new_centroid #(avg_x, avg_y)
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
    
    return clusters, centroids
            

# I copied this off of my original github commit
# This is before I broke it
# This should work for task 1
def k_means_github_restore(points, centroids, distance_metric, return_first_iteration=False):    
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


def reshape_for_distance_metrics(points, centroids):
    # Reshape to (1 x 784 x 10)
    centroids = centroids[::, np.newaxis]
    centroids = np.transpose(centroids, (1, 2, 0))
    
    # Reshape to (10000 x 784 x 1)
    points = points[::, np.newaxis]
    points = np.transpose(points, (0, 2, 1))
    
    return points, centroids
    

# Inputs are (10000 x 784) and (10 x 784)
# These distance metrics are built ground-up to work with task 2
# They take full advantage of numpy broadcasting to make all-pairs calculations
def euclidean(points, centroids):
    points, centroids = reshape_for_distance_metrics(points, centroids)

    # Broadcast to (10000 x 784 x 10), then flatten to (10000 x 10)
    dist_to_each_centroid = np.sqrt(np.sum((points - centroids) ** 2, axis=1))
    
    return dist_to_each_centroid


def cosine(points, centroids):
    points, centroids = reshape_for_distance_metrics(points, centroids)

    dot = np.sum(np.multiply(points, centroids), axis=1)
    
    point_norms = np.sqrt(np.sum(points ** 2, axis = 1))
    centroid_norms = np.sqrt(np.sum(centroids ** 2, axis = 1))
    norm_product = point_norms * centroid_norms

    cosine_similarity = dot / norm_product
    cosine_difference = 1 - cosine_similarity
    
    return cosine_difference


def jaccard(points, centroids):
    points, centroids = reshape_for_distance_metrics(points, centroids)
    
    mins = np.minimum(points, centroids)
    mins = np.sum(mins, axis=1)
    
    maxs = np.maximum(points, centroids)
    maxs = np.sum(maxs, axis=1)
    
    jaccard_similarity = mins/maxs
    jaccard_difference = 1 - jaccard_similarity
    
    return jaccard_difference


def pick_distant_starting_points(points, k, distance_metric):
    first_point = np.random.randint(points.shape[0])
    centroids = points[first_point]
    centroids = centroids[np.newaxis,:] # (1 x 784)
    points = np.delete(points, (first_point), axis=0)

    for _ in range(k - 1):
        distances = distance_metric(points, centroids) # all-pairs distances
        total_distance = np.sum(distances, axis=1) # sum distance to each center
        farthest_point_idx = np.argmax(total_distance)  # max sum distance
        farthest_point = points[farthest_point_idx] # select the actual point
        farthest_point = farthest_point[np.newaxis,:] # reshape to (1 x 784)
        centroids = np.append(centroids, farthest_point, axis=0) # add to centers
        points = np.delete(points, (farthest_point_idx), axis=0) # remove from points
        
    return centroids


# I'm starting over
# Points and centroids must be numpy arrays
# Distance metric must be 'bulk' if this is task 2
def k_means(points, centroids, distance_metric):
    num_points = points.shape[0]
    
    if isinstance(centroids, int):
        debug('Picking distant starting points')
        centroids = pick_distant_starting_points(points, centroids, distance_metric)

    k = centroids.shape[0]
    cluster_labels = np.full(num_points, -1)
    
    exit_code = 'Iterations'
    sse = 0
    
    for i in range(NUM_ITERATIONS):
        debug(f'Beginning iteration {i}')
        
        # Assignment to closest centroid
        distances = distance_metric(points, centroids)
        new_cluster_labels = np.argmin(distances, axis=1)
        
        if np.all(cluster_labels == new_cluster_labels):
            debug(f'Breaking at iteration {i}: Labels unchanged')
            #break
        else:
            cluster_labels = np.copy(new_cluster_labels)
        
        # Update centroids to the average of their members
        break_flag = True
        for cluster in range(k):
            cluster_points = points[cluster_labels == cluster]
            new_center = calculate_center_of_cluster(cluster_points)
            
            differences = np.abs(centroids[cluster] - new_center) # old - new
            unchanged_features = differences < 1 # Change sufficiently small
            
            if not np.all(unchanged_features): # not all unchanged = some changed
                break_flag = False
            centroids[cluster] = new_center
            
        if break_flag:
            debug(f'Breaking at iteration {i}: Centroids unchanged')
            exit_code = 'Centroids'
            break
        
        # Check if the SSE value increased
        new_sse = sum_of_squared_errors(points, cluster_labels, k, distance_metric)
        if new_sse > sse and sse > 0:
            debug(f'Breaking at iteration {i}: SSE increased')
            exit_code = 'SSE'
            break
        
        sse = new_sse
        
    return cluster_labels, i, exit_code
    
        

    
    
if __name__ == "__main__":
    pass
