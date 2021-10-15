from kmeans_library import k_means_github_restore as k_means
from kmeans_library import manhattan_distance_tuples as manhattan_distance
from kmeans_library import euclidean_distance_tuples as euclidean_distance


def task_1():
    data = [
        (3,5),
        (3,4),
        (2,8),
        (2,3),
        (6,2),
        (6,4),
        (7,3),
        (7,4),
        (8,5),
        (7,6)
    ]
    
    centroid_groups = [
        [(4,6), (5,4)],
        [(4,6), (5,4)],
        [(3,3), (8,3)],
        [(3,2), (4,8)]
    ]
    
    distance_metrics = [
        manhattan_distance,
        euclidean_distance,
        manhattan_distance,
        manhattan_distance
    ]
    
    for i in (1,2,3,4):
        clusters, centroids, first_iteration = k_means(data, centroid_groups[i-1], distance_metrics[i-1], True)
        team_clusters = [ [ f"X{data.index(point) + 1}" for point in cluster] for cluster in clusters]
        print(f'({i}) clusters:  {str(team_clusters)}')
        print(  f'    centroids: {str(centroids)}')
        print(  f'    one iter:  {str(first_iteration)}')
    
    
if __name__ == "__main__":
    task_1()
