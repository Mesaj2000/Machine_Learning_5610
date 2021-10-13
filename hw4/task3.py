from kmeans_library import euclidean_distance as dist
        

def task_3():
    reds = [
        (4.7, 3.2),
        (4.9, 3.1),
        (5.0, 3.0),
        (4.6, 2.9)
    ]
    
    blues = [
        (5.9, 3.2),
        (6.7, 3.1),
        (6.0, 3.0),
        (6.2, 2.8)   
    ]
    
    distances = {}

    for red in reds:
        for blue in blues:
            distances[f'{red} - {blue}'] = round(dist(red, blue), 4)
            
    sorted_dist = sorted(distances.items(), key=lambda x: x[1])
    
    total_distance = 0
    num_distances = len(sorted_dist)
    
    for k, v in sorted_dist:
        print(f'{k}: {v}')
        total_distance += v

    average_distance = total_distance / num_distances
    
    print(f'Average Distance: {average_distance:0.5}')


if __name__ == "__main__":
    task_3()
