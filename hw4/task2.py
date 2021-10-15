import kmeans_library as kml
import pandas as pd
import numpy as np
from tqdm import tqdm


DATA = 'data/data.csv'
LABELS = 'data/label.csv'
SAVED_LABELS = 'data/{}_{}.npy'
SAVE_NUMBER = 4
NUM_CLASSES = 10
NUM_FEATURES = 784


def load_data():
    column_names = [f'x{i:03d}' for i in range(NUM_FEATURES)]
    data = pd.read_csv(DATA, names=column_names)
    data = np.array(data)
    return data

def load_labels():
    column_names = ['Labels']
    data = pd.read_csv(LABELS, names=column_names)
    data = np.array(data)
    return data


def cluster_and_save(data, distance_metric):
    cluster_labels = kml.k_means(data, NUM_CLASSES, distance_metric)[0]
    
    np.save(SAVED_LABELS.format(SAVE_NUMBER, distance_metric.__name__), cluster_labels)
    
    return cluster_labels


def get_clustering_accuracy(cluster_labels, labels):
    overall_accuracy = 0
    overall_total = labels.size
    for cluster in range(NUM_CLASSES):
        labels_in_cluster = labels[cluster_labels == cluster]
        labels_in_cluster = np.squeeze(labels_in_cluster)
        label_counts = np.bincount(labels_in_cluster)
        most_common = np.argmax(label_counts)
        num_correct = label_counts[most_common]
        total = labels_in_cluster.size
        accuracy = num_correct / total
        wieght = total / overall_total
        overall_accuracy += wieght * accuracy
        
    return np.round(overall_accuracy * 100, 2)
        

# The data collected and printed by this function
# Can be used to answer every part of task 2
def get_avgs_of_many_runs():
    data = load_data()
    labels = load_labels()
    
    distance_metrics = [
        kml.euclidean,
        kml.cosine,
        kml.jaccard
    ]
    
    iterations = {}
    exits = {}
    sses = {}
    average_sses = {}
    accuracies = {}
    num_runs = 20
    
    for distance_metric in distance_metrics:
        iterations[distance_metric.__name__] = []
        exits[distance_metric.__name__] = []
        accuracies[distance_metric.__name__] = []
        
        for distance_metric_2 in distance_metrics:
            sses[distance_metric.__name__ + "-" + distance_metric_2.__name__] = []
            average_sses[distance_metric.__name__ + "-" + distance_metric_2.__name__] = 0
        
        # Run each distance metric num_runs times
        for _ in tqdm(range(num_runs), desc=distance_metric.__name__):
            cluster_labels, i, exit_code = kml.k_means(data, NUM_CLASSES, distance_metric)
            iterations[distance_metric.__name__].append(i)
            exits[distance_metric.__name__].append(exit_code)
            accuracies[distance_metric.__name__].append(get_clustering_accuracy(cluster_labels, labels))
            
            for distance_metric_2 in distance_metrics:
                sse = kml.sum_of_squared_errors(data, cluster_labels, NUM_CLASSES, distance_metric_2)
                sses[distance_metric.__name__ + "-" + distance_metric_2.__name__].append(sse)
                average_sses[distance_metric.__name__ + "-" + distance_metric_2.__name__] += sse / num_runs
            
        average_iterations = np.mean(iterations[distance_metric.__name__])
        codes = exits[distance_metric.__name__]    
        average_accuracy = np.mean(accuracies[distance_metric.__name__])    
        
        print(f'{distance_metric.__name__}')
        print(f'    Average Iterations: {average_iterations}')
        print(f'    Iterations: {iterations[distance_metric.__name__]}')
        print( '    Error Codes: ' + ', '.join(codes))
        
        print(f'    Average Accuracy: {average_accuracy}')
        print(f'    Accuracies: {accuracies[distance_metric.__name__]}')
        
        for distance_metric_2 in distance_metrics:
            avg = average_sses[distance_metric.__name__ + "-" + distance_metric_2.__name__]
            print(f'    Average SSE using {distance_metric_2.__name__}: {avg}')
            print(f'    SSEs: {sses[distance_metric.__name__ + "-" + distance_metric_2.__name__]}')
    

# SEE value for all 9 combinations
def question_1():
    data = load_data()
    
    distance_metrics = [
        kml.euclidean,
        kml.cosine,
        kml.jaccard
    ]
    
    for distance_metric in distance_metrics:
        print(distance_metric.__name__)
        
        
        try:
            cluster_labels = np.load(SAVED_LABELS.format(SAVE_NUMBER, distance_metric.__name__), allow_pickle=True)    
        except FileNotFoundError:
            cluster_labels = cluster_and_save(data, distance_metric)
        
        for distance_metric_2 in distance_metrics:
            sse = kml.sum_of_squared_errors(data, cluster_labels, NUM_CLASSES, distance_metric_2)
            print(distance_metric_2.__name__)
            print(sse)
            print()
        print()

# Accuracies
def question_2():
    data = load_data()
    labels = load_labels()
    
    distance_metrics = [
        kml.euclidean,
        kml.cosine,
        kml.jaccard
    ]
    
    for distance_metric in distance_metrics:
        print(distance_metric.__name__)
        
        try:
            cluster_labels = np.load(SAVED_LABELS.format(SAVE_NUMBER, distance_metric.__name__), allow_pickle=True)    
        except FileNotFoundError:
            cluster_labels = cluster_and_save(data, distance_metric)
            
        accuracy = get_clustering_accuracy(cluster_labels, labels)
        print(accuracy)
    

if __name__ == "__main__":
    question_1()
    question_2()
    #get_avgs_of_many_runs()
