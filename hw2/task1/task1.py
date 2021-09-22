import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from pprint import pprint
import re
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from more_itertools import powerset
from tqdm import tqdm

ALL_FEATURES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']
SELECTED_FEATURES = ['Pclass', 'Sex', 'Embarked']
GROUND_TRUTH = 'Survived'
NUM_VALIDATION_GROUPS = 5
FULL_SCORING_PROCESS = False
PLOT_TREE = False


def task_1():
    # Load the datasets and clean them up
    train = titanic_preprocess('train.csv')
    test = titanic_preprocess('test.csv')
    
    # Get the ideal tree (the tree trained with the ideal feature selection) and the corresponding 5-fold validation score
    regular_tree, regular_score = generate_ideal_tree(train, "Regular")
    random_forest, random_forest_score = generate_ideal_tree(train, "Random Forest")
    
    print("Regular Tree Score: {}".format(regular_score))
    print("Random Forest Score: {}".format(random_forest_score))


def generate_ideal_tree(df, method):
    if FULL_SCORING_PROCESS:
        # Rank ALL feature combinations by their average validation accuracy, select the best one
        feature_combinations = score_all_feature_combinations(df, method)
        #pprint(feature_combinations[:10])
        selected_features = feature_combinations[0][0]
    else:
        # Or, just hard-code the results after scoring it one time
        # It so happened that the best features are the same regardless of method
        selected_features = SELECTED_FEATURES
    
    # Train a tree with the selected features, and get it's score
    descision_tree = train_tree(df, selected_features, method)
    score = k_fold_validation(df, selected_features, method)
    
    # If this is the regular tree (not the random forest), output a plot of it 
    if method == "Regular" and PLOT_TREE:
        plot_tree(descision_tree, selected_features)
    
    return descision_tree, score
    

# Iterate over every possible combination of features, and find the one with the highest sore
# "Score" is the average of 5-fold validation accuracy
def score_all_feature_combinations(df, method):
    all_combinations = set(powerset(ALL_FEATURES))

    combination_with_score = []

    for combination in tqdm(all_combinations, desc="Finding Best Feature Combination"):
        if len(combination) == 0:
            continue

        combination = list(combination)

        score = k_fold_validation(df, combination, method)

        combination_with_score.append((combination, score))
        
    combination_with_score.sort(key=lambda x: x[1], reverse=True)

    return combination_with_score


# Split the dataset into groups
# Set aside one group for validation, train on the others
# Return the average validation accuracy
def k_fold_validation(df, features, method):
    total_rows = df.shape[0]
    split_size = total_rows // NUM_VALIDATION_GROUPS

    scores = []

    for i in range(NUM_VALIDATION_GROUPS):
        train_rows = np.r_[i * split_size : (i+1) * split_size]
        val_rows = list(set(np.r_[0:total_rows]) - set(train_rows))
        train = df.loc[train_rows]
        val = df.loc[val_rows]
        
        descision_tree = train_tree(train, features, method)
        score = score_tree(val, descision_tree, features)

        scores.append(score)

    return np.mean(scores)

def score_tree(df, descision_tree, features):
    x = df[features]
    y = df[GROUND_TRUTH]
    return descision_tree.score(x,y)


def train_tree(df, features, method="Regular"):
    x = df[features]
    y = df[GROUND_TRUTH]

    if method == "Regular":
        descision_tree = DecisionTreeClassifier(criterion="entropy")
    elif method == "Random Forest":
        descision_tree = RandomForestClassifier(criterion="entropy", n_estimators=500)
    descision_tree = descision_tree.fit(x,y)

    return descision_tree


def plot_tree(descision_tree, features):
    data = tree.export_graphviz(descision_tree, out_file=None, feature_names=features)
    graph = pydotplus.graph_from_dot_data(data)
    graph.write_png('mydecisiontree.png')

# Read the data in from a file
# Clean it up
def titanic_preprocess(file):
    df = pd.read_csv(file)

    # Remove 'Name' feature, as it likely had no bearing on who survived, and I have no idea how to use it in a decision tree
    del df['Name']

    # Remove 'Cabin' feature since about 3/4 of it is null
    del df['Cabin']

    # Remove the leading non-digit characters from the 'Ticket' (the ticket type?), since most of the records are missing it
    # Also cast ticket numbers to integers
    tickets = list(df['Ticket'])
    regex = re.compile('\d+$')
    tickets = [regex.search(ticket) for ticket in tickets]
    tickets = [int(ticket.group(0)) if ticket is not None else 0 for ticket in tickets]
    df['Ticket'] = tickets

    # Change 'Sex' from male/female strings to 0/1 integers
    df['Sex'] = [0 if sex == 'male' else 1 for sex in list(df['Sex'])]

    # Change 'Embarked' from C/Q/S strings to 0/1/2 integers
    df['Embarked'] = [2 if port == 'C' else 1 if port == 'Q' else 0 for port in list(df['Embarked'])]

    # About 1/6 of ages are null
    # Set them to the average age
    non_null_ages = df['Age'][~df['Age'].isnull()]
    average_age = round(np.average(non_null_ages), 2)
    df.loc[df['Age'].isnull(), 'Age'] = average_age


    # All features are now int or float, with no null values
    return df


if __name__ == "__main__":
    task_1()