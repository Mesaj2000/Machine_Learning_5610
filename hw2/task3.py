import pandas as pd
from math import log2

def task3():
    df = generate_dataframe()
    
    df_entropy = entropy(df)
    
    a_true, a_false = split(df, 'A')
    a_gain = df_entropy - merge_entropy(a_true, a_false)
    
    b_true, b_false = split(df, 'B')
    b_gain = df_entropy - merge_entropy(b_true, b_false)
    
    chosen_attribute = 'A' if a_gain > b_gain else 'B'
    
    print(f'Overall Entropy: {df_entropy}')
    print(f'A-split Gain:    {a_gain}')
    print(f'B-split Gain:    {b_gain}')
    print(f'Chosen Attr:     {chosen_attribute}')
    
    print_all_combos(df)


def print_all_combos(df):
    for a, b in ((True, True), (True, False), (False, True), (False, False)):
        print()
        print(df[(df['A'] == a) & (df['B'] == b)])

    
def merge_entropy(df1, df2):
    n1 = len(df1)
    n2 = len(df2)
    n = n1 + n2
    merged_entropy = entropy(df1) * n1 / n + entropy(df2) * n2 / n
    print(merged_entropy)
    return merged_entropy


def entropy(df):
    total = len(df)
    plus = len(df[df['Class'] == '+']) / total
    minus = len(df[df['Class'] == '-']) / total
    print(plus * total, minus * total, total)

    return - (plus * log2(plus) if plus > 0 else 0) - (minus * log2(minus) if minus > 0 else 0)


def split(df, splitting_class):
    true_split = df[df[splitting_class]]
    false_split = df[~df[splitting_class]]
    
    return true_split, false_split


def generate_dataframe():
    a_vals = [True, True, True, True, True, False, False, False, True, True]
    b_vals = [False, True, True, False, True, False, False, False, True, True]
    label = list('+++-+-----')

    data = {
        'A': a_vals,
        'B': b_vals,
        'Class': label
    }

    return pd.DataFrame(data)




task3()