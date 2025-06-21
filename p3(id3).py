import pandas as pd
import math
from collections import Counter

data = pd.read_csv("play_tennis.csv")

def entropy(target_col):
    values, counts = Counter(target_col).keys(), Counter(target_col).values()
    total = sum(counts)
    return -sum((c/total) * math.log2(c/total) for c in counts)

def info_gain(data, split_attr, target_attr):
    total_entropy = entropy(data[target_attr])
    values = data[split_attr].unique()
    weighted_entropy = 0

    for v in values:
        subset = data[data[split_attr] == v]
        weight = len(subset) / len(data)
        weighted_entropy += weight * entropy(subset[target_attr])

    return total_entropy - weighted_entropy

def id3(data, features, target_attr):
    target_values = data[target_attr]
    
    if len(set(target_values)) == 1:
        return target_values.iloc[0]
    
    if not features:
        return target_values.mode()[0]

    gains = [info_gain(data, f, target_attr) for f in features]
    best_feature = features[gains.index(max(gains))]

    tree = {best_feature: {}}

    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        subtree = id3(subset, [f for f in features if f != best_feature], target_attr)
        tree[best_feature][value] = subtree

    return tree
features = list(data.columns[:-1])
target = "PlayTennis"
decision_tree = id3(data, features, target)

import pprint
print("Decision Tree:")
pprint.pprint(decision_tree)
def classify(instance, tree):
    if not isinstance(tree, dict):
        return tree
    attribute = next(iter(tree))
    value = instance[attribute]
    if value in tree[attribute]:
        return classify(instance, tree[attribute][value])
    else:
        return "Unknown"


new_sample = {'Outlook': 'Rainy', 'Temperature': 'Mild', 'Humidity': 'Normal', 'Wind': 'Strong'}
prediction = classify(new_sample, decision_tree)
print("\nNew Sample Prediction:", prediction)
