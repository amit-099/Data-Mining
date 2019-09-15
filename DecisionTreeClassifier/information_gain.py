import math


def info_data(attributes, data, target_attribute):
    freq = {}
    info_of_data = 0.0
    i = 0
    for entry in attributes:
        if target_attribute == entry:
            break
        i = i + 1
    for entry in data:
        if entry[i] in freq:
            freq[entry[i]] += 1.0
        else:
            freq[entry[i]] = 1.0
    for freq in freq.values():
        info_of_data += (-freq / len(data)) * math.log(freq / len(data), 2)
    return info_of_data


def information_gain(attributes, data, attr, target_attribute):
    freq = {}
    info_of_attribute = 0.0
    i = attributes.index(attr)
    for entry in data:
        if entry[i] in freq:
            freq[entry[i]] += 1.0
        else:
            freq[entry[i]] = 1.0
    for value in freq.keys():
        partition_weight = freq[value] / sum(freq.values())
        new_partition = [entry for entry in data if entry[i] == value]
        info_of_attribute += partition_weight * info_data(attributes, new_partition, target_attribute)
    return info_data(attributes, data, target_attribute) - info_of_attribute


def attribute_selection(data, attributes, target):
    best = attributes[0]
    maximum_gain = 0
    for attr in attributes:
        if attr == target:
            break
        new_gain = information_gain(attributes, data, attr, target)
        if new_gain > maximum_gain:
            maximum_gain = new_gain
            best = attr
    return best
