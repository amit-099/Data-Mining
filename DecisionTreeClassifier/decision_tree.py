import csv
from information_gain import attribute_selection


# Majority Function which tells which class has more entries in given data-set
def most_frequent_class(attributes, data, target):
    freq = {}
    index = attributes.index(target)
    for data_tuple in data:
        if data_tuple[index] in freq:
            freq[data_tuple[index]] += 1
        else:
            freq[data_tuple[index]] = 1
    maximum = 0
    frequent_class = ""
    for key in freq.keys():
        if freq[key] > maximum:
            maximum = freq[key]
            frequent_class = key
    return frequent_class


# This function will get unique values for that particular attribute from the given data
def get_values(data, attributes, attr):
    index = attributes.index(attr)
    values = []

    for entry in data:
        if entry[index] not in values:
            values.append(entry[index])
    return values


# This function will get all the rows of the data where the chosen "best" attribute has a value "val"
def get_data(data, attributes, best_attribute, val):
    new_data = [[]]
    index = attributes.index(best_attribute)
    for entry in data:
        if entry[index] == val:
            new_entry = []
            for i in range(0, len(entry)):
                if i != index:
                    new_entry.append(entry[i])
            new_data.append(new_entry)
    new_data.remove([])
    return new_data


def build_tree(data, attributes, target):
    target_class_values = [record[attributes.index(target)] for record in data]
    frequent_class = most_frequent_class(attributes, data, target)
    if not data or (len(attributes) - 1) <= 0:
        return frequent_class
    elif target_class_values.count(target_class_values[0]) == len(target_class_values):
        return target_class_values[0]
    else:
        best_attribute = attribute_selection(data, attributes, target)
        tree = {best_attribute: {}}

        for val in get_values(data, attributes, best_attribute):
            new_data = get_data(data, attributes, best_attribute, val)
            new_attributes = attributes[:]
            new_attributes.remove(best_attribute)
            subtree = build_tree(new_data, new_attributes, target)
            tree[best_attribute][val] = subtree
    return tree


def run_decision_tree():
    data = []
    with open("Datasets/book.tsv") as tsv:
        for line in csv.reader(tsv, delimiter=" "):
            data.append(tuple(line))
        print("Number of records: ", len(data))
    # attributes = ['Age', 'Income', 'Businessman', 'Credit_Rating', 'Loan_Decision']  # For ques.tsv
    attributes = ['Age', 'Income', 'Student', 'Credit_Rating', 'Buys_Computer']  # For book.tsv
    target = attributes[-1]
    tree = build_tree(data, attributes, target)
    print("\n\nFinal Decision tree:  ", tree)


if __name__ == "__main__":
    run_decision_tree()
