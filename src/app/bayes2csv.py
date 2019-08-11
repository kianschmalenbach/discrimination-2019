import pandas as pd
import random
from model.bayes import BayesNet
from model.dataframe import DataFrameWrapper


class Bayes2Csv:

    def __init__(self, name, index='Index', label='Label', rank='Rank'):
        self.__bn = BayesNet(name)
        self.__df = None
        self.__index = index
        self.__label = label
        self.__rank = rank
        self.__weights = None

    # network = [(A, B), (A, D), ...]
    # node_values = [[True, False], ...] (nodes in sorted order)
    def create_bn(self, network, node_values, weights):
        nodes = set()
        for edge in network:
            nodes.add(edge[0])
            nodes.add(edge[1])
        nodes = list(nodes)
        nodes.sort()
        dependencies = []
        for i in range(len(nodes)):
            node = nodes[i]
            dependencies.append(set())
            self.__bn.add_column(node, node_values[i])
        for i in range(len(network)):
            edge = network[i]
            dependencies[nodes.index(edge[1])].add(edge[0])
        for i in range(len(dependencies)):
            dep = list(dependencies[i])
            if len(dep) > 0:
                self.__bn.add_dependencies(nodes[i], dep)
        self.__weights = weights

    def add_cpt(self, node, probabilities):
        column = self.__bn.get_column(node)
        if column is None:
            print("Error 1")
            return False
        for entry in probabilities:
            value = entry[0]
            probability = entry[1]
            evidence = entry[2]
            if type(value) == bool:
                if value:
                    true_probability = round(probability, 5)
                    false_probability = round(1 - probability, 5)
                else:
                    true_probability = round(1 - probability, 5)
                    false_probability = round(probability, 5)
                self.__bn.add_probability(node, False, false_probability, evidence)
                self.__bn.add_probability(node, True, true_probability, evidence)
            else:
                self.__bn.add_probability(node, value, round(probability, 5), evidence)

    def create_samples(self, n):
        column = self.__bn.get_column(self.__label)
        if not column or not _bool_check(column):
            print("Error 2")
            return False
        evidences = []
        for i in range(len(column.dependencies())):
            dependency = column.dependencies()[i]
            if not _bool_check(dependency):
                print("Error 3")
                return False
            evidences.append(dependency.name())
        evidence_value = [False] * len(evidences)
        while evidence_value is not None:
            evidence = list(zip(evidences, evidence_value))
            self.__bn.add_probability(self.__label, False, 1.0, evidence)
            self.__bn.add_probability(self.__label, True, 0.0, evidence)
            evidence_value = _next_value(evidence_value)
        columns = []
        for column in self.__bn.columns():
            columns.append(column.name())
        columns.sort()
        columns = [self.__index] + columns
        self.__df = pd.DataFrame(columns=columns)
        for i in range(n):
            sample = self.__bn.bn().compute_sample()
            data = []
            for column in self.__df.columns:
                if column == self.__index:
                    data.append(i + 1)
                else:
                    data.append(sample[column])
            self.__df.loc[i] = data
        return True

    def compute_label_probability(self, threshold):
        dependencies = []
        for dependency in self.__bn.get_column(self.__label).dependencies():
            dependencies.append(dependency.name())
        dependencies.sort()
        if len(self.__weights) != len(dependencies) + 1:
            print("Error 4")
            return False
        weight_sum = 0.0
        for i in range(len(self.__weights)):
            self.__weights[i] = round(self.__weights[i], 3)
            weight_sum += self.__weights[i]
        if round(weight_sum, 3) != 1:
            print("Error 5")
            return False
        rank_values = []
        label_values = []
        for date in self.__df.iterrows():
            values = []
            for column in dependencies:
                values.append(date[1][column])
            evidence = list(zip(dependencies, values))
            rank_value = self.__label_function(evidence)
            rank_values.append((date[0], rank_value))
            label_value = (rank_value >= threshold)
            label_values.append((date[0], label_value))
        for entry in label_values:
            self.__df.at[entry[0], self.__label] = entry[1]
        for entry in rank_values:
            self.__df.at[entry[0], self.__rank] = entry[1]

    def export_csv(self, file):
        self.__make_complete()
        self.__df.to_csv(file, index=False)

    def __make_complete(self):
        dependencies = []
        for dependency in self.__bn.get_column(self.__label).dependencies():
            dependencies.append(dependency.name())
        dependencies.sort()
        columns = list(self.__df.columns)
        columns.remove(self.__index)
        columns.remove(self.__rank)
        values = [False] * len(columns)
        index = len(self.__df)
        while values is not None:
            res = self.__df
            for column in columns:
                res = res.loc[res[column] == values[columns.index(column)]]
            if len(res) == 0:
                evidence = list(zip(dependencies, values))
                data = [index + 1] + values + [self.__label_function(evidence)]
                self.__df.loc[index] = data
                index += 1
            values = _next_value(values)

    def calc_discrimination(self, d_attr, label):
        return DataFrameWrapper(df=self.__df).calc_discrimination(d_attr, label, "BEFORE")

    def __label_function(self, evidence):
        value = 0
        for i in range(len(self.__weights)):
            if i == 0:
                value += round((self.__weights[i] * random.uniform(0, 1)), 3)
            elif evidence[i - 1][1]:
                value += self.__weights[i]
        return round(value, 3)


def _bool_check(column):
    col_values = column.values()
    col_values.sort()
    return col_values == [False, True]


def _next_value(array):
    for i in range(len(array)):
        if not array[-i - 1]:
            array[-i - 1] = True
            return array
        else:
            array[-i - 1] = False
    return None
