from lib.bay.Network import Network


class BayesNet:

    def __init__(self, name="NETWORK"):
        self.__bn = Network(name)
        self.__name = name
        self.__cols = []

    def bn(self):
        return self.__bn

    def add_column(self, name, values):
        self.__cols.append(Column(self.__bn, name, values))
        return True

    def get_column(self, name):
        for col in self.__cols:
            if col.name() == name:
                return col
        return None

    def add_dependencies(self, column, dependencies):
        column = self.get_column(column)
        if column not in self.__cols or not column.open():
            print('Error 1')
            return False
        for dep in dependencies:
            dep_col = self.get_column(dep)
            if dep_col is not None and dep_col != column and column not in dep_col.dependencies():
                column.dependencies().append(dep_col)
            else:
                print('Error 2')
                return False
        return True

    def add_probability(self, column, value, probability, evidence=None):
        if evidence is None:
            evidence = []
        column = self.get_column(column)
        if column is None or value not in column.values() \
                or probability < 0 or probability > 1 \
                or len(evidence) != len(column.dependencies()):
            print('Error 3')
            return False
        entry_cols = []
        for entry in evidence:
            entry_col = self.get_column(entry[0])
            if entry_col not in column.dependencies() or entry[1] not in entry_col.values() or entry_col in entry_cols:
                print('Error 4')
                return False
            entry_cols.append(entry_col)
        column.close()
        return column.cpt().add_probability(value, probability, evidence)

    def print(self):
        print('NETWORK \"' + self.__name.upper() + '\"\n')
        for col in self.__cols:
            col.print()
        print()

    def columns(self):
        return self.__cols


class Column:

    def __init__(self, bn, name, values, type='discrete'):
        values.sort()
        self.__bn = bn
        self.__bn.add_node(name, type, len(values), values)
        self.__name = name
        self.__values = values
        self.__dependencies = []
        self.__open = True
        self.__probabilities = None
        self.__cpt = None

    def bn(self):
        return self.__bn

    def name(self):
        return self.__name

    def values(self):
        return self.__values

    def add_dependencies(self, dependencies):
        self.__dependencies.append(dependencies)

    def dependencies(self):
        return self.__dependencies

    def close(self):
        if self.__open:
            self.__open = False
            self.__cpt = CPT(self)

    def open(self):
        return self.__open

    def cpt(self):
        return self.__cpt

    def print(self, depth=1):
        prefix = '' + ('\t' * depth)
        output = prefix + 'Column ' + self.__name + '\n'
        output += prefix + 'Values = ' + str(self.__values) + '\n'
        output += prefix + 'open = ' + str(self.__open) + '\n'
        if self.__cpt is not None:
            output += prefix + 'CPT = ' + str(self.__cpt.print()) + '\n'
        output += prefix + 'Dependencies:'
        for dep in self.__dependencies:
            output += '\n' + dep.print(depth + 1)
        if len(self.__dependencies) == 0:
            output += ' None\n'
        if depth == 1:
            print(output)
        else:
            return output


class CPT:

    def __init__(self, column):
        self.__probabilities = []
        self.__column = column
        self.__deps = []
        self.__missing = len(column.values())
        for dep_col in column.dependencies():
            self.__deps.append([dep_col.name(), dep_col.values()])
            self.__missing *= len(dep_col.values())

    def add_probability(self, value, probability, evidence, compensate=True):
        if self.__get_single_probability(value, evidence) != -1:
            print("Error 5")
            return False
        aggr_prob = self.__get_aggr_probability(evidence)
        if (probability + aggr_prob[0]) > 1 or aggr_prob[1] == 0:
            print("Error 6")
            return False
        if aggr_prob[1] == 1 and (aggr_prob[0] + probability) != 1:
            print("Error 7")
            if not compensate:
                return False
            probability = 1 - aggr_prob[0]
        entry = (value, probability, evidence)
        self.__probabilities.append(entry)
        self.__missing -= 1
        if self.__missing == 0:
            self.__fill_bayes()
        return True

    def __get_single_probability(self, value, evidence):
        for entry in self.__probabilities:
            if value == entry[0] and equality_check(evidence, entry[2]):
                return entry[1]
        return -1

    def __get_aggr_probability(self, evidence):
        prob = 0.0
        empty = 0
        for value in self.__column.values():
            val_prob = self.__get_single_probability(value, evidence)
            if val_prob != -1:
                prob += val_prob
            else:
                empty += 1
        return prob, empty

    def __fill_bayes(self):
        parents = []
        if len(self.__deps) == 0:
            parents.append(None)
        else:
            for dep in self.__deps:
                parents.append(dep[0])
        probabilities = []
        parent_combinations = []
        for entry in self.__probabilities:
            if entry[2] not in parent_combinations:
                parent_combinations.append(entry[2])
        for combination in parent_combinations:
            values = []
            for entry in self.__probabilities:
                if entry[2] == combination:
                    values.append((entry[0], entry[1]))
            parent_values = []
            for element in combination:
                parent_values.append(element[1])
            if not parent_values:
                parent_values.append(None)
            parent_values = tuple(parent_values)
            if len(parent_values) == 1:
                parent_values = parent_values[0]
            probabilities.append((parent_values, self.__combination_probability(values)))
        self.__column.bn().add_probability(self.__column.name(), parents, probabilities)

    def __combination_probability(self, probabilities):
        output = []
        values = self.__column.values()
        for value in values:
            for probability in probabilities:
                if value == probability[0]:
                    output.append(probability[1])
                    break
        return tuple(output)

    def print(self):
        return self.__probabilities


def equality_check(a, b):
    if len(a) != len(b):
        return False
    for entry in a:
        if entry not in b:
            return False
    return True
