import pandas as pd


class DataFrameWrapper:

    def __init__(self, file=None, df=None):
        if file is not None:
            self.__df = pd.read_csv(file, true_values=['True'], false_values=['False'])
        else:
            self.__df = df
        self.__columns = list(self.__df)
        self.__index_column = self.__columns[0]
        self.__columns.remove(self.__index_column)
        self.__max_index = self.count()

    def get_impacts(self, fr):
        impacts = []
        for row in self.__df.iterrows():
            index = row[1][self.__index_column]
            count = 0
            for frequent_rule in fr:
                premise = frequent_rule[0]
                supports = False
                for entry in premise:
                    if row[1][entry[0]] == entry[1]:
                        supports = True
                        break
                if supports:
                    count += 1
            impacts.append([index, count])
        return impacts

    def count(self):
        return self.__df.count()[self.__index_column]

    def df(self):
        return self.__df

    def modify(self, index, request):
        key = self.__df.index[self.__df[self.__index_column] == index].tolist()[0]
        for entry in request:
            self.__df.at[key, entry[0]] = entry[1]

    def delete(self, index):
        key = self.__df.index[self.__df[self.__index_column] == index].tolist()
        if len(key) == 0:
            return
        key = key[0]
        self.__df = self.__df[self.__df[self.__index_column] != key]

    def duplicate(self, index):
        key = self.__df.index[self.__df[self.__index_column] == index].tolist()
        if len(key) == 0:
            return
        key = key[0]
        row = self.__df[self.__df[self.__index_column] == index]
        row = row.rename(index={key: self.__max_index})
        self.__max_index += 1
        row.at[row.index[0], self.__index_column] = self.__max_index
        self.__df = self.__df.append(row)

    # evidence = [('A', True), ('B', False)]
    # columns = [('Label', True), ('D', False)]
    # returns (number of events, probability)
    def prob(self, columns, evidence=None):
        if evidence is None:
            evidence = []
        column_names = []
        column_values = []
        for entry in columns:
            if type(entry) is not tuple or entry[0] not in self.__columns or type(entry[1]) is not bool:
                print("Error 1")
                return -1, -1
            column_names.append(entry[0])
            column_values.append(entry[1])
        aggr = self.__df
        if evidence is not None:
            for entry in evidence:
                if type(entry) is not tuple or not entry[0] in aggr.columns or type(entry[1]) is not bool:
                    print("Error 2")
                    return -1, -1
                aggr = aggr[(aggr[entry[0]] == entry[1])]
        aggr = aggr.groupby(column_names).count()
        positive_count = aggr[self.__index_column]
        all_count = aggr.sum()[self.__index_column]
        for column_value in column_values:
            try:
                positive_count = positive_count[column_value]
            except:
                return 0, 0
        return positive_count, positive_count / all_count

    def records(self, evidence):
        ret = self.__df
        for entry in evidence:
            ret = ret[ret[entry[0]] == entry[1]]
        return DataFrameWrapper(df=ret)

    def export(self, file):
        self.__df.to_csv(file, index=False, sep=',')

    def calc_discrimination(self, d_attr, label, time=""):
        if time != "":
            time = " " + time
        print("DISCRIMINATION CALCULATION" + time + ":")
        d_all = 0
        d_expl = 0
        for entry in d_attr:
            s = entry[0]
            d_all += self.prob([(label, True)], [(s, True)])[1] - self.prob([(label, True)], [(s, False)])[1]
            e_values = [False] * len(entry[1])
            while e_values is not None:
                evidence = list(zip(entry[1], e_values))
                p_star = (self.prob([(label, True)], [(s, True)] + evidence)[1]
                          + self.prob([(label, True)], [(s, False)] + evidence)[1]) / 2
                d_expl += (self.prob(evidence, [(s, True)])[1] - self.prob(evidence, [(s, False)])[1]) * p_star
                print("P*[" + s + "](+|" + str(evidence) + ") = " + str(p_star))
                e_values = _next_value(e_values)
        d_all /= len(d_attr)
        d_expl /= len(d_attr)
        d_ill = d_all - d_expl
        print('D_all = ' + str(d_all))
        print('D_expl = ' + str(d_expl))
        print('D_ill = ' + str(d_ill) + "\n")
        return str(d_all) + ',' + str(d_expl) + ',' + str(d_ill) + ','


def _next_value(array):
    for i in range(len(array)):
        if not array[-i - 1]:
            array[-i - 1] = True
            return array
        else:
            array[-i - 1] = False
    return None
