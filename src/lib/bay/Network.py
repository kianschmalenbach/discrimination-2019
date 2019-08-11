import numpy as np
import pandas as pd
from progress.bar import ChargingBar as Bar
from random import randint
from collections import Counter


class Network(object):

    def __init__(self, name):
        self.name = name
        self.node = dict()
        self.cpd = dict()
        return

    def add_node(self, name, node_type, levels, values):
        self.node[name] = {'type': node_type, 'n': levels, 'levels': values}
        return

    def add_probability(self, child, parents, table):
        self.node[child]['parents'] = parents
        indices = list()
        values = list()
        for row in table:
            indices.append(row[0])
            values.append(row[1])
        if len(parents) > 1:
            index = pd.MultiIndex.from_tuples(indices, names=parents)
            self.cpd[child] = pd.DataFrame(data=values, columns=self.node[child]['levels'], index=index)
        if len(parents) == 1 and parents[0] is not None:
            df = pd.DataFrame(data=values, columns=self.node[child]['levels'], index=indices)
            df.index.name = parents[0]
            self.cpd[child] = df
        if parents[0] is None:
            self.cpd[child] = pd.DataFrame(data=values, columns=self.node[child]['levels'])
        return

    def get_probability(self, name, evidence=dict(), value=None):
        cpd = self.cpd[name]
        if len(evidence) > 1:
            levels = list(evidence.keys())
            values = list(evidence.values())
            if value is not None:
                return cpd.xs(values, level=levels)[value]
            else:
                return cpd.xs(values, level=levels)
        elif len(evidence) == 1:
            values = list(evidence.values())
            if value is not None:
                return cpd.loc[values[0], value]
            else:
                return cpd.loc[values[0]]
        elif len(evidence) == 0:
            if value is not None:
                return cpd[value]
            else:
                return cpd

    def sample(self, name, given=dict(), r=None):
        parents = self.node[name]['parents']
        if parents[0] is None:
            cpd = self.get_probability(name)
        else:
            if not set(parents) == set(given.keys()):
                raise Exception('Parent samples need to be provided to sample child.')
            cpd = self.get_probability(name, evidence=given)
        if r is None:
            r = np.random.uniform()
        sum = 0
        if len(cpd.shape) == 1:
            cpd = dict(cpd)
            for key in cpd:
                sum += cpd[key]
                if r < sum:
                    return key
        if len(cpd.shape) == 2:
            for column in cpd:
                sum += list(cpd[column])[0]
                if r < sum:
                    return column

    def compute_sample(self, preset=dict()):
        unobserved = set(self.node.keys())
        sample = dict(preset)
        unobserved = unobserved - set(sample.keys())
        for node in unobserved:
            sample = self.sample_nodes([node], sample)
            unobserved = unobserved - set(sample.keys())
        return sample

    def random_sample(self, bn, preset=dict()):
        unobserved = set(self.node.keys())
        sample = dict(preset)
        unobserved = unobserved - set(sample.keys())
        for node in unobserved:
            values = bn.cpd[node]
            values = list(values)
            ind = randint(0, len(values)-1)
            sample[node] = values[ind]
        return sample

    def sample_nodes(self, nodes, sample):
        for node in nodes:
            if node not in sample.keys():
                parents = self.node[node]['parents']
                if parents[0] is None:
                    sample[node] = self.sample(node)
                elif set(parents) <= set(sample.keys()):
                    given = {key: sample[key] for key in parents}
                    sample[node] = self.sample(node, given)
                else:
                    sample = self.sample_nodes(nodes=parents, sample=sample)
                    given = {key: sample[key] for key in parents}
                    sample[node] = self.sample(node, given)
        return sample

    def rejection_sample(self, predict=dict(), given=dict(), n=10000):
        sum = 0
        bar = Bar('Sampling', max=n)
        for i in range(n):
            bar.next()
            sample = self.compute_sample()
            evidence = {key: sample[key] for key in given.keys()}
            if not given == evidence:
                continue
            evidence = {key: sample[key] for key in predict.keys()}
            if not predict == evidence:
                continue
            sum += 1
        bar.finish()
        return sum/n

    def likelihood_weighting(self, predict=dict(), given=dict(), n=10000):
        num = den = 0
        bar = Bar('Sampling', max=n)
        for i in range(n):
            bar.next()
            sample = self.compute_sample(preset=predict)
            for node in predict.keys():
                parents = self.node[node]['parents']
                given_pa = {key: sample[key] for key in parents}
                weight = float(self.get_probability(node, evidence=given_pa, value=predict[node]))
            evidence = {key: sample[key] for key in given.keys()}
            if given == evidence:
                num += weight
            den += weight
        bar.finish()
        return num/den

    def gibbs_sampling(self, bn, predict=dict(), given=dict(), n=10000, skip = 50):
        bar = Bar('Sampling', max=n)
        nodes = list(self.node.keys())
        sample = self.random_sample(bn, preset=given)
        count = 0
        sum = 0
        for i in range(n):
            last = None
            bar.next()
            node = None
            last = node
            while node is None or node in given.keys() or node == last:
                node = nodes[randint(0,len(nodes)-1)]
            parents = self.node[node]['parents']
            if parents[0] is None:
                sample[node] = self.sample(node)
            else:
                given = {key: sample[key] for key in parents}
                sample[node] = self.sample(node, given=given)
            if count == skip:
                evidence = {key: sample[key] for key in predict.keys()}
                if not predict == evidence:
                    continue
                sum += 1
            else:
                count += 1
        bar.finish()
        return sum/(n-count)

    def predict(self, bn, nodes, algorithm='gibbs', given=dict(), n=10000, skip=100):
        p = dict()
        for node in nodes:
            p[node] = dict()
            levels = self.node[node]['levels']
            for level in levels:
                predict = {node:level}
                if algorithm == 'gibbs':
                    p[node][level] = self.gibbs_sampling(bn, predict=predict, given=given, n=n, skip=skip)
                elif algorithm == 'reject':
                    p[node][level] = self.rejection_sample(predict=predict, given=given, n=n)
                elif algorithm == 'weighting':
                    p[node][level] = self.likelihood_weighting(predict=predict, given=given, n=n)
            p[node] = Counter(p[node]).most_common(1)
        return p
