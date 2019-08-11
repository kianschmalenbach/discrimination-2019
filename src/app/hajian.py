import copy
import app.frequent_rules as frq

from model.dataframe import DataFrameWrapper


def hajian(input_file, output_file, min_sup, min_conf, alpha, attr, d_attr, d_attr_dep, label):
    db = DataFrameWrapper(input_file)
    rules = frq.get_frequent_rules(db, attr, min_sup, min_conf)
    fr = frq.get_frequent_classification_rules(rules.copy(), label)
    br = frq.get_frequent_background_rules(rules.copy(), d_attr)
    rr = frq.get_redlining_rules(db, fr, br, d_attr, alpha, label)
    mr = frq.get_discriminatory_rules(db, fr, d_attr, alpha)
    db = massage(db, copy.deepcopy(fr), copy.deepcopy(rr), copy.deepcopy(mr), alpha, d_attr, label)
    rules_new = frq.get_frequent_rules(db, attr, min_sup, min_conf)
    fr_new = frq.get_frequent_classification_rules(rules_new.copy(), label)
    br_new = frq.get_frequent_background_rules(rules_new.copy(), d_attr)
    rr_new = frq.get_redlining_rules(db, fr_new, br_new, d_attr, alpha, label)
    mr_new = frq.get_discriminatory_rules(db, fr_new, d_attr, alpha)
    output = db.calc_discrimination(d_attr_dep, label, "AFTER (HAJIAN)")
    output += frq.measure_utility(fr, rr, mr, fr_new, rr_new, mr_new, label)
    db.export(output_file)
    return output


def massage(db, fr, rr, mr, alpha, d_attr, label):
    for r in rr:
        premise = r[0]
        c = r[1]
        if c[0] != label:
            continue
        for i in range(len(premise)):
            b = premise[i]
            d = []
            for element in premise:
                if element != b:
                    d.append(element)
            gamma = r[2]
            inner_rules = []
            for att in d_attr:
                inner_premise = [(att, c[1]), b]
                inner_rules.append([inner_premise, c, db.prob(inner_premise + [c], inner_premise)[1]])
            for r2 in inner_rules:
                inner_premise = r2[0]
                a = []
                for element in inner_premise:
                    if element[0] in d_attr and element[1] == c[1]:
                        a.append(element)
                rb2 = [premise, a, db.prob(premise + a, premise)[1]]
                beta2 = rb2[2]
                delta1 = db.prob(premise + a)[1]
                delta = db.prob([b, c], [b])[1]
                delta2 = db.prob([b] + a)[1]
                beta1 = delta1 / delta2
                evidence = []
                for a_entry in a:
                    evidence.append([a_entry[0], not a_entry[1]])
                evidence.append([b[0], b[1]])
                evidence.append([c[0], not c[1]])
                for d_entry in d:
                    evidence.append([d_entry[0], not d_entry[1]])
                dbc = db.records(evidence)
                impacts = dbc.get_impacts(fr)
                if r2 in mr:
                    while delta <= (beta1 * (beta2 + gamma - 1)) / (beta2 * alpha) or delta <= (r2[2] / alpha):
                        max_impact = __find_max_impact(impacts)
                        if max_impact is None:
                            break
                        impacts = max_impact[0]
                        if len(impacts) <= 2:
                            break
                        db.modify(max_impact[1], [c])
                        delta = db.prob([b, c], [b])[1]
                        r2[2] = db.prob(r2[0] + [r2[1]], r2[0])[1]
                else:
                    while delta <= (beta1 * (beta2 + gamma - 1)) / (beta2 * alpha):
                        max_impact = __find_max_impact(impacts)
                        if max_impact is None:
                            break
                        impacts = max_impact[0]
                        if len(impacts) <= 2:
                            break
                        db.modify(max_impact[1], [c])
                        delta = db.prob([b, c], [b])[1]
    mr2 = mr.copy()
    for rule in rr:
        if __contains(mr2, rule):
            mr2.remove(rule)
    for r in mr2:
        premise = r[0]
        c = r[1]
        if c[0] != label:
            continue
        a = []
        b = []
        for element in premise:
            if element[0] in d_attr and element[1] == c[1]:
                a.append(element)
            else:
                b.append(element)
        if len(a) == 0:
            print("Error 1")
            return None
        if len(b) == 0:
            delta = db.prob([c])[1]
        else:
            delta = db.prob(b + [c], b)[1]
        evidence = []
        for a_entry in a:
            evidence.append([a_entry[0], not a_entry[1]])
        for b_entry in b:
            evidence.append(b_entry)
        evidence.append([c[0], not c[1]])
        dbc = db.records(evidence)
        impacts = dbc.get_impacts(fr)
        while delta <= (r[2] / alpha):
            max_impact = __find_max_impact(impacts)
            if max_impact is None:
                break
            impacts = max_impact[0]
            if len(impacts) <= 2:
                break
            db.modify(max_impact[1], [c])
            if len(b) == 0:
                delta = db.prob([c])[1]
            else:
                delta = db.prob(b + [c], b)[1]
            r[2] = db.prob(r[0] + [r[1]], r[0])[1]
    return db


def __find_max_impact(impacts):
    maximum = 0
    index = 0
    if impacts is None:
        return [], 0
    for i in range(len(impacts)):
        if impacts[i][1] > maximum:
            index = i
            maximum = impacts[i][1]
    if len(impacts) == 0 or len(impacts[index]) == 0:
        return None
    ret = impacts[index][0]
    del impacts[index]
    return impacts, ret


def __contains(rules, rule):
    for r in rules:
        if r == rule:
            return True
        if r[1] != rule[1] or len(r[0]) != len(rule[0]):
            continue
        answer = True
        for p in r[0]:
            if p not in rule[0]:
                answer = False
                break
        if answer:
            return True
    return False
