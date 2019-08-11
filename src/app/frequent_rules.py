from itertools import chain, combinations


def get_frequent_classification_rules(rules, label):
    to_remove = []
    for rule in rules:
        if rule[1][0] != label:
            to_remove.append(rule)
    for rule in to_remove:
        rules.remove(rule)
    return rules


def get_frequent_background_rules(rules, d_attr):
    to_remove = []
    for rule in rules:
        if rule[1][0] not in d_attr or rule[1][1]:
            to_remove.append(rule)
            continue
        for premise in rule[0]:
            if premise[0] in d_attr:
                to_remove.append(rule)
                continue
    for rule in to_remove:
        rules.remove(rule)
    return rules


# returns [([('B', True), ('D', True)], True), ()]
def get_frequent_rules(db, attr, min_sup, min_conf):
    itemsets = get_frequent_itemsets(attr, db, min_sup)
    rules = []
    for itemset in itemsets:
        powerset = __powerset(itemset[0])
        del powerset[-1]
        if len(powerset) == 0:
            continue
        for subset in powerset:
            if len(subset) != len(itemset[0]) - 1:
                continue
            complement = itemset[0].copy()
            for element in subset:
                complement.remove(element)
            sup_itemset = itemset[1]
            sup_implication = db.prob(complement)[0]
            conf_rule = sup_itemset / sup_implication
            if conf_rule >= min_conf:
                rules.append([subset, complement[0], conf_rule])
    return rules


# returns [[[('A', True)], 30], [[('A', False)], 30]]
def get_frequent_itemsets(attr, db, min_sup):
    itemsets = []
    new_itemsets = []
    for att in attr:
        new_itemsets.append([[(att, True)], 0])
        new_itemsets.append([[(att, False)], 0])
    finished = False
    while not finished:
        finished = True
        old_itemsets = []
        to_remove = []
        for new_itemset in new_itemsets:
            count = db.prob(new_itemset[0])[0]
            if count < (min_sup * db.count()):
                to_remove.append(new_itemset)
            else:
                new_itemset[1] = count
                old_itemsets.append(new_itemset[0])
                finished = False
        for item in to_remove:
            new_itemsets.remove(item)
        if finished:
            break
        itemsets += new_itemsets
        new_itemsets = []
        for itemset1 in old_itemsets:
            for itemset2 in old_itemsets:
                new_itemset = __union(itemset1, itemset2)
                if len(new_itemset) > len(itemset1) and new_itemset not in new_itemsets:
                    new_itemsets.append(new_itemset)
        new_values = [0] * len(new_itemsets)
        new_itemsets = list(map(list, zip(new_itemsets, new_values)))
    return itemsets


def get_discriminatory_rules(db, rules, d_attr, alpha):
    ret = []
    for rule in rules:
        suitable = False
        for premise in rule[0]:
            if premise[0] in d_attr and rule[1][1] == premise[1]:
                suitable = True
                break
        if suitable and elift(db, rule, d_attr) > alpha:
            ret.append(rule)
    return ret


def elift(db, rule, d_attr):
    base_premise = []
    for entry in rule[0]:
        if entry[0] not in d_attr:
            base_premise.append(entry)
    return rule[2] / db.prob([rule[1]], base_premise)[1]


def get_redlining_rules(db, rules, br, d_attr, alpha, label):
    ret = []
    for rule in rules:
        if rule[1][0] != label:
            continue
        suitable = True
        for premise in rule[0]:
            if premise[0] in d_attr:
                suitable = False
                break
        if suitable:
            elb = elb_check(db, rule, br, alpha)
            if len(elb) > 0:
                ret.append(rule)
                ret += elb
    return ret


def elb_check(db, rule, br, alpha):
    ret = []
    premise = rule[0]
    for i in range(len(premise)):
        b = premise[i]
        d = []
        for element in premise:
            if element != b:
                d.append(element)
        gamma = rule[2]
        delta = db.prob([rule[1]], [b])[1]
        for background_rule in br:
            if len(background_rule[0]) != len(premise):
                continue
            suitable = True
            for element in background_rule[0]:
                if element not in premise or background_rule[1][1]:
                    suitable = False
                    break
            if not suitable:
                continue
            a = background_rule[1]
            beta2 = background_rule[2]
            param = [b, a]
            if len(d) > 0:
                param += d
            beta1 = db.prob(param)[1] / db.prob([b, a])[1]
            f = (beta1 / beta2) * (beta2 + gamma - 1)
            elb = 0
            if f > 0:
                elb = f / delta
            if elb >= alpha:
                ret.append(background_rule)
    return ret


def measure_utility(fr, rr, mr, fr_new, rr_new, mr_new, label):
    ddpd = __calc_prev_degr(mr, mr_new, label)
    idpd = __calc_prev_degr(rr, rr_new, label)
    mc = __calc_fr_diff(fr, fr_new) * 100
    gc = __calc_fr_diff(fr_new, fr) * 100
    print("DDPD = " + str(ddpd))
    output = str(ddpd) + ','
    print("IDPD = " + str(idpd))
    output += str(idpd) + ','
    print("MC = " + str(mc))
    output += str(mc) + ','
    print("GC = " + str(gc))
    output += str(gc)
    print()
    return output


def __calc_prev_degr(base, comp, label):
    base_count = len(base)
    if base_count == 0:
        return 100.0
    comp_count = 0
    for rule in comp:
        if rule[1][0] != label:
            continue
        for rule2 in base:
            if rule[0] == rule2[0] and rule[1] == rule2[1]:
                comp_count += 1
    pd = (100.0 * (base_count - comp_count)) / base_count
    if pd < 0.0:
        return 0.0
    return pd


def __calc_fr_diff(base, comp):
    count = 0.0
    for rule in base:
        found = False
        for rule2 in comp:
            if rule[0] == rule2[0] and rule[1] == rule2[1]:
                found = True
                break
        if not found:
            count += 1
    return count / len(base)


def __powerset(input_list):
    ret = list(chain.from_iterable(combinations(input_list, n) for n in range(len(input_list) + 1)))
    ret.remove(())
    for i in range(len(ret)):
        ret[i] = list(ret[i])
    return ret


def __union(set1, set2):
    columns1 = []
    for entry in set1:
        columns1.append(entry[0])
    for entry in set2:
        if entry[0] in columns1 and entry[1] != set1[columns1.index(entry[0])][1]:
            return []
    ret = list(set(set1 + set2))
    ret.sort()
    return ret
