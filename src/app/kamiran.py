import app.frequent_rules as frq
from model.dataframe import DataFrameWrapper


def kamiran(input_file, output_file, min_sup, min_conf, alpha, attr, d_attr, d_attr_dep, index, label, rank):
    db = DataFrameWrapper(input_file)
    rules = frq.get_frequent_rules(db, attr, min_sup, min_conf)
    fr = frq.get_frequent_classification_rules(rules.copy(), label)
    br = frq.get_frequent_background_rules(rules.copy(), d_attr)
    rr = frq.get_redlining_rules(db, fr, br, d_attr, alpha, label)
    mr = frq.get_discriminatory_rules(db, fr, d_attr, alpha)
    db = massage(db, d_attr_dep, label, rank, index)
    rules_new = frq.get_frequent_rules(db, attr, min_sup, min_conf)
    fr_new = frq.get_frequent_classification_rules(rules_new.copy(), label)
    br_new = frq.get_frequent_background_rules(rules_new.copy(), d_attr)
    rr_new = frq.get_redlining_rules(db, fr_new, br_new, d_attr, alpha, label)
    mr_new = frq.get_discriminatory_rules(db, fr_new, d_attr, alpha)
    output = db.calc_discrimination(d_attr_dep, label, "AFTER (KAMIRAN)")
    output += frq.measure_utility(fr, rr, mr, fr_new, rr_new, mr_new, label)
    db.export(output_file)
    return output


def massage(db, d_attr, label, rank, index_column):
    for d_att in d_attr:
        s = d_att[0]
        expl_attr = d_att[1]
        values = [False] * len(d_att[1])
        while values is not None:
            evidence = list(zip(expl_attr, values))
            values = _next_value(values)
            grouped = db.df()
            for entry in evidence:
                grouped = grouped.loc[grouped[entry[0]] == entry[1]]
            for s_val in (True, False):
                grouped_s = grouped.loc[grouped[s] == s_val]
                delta_s = delta(db, (s, s_val), evidence, label)
                grouped_del = grouped_s.loc[grouped_s[label] == s_val].sort_values(rank, ascending=s_val)
                grouped_dup = grouped_s.loc[grouped_s[label] == (not s_val)].sort_values(rank, ascending=(not s_val))
                delta_pos = delta_s
                delta_neg = delta_s
                for row in grouped_del.iterrows():
                    if delta_pos == 0:
                        break
                    db.delete(row[1][index_column])
                    delta_pos -= 1
                for row in grouped_dup.iterrows():
                    if delta_neg == 0:
                        break
                    db.duplicate(row[1][index_column])
                    delta_neg -= 1
    return db


def delta(db, s, e, label):
    p_star = 0.5*(db.prob([(label, True)], e + [(s[0], True)])[1] + db.prob([(label, True)], e + [(s[0], False)])[1])
    return int(db.prob([s], e)[0] * abs(db.prob([(label, True)], [s]+e)[1] - p_star))


def _next_value(array):
    for i in range(len(array)):
        if not array[-i - 1]:
            array[-i - 1] = True
            return array
        else:
            array[-i - 1] = False
    return None
