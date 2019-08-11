import app.bayes2csv as bc
import app.hajian as hj
import app.kamiran as km


def generate_dataset(file, network, node_values, weights, cpts, n, theta, d_attr_dep, label):
    bn = bc.Bayes2Csv("HR")
    bn.create_bn(network, node_values, weights)
    for cpt in cpts:
        bn.add_cpt(cpt[0], cpt[1])
    bn.create_samples(n)
    bn.compute_label_probability(theta)
    bn.export_csv(file)
    return bn.calc_discrimination(d_attr_dep, label)


def perform_massaging(s_file, h_file, k_file, o_file, alphas, min_sup, min_conf, attr, d_attr, d_attr_dep,
                      index, label, rank, output):
    alpha = alphas[0]
    output = "alpha,algorithm,D_all,D_expl,D_ill,DDPD,IDPD,MC,GC\n" + "-,initial, " + output + "-,-,-,-\n"
    while alpha <= alphas[2]:
        output += str(alpha) + ",hajian,"
        print("\nalpha = " + str(alpha) + '\n')
        output += hj.hajian(s_file, h_file, min_sup, min_conf, alpha, attr, d_attr, d_attr_dep, label)
        output += '\n' + str(alpha) + ",kamiran,"
        output += km.kamiran(s_file, k_file, min_sup, min_conf, alpha, attr, d_attr, d_attr_dep, index, label, rank)
        alpha = round(alpha + alphas[1], 3)
        output += '\n'
    file = open(o_file, "w+")
    file.write(output)
    file.close()


def main():
    network = [('A', 'C'), ('A', 'D'), ('B', 'D'), ('A', 'Label'), ('B', 'Label'), ('C', 'Label'), ('D', 'Label')]
    node_values = [[True, False]] * 5
    weights = [0.25, 0.125, 0.075, 0.225, 0.325]
    cpts = [['A', [(True, 0.5, [])]],
            ['B', [(True, 0.4, [])]],
            ['C', [(True, 0.4, [('A', False)]), (True, 0.6, [('A', True)])]],
            ['D', [(True, 0.4, [('A', False), ('B', False)]), (True, 0.7, [('A', False), ('B', True)]),
                   (True, 0.5, [('A', True), ('B', False)]), (True, 0.9, [('A', True), ('B', True)])]]]
    n = 5000
    theta = 0.6
    s_file = '../data/sample.csv'
    h_file = '../data/hajian.csv'
    k_file = '../data/kamiran.csv'
    o_file = '../data/eval.csv'
    min_sup = 0.05
    min_conf = 0.1
    alphas = [0.5, 0.1, 2.0]
    attr = ['A', 'B', 'C', 'D', 'Label']
    d_attr_dep = [('A', ['C', 'D']), ('B', ['D'])]
    d_attr = []
    for entry in d_attr_dep:
        d_attr.append(entry[0])
    index = 'Index'
    label = 'Label'
    rank = 'Rank'
    output = generate_dataset(s_file, network, node_values, weights, cpts, n, theta, d_attr_dep, label)
    perform_massaging(s_file, h_file, k_file, o_file, alphas, min_sup, min_conf, attr, d_attr, d_attr_dep,
                      index, label, rank, output)


if __name__ == "__main__":
    main()
