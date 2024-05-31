"""
Graph class:

    layers: the layers of this graph
"""
from scheme_explanationlayer import explanatorylayer


class explantorygraph:

    def __init__(self):
        self.layers = []

    # print graph
    def print_graph(self):
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        for i in range(len(self.layers)):
            print(self.layers[i].name, "=", self.layers[i].nodes_weights)
            # print(self.layers[i].name,"\t",self.layers[len(self.layers)- 3 + i ].nodes_weights)

    # refresh graph
    def add_layer(self, layer):
        find = False
        for i in range(len(self.layers)):
            if layer.name == self.layers[i].name:
                find = True
                for j in range(len(self.layers[i].nodes_weights)):
                    self.layers[i].nodes_weights[j] = self.layers[i].nodes_weights[j] + layer.nodes_weights[j]
        if not find:
            self.layers.append(layer)

    def save_target_class(self, args):
        file = open("important" + "/" + args.run_name + "_" + args.dataset + "_" + str(args.target_class) + '_target_important_dict.txt', 'w')
        for i in range(len(self.layers)):
            file.write(self.layers[i].name + "\t" + str(self.layers[i].nodes_weights) + '\n')
        file.close()

    def save_only_balance(self, args):
        file = open("important" + "/" + args.run_name + "_" + args.dataset + "_" + str(args.target_class) + '_target_only_balance.txt', 'w')
        for i in range(len(self.layers)):
            file.write(self.layers[i].name + "\t" + str(self.layers[i].nodes_weights) + '\n')
        file.close()

    def save_no_balance(self, args):
        file = open(
            "important" + "/" + args.run_name + "_" + args.dataset + "_" + str(
                args.target_class) + '_target_non_balance.txt', 'w')
        for i in range(len(self.layers)):
            file.write(self.layers[i].name + "\t" + str(self.layers[i].nodes_weights) + '\n')
        file.close()

    def load_target_class(self, args):
        self.layers = []
        file = open(
            "important" + "/" + args.run_name + "_" + args.dataset + "_" + str(
                args.target_class) + '_target_important_dict.txt', 'r')
        for line in file.readlines():
            line = line.strip()
            k = line.split('\t')[0]
            v = line.split('\t')[1]
            int_list = list(map(float, v.replace("[", "").replace("]", "").split(",")))
            my_explanatorylayer = explanatorylayer(k, len(int_list))
            my_explanatorylayer.set_nodes_weights(int_list)
            self.add_layer(my_explanatorylayer)
        file.close()

    def load_only_balance(self, args):
        self.layers = []
        file = open(
            "important" + "/" + args.run_name + "_" + args.dataset + "_" + str(
                args.target_class) + '_target_only_balance.txt', 'r')
        for line in file.readlines():
            line = line.strip()
            k = line.split('\t')[0]
            v = line.split('\t')[1]
            int_list = list(map(float, v.replace("[", "").replace("]", "").split(",")))
            my_explanatorylayer = explanatorylayer(k, len(int_list))
            my_explanatorylayer.set_nodes_weights(int_list)
            self.add_layer(my_explanatorylayer)
        file.close()

    def load_no_balance(self, args):
        self.layers = []
        file = open(
            "important" + "/" + args.run_name + "_" + args.dataset + "_" + str(
                args.target_class) + '_target_non_balance.txt', 'r')
        for line in file.readlines():
            line = line.strip()
            k = line.split('\t')[0]
            v = line.split('\t')[1]
            int_list = list(map(float, v.replace("[", "").replace("]", "").split(",")))
            my_explanatorylayer = explanatorylayer(k, len(int_list))
            my_explanatorylayer.set_nodes_weights(int_list)
            self.add_layer(my_explanatorylayer)
        file.close()
