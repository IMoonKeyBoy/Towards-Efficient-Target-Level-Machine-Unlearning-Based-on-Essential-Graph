"""
Layer class:

    name: the name of this layer
    layer_and_layer_relevances: the relevance between this layer's nodes and next layer's nodes,
        the size of this variable is number_of_nodes * number_of_next_layer_nodes

"""


class explanatorylayer:

    # init function
    def __init__(self, layername, number_of_nodes):
        self.name = layername
        self.nodes_weights = []
        for i in range(number_of_nodes):
            self.nodes_weights.append(0)  # [0 for j in range(number_of_next_layer_nodes)]

    # sort function, which sort the relevances of given node(index_nodes) from max to min
    def sort(self) -> list:
        sorted_id = sorted(range(len(self.nodes_weights)),
                           key=lambda k: self.nodes_weights[k], reverse=True)
        return sorted_id

    def set_nodes_weights(self, nodes_weights_para):
        self.nodes_weights = nodes_weights_para