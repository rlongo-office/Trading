import json

class DecisionNode:
    def __init__(self, feature_index, threshold, left, right):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right

    def is_leaf(self):
        return False

class LeafNode:
    def __init__(self, value):
        self.value = value

    def is_leaf(self):
        return True

def save_tree(node):
    if node.is_leaf():
        return {"value": node.value}
    else:
        return {
            "feature_index": node.feature_index,
            "threshold": node.threshold,
            "left": save_tree(node.left),
            "right": save_tree(node.right)
        }

def save_gbm_model(trees, initial_prediction, learning_rate, filename='model.json'):
    model_data = {
        "initial_prediction": initial_prediction,
        "learning_rate": learning_rate,
        "trees": [save_tree(tree) for tree in trees]  # ensure that trees are iterable and correctly passed
    }
    with open(filename, 'w') as f:
        json.dump(model_data, f, indent=4)

def load_tree(node_data):
    """ Recursively load a tree from a nested dictionary. """
    if "value" in node_data:
        # This is a leaf node
        return LeafNode(value=node_data["value"])
    else:
        # This is a decision node
        left_node = load_tree(node_data["left"]) if "left" in node_data else None
        right_node = load_tree(node_data["right"]) if "right" in node_data else None
        return DecisionNode(
            feature_index=node_data["feature_index"], 
            threshold=node_data["threshold"], 
            left=left_node, 
            right=right_node
        )


def load_gbm_model(filename='model.json'):
    with open(filename, 'r') as f:
        model_data = json.load(f)
    trees = [load_tree(tree_data) for tree_data in model_data["trees"]]
    return trees, model_data["initial_prediction"], model_data["learning_rate"]

# The line below should be used explicitly where needed and not run on import
# trees, initial_pred, lr = load_gbm_model()
