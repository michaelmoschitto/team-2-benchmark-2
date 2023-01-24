from sklearn.base import RegressorMixin
import math
import numpy as np
import copy

def entropy(y):
    e = 0
    val_counts = y.value_counts()
    for val in val_counts:
        prob = val / y.shape[0]
        e -=(prob * math.log(prob, 2))
    return e



def gain(y,x):
    g = 0
    for val in x.unique():
        restricted_y = y.loc[x == val]
        g += entropy(restricted_y) * (len(restricted_y) / len(x))
    return entropy(y) - g

def gain_ratio(y,x):
    g = gain(y,x)
    return g/entropy(y)

def helper(tree):

    if tree == 0 or tree == 1:
        return ([tree], "visited")
    
    key_choice = list(tree.keys())[0]
    
    val_choice = None
    for val in (list(tree[key_choice].keys())):
        if tree[key_choice][val] != "visited":
            val_choice = val
            break
            
    if val_choice!= None:
        rule = [(key_choice, val_choice)]
        new_rule, tree[key_choice][val_choice] = helper(tree[key_choice][val_choice])
        if new_rule != []:
            rule += new_rule
        else:
            return ([], tree)
        return (rule, tree)
    return ([], "visited")

class DecisionTree(RegressorMixin):
    def __init__(self) -> None:
        self._tree = {}
        self._rules = []
        self._default = 0
        self._gain = 0
        self._min_split_count = 5



    def select_split(self, X,y):
        col = None
        gr = -1
        bestSplit = None
        for column in X.columns:
            if X[column].dtypes == float:
                vals = sorted(X[column].unique())
                splits = []
                for i in range(len(vals) - 1):
                    splits.append(round((vals[i] + vals[i + 1])/2, 2))
                
                for split in splits:
                    convertedX = X[column] < split
                    gain = gain_ratio(y, convertedX)
                    if gain > gr:
                        gr = gain
                        col = column
                        bestSplit = split            
            else: 
                gain = gain_ratio(y, X[column])
                if gain > gr:
                    gr = gain
                    col = column
                    bestSplit = None

        return col, gr, bestSplit

    def generate_tree(self, X, y):
        tree = {}
        val_tree = {}
        columns = X.columns
        
        #Pre-processing
        for column in columns:
            if len(X[column].unique()) == 1:
                X = X.drop(columns=column)
        
        if len(y) < self._min_split_count:
            return y.value_counts(normalize=True).idxmax()
            
        #Base Case 1: ALL REMAINING PREDICTIONS ARE ALIVE or DEAD 
        if len(y.unique()) == 1:
            return y.unique()[0]
        
        #Base Case 2: NO COLUMNS REMAINING
        if len(X.columns) == 0:
            return y.value_counts(normalize=True).idxmax()
        
        else:
            #DETERMINE NEXT BEST FEATURE
            next_choice = self.select_split(X, y)
            column_name, gr, split = next_choice
            
            if gr < .000001:
                return y.value_counts(normalize=True).idxmax()
            
            if split != None:
                new_column_name = "%s<%.2f"%(column_name, split)
                X[new_column_name] = X[column_name] < split
                X.drop(column_name, axis=1, inplace=True)
                column_name = new_column_name
            
            vals = X[column_name].unique()
            #FOR EACH VALUE IN THAT FEATURE
            for val in vals:
                #CREATE NEW TABLES THAT RESTRICT YOUR TABLES TO ROWS THAT HAVE THAT VALUE
                restricted_y = y.loc[X[column_name] == val]
                restricted_X = X.loc[X[column_name] == val].drop(column_name,axis=1)
                #GET NEXT BRANCH WITH THAT RESTRICTION
                if type(val) == np.bool_:
                    val = str(val)
                val_tree[val] = self.generate_tree(restricted_X, restricted_y)
            column_name = str(column_name)
            tree = {
                column_name: val_tree
            }
        return tree


    def get_decision_rules(self):
        rules = []
        new_tree = copy.deepcopy(self._tree)
        counter = 0
        while new_tree != "visited":
            rule, new_tree = helper(new_tree)
            if rule != []:
                rules.append(rule)
        return rules

    def fit(self, X, y):
        self._tree = self.generate_tree(X, y)
        self._rules = self.get_decision_rules()

        return self

    def make_prediction(self,x):
        x_rules = []
        
        for index, val in x.items():
            x_rules.append((index, val))
        
        for rule in self._rules:
            for cond in rule:
                if cond == 1 or cond == 0:
                    return cond
                elif cond in x_rules:
                    continue
                
                elif "<" in cond[0]:
                    args = cond[0].split("<")
                    col = args[0]
                    found = False
                    for x_rule in x_rules:
                        if x_rule[0] == col:
                            val = x_rule[1]
                            
                            if (str(float(val) < float(args[1])) == cond[1]):
                               
                                found = True
                                break
                    if found == False:
                        break
                else:
                    break
        return self._default

    def predict(self, X):
        y_pred = X.apply(lambda x: self.make_prediction(x),axis=1)
        return y_pred

