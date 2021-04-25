# Desicion-Tree-Code

Decision Tree

ZOOM2 = pd.read_csv("ZOOM2.csv")
ZOOM2.head(1)
ZOOM2.dtypes
cat_col_names = [col_name for col_name in ZOOM2.columns if ZOOM2[col_name].dtype=='object']
cat_col_names

for col_name in cat_col_names:
    print(col_name, '\n', ZOOM2[col_name].value_counts(), '\n', sep='')

cross_freq = pd.crosstab(ZOOM2.REPORTED_SATISFACTION, ZOOM2.REPORTED_USAGE_LEVEL)
cross_freq

# rearrange rows and columns
sorted_a = ['very_unsat', 'unsat', 'avg', 'sat', 'very_sat']
sorted_b = ['very_little', 'little', 'avg', 'high', 'very_high']
cross_freq = cross_freq.loc[sorted_a, sorted_b]
cross_freq

# initialize plotting environment
sns.set()
In [10]:
ax = sns.heatmap(cross_freq)

def check_cross_freq(var_a, var_b, sorted_a=None, sorted_b=None):
    cross_freq = pd.crosstab(ZOOM2[var_a], ZOOM2[var_b])
    if sorted_a:
        cross_freq = cross_freq.loc[sorted_a, :]
    if sorted_b:
        cross_freq = cross_freq.loc[sorted_b]
    display(cross_freq)
    ax = sns.heatmap(cross_freq)
    plt.show()

check_cross_freq('REPORTED_SATISFACTION', 'LEAVE', ['very_unsat', 'unsat', 'avg', 'sat', 'very_sat'])

import warnings
with warnings.catch_warnings():
    # sns.set() # for testing purpose
    # suppress the "deprecated" warnings
    warnings.simplefilter("ignore")
    # restore to the default plot settings
    sns.reset_orig();

# collect column names for numeric columns
num_col_names = ZOOM2.select_dtypes(include=['int64']).columns.values
num_col_names

ZOOM2[num_col_names].describe()

ZOOM2.hist(column=num_col_names, figsize=(20, 10), layout=(2, 4));

from sklearn.preprocessing import OneHotEncoder
In [18]:
enc = OneHotEncoder(sparse=False)
In [19]:
enc.fit(ZOOM2[cat_col_names])

cat_col_names

enc.categories_

X_cat = enc.transform(ZOOM2[cat_col_names])
print(X_cat.shape)
X_cat

enc = OneHotEncoder(sparse=False)
X_cat = enc.fit_transform(ZOOM2[cat_col_names])
print(X_cat.shape)
X_cat

X_cat_features = X_cat[:, :-2]
y = X_cat[:, -2]

cat_var_name_l = []
for var_name, one_var in zip(cat_col_names, enc.categories_):
    for level in one_var:
        # cat_var_name_l.append(f"{var_name}_is_{level}")
        cat_var_name_l.append("{}_is_{}".format(var_name, level))
print(cat_var_name_l[:5], "...", cat_var_name_l[-5:])
print('number of features:', len(cat_var_name_l))

cat_feature_name_l = cat_var_name_l[:-2]
y_name_l = cat_var_name_l[-1:-3:-1]
print(cat_feature_name_l[:5])
print(y_name_l)

#GROW TREE
from sklearn import tree
from sklearn.metrics import accuracy_score 

clf_entropy = tree.DecisionTreeClassifier( 
            criterion = "entropy", 
            random_state = 100) 
In [29]:
clf_entropy.fit(X_cat_features, y)

y_hat = clf_entropy.predict(X_cat_features)
print(accuracy_score(y, y_hat))

np.mean(y == y_hat)

plt.figure(figsize=(10, 10))
tree.plot_tree(clf_entropy, feature_names=cat_feature_name_l, class_names=y_name_l, max_depth=1, fontsize=10);

clf_entropy.fit(X_num_features, y);
clf_entropy.fit(X_num_features, y);
In [37]:
y_hat = clf_entropy.predict(X_num_features)
print(accuracy_score(y, y_hat))

In [37]:
y_hat = clf_entropy.predict(X_num_features)
print(accuracy_score(y, y_hat))

clf_entropy.tree_.max_depth


clf_entropy = tree.DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth=5, min_samples_leaf=20, min_samples_split=100)
clf_entropy.fit(X_num_features, y);
y_hat = clf_entropy.predict(X_num_features)
print(accuracy_score(y, y_hat))

fi_pd = pd.DataFrame({'feature': num_col_names, 'importance': clf_entropy.feature_importances_})
fi_pd.sort_values('importance', ascending=False)
plt.figure(figsize=(8, 8))
tree.plot_tree(clf_entropy, feature_names=num_col_names, class_names=y_name_l, max_depth=1);

p1=5527/(5527+7703); p2 = 1-p1; -p1*np.log2(p1)-p2*np.log2(p2)

import graphviz
dot_data = tree.export_graphviz(
    clf_entropy, 
    feature_names=num_col_names,
    class_names=y_name_l, 
    filled=True, # for eyeball identifying paths with high purity 
    rounded=True, 
    rotate=True, # plot root to leaf from left to right
    out_file=None) 
graph = graphviz.Source(dot_data) 

graph.render("full_tree")

![image](https://user-images.githubusercontent.com/80474158/116000227-577ace00-a5bd-11eb-85a3-8b64dd480bce.png)
