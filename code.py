
# coding: utf-8

# # Voleon Interview Problem

# ### Summary
#
# This is a regression learning problem and the hypothesis set is assumed to be a first degree polynomials. We will estimate the coefficients of the model by minimizing the loss function. Three loss functions will be used: Ordinary Least Square (OLS), OLS with $\mathscr{l}_1$ penalty, and OLS with $\mathscr{l}_2$ penalty. We will then use K-Fold cross-validation to average the estimated weights and compare the results.
#
#
# ### General learning problem
#
# Data points $(x_n, y_n)$ are generated from the joint distribution $$P(x,y) = P(x) P(y|x).$$
#
# $P(x)$ is the distribution of the independent variable $x$ and $$P(y|x) = f(x) + P(e|x).$$
#
# $f(x)$ is the deterministic target function from input space $\mathcal{X}$ to output space $\mathcal{Y}$ ($f: \mathcal{X}\to \mathcal{Y}$), and e is the noise in the target function. The target function can be found by: $$f(x) = \mathbb{E}[y|x].$$
#
# Moreover, the noise in the target has the property $$\mathbb{E}[e|x] = 0$$
#
# As instructed in the statement of the problem, we assume a hypothesis set $\mathcal{H}$ with first degree polynomials:
#
# $$\mathcal{H} = \{h \mid h(x) = w_0 + w_1 x ~ \forall ~ w_0, w_1 \in \mathbb{R}\}$$
#
# The goal is to find the best hypothesis $g$ from our hypothesis set $\mathcal{H}$ that minimizes the error between $g(x)$ and $f(x)$.
#
#
# ### Model Formulation with linear regression algorithm
#
# As mentioned before, we assume our model (hypothesis) function to be a linear combination of the independent variable $x$:
#
# $$
# h(\mathbf{x}) = \mathbf{w}^\mathrm{T} \mathbf{x}
# \\[0.3em]
# \mathbf{w} = \begin{bmatrix}
#            w_0 \\[0.3em]
#            w_1 \\[0.3em]
#           \end{bmatrix}
# ~ , ~
# \mathbf{x} = \begin{bmatrix}
#            1 \\[0.3em]
#            x \\[0.3em]
#            \end{bmatrix}
# $$
#
# Our learning algorithm is to find the weights $a = w_0$ and $b = w_1$ such that they minimize the error. In order to find the best values, we will use three loss functions as our error measures:
#
# 1. Mean Squared Error (MSE): Ordinary Least Square (OLS) linear regression
# 2. MSE + $\mathscr{l}_1$ regularizer: Lasso linear regression
# 3. MSE + $\mathscr{l}_2$ regularizer: Ridge linear regression
#
#
# **Note**: General Method of Moments (GMM) will result in the same error measure as using Mean Squared Error (MSE).
#
# ### Optimization problem
#
# #### OLS linear regression:
#
# $$
# Loss = \frac{1}{N} \sum^N_{n=1} (\mathbf{w}^\mathrm{T} \mathbf{x}_n - y_n)^2
# $$
#
# MSE loss function is beneficial when we would like to accomodate effect of every data point. It is a very simple model and minimization of the loss function has a closed form solution:
#
# $$
# \mathbf{w} = \mathbf{X}^\dagger \mathbf{y}
# $$
#
# where:
# $$
# \mathbf{X} = \begin{bmatrix}
#            \mathbf{x}^\mathrm{T}_1 \\[0.3em]
#            \mathbf{x}^\mathrm{T}_2 \\[0.3em]
#            \vdots \\[0.3em]
#            \mathbf{x}^\mathrm{T}_N \\[0.3em]
#            \end{bmatrix}
# ~ , ~
# \mathbf{y} = \begin{bmatrix}
#            y_1 \\[0.3em]
#            y_2 \\[0.3em]
#            \vdots \\[0.3em]
#            y_N \\[0.3em]
#            \end{bmatrix}
# $$
#
# $$
# \mathbf{X}^\dagger = \left( \mathbf{X}^\mathrm{T} \mathbf{X}\right)^{-1} \mathbf{X}^\mathrm{T}
# $$
#
#
# #### Lasso linear regression:
#
# $$
# Loss = \frac{1}{N} \sum^N_{n=1} (\mathbf{w}^\mathrm{T} \mathbf{x}_n - y_n)^2 + \alpha \| \mathbf{w} \|
# $$
#
#
# Lasso is more robust to the effect of outliers. I will use the coordinate descent optimization method (the default method for Lasso regression in python) to minimize the loss function and find weight parameters $a$ and $b$.
#
#
# #### Ridge linear regression:
# $$
# Loss = \frac{1}{N} \sum^N_{n=1} (\mathbf{w}^\mathrm{T} \mathbf{x}_n - y_n)^2 + \alpha \mathbf{w}^\mathrm{T}\mathbf{w}
# $$
#
# Ridge linear regression has a smooth loss function but it is less robust to the effect of outliers. Conjugate gradient method will be used to minimize the loss function and find the weight parameters $a$ and $b$.

# ### 1. Importing the libraries

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from IPython.display import display

sns.set()
sns.set_style("whitegrid")
# sns.set(style="ticks")
# get_ipython().magic(u'matplotlib inline')


# ### 2. Loading data:

# In[2]:

data = {}
data[1] = pd.read_csv('data_1_1.csv')
data[2] = pd.read_csv('data_1_2.csv')
data[3] = pd.read_csv('data_1_3.csv')
data[4] = pd.read_csv('data_1_4.csv')
data[5] = pd.read_csv('data_1_5.csv')


# ### 3. Initial exploratory data visualization

# In[3]:

fig = plt.figure(figsize=[16,24])
ax1 = fig.add_subplot(321)
data[1].plot.scatter('x','y', ax=ax1);
ax2 = fig.add_subplot(322)
data[2].plot.scatter('x','y', ax=ax2, color='red');
ax3 = fig.add_subplot(323)
data[3].plot.scatter('x','y', ax=ax3, color='green');
ax4 = fig.add_subplot(324)
data[4].plot.scatter('x','y', ax=ax4, color='orange');
ax5 = fig.add_subplot(325)
data[5].plot.scatter('x','y', ax=ax5, color='purple');

# fig, axes = plt.subplots(3, 2, figsize=[16,24])
# data1.plot.scatter('x','y', ax=axes[0,0]);
# data2.plot.scatter('x','y', ax=axes[0,1], color='red');
# data3.plot.scatter('x','y', ax=axes[1,0], color='green');
# data4.plot.scatter('x','y', ax=axes[1,1], color='orange');
# data5.plot.scatter('x','y', ax=axes[2,0], color = 'black');


# ### 4. Modeling data
#
# #### 4.1 Getting data statistics

# In[4]:

sns.jointplot(x='x', y='y', data=data[1], kind='reg');
data[1].describe()


# In[5]:

sns.jointplot(x='x', y='y', data=data[2], kind='reg', color='red');
data[2].describe()


# In[6]:

sns.jointplot(x='x', y='y', data=data[3], kind='reg', color='green');
data[3].describe()


# In[7]:

sns.jointplot(x='x', y='y', data=data[4], kind='reg', color='orange');
data[4].describe()


# In[8]:

sns.jointplot(x='x', y='y', data=data[5], kind='reg', color='purple');
data[5].describe()


# #### 4.2 Modeling data:

# In[9]:

for i in xrange(1, len(data)+1):
    print 'Modeling data in file data1_%d.csv' % i
    X = data[i].x.values.reshape(-1, 1)
    y = data[i].y.values#.reshape(-1, 1)

    print 'Linear regression model: y = ax + b'
    ols = LinearRegression()
    kf = KFold(5)
    w_0 = np.array([])
    w_1 = np.array([])
    error = np.array([])
    r2 = np.array([])
    for k, (train, test) in enumerate(kf.split(X, y)):
        ols.fit(X[train], y[train])
        w_0 = np.append(w_0, ols.intercept_)
        w_1 = np.append(w_1, ols.coef_[0])
        error = np.append(error, mean_squared_error(y[test], ols.predict(X[test])))
        r2 = np.append(r2, ols.score(X[test], y[test]))
    a = w_1.mean()
    b = w_0.mean()
    e = error.mean()
    print 'OLS   lin Reg: a = %.2f , b = %.2f , MSE = %.3f, R^2 = %.3f' % (a, b, e, r2.mean())
    # clf = GridSearchCV(LinearRegression(), cv=5)
    # clf.fit(data1.x, data1.y)


    lso = LassoCV(cv=5)
    kf = KFold(5)
    w_0 = np.array([])
    w_1 = np.array([])
    error = np.array([])
    r2 = np.array([])
    for k, (train, test) in enumerate(kf.split(X, y)):
        lso.fit(X[train], y[train])
        w_0 = np.append(w_0, lso.intercept_)
        w_1 = np.append(w_1, lso.coef_[0])
        error = np.append(error, mean_squared_error(y[test], lso.predict(X[test])))
        r2 = np.append(r2, lso.score(X[test], y[test]))
    #     print lso.alpha_
    a = w_1.mean()
    b = w_0.mean()
    e = error.mean()
    print 'LASSO lin reg: a = %.2f , b = %.2f , MSE = %.3f, R^2 = %.3f' % (a, b, e, r2.mean())

    alphas = np.logspace(-4, 4, 30)
    rdg = RidgeCV(alphas=alphas, cv=5)
    kf = KFold(5)
    w_0 = np.array([])
    w_1 = np.array([])
    error = np.array([])
    r2 = np.array([])
    for k, (train, test) in enumerate(kf.split(X, y)):
        rdg.fit(X[train], y[train])
        w_0 = np.append(w_0, rdg.intercept_)
        w_1 = np.append(w_1, rdg.coef_[0])
        error = np.append(error, mean_squared_error(y[test], rdg.predict(X[test])))
        r2 = np.append(r2, rdg.score(X[test], y[test]))
    #     print rdg.alpha_
    a = w_1.mean()
    b = w_0.mean()
    e = error.mean()
    print 'Ridge lin reg: a = %.2f , b = %.2f , MSE = %.3f, R^2 = %.3f' % (a, b, e, r2.mean())
    print


# ### Observation:
#
# Based on the current datasets, and more specifically for datasets from files data1_4.csv and data1_5.csv, a first degree polynomial hypothesis set may not be the best choice to model our samples data generation process. ***This is a high bias problem!***
