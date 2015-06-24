pyrvm: Ranking Vector Machines in Python
========================================

Learn and predict orderings of vectors using large-margin criteria.

This is an implementation of the ranking-vector-machine algorithm from
Yu, Hwanjo and Kim, Sungchul. "SVM Tutorial: Classification, Regression and
Ranking." Handbook of Natural Computing. Springer Berlin Heidelberg, 2012.

RVMs are similar to 1-norm SVMs, but can be trained more efficiently.

Installation
------------

`pip install pyrvm`

Examples
--------

```python
>>> from math import cos, sin, pi
>>> import numpy as np
>>> from pyrvm import RVM
>>> # Create points that spiral along the z axis, with higher rankings at lower
>>> # values of z.
>>> n_points = 50
>>> points = [[cos(2*pi*5*t), sin(2*pi*5*t), t] for t in np.linspace(0, 1, n_points)]
>>> X = np.array(points)
>>> # This algorithm is sensitive to constant shifts. To see this, uncomment
>>> # this mean subtraction; we'll see the algorithm fail.
>>> X = X - X.mean(0)
>>> y = range(n_points)
>>> # Train a linear RVM using half of the data. We keep the slack penalty C high
>>> # here because we know that the points can be ranked linearly with no errors.
>>> ranker = RVM(C=100.0)
>>> ranker.fit(X[0::2, :], y[0::2])
Out<rvm.RVM at 0x116f2d0d0>
>>> # Since we used a linear kernel, we can determine the weight vector in the
>>> # original space that determines ranking. It should be in the direction
>>> # of -z.
>>> print sum(ranker._alpha[ranker._alpha != 0, np.newaxis] * ranker._rank_vectors, 0)
[  1.88497489e-05   1.58543033e-05  -2.45000255e+01]
>>> # Now let's see how we do on the other half of the data.
>>> print ranker.predict(X[1::2, :])
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
```

Documentation
-------------

The API reference can be found [here](doc/_build/html/pyrvm.html).

Dependencies
------------

`pyrvm` depends on `numpy`, `sklearn`, `pulp`, and (assuming default use)
`GLPK`.

