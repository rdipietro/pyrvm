import itertools
import numpy as np
from sklearn.metrics import pairwise
import pulp

class RVM(object):
    """Ranking Vector Machine
    
    Learn and predict orderings of vectors using large-margin criteria.
    
    This is an implementation of the ranking-vector-machine algorithm from
    Yu, Hwanjo and Kim, Sungchul. "SVM Tutorial: Classification, Regression and
    Ranking." Handbook of Natural Computing. Springer Berlin Heidelberg, 2012.
    
    Parameters
    ----------
    C : float, optional (default=1.0)
        Slack parameter.
    
    kernel : string, optional (default='linear')
        Specifies the kernel type to be used in the algorithm. 'linear',
        'rbf', 'chi2', or a callable are common options. See
        `sklearn.metrics.pairwise.pairwise_kernels`.
        
    solver : pulp.solvers.LpSolver, optional (default=pulp.solvers.GLPK(msg=0))
        The solver used for the linear program. See `pulp.solvers`.
        
    verbose : boolean, optional (default=False)
    
    Examples
    --------
    >>> from math import cos, sin, pi
    >>> import numpy as np
    >>> from rvm import RVM
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
    
    """
    
    def __init__(self, C=1.0, kernel='linear', 
                 solver=pulp.solvers.GLPK(msg=0), verbose=False):
        
        self.C = C
        self.kernel = kernel
        self.solver = solver
        self.verbose = verbose
        self._linprog = None
        self._alpha = None
        self._rank_vectors = None
    
    def decision_function(self, X):
        """Scores related to the ordering of the samples X.
        
        Note that higher scores correspond to higher rankings. For example,
        for three ordered samples (say ranks 1, 2, 3) we would expect the
        corresponding scores to decrease (say 9.5, 6.2, 3.5).
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors.
            
        Returns
        -------
        scores : array-like, shape = [n_samples]
            The higher the score, the higher the rank. For example,
            if the x_1's rank is 1 and x_2's rank is 2, then
            x_1's score will be higher than x_2's score.        
            
        """
        
        if self._rank_vectors is None:
            raise Exception('Attempted to predict before fitting model')

        alpha = self._alpha
        gram_matrix = pairwise.pairwise_kernels(self._rank_vectors, X, metric=self.kernel)
        scores = np.sum(alpha[alpha != 0, np.newaxis] * gram_matrix, 0)
        return scores
    
    def fit(self, X, y):
        """Fit the RVM model to the given training data.
        
        Pairs of unequal ordering are used for training. For example, if
        rank(x_1) = rank(x_2) = 1 and rank(x_3) = 2, then the pairs
        (x_1, x_3) and (x_2, x_3) will be used to train the model.
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors.
        
        y : array-like, shape = [n_samples]
            Training ordering with one rank per sample.
        
        Returns
        -------
        self : object
            Returns self.
            
        """
        
        gram_matrix = pairwise.pairwise_kernels(X, metric=self.kernel)
        
        index_pairs = ranked_index_pairs(y)
        
        n_points = X.shape[0]
        n_pairs = len(index_pairs)
        
        if self.verbose:
            print 'rvm: Setting up the linear program and its objective..'
        
        # Set up the linear program and specify the variables, their domains,
        # and the objective. (We only specify lower bounds because the upper
        # bound is infinity by default.)
        self._linprog = pulp.LpProblem('RVM', pulp.LpMinimize)
        alpha = [pulp.LpVariable('alpha%d' % i, 0) for i in xrange(n_points)]
        alpha_coefs = [1.0]*n_points
        xi = [pulp.LpVariable('xi%d' % i, 0) for i in xrange(n_pairs)]
        xi_coefs = [self.C]*n_pairs
        self._linprog += pulp.lpDot(alpha_coefs + xi_coefs, alpha + xi)
        
        if self.verbose:
            print 'rvm: Adding constraints to the linear program..'
        
        # Add the constraints, one for each pair formed above.
        for l, (u, v) in enumerate(index_pairs):
            variables = alpha + [xi[l]]
            coefs = list(gram_matrix[:, u] - gram_matrix[:, v]) + [1.0]
            # LpAffineExpression isn't as clear as using lpDot, but for some
            # reason it's faster.
            self._linprog += pulp.LpAffineExpression(zip(variables, coefs)) >= 1
        
        if self.verbose:
            print "rvm: Solving linear program.."
        
        status = self._linprog.solve(self.solver)
        if status != 1:
            raise Exception('rvm: Unknown error occurred while trying to solve linear program')
        
        self._alpha = np.array([pulp.value(a) for a in alpha])
        self._rank_vectors = X[self._alpha != 0, :]
        
        if self.verbose:
            print 'rvm: Fit complete.'
        
        return self
    
    def predict(self, X):
        """Compute the ordering of the samples X.
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        
        Returns
        -------
        y : array-like, shape = [n_samples]
        
        """
        
        scores = self.decision_function(X)
        return np.argsort(-scores)
    
    def score(self, X, y):
        """Performance metric based on Kendall's tau metric.
        
        This is (the number of true pairs we predicted correctly) / (the total
        number of true pairs).
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        
        y : array-like, shape = [n_samples]
            True ordering for X.
            
        Returns
        -------
        score : float
        
        """
        
        y_pred = self.predict(X)
        return kendall_tau_metric(y, y_pred)
    
def ranked_index_pairs(y):
    """Return all index pairs that satisfy y[i] < y[j].
    
    Parameters
    ----------
    y : array-like, shape = [n_samples]
        An ordering
    
    Returns
    -------
    index_pairs : list
        List of tuples, each being one index pair (i, j)
        
    """
    
    index_pairs = []
    for ranking in sorted(np.unique(y)):
        index_pairs += itertools.product(np.flatnonzero(y == ranking), np.flatnonzero(ranking < y))
    
    return index_pairs

def kendall_tau_metric(y_true, y_pred):
    """Performance metric based on Kendall's tau metric.
    
    This is (the number of true pairs we predicted correctly) / (the total
    number of true pairs).
    
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        True rankings.
    y_pred : array-like, shape = [n_samples]
        Predicted rankings.
    
    Returns
    -------
    score : float
    
    """
    
    true_pairs = ranked_index_pairs(y_true)
    predicted_pairs = ranked_index_pairs(y_pred)        
    incorrect_pairs = list(set(true_pairs) - set(predicted_pairs))
    
    return 1 - float(len(incorrect_pairs))/len(true_pairs)
    
    
    
    
    
    
    
    
    
    
