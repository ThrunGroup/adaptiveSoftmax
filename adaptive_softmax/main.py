import numpy as np

from sftm import SFTM
from tests.test_utils import construct_high_variance_example

A, x = construct_high_variance_example(10000, 10000)
sftm = SFTM(A)

best_arm, prob, _ = sftm.adaptive_softmax(x)
print(best_arm, prob)
print(np.sum(sftm.bandits.it) / (sftm.n * sftm.max_pulls))
print(np.max(sftm.bandits.it))

sm = sftm.softmax(x)
print(np.argmax(sm))
print(np.max(sm))
