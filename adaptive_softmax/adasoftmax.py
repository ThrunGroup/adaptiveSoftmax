import torch
torch.manual_seed(0)
import numpy as np
import math

from time import time

from numba import jit
from numba.experimental import jitclass

@jitclass
class AdaSoftmax():
    def __init__(self,):
        pass

    def compute_mip_batch_topk_ver2_warm(self, atoms, query, sigma, delta, batch_size=16, k=1, mu=None, budget_vec=None):
        """
        does same thing as previous, but instead of doing multiplication between single element of A and x,
        it sequentially slices 'batch_size' elements from left to right, and performs inner product to
        pull an arm.
        """

        dim = len(query)
        n_atoms = len(atoms)

        best_ind = np.empty(n_atoms, dtype=np.int64)
        found_indices_num = 0

        # case where no best-arm identification is needed
        if k == n_atoms:
            return np.arange(n_atoms), 0, mu, budget_vec

        d_used = budget_vec
        n_samples = 0  # instrumentation

        mu = mu
        max_index = np.argmax(mu)
        max_mu = mu[max_index]
        C = np.divide(sigma * np.sqrt(2 * np.log(4 * n_atoms * d_used ** 2 / delta)),
                      d_used + 1) if d_used is not None else np.zeros(n_atoms)

        solution_mask = np.ones(n_atoms, dtype="bool") & (
                    mu + C >= max_mu - C[max_index]) if mu is not None else np.ones(n_atoms, dtype="bool")
        solutions = np.nonzero(solution_mask)[0]
        # topk_indices = np.array([], dtype=np.int64)

        if len(solutions) <= k:
            # topk_indices = np.append(topk_indices, solutions)
            best_ind[found_indices_num: found_indices_num + len(solutions)] = solutions
            found_indices_num += len(solutions)
            solution_mask = np.logical_not(solution_mask)
            solutions = np.nonzero(solution_mask)[0]
            max_index = solutions[np.argmax(mu[solution_mask])]
            max_mu = mu[max_index]

        C = np.divide(sigma * np.sqrt(2 * np.log(4 * n_atoms * d_used ** 2 / delta)), d_used + 1)

        solution_mask_before = solution_mask

        brute_force_threshold = math.ceil(atoms.shape[0] * 0.05)

        while (len(solutions) > brute_force_threshold and found_indices_num < k and np.max(
                d_used) < dim - batch_size):  # TODO: computing max everytime may degrade performance

            tmp = np.empty(np.sum(solution_mask))

            for i, atom_index in enumerate(solutions):
                tmp[i] = atoms[atom_index, d_used[atom_index]: d_used[atom_index] + batch_size] @ query[
                                                                                                  d_used[atom_index]:
                                                                                                  d_used[
                                                                                                      atom_index] + batch_size]

            mu[solutions] = np.divide(np.multiply(d_used[solution_mask], mu[solutions]) + tmp * dim,
                                      d_used[solution_mask] + batch_size)
            n_samples += len(solutions) * batch_size

            C = np.divide(sigma * np.sqrt(2 * np.log(4 * n_atoms * d_used ** 2 / delta)),
                          d_used + 1)  # TODO: update confidence bound. This is when we're sampling one A_iJ * x_J at each round. Can and should be tighter than this -> divide with + batch size somehow?

            max_index = solutions[np.argmax(mu[solution_mask])]
            max_mu = mu[max_index]

            d_used[solutions] += batch_size

            solution_mask_before = solution_mask
            solution_mask = solution_mask & (mu + C >= max_mu - C[max_index])
            solutions = np.nonzero(solution_mask)[0]

            if len(solutions) <= k - found_indices_num:
                # topk_indices = np.append(topk_indices, solutions)
                best_ind[found_indices_num: found_indices_num + len(solutions)] = solutions
                found_indices_num += len(solutions)
                solution_mask = np.logical_xor(solution_mask_before,
                                               solution_mask)  # TODO: Does xor work even in the worst case scenario? or should we revive all candidates?
                solutions = np.nonzero(solution_mask)[0]
                max_index = solutions[np.argmax(mu[solution_mask])]
                max_mu = mu[max_index]

        # need to check if this is correct?
        if found_indices_num < k:
            # required_indices_num = k - found_indices_num
            # print("k - topk_len:", required_indices_num)

            mu_exact = np.multiply(d_used[solution_mask], mu[solution_mask])

            """
            def sample_exact(a, x, start_point):
              inner_product = a[start_point:].dot(x[start_point:])
              return inner_product

            sample_exact_mat = np.vectorize(pyfunc=sample_exact, signature="(n),(d),()->()")

            mu_exact = (mu_exact + dim * sample_exact_mat(atoms[solution_mask], query, d_used[solution_mask])) / dim
            """
            tmp = np.empty(np.sum(solution_mask))

            for i, atom_index in enumerate(solutions):
                tmp[i] = atoms[atom_index, d_used[atom_index]:] @ query[d_used[atom_index]:]

            mu_exact = (mu_exact + tmp) / dim

            # TODO: is there a way to avoid copy?
            mu_exact_search = mu_exact.copy()

            while found_indices_num < k:
                best_index = np.argmax(mu_exact_search)
                best_ind[found_indices_num] = best_index
                found_indices_num += 1
                mu_exact_search[best_index] = -np.inf

            # best_indices_tail_np = np.array(best_indices_tail)

            # best_ind = np.append(topk_indices, best_indices_tail_np)
            mu[solutions] = mu_exact

            n_samples += np.sum(dim - d_used[solution_mask])
            d_used[solutions] = dim

        return best_ind, n_samples, mu, d_used

    def estimate_softmax_normalization_warm(self, atoms, query, beta, epsilon, delta, sigma, bruteforce=False):
        n = atoms.shape[0]
        d = query.shape[0]
        used_samples = 0

        T0 = min(math.ceil(17 * beta ** 2 * sigma ** 2 * np.log(6 * n / delta)), d)

        mu_hat = (d / T0) * (atoms[:, :T0] @ query[:T0])
        C = (2 * sigma ** 2 * np.log(6 * n / delta) / T0) ** 0.5

        #Maybe this is better?
        #mu_hat -= np.min(mu_hat)

        mu_hat_aux = (mu_hat - np.max(mu_hat) - C) * beta

        # mu_hat_exp = np.exp((mu_hat - C) * beta, dtype=np.float64)
        mu_hat_exp_alpha = np.exp(mu_hat_aux)
        alpha = mu_hat_exp_alpha / np.sum(mu_hat_exp_alpha)

        mu_hat_exp_gamma = np.exp(mu_hat_aux / 2)
        gamma = mu_hat_exp_gamma / np.sum(mu_hat_exp_gamma)

        T = beta**2 * sigma**2 (
                17 * np.log((6 * n) / delta) * n
                + (32 * np.log((6 * n) / delta) * n) / epsilon
                + (16 * np.log(12 / delta)) / epsilon ** 2
        )

        n_samples = np.ceil(np.minimum((alpha + gamma) * T, d)).astype(np.int64)

        if bruteforce:
            n_samples = np.zeros(n, dtype=np.int64)
            n_samples += d

        mu_hat_refined_aux = np.empty(n)

        for i in range(n):
            mu_hat_refined_aux[i] = atoms[i, T0:T0 + n_samples[i]] @ query[T0:T0 + n_samples[i]]

        mu_hat_refined = np.divide(mu_hat * T0 + mu_hat_refined_aux * d, n_samples)

        #mu_hat_max_element = np.max(mu_hat_refined)

        mu_hat_refined -= np.max(mu_hat_refined)

        # mu_hat_refined_exp = np.exp(beta * mu_hat_refined, dtype=np.float64)
        mu_hat_refined_exp = np.exp(beta * mu_hat_refined)
        S_hat = np.sum(mu_hat_refined_exp)

        return S_hat, mu_hat_refined, n_samples

        #Potential hazard: max value of mu_hat for alpha construction and max value of mu hat for S_hat estimation is different, which may result in incorrect sampling or other problems?

    # adaSoftmax with warm start
    def ada_softmax(self, A, x, beta, epsilon, delta, sigma, k, bruteforce=False, return_estimates=False):

        S_hat, mu_hat, budget_vec = self.estimate_softmax_normalization_warm(A, x, beta, epsilon / 2, delta / 3, sigma,
                                                                        bruteforce=bruteforce)

        normalization_budget = np.sum(budget_vec)

        # print("normalization budget:", normalization_budget)

        best_index_hat, budget_mip, mu_hat, budget_vec = self.compute_mip_batch_topk_ver2_warm(A, x, sigma, delta / 3,
                                                                                          batch_size=16, k=k, mu=mu_hat,
                                                                                          budget_vec=budget_vec)

        best_index_hat = best_index_hat[:k]

        budget_mip = np.sum(budget_vec) - normalization_budget

        # print("mip budget:", budget_mip)

        n_arm_pull = min(
            math.ceil(8 * sigma ** 2 * beta ** 2 * np.log(6 / delta) / epsilon ** 2),
            x.shape[0]
        )

        if bruteforce:
            n_arm_pull = x.shape[0]

        # budget_spent_best_arm = budget_vec[best_index_hat]

        # A_subset = A[best_index_hat]
        #mu_best_hat = mu_hat[best_index_hat]

        mu_additional = np.empty(k)

        for i, arm_index in enumerate(best_index_hat):
            mu_additional[i] = A[arm_index, budget_vec[arm_index]: n_arm_pull] @ x[budget_vec[arm_index]: n_arm_pull]

        mu_hat[best_index_hat] += np.divide(x.shape[0] * mu_additional, np.maximum(1, n_arm_pull - budget_vec[best_index_hat]))

        budget_vec[best_index_hat] = np.maximum(budget_vec[best_index_hat], n_arm_pull)

        # y_best_hat = np.exp(beta * (mu_best_hat), dtype=np.float64)
        y_best_hat = np.exp(beta * (mu_hat))
        budget = np.sum(budget_vec)

        return best_index_hat, y_best_hat / S_hat, budget