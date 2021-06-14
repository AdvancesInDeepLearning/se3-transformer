import numpy as np
import matplotlib.pyplot as plt
import time

from numpy.lib.function_base import select


class SpringSim(object):
    def __init__(self, n_balls=5, box_size=5., loc_std=.5, vel_norm=.5,
                 interaction_strength=.1, noise_var=0.,
                 dim=2):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        self.dim = dim

        self._spring_types = np.array([0., 0.5, 1.])
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

    def _energy(self, loc, vel, edges):
        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.interaction_strength * edges[
                            i, j] * (dist ** 2) / 2
            return U + K, U, K

    def _clamp(self, loc, vel):
        '''
        :param loc: dim x N location at one time stamp
        :param vel: dim x N velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        clamp = np.zeros([loc.shape[1]])
        if self.box_size > 1e-5:
            assert (np.all(loc < self.box_size * 3))
            assert (np.all(loc > -self.box_size * 3))

            over = loc > self.box_size
            loc[over] = 2 * self.box_size - loc[over]
            assert (np.all(loc <= self.box_size))

            # assert(np.all(vel[over]>0))
            vel[over] = -np.abs(vel[over])

            under = loc < -self.box_size
            loc[under] = -2 * self.box_size - loc[under]
            # assert (np.all(vel[under] < 0))
            assert (np.all(loc >= -self.box_size))
            vel[under] = np.abs(vel[under])

            clamp[over[0, :]] = 1
            clamp[under[0, :]] = 1

        return loc, vel, clamp

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matrix
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def sample_trajectory(self, T=10000, sample_freq=10,
                          spring_prob=[1. / 2, 0, 1. / 2]):
        n = self.n_balls
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0
        # Sample edges
        edges = np.random.choice(self._spring_types,
                                 size=(self.n_balls, self.n_balls),
                                 p=spring_prob)
        edges = np.tril(edges) + np.tril(edges, -1).T
        np.fill_diagonal(edges, 0)
        # Initialize location and velocity
        loc = np.zeros((T_save, self.dim, n))
        vel = np.zeros((T_save, self.dim, n))
        clamp = np.zeros((T_save, n))
        loc_next = np.random.randn(self.dim, n) * self.loc_std
        vel_next = np.random.randn(self.dim, n)
        clamp_next = np.zeros([n,])
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :], clamp[0, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            forces_size = - self.interaction_strength * edges
            np.fill_diagonal(forces_size,
                             0)  # self forces are zero (fixes division by zero)
            if self.dim == 2:
                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n, n)))).sum(
                    axis=-1)
            elif self.dim == 3:
                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[2, :],
                                           loc_next[2, :]).reshape(1, n, n)))).sum(
                    axis=-1)

            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                loc_next, vel_next, clamp_this_i = self._clamp(loc_next, vel_next)

                # Update clamping.
                clamp_next[clamp_this_i == 1] = 1

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :], clamp[counter, :] \
                        = loc_next, vel_next, clamp_next
                    clamp_next *= 0
                    counter += 1

                forces_size = - self.interaction_strength * edges
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)
                if self.dim == 2:
                    F = (forces_size.reshape(1, n, n) *
                         np.concatenate((
                             np.subtract.outer(loc_next[0, :],
                                               loc_next[0, :]).reshape(1, n, n),
                             np.subtract.outer(loc_next[1, :],
                                               loc_next[1, :]).reshape(1, n,
                                                                       n)))).sum(
                        axis=-1)
                elif self.dim == 3:
                    F = (forces_size.reshape(1, n, n) *
                         np.concatenate((
                             np.subtract.outer(loc_next[0, :],
                                               loc_next[0, :]).reshape(1, n, n),
                             np.subtract.outer(loc_next[1, :],
                                               loc_next[1, :]).reshape(1, n, n),
                             np.subtract.outer(loc_next[2, :],
                                               loc_next[2, :]).reshape(1, n,
                                                                       n)))).sum(
                        axis=-1)

                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
            # Add noise to observations
            loc += np.random.randn(T_save, self.dim, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, self.dim, self.n_balls) * self.noise_var
            return loc, vel, edges, clamp


class ChargedParticlesSim(object):
    def __init__(self, n_balls=5, box_size=5., loc_std=1., vel_norm=0.5,
                 interaction_strength=1., noise_var=0.,
                 dim = 2):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        self.dim = dim

        self._charge_types = np.array([-1., 0., 1.])
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matrix
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:] - B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def _energy(self, loc, vel, edges):

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.interaction_strength * edges[
                            i, j] / dist
            return U + K, U, K

    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        clamp = np.zeros([loc.shape[1]])
        if self.box_size > 1e-6:
            assert (np.all(loc < self.box_size * 3))
            assert (np.all(loc > -self.box_size * 3))

            over = loc > self.box_size
            loc[over] = 2 * self.box_size - loc[over]
            assert (np.all(loc <= self.box_size))

            # assert(np.all(vel[over]>0))
            vel[over] = -np.abs(vel[over])

            under = loc < -self.box_size
            loc[under] = -2 * self.box_size - loc[under]
            # assert (np.all(vel[under] < 0))
            assert (np.all(loc >= -self.box_size))
            vel[under] = np.abs(vel[under])

            clamp[over[0, :]] = 1
            clamp[under[0, :]] = 1

        return loc, vel, clamp

    def sample_trajectory(self, T=10000, sample_freq=10,
                          charge_prob=[1. / 2, 0, 1. / 2]):
        n = self.n_balls

        # T_save is number of (saved) measurements/observations
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        counter = 0  # count number of measurements

        # create matrix of 1s with 0s on diag
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)

        # Sample charges and get edges
        charges = np.random.choice(self._charge_types, size=(self.n_balls, 1),
                                   p=charge_prob)
        edges = charges.dot(charges.transpose())

        # Initialize location and velocity
        loc = np.zeros((T_save, self.dim, n))
        vel = np.zeros((T_save, self.dim, n))
        clamp = np.zeros((T_save, n))
        loc_next = np.random.randn(self.dim, n) * self.loc_std
        vel_next = np.random.randn(self.dim, n)
        clamp_next = np.zeros([n,])
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :], clamp[0, :] = self._clamp(loc_next, vel_next)

        # count number of times forces were capped
        count_maxedout = 0

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):
            # half step leapfrog
            print(loc_next.shape)
            l2_dist_power3 = np.power(
                self._l2(loc_next.transpose(), loc_next.transpose()), 3. / 2.)

            # size of forces up to a 1/|r| factor
            # since I later multiply by an unnormalized r vector
            forces_size = self.interaction_strength * edges / l2_dist_power3
            np.fill_diagonal(forces_size,
                             0)  # self forces are zero (fixes division by zero)
            assert (np.abs(forces_size[diag_mask]).min() > 1e-10)
            if self.dim == 2:
                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n, n)))).sum(
                    axis=-1)
            elif self.dim == 3:
                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[2, :],
                                           loc_next[2, :]).reshape(1, n, n)))).sum(
                    axis=-1)
            # cap maximum force strength
            count_maxedout += np.sum(F > self._max_F) + np.sum(F < -self._max_F)
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                loc_next, vel_next, clamp_this_i = self._clamp(loc_next, vel_next)

                # Update clamping.
                clamp_next[clamp_this_i == 1] = 1

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :], clamp[counter, :] \
                        = loc_next, vel_next, clamp_next
                    clamp_next *= 0
                    counter += 1

                l2_dist_power3 = np.power(
                    self._l2(loc_next.transpose(), loc_next.transpose()),
                    3. / 2.)
                forces_size = self.interaction_strength * edges / l2_dist_power3
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)
                if self.dim == 2:
                    F = (forces_size.reshape(1, n, n) *
                         np.concatenate((
                             np.subtract.outer(loc_next[0, :],
                                               loc_next[0, :]).reshape(1, n, n),
                             np.subtract.outer(loc_next[1, :],
                                               loc_next[1, :]).reshape(1, n,
                                                                       n)))).sum(
                        axis=-1)
                elif self.dim == 3:
                    F = (forces_size.reshape(1, n, n) *
                         np.concatenate((
                             np.subtract.outer(loc_next[0, :],
                                               loc_next[0, :]).reshape(1, n, n),
                             np.subtract.outer(loc_next[1, :],
                                               loc_next[1, :]).reshape(1, n, n),
                             np.subtract.outer(loc_next[2, :],
                                               loc_next[2, :]).reshape(1, n,
                                                                       n)))).sum(
                        axis=-1)

                # cap maximum force strength
                count_maxedout += np.sum(F > self._max_F) + np.sum(
                    F < -self._max_F)
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
            # Add noise to observations
            loc += np.random.randn(T_save, self.dim, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, self.dim, self.n_balls) * self.noise_var
            print(count_maxedout)
            return loc, vel, edges, charges, clamp



class ArgonSim(object):
    def __init__(self, n_balls=28, box_size=5., loc_std=1., vel_norm=0.5,
                 interaction_strength=1., noise_var=0.,
                 dim = 3, temperature=1.0, density=0.8):
        # THREE CASES THAT ARE EASY to use:
        # DENSITY 1.2, TEMP 0.5: SOLID
        # DENSITY 0.8, TEMP 1.0: LIQUID
        # DENSITY 0.3, TEMP 3.0: GAS
        # self.n_balls = n_balls
        # self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength # 
        self.noise_var = noise_var # noise that can be added to the observations
        # self.box_size = 100
        self.dim = dim
        self.number_of_cells = 2
        self.density = 0.8
        self.init_pos = InitializeFCC(self.number_of_cells, self.density)
        self.box_size = self.init_pos.get_boxsize()
        self.n_balls = self.init_pos.get_n_particles()

        self.sigma = 1.0 # natural constant 
        self.kb = 1.0 # natural constant
        self.epsilon_kb = 1.-0 # another natural constant
        self.eps = 1e-6 # smoothing factor to avoid NaNs.
        self._charge_types = np.array([1])
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T
        self.verbose = False

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matrix
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:] - B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def _energy(self, loc, vel, edges):

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.interaction_strength * edges[
                            i, j] / dist
            return U + K, U, K

    def _difference_matrix(self, x0, x1):
        """Get the closest neighbours given a list of translations
        Args:
            x0 (np.array): coordinate of particle shape(n_dimensions)
            x1 (np.array: coordinate of neighbouring particle shape(n_dimensions)
        Returns:
            (np.array): coordinates of the closest neighbour in periodic boundary conditions
        """
        delta = []
        # For each dimension, compute the difference and then combine all dimensions.
        # dim:dim+1 is used in order to maintain the dimensionality of the array
        # ((x0 - x1 + 0.5 * L) % L) - 0.5 L is the closest image trick
        for dim in range(self.dim):
            delta.append(
                (
                    (x0[dim : dim + 1, :] - x1[dim : dim + 1, :].T + 0.5 * self.box_size)
                    % self.box_size
                )
                - 0.5 * self.box_size
            )
        if self.verbose: print("delta", np.stack(delta, axis=0).shape)
        return np.stack(delta, axis=0)

    def get_gradient(self, r_vec):
        """Compute the gradient of the Lennard Jones Potential
        Args:
            r_vec (np.array): A difference vector for which to get the gradient
        Returns:
            (np.array): The gradient of the lennard jones potential
        """
        # Compute the distances (a scalar), add a value to not get overflow errors
        r = np.expand_dims(np.linalg.norm(r_vec, axis=0), 0) + self.eps
        grad_lennard_jones = (
            4
            * self.epsilon_kb
            * (-12 * (r / self.sigma) ** (-13) + 6 * (r / self.sigma) ** (-7))
            * r_vec
            / r
        )
        if self.verbose: print("grad_lennard_jones", grad_lennard_jones.shape)
        return grad_lennard_jones

    def get_force(self, state):
        """Compute the force for some index based on the state
        Args:
            index (int): The timestep index for which to compute the forces
            time (int): the actual time, for solving time dependent forces (i.e. a changing potential.)
            state (np.array): The state vector
        Returns:
            np.array: The state array.
        """
        # Get the differences in positions R = (x_j - x_i)
        differences = self._difference_matrix(
            state, state
        )
        # Compute the lennard jones gradient and apply it to each particle, this gives the acceleration.
        if self.verbose: print("differences", differences.shape)
        if self.verbose: print("forces", np.sum(self.get_gradient(differences), axis=-1))
        return np.sum(self.get_gradient(differences), axis=-1)


    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        clamp = np.zeros([loc.shape[1]])
        # if self.box_size > 1e-6:
        #     assert (np.all(loc < self.box_size * 3))
        #     assert (np.all(loc > -self.box_size * 3))

        #     over = loc > self.box_size
        #     loc[over] = 2 * self.box_size - loc[over]
        #     assert (np.all(loc <= self.box_size))

        #     # assert(np.all(vel[over]>0))
        #     vel[over] = -np.abs(vel[over])

        #     under = loc < -self.box_size
        #     loc[under] = -2 * self.box_size - loc[under]
        #     # assert (np.all(vel[under] < 0))
        #     assert (np.all(loc >= -self.box_size))
        #     vel[under] = np.abs(vel[under])

        #     clamp[over[0, :]] = 1
        #     clamp[under[0, :]] = 1

        return loc, vel, clamp

    def sample_trajectory(self, T=10000, sample_freq=10,
                          charge_prob=[1.]):
        n = self.n_balls

        # T_save is number of (saved) measurements/observations
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        counter = 0  # count number of measurements

        # create matrix of 1s with 0s on diag
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)

        # Sample charges and get edges
        charges = np.ones((self.n_balls, 1))
        edges = charges.dot(charges.transpose())

        # Initialize location and velocity
        loc = np.zeros((T_save, self.dim, n))
        vel = np.zeros((T_save, self.dim, n))
        clamp = np.zeros((T_save, n))
        # loc_next = np.random.randn(self.dim, n) * self.loc_std # implement FCC lattice here
        loc_next = self.init_pos()
        vel_next = np.random.randn(self.dim, n) # implement boltzmann velocity profile here ? Maybe not needed if we use Rescaling.
        clamp_next = np.zeros([n,])
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        # print("before clamp: ", loc_next[0])
        loc[0, :, :], vel[0, :, :], clamp[0, :] = self._clamp(loc_next, vel_next)
        # count number of times forces were capped
        count_maxedout = 0
        # disables division by zero warning, since I fix it with fill_diagonal
        # with np.errstate(divide='ignore'):
        if True:
            # half step leapfrog
            #
            ## FORCE COMPUTATION:
            F = self.get_force(loc_next)
            # l2_dist_power3 = np.power(
            #     self._l2(loc_next.transpose(), loc_next.transpose()), 3. / 2.)

            # # size of forces up to a 1/|r| factor
            # # since I later multiply by an unnormalized r vector
            # forces_size = self.interaction_strength * edges / l2_dist_power3
            # np.fill_diagonal(forces_size,
            #                  0)  # self forces are zero (fixes division by zero)
            # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)
            # if self.dim == 2:
            #     F = (forces_size.reshape(1, n, n) *
            #          np.concatenate((
            #              np.subtract.outer(loc_next[0, :],
            #                                loc_next[0, :]).reshape(1, n, n),
            #              np.subtract.outer(loc_next[1, :],
            #                                loc_next[1, :]).reshape(1, n, n)))).sum(
            #         axis=-1)
            # elif self.dim == 3:
            #     F = (forces_size.reshape(1, n, n) *
            #          np.concatenate((
            #              np.subtract.outer(loc_next[0, :],
            #                                loc_next[0, :]).reshape(1, n, n),
            #              np.subtract.outer(loc_next[1, :],
            #                                loc_next[1, :]).reshape(1, n, n),
            #              np.subtract.outer(loc_next[2, :],
            #                                loc_next[2, :]).reshape(1, n, n)))).sum(
            #         axis=-1)
            # # cap maximum force strength
            # count_maxedout += np.sum(F > self._max_F) + np.sum(F < -self._max_F)
            # F[F > self._max_F] = self._max_F
            # F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                loc_next, vel_next, clamp_this_i = self._clamp(loc_next, vel_next)

                # # Update clamping.
                # clamp_next[clamp_this_i == 1] = 1

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :], clamp[counter, :] \
                        = loc_next, vel_next, clamp_next
                    clamp_next *= 0
                    # plt.figure()
                    # plt.scatter(loc[counter, 0, :], loc[counter, 1, :])
                    # plt.show()
                    # plt.close()
                    counter += 1
                ## FORCE COMPUTATION
                F = self.get_force(loc_next)
                # l2_dist = self._l2(loc_next.transpose(), loc_next.transpose())
                # forces_size = (-12.0 * edges * l2_dist**-13 + 6.0* edges * l2_dist**-7)
                # np.fill_diagonal(forces_size, 0)
                # # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)
                # if self.dim == 2:
                #     F = (forces_size.reshape(1, n, n) *
                #          np.concatenate((
                #              np.subtract.outer(loc_next[0, :],
                #                                loc_next[0, :]).reshape(1, n, n),
                #              np.subtract.outer(loc_next[1, :],
                #                                loc_next[1, :]).reshape(1, n,
                #                                                        n)))).sum(
                #         axis=-1)
                # elif self.dim == 3:
                #     F = (forces_size.reshape(1, n, n) *
                #          np.concatenate((
                #              np.subtract.outer(loc_next[0, :],
                #                                loc_next[0, :]).reshape(1, n, n),
                #              np.subtract.outer(loc_next[1, :],
                #                                loc_next[1, :]).reshape(1, n, n),
                #              np.subtract.outer(loc_next[2, :],
                #                                loc_next[2, :]).reshape(1, n,
                #                                                        n)))).sum(
                #         axis=-1)

                # cap maximum force strength
                # count_maxedout += np.sum(F > self._max_F) + np.sum(
                #     F < -self._max_F)
                # F[F > self._max_F] = self._max_F
                # F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F                    
            # Add noise to observations
            # loc += np.random.randn(T_save, self.dim, self.n_balls) * self.noise_var
            # vel += np.random.randn(T_save, self.dim, self.n_balls) * self.noise_var
            # print(count_maxedout)
            return loc, vel, edges, charges, clamp


if __name__ == '__main__':
    sim = SpringSim()
    # sim = ChargedParticlesSim()

    t = time.time()
    loc, vel, edges, clamp = sim.sample_trajectory(T=5000, sample_freq=100)

    print(edges)
    print("Simulation time: {}".format(time.time() - t))
    vel_norm = np.sqrt((vel ** 2).sum(axis=1))
    plt.figure()
    axes = plt.gca()
    axes.set_xlim([-5., 5.])
    axes.set_ylim([-5., 5.])
    for i in range(loc.shape[-1]):
        plt.plot(loc[:, 0, i], loc[:, 1, i])
        plt.plot(loc[0, 0, i], loc[0, 1, i], 'd')
    plt.figure()
    energies = [sim._energy(loc[i, :, :], vel[i, :, :], edges) for i in
                range(loc.shape[0])]
    plt.plot(energies)
    plt.show()


class InitializeFCC():
    def __init__(self, number_of_cells, density, noise_level=0.1):
        """Create a fcc lattice for some number of unit cells
        An fcc lattice with period 1 in periodic boundary conditions has 4 points:
        # fcc (0,0,0), (0.5, 0.5, 0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)
        Args:
            number_of_cells (int): The number of unit cells to use per (x,y,z), so total number of cells is number_of_cells**3
            density (float): The density of the system, determines the box size
        """
        self.number_of_cells = number_of_cells
        self.density = density
        self.n_unit_cell = 4
        self.name = "fcc"
        self.n_particles = self.n_unit_cell * self.number_of_cells ** 3
        self.boxsize = np.cbrt(self.n_particles / self.density)
        print("The size of the box is: {}".format(self.boxsize))
        self.noise_level = noise_level

    def get_n_particles(self):
        return self.n_particles
    
    def get_boxsize(self):
        return self.boxsize

    def __call__(self):
        """Generate a Face Centered Cubic lattice.

        Returns:
            (np.array, int, float): The positions, the number of particles and the edge size of the box
        """
          # cubic root of N/density
        init_pos = []
        fcc_offset = 0.5 * self.boxsize / self.number_of_cells
        for x in np.linspace(0, self.boxsize, self.number_of_cells, endpoint=False):
            for y in np.linspace(0, self.boxsize, self.number_of_cells, endpoint=False):
                for z in np.linspace(0, self.boxsize, self.number_of_cells, endpoint=False):
                    init_pos.append([x, y, z])
                    init_pos.append([x + fcc_offset, y + fcc_offset, z])
                    init_pos.append([x + fcc_offset, y, z + fcc_offset])
                    init_pos.append([x, y + fcc_offset, z + fcc_offset])
        new_arr = np.array(init_pos).T
        # add a little bit of noise to the initial conditions to ensure no two samples are the same
        new_arr += np.random.uniform(-self.noise_level, self.noise_level, size=(new_arr.shape))
        return new_arr