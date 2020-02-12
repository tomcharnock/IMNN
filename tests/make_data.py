import numpy as np

input_shape = (10,)
n_summaries = 2
n_params = 2
n_s = 1000
n_d_large = 1000
n_d_small = 100

fiducial = np.array([0., 1.])
delta = np.array([0.1, 0.1])


np.savez("data/large_details.npz",
         input_shape=input_shape,
         n_params=n_params,
         n_summaries=n_summaries,
         n_s=n_s,
         n_d=n_d_large,
         fiducial=fiducial,
         delta=(2. * delta))
np.savez("data/small_details.npz",
         input_shape=input_shape,
         n_params=n_params,
         n_summaries=n_summaries,
         n_s=n_s,
         n_d=n_d_small,
         fiducial=fiducial,
         delta=(2. * delta))

a_0 = np.random.normal(fiducial[0], np.sqrt(fiducial[1]), (n_s,) + input_shape)
a_1 = np.random.normal(fiducial[0], np.sqrt(fiducial[1]), (n_s,) + input_shape)

np.save("data/fiducial_data.npz", a_0)
np.save("data/fiducial_validation_data.npz", a_1)

seed_0 = np.random.randint(1e6)
seed_1 = np.random.randint(1e6)

np.random.seed(seed_0)
b_0 = np.random.normal(fiducial[0] - delta[0], np.sqrt(fiducial[1]), (n_d_large,) + input_shape)
np.random.seed(seed_1)
b_1 = np.random.normal(fiducial[0] - delta[0], np.sqrt(fiducial[1]), (n_d_large,) + input_shape)
np.random.seed(seed_0)
c_0 = np.random.normal(fiducial[0] + delta[0], np.sqrt(fiducial[1]), (n_d_large,) + input_shape)
np.random.seed(seed_1)
c_1 = np.random.normal(fiducial[0] + delta[0], np.sqrt(fiducial[1]), (n_d_large,) + input_shape)
np.random.seed(seed_0)
d_0 = np.random.normal(fiducial[0], np.sqrt(fiducial[1] - delta[1]), (n_d_large,) + input_shape)
np.random.seed(seed_1)
d_1 = np.random.normal(fiducial[0], np.sqrt(fiducial[1] - delta[1]), (n_d_large,) + input_shape)
np.random.seed(seed_0)
e_0 = np.random.normal(fiducial[0], np.sqrt(fiducial[1] + delta[1]), (n_d_large,) + input_shape)
np.random.seed(seed_1)
e_1 = np.random.normal(fiducial[0], np.sqrt(fiducial[1] + delta[1]), (n_d_large,) + input_shape)

f_0 = np.stack((np.stack((b_0, c_0)), np.stack((d_0, e_0)))).transpose(2, 1, 0, 3)
f_1 = np.stack((np.stack((b_1, c_1)), np.stack((d_1, e_1)))).transpose(2, 1, 0, 3)
               
np.save("data/large_derivative_data.npy", f_0)
np.save("data/small_derivative_data.npy", f_0[:n_d_small])
np.save("data/large_derivative_validation_data.npy", f_1)
np.save("data/small_derivative_validation_data.npy", f_1[:n_d_small])