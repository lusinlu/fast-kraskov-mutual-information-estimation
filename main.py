from kmu import compute_kmu
import numpy as np



if __name__ == "__main__":
    batch_size = 100
    continious_tensor = np.repeat(np.random.rand(int(batch_size / 10), 10, 64, 64).astype(np.float32), 10, axis=0)
    discrete_variable = np.random.randint(0, 5, batch_size)

    mu_full_tensor = compute_kmu(x=continious_tensor, y=discrete_variable, per_filter=False)
    # should output zero(or near values), as tensors completely random
    print('mutual information for full tensor - ', str(mu_full_tensor))

    mu_per_filter_tensor = compute_kmu(x=continious_tensor, y=discrete_variable, per_filter=True, avarage=False)
    # should output zero(or near values), as tensors completely random
    print('mutual information for of each filter of the tensor - ', str(mu_per_filter_tensor))