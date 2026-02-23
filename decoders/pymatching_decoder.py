import numpy as np
from pymatching import Matching

def num_decoding_failures_vectorised(H, logicals, error_probability, num_shots):
    matching = Matching.from_check_matrix(
        H,
        weights=np.log((1 - error_probability) / error_probability),
        faults_matrix=logicals,
    )
    noise = (np.random.random((num_shots, H.shape[1])) < error_probability).astype(
        np.uint8
    )
    shots = (noise @ H.T) % 2
    actual_observables = (noise @ logicals.T) % 2
    predicted_observables = matching.decode_batch(shots)
    num_errors = np.sum(np.any(predicted_observables != actual_observables, axis=1))
    return num_errors