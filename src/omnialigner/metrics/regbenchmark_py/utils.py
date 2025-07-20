import numpy as np


def kpt_pairs_to_fiducialpoints(kpt_pairs, n_kpt_used=-1):
    fiducialpoints = []
    if len(kpt_pairs[0]) == 1:
        for pair in kpt_pairs:
            kpt_ref = pair[0]
            if len(kpt_ref) == 0:
                continue

            kpt_ref = np.array(kpt_ref)[0:n_kpt_used, :]            
            Y1 = kpt_ref[:, 1]
            X1 = kpt_ref[:, 0]
            combined = np.column_stack((Y1, X1))
            fiducialpoints.append(combined)

        return fiducialpoints


    for pair in kpt_pairs:
        kpt_ref, kpt_target = pair
        assert len(kpt_ref) == len(kpt_target), "different number of points in kpt_ref and kpt_target."
        if len(kpt_ref) == 0:
            continue

        kpt_ref = np.array(kpt_ref)[0:n_kpt_used, :]
        kpt_target = np.array(kpt_target)[0:n_kpt_used, :]

        Y1 = kpt_ref[:, 1]
        X1 = kpt_ref[:, 0]
        Y2 = kpt_target[:, 1]
        X2 = kpt_target[:, 0]

        combined = np.column_stack((Y1, X1, Y2, X2))
        fiducialpoints.append(combined)

    return fiducialpoints
