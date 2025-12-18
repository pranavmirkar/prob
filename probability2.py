def predict_motion_probs(Re, I_star, angle, param_matrix):
    """
    Predict motion probabilities using the trained multinomial logistic model.

    This uses the same feature scaling and structure as the training phase,
    so the outputs are consistent with the fitted regime boundaries.

    Parameters
    ----------
    Re : float
        Reynolds number
    I_star : float
        Dimensionless moment of inertia
    angle : float
        Launch angle in degrees
    param_matrix : ndarray
        3x4 array of fitted parameters:
        [a * log10(Re) + b * log10(I*) + d * (angle/90) + c]

    Returns
    -------
    dict
        Probabilities for Flutter, Chaotic, and Tumble
    """

    # Log-transform the variables that were fit in log space
    log_Re = np.log10(Re)
    log_I = np.log10(I_star)

    # Angle is scaled exactly the same way as during training
    angle_scaled = angle / 90.0

    # Assemble logits for each motion class
    logits = []
    for a, b, angle_coef, bias in param_matrix:
        z = (
            a * log_Re
            + b * log_I
            + angle_coef * angle_scaled
            + bias
        )
        logits.append(z)

    logits = np.array(logits)

    # Stable softmax (prevents overflow if logits get large)
    logits -= np.max(logits)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)

    motion_types = ["Flutter", "Chaotic", "Tumble"]

    # Rounded for readability, but still returned as floats
    return {
        motion: round(float(prob), 4)
        for motion, prob in zip(motion_types, probs)
    }


# ------------------------------------------------------------
# Example prediction
# ------------------------------------------------------------

new_Re = 3303
new_I_star = 0.015
new_angle = 0

probs = predict_motion_probs(
    new_Re,
    new_I_star,
    new_angle,
    optimized_params
)

print("Predicted motion probabilities:")
for motion, prob in probs.items():
    print(f"{motion}: {prob}")
