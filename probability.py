import pandas as pd
import numpy as np
from scipy.optimize import minimize

# ------------------------------------------------------------
# Experimental dataset
# Only motion classes 1 (Flutter), 2 (Chaotic), 3 (Tumble)
# ------------------------------------------------------------

data = pd.DataFrame({
    'Re': [
        8914.3, 8914.3, 8914.3, 8914.3, 8914.3, 8914.3, 8914.3, 8914.3, 8914.3, 8914.3, 8914.3, 8914.3,
        8914.3, 8914.3, 8914.3, 8914.3, 8914.3, 8914.3,
        8763.3, 8763.3, 8763.3, 8763.3, 8763.3, 8763.3, 8763.3, 8763.3, 8763.3, 8763.3, 8763.3, 8763.3,
        8763.3, 8763.3, 8763.3, 8763.3, 8763.3, 8763.3,
        7479.5, 7479.5, 7479.5, 7479.5, 7479.5, 7479.5, 7479.5, 7479.5, 7479.5, 7479.5, 7479.5, 7479.5,
        7479.5, 7479.5, 7479.5, 7479.5, 7479.5, 7479.5,
        7881.5, 7881.5, 7881.5, 7881.5, 7881.5, 7881.5, 7881.5, 7881.5, 7881.5, 7881.5, 7881.5, 7881.5,
        7881.5, 7881.5, 7881.5, 7881.5, 7881.5, 7881.5,
        7061, 7061, 7061, 7061, 7061, 7061, 7061, 7061, 7061, 7061, 7061, 7061,
        7061, 7061, 7061, 7061, 7061, 7061,
        5640, 5640, 5640, 5640, 5640, 5640, 5640, 5640, 5640,
        5640, 5640, 5640, 5640, 5640, 5640, 5640, 5640, 5640
    ],

    'I_star': [
        0.0238, 0.0238, 0.0238, 0.0238, 0.0238, 0.0238, 0.0238, 0.0238, 0.0238, 0.0238, 0.0238, 0.0238,
        0.0238, 0.0238, 0.0238, 0.0238, 0.0238, 0.0238,
        0.027, 0.027, 0.027, 0.027, 0.027, 0.027, 0.027, 0.027, 0.027, 0.027, 0.027, 0.027,
        0.027, 0.027, 0.027, 0.027, 0.027, 0.027,
        0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018,
        0.018, 0.018, 0.018, 0.018, 0.018, 0.018,
        0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017,
        0.017, 0.017, 0.017, 0.017, 0.017, 0.017,
        0.0192, 0.0192, 0.0192, 0.0192, 0.0192, 0.0192, 0.0192, 0.0192, 0.0192, 0.0192, 0.0192, 0.0192,
        0.0192, 0.0192, 0.0192, 0.0192, 0.0192, 0.0192,
        0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028,
        0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028
    ],

    'angle': [
        0, 0, 0, 30, 30, 30, 90, 90, 90,
        0, 0, 0, 30, 30, 30, 90, 90, 90
    ] * 6,

    'motion_class': [
        2,2,2,3,2,3,3,3,3,2,1,3,3,3,2,3,3,3,
        2,2,1,2,3,3,3,2,3,2,2,2,3,3,2,2,3,3,
        2,1,2,3,2,2,3,3,3,2,1,2,2,3,2,3,3,3,
        2,2,2,3,3,3,3,3,3,2,3,1,3,3,2,3,3,3,
        2,2,3,3,3,3,3,3,3,2,2,3,2,3,2,3,3,3,
        2,2,2,3,3,3,3,3,3,3,2,2,3,3,3,3,3,3
    ]
})

# ------------------------------------------------------------
# Map physical labels to internal indices
# ------------------------------------------------------------

label_map = {1: 0, 2: 1, 3: 2}
motion_names = {0: "Flutter", 1: "Chaotic", 2: "Tumble"}

data['internal_class'] = data['motion_class'].map(label_map)

# ------------------------------------------------------------
# Model setup
# ------------------------------------------------------------

num_classes = 3  # Flutter, Chaotic, Tumble

def softmax(z):
    # Standard numerically stable softmax
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

def loss(params, Re, I_star, angle, labels, lambda_reg=1e-3):
    """
    Multinomial logistic loss with mild L2 regularization.
    Angle is scaled so it does not dominate the logs.
    """

    params = params.reshape(num_classes, 4)

    log_Re = np.log10(Re)
    log_I = np.log10(I_star)
    angle_scaled = angle / 90.0  # keeps angle on the same order as the logs

    N = len(labels)
    total_loss = 0.0

    for i in range(N):
        x = np.array([log_Re[i], log_I[i], angle_scaled[i], 1.0])

        logits = params @ x
        probs = softmax(logits)

        total_loss -= np.log(probs[labels[i]] + 1e-12)

    # L2 regularization (do not penalize the bias term)
    total_loss += lambda_reg * np.sum(params[:, :-1] ** 2)

    return total_loss / N

# ------------------------------------------------------------
# Optimization
# ------------------------------------------------------------

Re_vals = data['Re'].values
I_vals = data['I_star'].values
angle_vals = data['angle'].values
labels = data['internal_class'].values

initial_guess = np.zeros(num_classes * 4)

result = minimize(
    loss,
    initial_guess,
    args=(Re_vals, I_vals, angle_vals, labels),
    method='BFGS',
    options={'disp': True, 'maxiter': 500}
)

optimized_params = result.x.reshape(num_classes, 4)

# ------------------------------------------------------------
# Print final fitted models
# ------------------------------------------------------------

print("\nFitted regime models (logit form):\n")

for i in range(num_classes):
    a, b, angle_coef, bias = optimized_params[i]
    print(
        f"{motion_names[i]}: "
        f"z = {a:.3f}·log10(Re) + {b:.3f}·log10(I*) "
        f"+ {angle_coef:.3f}·(angle/90) + {bias:.3f}"
    )
