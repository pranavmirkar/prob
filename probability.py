import pandas as pd
import numpy as np
from scipy.optimize import minimize

# ------------------------------------------------------------
# Experimental dataset
# Sirf 3 motion types rakhe hain:
# 1 = Flutter, 2 = Chaotic, 3 = Tumble
# ------------------------------------------------------------

data = pd.DataFrame({
    'Re': [
        # Reynolds number – flow ka strength samajh le
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
        # Dimensionless moment of inertia – body kitna heavy / spread out hai
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
        # Launch angle in degrees
        0, 0, 0, 30, 30, 30, 90, 90, 90,
        0, 0, 0, 30, 30, 30, 90, 90, 90
    ] * 6,

    'motion_class': [
        # Actual observed motion from experiments
        2,2,2,3,2,3,3,3,3,2,1,3,3,3,2,3,3,3,
        2,2,1,2,3,3,3,2,3,2,2,2,3,3,2,2,3,3,
        2,1,2,3,2,2,3,3,3,2,1,2,2,3,2,3,3,3,
        2,2,2,3,3,3,3,3,3,2,3,1,3,3,2,3,3,3,
        2,2,3,3,3,3,3,3,3,2,2,3,2,3,2,3,3,3,
        2,2,2,3,3,3,3,3,3,3,2,2,3,3,3,3,3,3
    ]
})

# ------------------------------------------------------------
# Labels ko ML-friendly numbering me convert kar rahe hain
# Flutter -> 0, Chaotic -> 1, Tumble -> 2
# ------------------------------------------------------------

label_map = {1: 0, 2: 1, 3: 2}
motion_names = {0: "Flutter", 1: "Chaotic", 2: "Tumble"}

data['internal_class'] = data['motion_class'].map(label_map)

# ------------------------------------------------------------
# Model setup
# 3-class multinomial logistic regression
# ------------------------------------------------------------

num_classes = 3  # total motion regimes

def softmax(z):
    # Softmax – scores ko probability me convert karta hai
    z = z - np.max(z)  # numerical stability ke liye
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

def loss(params, Re, I_star, angle, labels, lambda_reg=1e-3):
    """
    Bhai ye main loss function hai:
    - Multiclass logistic loss
    - Thoda sa L2 regularization overfitting se bachane ke liye
    """

    # Parameters ko 3x4 matrix me reshape kar rahe hain
    params = params.reshape(num_classes, 4)

    # Re aur I* ka log liya – physics me power-law type behavior hota hai
    log_Re = np.log10(Re)
    log_I = np.log10(I_star)

    # Angle ko scale kar diya taaki dominate na kare
    angle_scaled = angle / 90.0

    N = len(labels)
    total_loss = 0.0

    for i in range(N):
        # Feature vector: [log(Re), log(I*), angle, bias]
        x = np.array([log_Re[i], log_I[i], angle_scaled[i], 1.0])

        # Har class ka score (logit)
        logits = params @ x

        # Probability nikaal rahe hain
        probs = softmax(logits)

        # Correct class ki negative log likelihood
        total_loss -= np.log(probs[labels[i]] + 1e-12)

    # L2 regularization (bias term pe penalty nahi)
    total_loss += lambda_reg * np.sum(params[:, :-1] ** 2)

    return total_loss / N

# ------------------------------------------------------------
# Optimization – BFGS se best parameters dhundh rahe hain
# ------------------------------------------------------------

Re_vals = data['Re'].values
I_vals = data['I_star'].values
angle_vals = data['angle'].values
labels = data['internal_class'].values

# Starting point – sab zero se
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
# Final fitted equations print kar rahe hain
# ------------------------------------------------------------

print("\nFitted regime models (logit form):\n")

for i in range(num_classes):
    a, b, angle_coef, bias = optimized_params[i]
    print(
        f"{motion_names[i]}: "
        f"z = {a:.3f}·log10(Re) + {b:.3f}·log10(I*) "
        f"+ {angle_coef:.3f}·(angle/90) + {bias:.3f}"
    )

def predict_motion_probs(Re, I_star, angle, param_matrix):
    """
    Bhai ye function new case ke liye batata hai:
    Flutter / Chaotic / Tumble hone ke chances kya hain
    """

    log_Re = np.log10(Re)
    log_I = np.log10(I_star)
    angle_scaled = angle / 90.0

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

    # Softmax again
    logits -= np.max(logits)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)

    motion_types = ["Flutter", "Chaotic", "Tumble"]

    return {
        motion: round(float(prob), 4)
        for motion, prob in zip(motion_types, probs)
    }

# ------------------------------------------------------------
# Example prediction – ek naya experiment maan ke
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
