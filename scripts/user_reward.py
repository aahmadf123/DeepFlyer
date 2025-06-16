import numpy as np

def reward(state):
    """
    User-editable reward function for drone path following.
    Args:
        state: dict with at least 'cross_track_error' (meters) and 'heading_error' (radians)
    Returns:
        float: reward value (higher is better)
    """
    # Path following term (cross-track error)
    cross_track_error = abs(state.get("cross_track_error", 0.0))
    max_error = 2.0  # meters
    path_term = 1.0 * (1 - min(cross_track_error / max_error, 1.0))

    # Heading alignment term (heading error)
    heading_error = abs(state.get("heading_error", 0.0))
    max_heading_error = np.pi  # radians
    heading_term = 0.1 * (-min(heading_error / max_heading_error, 1.0))

    return path_term + heading_term

# Edit the weights or normalization above to experiment! 