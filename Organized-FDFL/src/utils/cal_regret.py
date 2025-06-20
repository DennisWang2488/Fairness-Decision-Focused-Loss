# This module contains the function to calculate the regret based on the predictions from a model and the optimal model.
def objValue(d, b, alpha):
    """
    Calculate the objective value based on the data and utility predictions.
    """
    objval = None
    return objval

def calRegret(predictor, optModel, data, alpha):
    """
    Calculate the regret based on the predictions from the predictor model and the optimal model.

    Args:
        predictor: The prediction model that outputs util predictions.
        optModel: The optimal model that computes the optimal decisions.
        data: dataloder containing the input data, including true solutions and objective values.
    Returns:
        The normalized regret value.
    """
    # Get the predictions from the predictor model
    preds = predictor(data)

    # Use it from the data if available
    true_sol, true_obj = data['true_sol'], data['true_obj']

    # Calculate the predicted solution and objective value using the optimization model
    pred_sol, _ = optModel(preds)

    # Calculate the predicted objective value of the predicted solution, and the true benefit.
    pred_obj = objValue(pred_sol, data['benefit'], alpha)
    normalized_regret = (true_obj - pred_obj) / (abs(true_obj) + 1e-7)

    return normalized_regret