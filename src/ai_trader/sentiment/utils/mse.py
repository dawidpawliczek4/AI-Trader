def mse(y_pred, y_true):
    ans = 0
    for u, w in zip(y_pred, y_true):
        ans += (u - w) ** 2
    
    return ans / len(y_pred)