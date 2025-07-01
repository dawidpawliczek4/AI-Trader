def mae(y_pred, y_true):
    ans = 0
    for u, w in zip(y_pred, y_true):
        ans += abs(u - w)
    
    return ans / len(y_pred)