#New toy example for FDFL
#Prediction fairness: accuracy disparity

import numpy as np
import cvxpy as cp

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from scipy.optimize import minimize

# features
f1 = np.array([5, 10, 14, 17]) # group 1
f2 = np.array([4, 6, 9, 12]) # group 2

n = len(f1)  # number of individuals per group

# Benefits
a = 1/4 * f1 + 1  # group 1, resource 1
c = 1/3 * f1 + 2  # group 1, resource 2

print ("Group 1, resource 1: ", a, np.mean(a))
print ("Group 1, resource 2: ", c, np.mean(c))

b = 1/2 * f2 + 2 # group 2, resource 1
d = 3/16 * f2 + 3  # group 2, resource 2

print ("Group 2, resource 1: ", b, np.mean(b))
print ("Group 2, resource 2: ", d, np.mean(d))

# MSE regression
X = np.concatenate([f1, f2]).reshape(-1, 1)

y1 = np.concatenate([a, b])   # outcome for resource 1
y2 = np.concatenate([c, d])   # outcome for resource 2

# Standard regression for resource 1
reg1 = LinearRegression().fit(X, y1)
pred1 = reg1.predict(X)
mse1 = mean_squared_error(y1, pred1)

ahat_mse = pred1[:n]
bhat_mse = pred1[n:]

# Standard regression for resource 2
reg2 = LinearRegression().fit(X, y2)
pred2 = reg2.predict(X)
mse2 = mean_squared_error(y2, pred2)

chat_mse = pred2[:n]
dhat_mse = pred2[n:]

print("Standard MSE - Resource 1:", mse1)
print("MSE by group - G1, G2 ", np.linalg.norm(ahat_mse-a), np.linalg.norm(bhat_mse-b))

print("Standard MSE - Resource 2:", mse2)
print("MSE by group - G1, G2 ", np.linalg.norm(chat_mse-c), np.linalg.norm(dhat_mse-d))

# Fair loss function with accuracy disparity
initial_guess = [1.0, 1.0]
bounds = [(-10, 10), (-10,10)]

def y_mse_loss_fair(params, x, y, x1, y1, x2, y2, lam):
    slope, intercept = params
    yhat = slope * x + intercept
    yg1 = yhat[:len(x1)]
    yg2 = yhat[len(x1):]
    disparity = np.abs(np.linalg.norm(yg1 - y1) - np.linalg.norm(yg2 - y2))
    mse = np.mean((y - yhat) ** 2)
    return mse + lam * disparity

# Wrapper to train a fair-aware regression model
def train_fair_regression_scipy(x, y, x1, y1, x2, y2, lam):
    result = minimize(y_mse_loss_fair, initial_guess, args=(x, y, x1, y1, x2, y2, lam), bounds=bounds, method='L-BFGS-B')
    return result.x, result.fun

# Train fair regression model for Resource 1
lam = 1
X = np.concatenate([f1, f2])

w1, obj_val1 = train_fair_regression_scipy(X, y1, f1, a, f2, b, lam)

# Train fair regression model for Resource 2
w2, obj_val2 = train_fair_regression_scipy(X, y2, f1, c, f2, d, lam)

# Display results
# print("Fair-aware regression for Resource 1:")
# print("  Slope:", w1[0])
# print("  Intercept:", w1[1])
# print("  Objective value (MSE + fairness penalty):", obj_val1)

# print("\nFair-aware regression for Resource 2:")
# print("  Slope:", w2[0])
# print("  Intercept:", w2[1])
# print("  Objective value (MSE + fairness penalty):", obj_val2)

fpred1 = w1[0]*X + w1[1]
fmse1 = mean_squared_error(y1, fpred1)

ahat_fair = fpred1[:n].reshape(n)
bhat_fair = fpred1[n:].reshape(n)

fpred2 = w2[0]*X + w2[1]
fmse2 = mean_squared_error(y2, fpred2)

chat_fair = fpred2[:n].reshape(n)
dhat_fair = fpred2[n:].reshape(n)

print("MSE when fair - Resource 1:", fmse1)
print("MSE by group - G1, G2 ", np.linalg.norm(ahat_fair-a), np.linalg.norm(bhat_fair-b))

print("MSE when fair - Resource 2:", fmse2)
print("MSE by group - G1, G2 ", np.linalg.norm(chat_fair-c), np.linalg.norm(dhat_fair-d))

# function for computing optimal decisions
def opt_dec(a,b,c,d):
    # Decision variables: binary
    x = cp.Variable(n, boolean=True)  # group 1, resource 1
    y = cp.Variable(n, boolean=True)  # group 1, resource 2
    s = cp.Variable(n, boolean=True)  # group 2, resource 1
    t = cp.Variable(n, boolean=True)  # group 2, resource 2
    
    # Total benefit for each group
    u_G1 = a @ x + c @ y
    u_G2 = b @ s + d @ t
    
    # Alpha-fair utility function (alpha ≠ 1)
    # alpha = 2
    # objective_expr = (1 / (1 - alpha)) * (cp.power(u_G1, 1 - alpha) + cp.power(u_G2, 1 - alpha))
    # objective = cp.Maximize(objective_expr)
    
    # Minimize utility difference
    objective_expr = (u_G1 - u_G2)**2
    objective = cp.Minimize(objective_expr)
    
    # Constraints
    constraints = []
    
    # Each individual gets exactly one resource
    constraints += [x[i] + y[i] == 1 for i in range(n)]
    constraints += [s[i] + t[i] == 1 for i in range(n)]
    
    # Total allocations per resource
    constraints += [cp.sum(x) + cp.sum(s) == n]
    constraints += [cp.sum(y) + cp.sum(t) == n]
    
    # Solve problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS_BB)  # Or use solver='ECOS_BB' if GUROBI is not available
    
    # Output results
    # print("Status:", prob.status)
    # print("Objective value:", prob.value)
    # print("x (G1 R1):", x.value.round())
    # print("y (G1 R2):", y.value.round())
    # print("s (G2 R1):", s.value.round())
    # print("t (G2 R2):", t.value.round())
    
    return x.value.round(), y.value.round(), s.value.round(), t.value.round()
    
print ("True values")
x,y,s,t = opt_dec(a, b, c, d)
print ("Group 1, true utility ", np.dot(a,x) + np.dot(c, y))
print ("Group 2, true utility ", np.dot(b,s) + np.dot(d, t))

print ("MSE predictions")
xp, yp, sp, tp = opt_dec(ahat_mse, bhat_mse, chat_mse, dhat_mse)
print ("Group 1, predicted utility ", np.dot(a,xp) + np.dot(c, yp))
print ("Group 2, predicted utility ", np.dot(b,sp) + np.dot(d, tp))
print ("Decision error ", np.linalg.norm(x-xp) + np.linalg.norm(y-yp) + np.linalg.norm(s-sp) + np.linalg.norm(t-tp))

print ("fair predictions")
xf, yf, sf, tf = opt_dec(ahat_fair, bhat_fair, chat_fair, dhat_fair)
print ("Group 1, fair predicted utility ", np.dot(a,xf) + np.dot(c, yf))
print ("Group 2, fair predicted utility ", np.dot(b,sf) + np.dot(d, tf))
print ("Decision error ", np.linalg.norm(x-xf) + np.linalg.norm(y-yf) + np.linalg.norm(s-sf) + np.linalg.norm(t-tf))


# end-to-end methods
initial_guess = [1,1,1,1]
bounds = [(0,10),(0,10),(0,10),(0,10)]

def decision_loss(params, X, a, b, c, d):
    slope_r1, intercept_r1, slope_r2, intercept_r2 = params
    yhat_r1 = slope_r1 * X + intercept_r1
    yhat_r2 = slope_r2 * X + intercept_r2

    ap = yhat_r1[:n]
    bp = yhat_r1[n:]
    cp = yhat_r2[:n]
    dp = yhat_r2[n:]
    
    x,y,s,t = opt_dec(a, b, c, d)
    xp,yp,sp,tp = opt_dec(ap, bp, cp, dp)
    
    d_mse = np.linalg.norm(x-xp) + np.linalg.norm(y-yp) + np.linalg.norm(s-sp) + np.linalg.norm(t-tp)
    
    # pred_mse = np.linalg.norm(a-ap) + np.linalg.norm(b-bp) + np.linalg.norm(c-cp) + np.linalg.norm(d-dp)
    
    return d_mse

def e2etrain_scipy(X, a, b, c, d):
    result = minimize(decision_loss, initial_guess, args=(X, a, b, c, d), bounds=bounds, method='L-BFGS-B')
    return result.x, result.fun

k1_e2e, b1_e2e, k2_e2e, b2_e2e = e2etrain_scipy(X,a,b,c,d)[0]

y1_e2e = k1_e2e*X + b1_e2e
e2e_r1 = mean_squared_error(y1, y1_e2e)

ahat_e2e = y1_e2e[:n].reshape(n)
bhat_e2e = y1_e2e[n:].reshape(n)

y2_e2e = k2_e2e*X + b2_e2e
e2e_r2 = mean_squared_error(y2, y2_e2e)

chat_e2e = y2_e2e[:n].reshape(n)
dhat_e2e = y2_e2e[n:].reshape(n)

print("E2E - Resource 1:", e2e_r1)
print("MSE by group - G1, G2 ", np.linalg.norm(ahat_e2e-a), np.linalg.norm(bhat_e2e-b))

print("E2E - Resource 2:", e2e_r2)
print("MSE by group - G1, G2 ", np.linalg.norm(chat_e2e-c), np.linalg.norm(dhat_e2e-d))

print ("E2E predictions")
xe,ye,se,te = opt_dec(ahat_e2e, bhat_e2e, chat_e2e, dhat_e2e)
print ("Group 1, predicted utility ", np.dot(a,xe) + np.dot(c, ye))
print ("Group 2, predicted utility ", np.dot(b,se) + np.dot(d, te))
print ("Decision error ", np.linalg.norm(x-xe) + np.linalg.norm(y-ye) + np.linalg.norm(s-se) + np.linalg.norm(t-te))

# end-to-end methods with prediction fairness
initial_guess = [1,1,1,1]
bounds = [(0,10),(0,10),(0,10),(0,10)]

def fair_decision_loss(params, X, a, b, c, d, lam):
    slope_r1, intercept_r1, slope_r2, intercept_r2 = params
    yhat_r1 = slope_r1 * X + intercept_r1
    yhat_r2 = slope_r2 * X + intercept_r2

    ap = yhat_r1[:n]
    bp = yhat_r1[n:]
    cp = yhat_r2[:n]
    dp = yhat_r2[n:]
    
    x,y,s,t = opt_dec(a, b, c, d)
    xp,yp,sp,tp = opt_dec(ap, bp, cp, dp)
    
    d_mse = np.linalg.norm(x-xp) + np.linalg.norm(y-yp) + np.linalg.norm(s-sp) + np.linalg.norm(t-tp)
    acc_disparity = np.abs(np.linalg.norm(a - ap) - np.linalg.norm(b - bp)) + np.abs(np.linalg.norm(c - cp) - np.linalg.norm(d - dp))
    
    # pred_mse = np.linalg.norm(a-ap) + np.linalg.norm(b-bp) + np.linalg.norm(c-cp) + np.linalg.norm(d-dp)
    
    return d_mse + lam*acc_disparity

def fair_e2etrain_scipy(X, a, b, c, d, lam):
    result = minimize(fair_decision_loss, initial_guess, args=(X, a, b, c, d, lam), bounds=bounds, method='L-BFGS-B')
    return result.x, result.fun

lam_fe2e = 1

k1_e2ef, b1_e2ef, k2_e2ef, b2_e2ef = fair_e2etrain_scipy(X,a,b,c,d,lam_fe2e)[0]

y1_e2ef = k1_e2ef*X + b1_e2ef
e2ef_r1 = mean_squared_error(y1, y1_e2ef)

ahat_e2ef = y1_e2ef[:n].reshape(n)
bhat_e2ef = y1_e2ef[n:].reshape(n)

y2_e2ef = k2_e2ef*X + b2_e2ef
e2ef_r2 = mean_squared_error(y2, y2_e2ef)

chat_e2ef = y2_e2ef[:n].reshape(n)
dhat_e2ef = y2_e2ef[n:].reshape(n)

print("fair E2E - Resource 1:", e2ef_r1)
print("MSE by group - G1, G2 ", np.linalg.norm(ahat_e2ef-a), np.linalg.norm(bhat_e2ef-b))

print("fair E2E - Resource 2:", e2ef_r2)
print("MSE by group - G1, G2 ", np.linalg.norm(chat_e2ef-c), np.linalg.norm(dhat_e2ef-d))

print ("fair E2E predictions")
xef,yef,sef,tef = opt_dec(ahat_e2ef, bhat_e2ef, chat_e2ef, dhat_e2ef)
print ("Group 1, predicted utility ", np.dot(a,xef) + np.dot(c, yef))
print ("Group 2, predicted utility ", np.dot(b,sef) + np.dot(d, tef))
print ("Decision error ", np.linalg.norm(x-xef) + np.linalg.norm(y-yef) + np.linalg.norm(s-sef) + np.linalg.norm(t-tef))
