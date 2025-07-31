#Try: another toy example for FDFL
#single resource
#Prediction fairness: group-based accuracy disparity
#Decision fairness: new group-based alpha fairness
#DFL: regret based FDFL

import numpy as np
import cvxpy as cp

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution


def pred_acc(rhat, r):
    return mean_squared_error(rhat, r)

def pred_fair(rhat_G1, rhat_G2, r_G1, r_G2):
    mse_G1 = mean_squared_error(rhat_G1, r_G1)
    mse_G2 = mean_squared_error(rhat_G2, r_G2)
    
    return np.abs(mse_G1 - mse_G2)

def dec_acc(dhat, d):
    return mean_squared_error(dhat, d)

def alpha_fair(u, alpha):
    if alpha == 1:
        return sum(np.log(u + 0.000001))
    else:
        if alpha < 1:
            return 1/(1-alpha)*sum(u**(1-alpha))
        else:
            return (alpha-1)/sum(u**(1-alpha))

def dec_fair(r_G1, r_G2, dhat_G1, dhat_G2, alpha):
    u_G1 = r_G1 * dhat_G1
    u_G2 = r_G2 * dhat_G2
     
    fair_G1 = alpha_fair(u_G1, alpha)
    fair_G2 = alpha_fair(u_G2, alpha)
    
    if alpha == 1:
        return np.log(fair_G1+0.000001) + np.log(fair_G2 + 0.000001)
    else:
        return 1/(1-alpha)*(fair_G1**(1-alpha) + fair_G2**(1-alpha))

alpha = 0.5
Q = 20

# features

# # an example that works: 
# f1 = np.array([5,9,14,22,25]) # group 1
# # # features when fpto is less decision fair than pto
# # f1 = np.array([5,9,18,22,25]) # group 1
# f2 = np.array([4,6,7,17,18]) # group 2

# n1 = len(f1)  # number of individuals per group
# n2 = len(f2)

# a = np.array([3.3, 2.7, 3.9, 5.9, 6.3])
# b = np.array([2.2, 3.9, 5.7, 8.1, 8.4])

# print ("Group 1, resource 1: ", a, np.mean(a))
# print ("Group 2, resource 1: ", b, np.mean(b))


# # # what will be used in writing
# f1 = np.array([5,9,14,22,25]) # group 1
# f2 = np.array([4,6,7,17,18]) # group 2

# n1 = len(f1)  # number of individuals per group
# n2 = len(f2)

# a = np.array([3.3, 2.7, 3.2, 5.9, 6.3])
# b = np.array([2.2, 3.9, 5.7, 9, 8.2])

# print ("Group 1, resource 1: ", a, np.mean(a))
# print ("Group 2, resource 1: ", b, np.mean(b))

# another example that works: 
# f1 = np.array([8,12,14,22,25]) # group 1
# f2 = np.array([5,6,7,17,18]) # group 2

# n1 = len(f1)  # number of individuals per group
# n2 = len(f2)

# a = np.array([3.3, 2.7, 6, 5.5, 7])
# b = np.array([3, 4, 5.7, 8, 8.5])

# print ("Group 1, resource 1: ", a, np.mean(a))
# print ("Group 2, resource 1: ", b, np.mean(b))

# # use in paper 
f1 = np.linspace(5,30,10)
n1 = len(f1)
a = 0.2*f1+5

f2 = np.array([6,8,10,20,22])
n2 = len(f2)
b = 0.8*f2+0.2


# MSE regression
X = np.concatenate([f1, f2]).reshape(-1, 1)

y1 = np.concatenate([a, b])   # outcome for resource 1

# Standard regression for resource 1
reg1 = LinearRegression().fit(X, y1)
pred1 = reg1.predict(X)
mse1 = mean_squared_error(y1, pred1)

ahat_mse = pred1[:n1]
bhat_mse = pred1[n1:]

k_pto = reg1.coef_[0]
b_pto = reg1.intercept_

# print("Standard MSE - Resource 1:", mse1)
# # print("MSE by group - G1, G2 ", np.linalg.norm(ahat_mse-a), np.linalg.norm(bhat_mse-b))
# print("MSE by group - G1, G2 ", mean_squared_error(ahat_mse,a), mean_squared_error(bhat_mse,b))

# Fair loss function with accuracy disparity
initial_guess = [1.0, 1.0]
bounds = [(-10, 10), (-10,10)]

def y_mse_loss_fair(params, x, y, x1, y1, x2, y2, lam):
    slope, intercept = params
    yhat = slope * x + intercept
    yg1 = yhat[:len(x1)]
    yg2 = yhat[len(x1):]
    # disparity = np.abs(np.linalg.norm(yg1 - y1) - np.linalg.norm(yg2 - y2))
    disparity = np.abs(np.mean((yg1-y1)**2) - np.mean((yg2-y2)**2))
    mse = np.mean((y - yhat) ** 2)
    
    return mse + lam * disparity

# Wrapper to train a fair-aware regression model
def train_fair_regression_scipy(x, y, x1, y1, x2, y2, lam):
    result = minimize(y_mse_loss_fair, initial_guess, args=(x, y, x1, y1, x2, y2, lam), bounds=bounds, method='L-BFGS-B')
    return result.x, result.fun

# Train fair regression model for Resource 1
lam = 0.5
X = np.concatenate([f1, f2])

w1, obj_val1 = train_fair_regression_scipy(X, y1, f1, a, f2, b, lam)

k_fpto, b_fpto = w1

# Display results
# print("Fair-aware regression for Resource 1:")
# print("  Slope:", w1[0])
# print("  Intercept:", w1[1])
# print("  Objective value (MSE + fairness penalty):", obj_val1)

fpred1 = w1[0]*X + w1[1]
fmse1 = mean_squared_error(y1, fpred1)

ahat_fair = fpred1[:n1].reshape(n1)
bhat_fair = fpred1[n1:].reshape(n2)

# print("MSE when fair - Resource 1:", fmse1)
# print("MSE by group - G1, G2 ", mean_squared_error(ahat_fair,a), mean_squared_error(bhat_fair,b))


# function for computing optimal decisions
def opt_dec(a,b,Q,alpha=1):
    n1 = len(a)
    n2 = len(b)
    d1 = np.zeros(n1)
    d2 = np.zeros(n2)
    if alpha == 1:
        d1 = np.ones(n1)*Q/(2*n1)
        d2 = np.ones(n2)*Q/(2*n2)
    else:
        S1 = sum((a**(1/alpha))**(1-alpha))
        S2 = sum((b**(1/alpha))**(1-alpha))
        H1 = sum(a**(1/alpha - 1))
        H2 = sum(b**(1/alpha - 1))
        
        if alpha < 1:
            # multiplier for group 1
            d1_m = (S1/(1-alpha))**(1/(-2+alpha))*Q/(H1*(S1/(1-alpha))**(1/(-2+alpha)) + H2*(S2/(1-alpha))**(1/(-2+alpha)))
            d1 = d1_m * (a**(1/alpha - 1))
            # multipler for group 2
            d2_m = (S2/(1-alpha))**(1/(-2+alpha))*Q/(H1*(S1/(1-alpha))**(1/(-2+alpha)) + H2*(S2/(1-alpha))**(1/(-2+alpha)))
            d2 = d2_m * (b**(1/alpha - 1))
        else:
            # multiplier for group 1
            d1_m = (S1/(alpha-1))**((-alpha+2)/(-2+2*alpha-alpha**2))*Q/(H1*(S1/(alpha-1))**((-alpha+2)/(-2+2*alpha-alpha**2)) + H2*(S2/(alpha-1))**((-alpha+2)/(-2+2*alpha-alpha**2)))
            d1 = d1_m * (a**(1/alpha - 1))
            # multipler for group 2
            d2_m = (S2/(alpha-1))**((-alpha+2)/(-2+2*alpha-alpha**2))*Q/(H1*(S1/(alpha-1))**((-alpha+2)/(-2+2*alpha-alpha**2)) + H2*(S2/(alpha-1))**((-alpha+2)/(-2+2*alpha-alpha**2)))
            d2 = d2_m * (b**(1/alpha - 1))
    
    return d1,d2
     
print ("True values")
x,s = opt_dec(a, b, Q, alpha)
print ("Group 1, true utility ", np.dot(a,x))
print ("Group 2, true utility ", np.dot(b,s))
print ("True dec fairness ", dec_fair(a,b,x,s,alpha))

print ("MSE predictions")
xp, sp = opt_dec(ahat_mse, bhat_mse, Q, alpha)

rhat = np.concatenate((ahat_mse, bhat_mse))
r = np.concatenate((a,b))

dec_hat = np.concatenate((xp,sp))
dec = np.concatenate((x,s))

pred_acc1 = pred_acc(rhat, r)
pred_fair1 = pred_fair(ahat_mse, bhat_mse, a, b)

dec_acc1 = dec_acc(dec_hat, dec)
dec_fair1 = dec_fair(a, b, xp, sp, alpha)

print (f"Prediction acc:{pred_acc1:.3f}, Prediction fair:{pred_fair1:.3f}, Decision acc:{dec_acc1:.3f}, Decision fair:{dec_fair1:.3f}")

pmse_g1_pto = pred_acc(ahat_mse, a)
pmse_g2_pto = pred_acc(bhat_mse, b)

# dfair_g1_pto = alpha_fair(a * xp, alpha)
# dfair_g2_pto = alpha_fair(b * sp, alpha)

dfair_g1_pto = dec_acc(xp, x)
dfair_g2_pto = dec_acc(sp, s)

print ("fair predictions")
xf, sf = opt_dec(ahat_fair, bhat_fair, Q, alpha)

rhat = np.concatenate((ahat_fair, bhat_fair))
r = np.concatenate((a,b))

dec_hat = np.concatenate((xf,sf))
dec = np.concatenate((x,s))

pred_acc2 = pred_acc(rhat, r)
pred_fair2 = pred_fair(ahat_fair, bhat_fair, a, b)

dec_acc2 = dec_acc(dec_hat, dec)
dec_fair2 = dec_fair(a,b,xf,sf, alpha)

print (f"Prediction acc:{pred_acc2:.3f}, Prediction fair:{pred_fair2:.3f}, Decision acc:{dec_acc2:.3f}, Decision fair:{dec_fair2:.3f}")

pmse_g1_fpto = pred_acc(ahat_fair, a)
pmse_g2_fpto = pred_acc(bhat_fair, b)

# dfair_g1_fpto = alpha_fair(a * xf, alpha)
# dfair_g2_fpto = alpha_fair(b * sf, alpha)

dfair_g1_fpto = dec_acc(xf, x)
dfair_g2_fpto = dec_acc(sf, s)

# end-to-end methods
initial_guess = [0.1,0.1]
bounds = [(0.000001,10),(0.000001,10)]

def decision_loss(params, X, a, b, Q, alpha = 1):
    slope_r1, intercept_r1 = params
    yhat_r1 = slope_r1 * X + intercept_r1

    ap = yhat_r1[:n1]
    bp = yhat_r1[n1:]
    
    x,s = opt_dec(a, b, Q, alpha)
    xp,sp = opt_dec(ap, bp, Q, alpha)
    
    uG1_hat = a * xp
    uG2_hat = b * sp
    
    if alpha == 1:
        fair_G1 = sum(np.log(uG1_hat + 0.000001))
        fair_G2 = sum(np.log(uG2_hat + 0.000001))
        d_mse = -np.log(fair_G1+0.000001) - np.log(fair_G2 + 0.000001)
    else:
        if alpha < 1:
            fair_G1 = 1/(1-alpha)*sum((uG1_hat)**(1-alpha))
            fair_G2 = 1/(1-alpha)*sum((uG2_hat)**(1-alpha))
        else:
            fair_G1 = (alpha-1)/sum((uG1_hat)**(1-alpha))
            fair_G2 = (alpha-1)/sum((uG2_hat)**(1-alpha))
        d_mse = -1/(1-alpha)*((fair_G1)**(1 - alpha) + (fair_G2)**(1 - alpha))
    
    acc_disparity = np.abs(mean_squared_error(a,ap) - mean_squared_error(b,bp))
     
    return d_mse + 0*acc_disparity
 
def e2etrain_scipy(X, a, b, Q, alpha):
    # result = minimize(decision_loss, initial_guess, args=(X, a, b, Q, alpha), bounds=bounds, method='SLSQP')
    # minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds, "args": (X, a, b, Q, alpha)}
    # result = basinhopping(decision_loss, initial_guess, minimizer_kwargs=minimizer_kwargs, niter=500)
    # result = differential_evolution(decision_loss, bounds, args=(X, a, b, Q, alpha), maxiter=2000, tol=1e-6)
    
    # Step 1: Global search
    de_result = differential_evolution(decision_loss, bounds, args=(X, a, b, Q, alpha), seed=22)
    
    # Step 2: Local refinement
    result = minimize(
        decision_loss,
        de_result.x,
        args=(X, a, b, Q, alpha),
        method='L-BFGS-B',
        bounds=bounds
    )
    
    return result.x, result.fun

k1_e2e, b1_e2e = e2etrain_scipy(X,a,b,Q,alpha)[0]

y1_e2e = k1_e2e*X + b1_e2e
e2e_r1 = mean_squared_error(y1, y1_e2e)

ahat_e2e = y1_e2e[:n1].reshape(n1)
bhat_e2e = y1_e2e[n1:].reshape(n2)

print ("E2E predictions")
xe,se = opt_dec(ahat_e2e, bhat_e2e, Q, alpha)

rhat = np.concatenate((ahat_e2e, bhat_e2e))
r = np.concatenate((a,b))

dec_hat = np.concatenate((xe,se))
dec = np.concatenate((x,s))

pred_acc3 = pred_acc(rhat, r)
pred_fair3 = pred_fair(ahat_e2e, bhat_e2e, a, b)

dec_acc3 = dec_acc(dec_hat, dec)
dec_fair3 = dec_fair(a, b, xe, se, alpha)

print (f"Prediction acc:{pred_acc3:.3f}, Prediction fair:{pred_fair3:.3f}, Decision acc:{dec_acc3:.3f}, Decision fair:{dec_fair3:.3f}")

pmse_g1_dfl = pred_acc(ahat_e2e, a)
pmse_g2_dfl = pred_acc(bhat_e2e, b)

# dfair_g1_dfl = alpha_fair(a * xe, alpha)
# dfair_g2_dfl = alpha_fair(b * se, alpha)

dfair_g1_dfl = dec_acc(xe, x)
dfair_g2_dfl = dec_acc(se, s)

# end-to-end methods with prediction fairness
initial_guess = [0.1,0.1]
bounds = [(0.000001,10),(0.000001,10)]

def fair_decision_loss(params, X, a, b, Q, lam, alpha=1):
    slope_r1, intercept_r1 = params
    yhat_r1 = slope_r1 * X + intercept_r1
    
    ap = yhat_r1[:n1]
    bp = yhat_r1[n1:]
    
    x,s = opt_dec(a, b, Q, alpha)
    xp,sp = opt_dec(ap, bp, Q, alpha)
    
    uG1_hat = a * xp
    uG2_hat = b * sp
    
    if alpha == 1:
        fair_G1 = sum(np.log(uG1_hat + 0.000001))
        fair_G2 = sum(np.log(uG2_hat + 0.000001))
        d_mse = -np.log(fair_G1+0.000001) - np.log(fair_G2 + 0.000001)
    else:
        if alpha < 1:
            fair_G1 = 1/(1-alpha)*sum((uG1_hat)**(1-alpha))
            fair_G2 = 1/(1-alpha)*sum((uG2_hat)**(1-alpha))
        else:
            fair_G1 = (alpha-1)/sum((uG1_hat)**(1-alpha))
            fair_G2 = (alpha-1)/sum((uG2_hat)**(1-alpha))
        d_mse = -1/(1-alpha)*((fair_G1)**(1 - alpha) + (fair_G2)**(1 - alpha))

    acc_disparity = np.abs(mean_squared_error(a,ap) - mean_squared_error(b,bp))
    
    # print (d_mse, acc_disparity)
    return d_mse + lam*acc_disparity

def fair_e2etrain_scipy(X, a, b, Q, lam,alpha):
    # result = minimize(fair_decision_loss, initial_guess, args=(X, a, b, Q, lam, alpha), bounds=bounds, method='SLSQP')
    # minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds, "args": (X, a, b, Q, lam, alpha)}
    # result = basinhopping(fair_decision_loss, initial_guess, minimizer_kwargs=minimizer_kwargs, niter=500)
    # result = differential_evolution(fair_decision_loss, bounds, args=(X, a, b, Q, lam, alpha), maxiter=2000, tol=1e-6)
    
    # Step 1: Global search
    de_result = differential_evolution(fair_decision_loss, bounds, args=(X, a, b, Q, lam, alpha), seed=22)
    
    # Step 2: Local refinement
    result = minimize(
        fair_decision_loss,
        de_result.x,
        args=(X, a, b, Q, lam, alpha),
        method='L-BFGS-B',
        bounds=bounds
    )
    
    return result.x, result.fun

lam_fe2e = 0.5

k1_e2ef, b1_e2ef = fair_e2etrain_scipy(X,a,b,Q,lam_fe2e,alpha)[0]

y1_e2ef = k1_e2ef*X + b1_e2ef
e2ef_r1 = mean_squared_error(y1, y1_e2ef)

ahat_e2ef = y1_e2ef[:n1].reshape(n1)
bhat_e2ef = y1_e2ef[n1:].reshape(n2)

print ("fair E2E predictions")
xef,sef = opt_dec(ahat_e2ef, bhat_e2ef, Q,alpha)

rhat = np.concatenate((ahat_e2ef, bhat_e2ef))
r = np.concatenate((a,b))

dec_hat = np.concatenate((xef,sef))
dec = np.concatenate((x,s))

pred_acc4 = pred_acc(rhat, r)
pred_fair4 = pred_fair(ahat_e2ef, bhat_e2ef, a, b)

dec_acc4 = dec_acc(dec_hat, dec)
dec_fair4 = dec_fair(a, b, xef, sef, alpha)

print (f"Prediction acc:{pred_acc4:.3f}, Prediction fair:{pred_fair4:.3f}, Decision acc:{dec_acc4:.3f}, Decision fair:{dec_fair4:.3f}")

pmse_g1_fdfl = pred_acc(ahat_e2ef, a)
pmse_g2_fdfl = pred_acc(bhat_e2ef, b)

# dfair_g1_fdfl = alpha_fair(a * xef, alpha)
# dfair_g2_fdfl = alpha_fair(b * sef, alpha)

dfair_g1_fdfl = dec_acc(xef, x)
dfair_g2_fdfl = dec_acc(sef, s)

import matplotlib.pyplot as plt

features = np.concatenate((f1,f2))
# 8. Visualization
# Plot the results
plt.figure(figsize=(12,8))
plt.scatter(f1, a, color='#1f77b4', label='Group 1',s=100)
plt.scatter(f2, b, color='#ff7f0e', label='Group 2',s=100)
plt.plot(features, k_pto * features + b_pto, color='#2ca02c', label='PTO', linewidth=2)
plt.plot(features, k_fpto * features + b_fpto, color='#bcbd22', linestyle='--', label='FPTO', linewidth=2)
plt.plot(features, k1_e2e * features + b1_e2e, color='#9467bd', linestyle='-.', label='DFL', linewidth=2)
plt.plot(features, k1_e2ef * features + b1_e2ef, color='#8c564b', linestyle=':', label='FDFL', linewidth=2)
plt.xlabel(r'Features $x_i$', fontsize=24)
plt.ylabel(r'Prediction Labels: $r_i$ or $\hat{r}_i$', fontsize=24)
plt.legend(fontsize=24)
plt.title('Regression Lines with Different Methods',fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.tight_layout()
plt.show()

# # visualize prediction MSE
x_labels = ['PTO', 'FPTO', 'DFL', 'FDFL']
x_pos = np.arange(len(x_labels))
bar_width = 0.3
plt.figure(figsize=(12,8))
plt.bar(x_pos - bar_width / 2, [pmse_g1_pto, pmse_g1_fpto, pmse_g1_dfl, pmse_g1_fdfl], bar_width, label='Group 1', alpha=0.8)
plt.bar(x_pos + bar_width / 2, [pmse_g2_pto, pmse_g2_fpto, pmse_g2_dfl, pmse_g2_fdfl], bar_width, label='Group 2', alpha=0.8)
# Labels and title
plt.xticks(x_pos, x_labels,fontsize=24)
plt.yticks(fontsize=24)
plt.ylabel(r'Prediction MSE: $\frac{1}{n} \sum_i (r_i - \hat{r}_i)^2$',fontsize=24)
plt.title('Prediction Errors with Different Methods',fontsize=24)
plt.legend(fontsize=24)
plt.tight_layout()
plt.show()

# visualize decision fairness
x_labels = ['PTO', 'FPTO', 'DFL', 'FDFL']
x_pos = np.arange(len(x_labels))
bar_width = 0.3
plt.figure(figsize=(12,8))
plt.bar(x_pos - bar_width / 2, [dfair_g1_pto, dfair_g1_fpto, dfair_g1_dfl, dfair_g1_fdfl], bar_width, label='Group 1', alpha=0.8)
plt.bar(x_pos + bar_width / 2, [dfair_g2_pto, dfair_g2_fpto, dfair_g2_dfl, dfair_g2_fdfl], bar_width, label='Group 2', alpha=0.8)
# Labels and title
plt.xticks(x_pos, x_labels,fontsize=24)
plt.yticks(fontsize=24)
plt.ylabel(r'Decision MSE: $\frac{1}{n} \sum_i (d_i - \hat{d}_i)^2$',fontsize=24)
plt.title('Decision Errors with Different Methods',fontsize=24)
plt.legend(fontsize=24)
plt.tight_layout()
plt.show()

