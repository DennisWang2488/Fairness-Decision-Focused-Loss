# The code repo for our working paper Fairness Decision Focused Learning

We explore a dataset of medical cost minimization with racial bias, with the following experiment setup:

Utility $u_i = (r_i g_i d_i )^p$. $p=1$. 

Here r is the risk, g is a factor, and d is the decision of enrollment (propensity score?).

---

- The optimization objective is alpha fairness with $\alpha$ = {0, 0.5, 1, 2, $\infty$}.

- The constraints are:
    1. Knapsack constraints on $u$ and non-negativity on $u$. $\sum c_i u_i \leq Q$.
    2. Same constraints but on $d$, i.e.  $\sum c_i d_i \leq Q$.

---

We calculate each individual's $g_i$ from the above equation, with avoidable cost.

 The first algorithm to implement is the 2-stage predict then optimize (PTO), or so called
 prediction focused learning (PFL).

We need to create a optimization model, a prediction model, and a regret loss calculation function to evaluate.

---

1. We have a set of features $x$, we use them to train a prediction model $g(x;\theta)$ predict a risk score $r$.

2. Calculate the utility with $u_i = r_i g_i d_i$, and maximize the optimization objective w.r.t decision variable $d_i$.  

    (Intuitively, we are maximizing the fairness measured avoidable cost for each patient on whether we should enroll them into a program based on their medical history)

3. Evaluate regret loss.



