import numpy as np

def alphaFairness(utility, alpha):
    """
    Calculate alpha fairness
    """
    return np.sum(utility) ** (1 - alpha) / (1 - alpha) if alpha != 1 else np.sum(np.log(np.abs(utility)))


def utility(risk, gFactor, decision):
    """
    Calculate utility
    """
    return risk * gFactor * decision


def regret(data, optModel, predModel):
    """
    Calculate regret
    """

def main():
    risk, gFactor, decision = 2, 5, 1
    myUtil = utility(risk, gFactor, decision)
    print(myUtil)


    print(alphaFairness(myUtil, 0.5))


if __name__ == "__main__":
    main()

