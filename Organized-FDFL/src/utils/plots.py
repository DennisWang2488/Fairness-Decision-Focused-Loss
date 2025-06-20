import matplotlib.pyplot as plt

def visLearningCurve(loss_log, loss_log_regret, mse_loss_log, fairness_log=None):
    if fairness_log is not None:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 4))
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))
    # Plot original loss log
    ax1.plot(loss_log, color="c", lw=2)
    ax1.tick_params(axis="both", which="major", labelsize=12)
    ax1.set_xlabel("Iters", fontsize=16)
    ax1.set_ylabel("Loss", fontsize=16)
    ax1.set_title("Training Loss Curve", fontsize=16)
    ax1.grid(True, alpha=0.3)

    # Plot regret log
    ax2.plot(loss_log_regret, color="royalblue", ls="--", alpha=0.7, lw=2)
    ax2.tick_params(axis="both", which="major", labelsize=12)
    ax2.set_xlabel("Epochs", fontsize=16)
    ax2.set_ylabel("Regret", fontsize=16)
    ax2.set_title("Learning Curve (Test Regret)", fontsize=16)
    ax2.grid(True, alpha=0.3)

    # Plot MSE loss log
    ax3.plot(mse_loss_log, color="orange", lw=2)
    ax3.tick_params(axis="both", which="major", labelsize=12)
    ax3.set_xlabel("Iters", fontsize=16)
    ax3.set_ylabel("MSE Loss", fontsize=16)
    ax3.set_title("Learning Curve (MSE Loss)", fontsize=16)
    ax3.grid(True, alpha=0.3)

    # Plot fairness log if provided
    if fairness_log is not None:
        ax4.plot(fairness_log, color="green", lw=2)
        ax4.tick_params(axis="both", which="major", labelsize=12)
        ax4.set_xlabel("Iters", fontsize=16)
        ax4.set_ylabel("Fairness", fontsize=16)
        ax4.set_title("Learning Curve (Fairness)", fontsize=16)
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()