import numpy as np
import torch

def regret(predmodel, optmodel, dataloader, closed=False, alpha=0.5):
    """
    A function to evaluate model performance with normalized true regret

    Args:
        predmodel (nn): a regression neural network for cost prediction
        optmodel (optModel): an PyEPO optimization model
        dataloader (DataLoader): Torch dataloader from optDataSet

    Returns:
        float: true regret loss
    """
    # eval
    predmodel.eval()
    loss = 0
    optsum = 0

    if not closed:
    # load data
        for data in dataloader:
            x,r,c,opt_sol,opt_obj,_,_ = data
            # cuda
            x,r,c,opt_sol,opt_obj = x.cuda(),r.cuda(),c.cuda(),opt_sol.cuda(),opt_obj.cuda()
            # predict
            with torch.no_grad():
                pred_r = predmodel(x).to('cpu').detach().numpy()
            # solve
            for j in range(pred_r.shape[0]):
                loss += calRegret(optmodel, c[j], pred_r[j], r[j].to("cpu").detach().numpy(), opt_obj[j].item(), alpha)

                optsum += abs(opt_obj[j].item())
    # turn back to train mode
    predmodel.train()

    # normalize
    return loss / (optsum+1e-7)



def calRegret(optmodel, true_c, pred_r, true_r, true_obj, alpha):
    """
    A function to calculate the normalized true regret for a batch

    Args:
        optmodel(optModel): optimization model
        pred_r(torch.tensor): predicted r values
        true_r(torch.tensor): true r values
        true_obj(torch.tensor): true objective values

    Returns:predmodel
        float:true regret losses
    """
    # opt solution for predicted r

    optmodel.setObj(r=pred_r,c=true_c)
    sol, _ = optmodel.solve()
    # obj value with true cost and predicted r
    if alpha == 1:
        obj = np.sum(np.log(pred_r * sol))
    else:
        obj = np.sum((pred_r * sol)**(1 - alpha)) / (1 - alpha)

    # loss
    loss = true_obj - obj

    return loss
        