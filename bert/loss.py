import torch

# Focal loss
def focal_loss(alpha, gamma=2.0, reduction="none"):
    """
    alpha= 0.25,
    gamma: float = 2,
    reduction: str = "none",
    """
    
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gamma =  torch.tensor(gamma, device=device)
    def categorical_focal_loss(inputs, targets):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        epsilon = 1e-10
        epsilon = torch.tensor(epsilon, device=device)
        inputs = inputs.to(device)
        targets = targets.to(device)
        ce_loss = -targets * torch.log(inputs+epsilon)
        loss = alpha * torch.pow(1 - inputs, gamma) * ce_loss 

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss
    
    return categorical_focal_loss