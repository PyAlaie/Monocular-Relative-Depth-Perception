import torch


def sample_pairs(depth, num_pairs=2000):
    H = depth.shape[0]
    W = depth.shape[1]
    idx1 = torch.randint(0, H*W, (num_pairs,))
    idx2 = torch.randint(0, H*W, (num_pairs,))
    return idx1, idx2


def ranking_loss(pred, target, margin=0.0, num_pairs=1000):
    B, _, H, W = pred.shape

    pred_flat   = pred.view(B, -1)
    target_flat = target.view(B, -1)

    total_loss = 0.0

    for b in range(B):
        idx1, idx2 = sample_pairs(target[b], num_pairs)

        unequal_pairs_loss = []

        for point1, point2 in zip(idx1, idx2):
            # ground truth
            g_i = target_flat[b][point1]
            g_j = target_flat[b][point2]

            sigma = 0.02

            l_k = 0
            if g_i / g_j > 1 + sigma:
                l_k = 1
            elif g_j / g_i > 1 + sigma:
                l_k = -1
            else:
                pass # l_k is 0 by defualt

            # Predicted depth
            p1 = pred_flat[b][point1]
            p2 = pred_flat[b][point2]

            loss = 0
            if l_k == 0: # two points have reletivly the same depth
                loss = (p1 - p2)**2
                total_loss += loss
            else:
                loss = torch.log(1 + torch.exp((p2 - p1) * l_k))
                unequal_pairs_loss.append(loss)

        # exclude least 25% of unequal pairs loss
        unequal_pairs_loss.sort()
        bottom_25 = int(len(unequal_pairs_loss) * .25)
        unequal_pairs_loss = unequal_pairs_loss[bottom_25:]

        total_loss += sum(unequal_pairs_loss)

    return total_loss