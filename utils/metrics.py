import torch
import torch.nn as nn


class EdgeAccuracy(nn.Module):
    """
    Measures the accuracy of the edge map
    """
    def __init__(self, threshold=0.5):
        super(EdgeAccuracy, self).__init__()
        self.threshold = threshold

    def __call__(self, inputs, outputs):
        labels = (inputs > self.threshold)
        outputs = (outputs > self.threshold)

        relevant = torch.sum(labels.float())
        selected = torch.sum(outputs.float())

        if relevant == 0 and selected == 0:
            return torch.tensor(1), torch.tensor(1)

        true_positive = ((outputs == labels) * labels).float()
        recall = torch.sum(true_positive) / (relevant + 1e-8)
        precision = torch.sum(true_positive) / (selected + 1e-8)

        return precision, recall


class PSNR(nn.Module):
    def __init__(self, max_val):
        super(PSNR, self).__init__()
        self.identity_PSNR = 100
        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        # mse = torch.mean((a.float() - b.float()) ** 2)
        lst, lmse = self.with_mse(a,b)

        psnr = sum(lst) / len(lst)
        # mse = sum(lmse) / len(lmse)

        return torch.tensor(psnr,device=a.device)

    def with_mse(self,a,b):
        # mse = torch.mean((a.float() - b.float()) ** 2)
        lst, lmse = [], []
        for i in range(a.shape[0]):
            mse = torch.mean((a[i].float() - b[i].float()) ** 2)
            if mse == 0:
                psnr = self.identity_PSNR
            else:
                psnr = self.max_val - 10 * torch.log(mse) / self.base10
                psnr = psnr.item()
            lst.append(psnr)
            lmse.append(mse.item())

        return lst, lmse

    def from_error_map_to_psnr(self, predicted_mse_map):
        lst, lmse = [], []
        for i in range(predicted_mse_map.shape[0]):
            mse = torch.mean(((255*predicted_mse_map[i]).float()) ** 2)
            if mse == 0:
                psnr = self.identity_PSNR
            else:
                psnr = self.max_val - 10 * torch.log(mse) / self.base10
                psnr = min(self.identity_PSNR, psnr.item())
            lst.append(psnr)

        # psnr = sum(lst) / len(lst)
        return lst

    def from_mse_to_psnr(self, pred_mse):
        lst, lmse = [], []
        for i in range(pred_mse.shape[0]):
            mse = pred_mse[i] * (255**2)
            if mse == 0:
                psnr = self.identity_PSNR
            else:
                psnr = self.max_val - 10 * torch.log(mse) / self.base10
                psnr = min(self.identity_PSNR, psnr.item())
            lst.append(psnr)

        # psnr = sum(lst) / len(lst)
        return lst
