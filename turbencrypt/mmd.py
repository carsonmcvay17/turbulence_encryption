# imports
import torch
import numpy as np

class MMD:
    """
    Computes the Maximum Mean Discrepancy (MMD) between two distributions.
    """

    def compute_mmd(self, x, y, kernel):
        """
        The lower the result the more similar the distributions are
        Args:
        x-first sample, distribution p
        y-second sample, distribution q
        kernel-kernel type such as multiscale or rbf
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        xx, yy, zz = torch.mm(x,np.transpose(x)), torch.mm(y,np.transpose(y)), torch.mm(x,np.transpose(y))
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))


        dxx = np.transpose(rx) + rx - 2. * xx
        dyy = np.transpose(ry) + ry - 2. * yy
        dxy = np.transpose(rx) + ry - 2. * zz

        XX, YY, XY = (torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device))
        
        if kernel == "multiscale":
            bandwith_range = [0.2, 0.5, 0.9, 1.3]
            for a in bandwith_range:
                XX += a**2 * (a**2 + dxx)**-1
                YY += a**2 * (a**2 + dyy)**-1
                XY += a**2 * (a**2 + dxy)**-1
        if kernel == "rbf":
            bandwith_range = [10, 15, 20, 50]
            for a in bandwith_range:
                XX += torch.exp(-.5*dxx/a)
                YY += torch.exp(-.5*dyy/a)
                XY += torch.exp(-.5*dxy/a)

        computed_mmd =  torch.mean(XX + YY - 2. * XY)
        print(computed_mmd)
        return computed_mmd
    
    def extract_features(self, data_loader, model):
        features = []
        with torch.no_grad():
            for images, _ in data_loader:
                images = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                outputs = model(images)
                features.append(outputs.cpu())
        return torch.cat(features, dim=0)
        