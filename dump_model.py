import numpy as np
import torch
from pytorch_bcnn.models import UNet, BayesianUNet
from pytorch_bcnn.links import MCSampler

def main():

    device='cuda'
    out = 'model.pth'

    predictor = BayesianUNet(ndim=2, in_channels=1, nlayer=3, out_channels=10) # NOTE: minimal model
    model = MCSampler(predictor, mc_iteration=10)
    model.to(device)
    model.eval()

    x = np.ones((1, 1, 40, 60)).astype(np.float32)
    x = torch.as_tensor(x, device=device)

    with torch.no_grad():
        traced_model = torch.jit.trace(model, x)

        print(type(traced_model))
        print(traced_model(x))

    traced_model.save(out)

if __name__ == '__main__':
    main()
