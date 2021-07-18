import torch
torch.set_default_dtype(torch.float64)

import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from pyro.infer import Predictive
import pyro.contrib.gp.kernels as kernels


class HMC_GP():
    def __init__(self, num_samples=10, warmup_steps=50, num_chains=1):
        self.num_samples = num_samples
        self.warmup_steps = warmup_steps
        self.num_chains = num_chains
        self.samples = None

    def get_predictions(self, x, mu, eta):
        K = self.kernel.forward(x, x) + torch.eye(x.size()[0]) * 1e-6
        L = K.cholesky()
        return mu + torch.mv(L, eta)

    def get_kernel(self, p):
        k1 = kernels.RBF(1, variance=p["s1"], lengthscale=p["l1"])
        k2 = kernels.Product(kernels.RBF(1, variance=p["s2"], lengthscale=p["l2"]),
                             kernels.Periodic(1, variance=torch.tensor([1.]), lengthscale=p["lp2"], period=p["p2"]))
        k3 = kernels.RationalQuadratic(1, variance=p["s3"], lengthscale=p["l3"], scale_mixture=p["m3"])
        #k4 = kernels.WhiteNoise(1, variance=p["v4"])

        return kernels.Sum(kernels.Sum(k1, k2), k3)

    def model(self, x):
        # Kernel parameters
        p = dict()
        p["s1"] = pyro.sample("s1", dist.LogNormal(0, 1))
        p["l1"] = pyro.sample("l1", dist.LogNormal(0, 1))
        
        p["s2"] = pyro.sample("s2", dist.LogNormal(0, 1))
        p["l2"] = pyro.sample("l2", dist.LogNormal(0, 1))
        p["lp2"] = pyro.sample("lp2", dist.LogNormal(0, 1))
        p["p2"] = pyro.sample("p2", dist.LogNormal(0, 1))

        p["s3"] = pyro.sample("s3", dist.LogNormal(0, 1))
        p["l3"] = pyro.sample("l3", dist.LogNormal(0, 1))
        p["m3"] = pyro.sample("m3", dist.LogNormal(0, 1))

        #p["v4"] = pyro.sample("v4", dist.LogNormal(0, 1))

        self.kernel = self.get_kernel(p)
        
        with pyro.plate("etas", x.size()[0]):
            eta = pyro.sample("eta", dist.Normal(0, 1))
        
        mu = pyro.sample("mu", dist.Normal(0, 1))
        noise = pyro.sample("noise", dist.LogNormal(0, 1))

        y = pyro.sample("y", dist.Normal(self.get_predictions(x, mu, eta), noise))
            
        return y

    def fit(self, X, y):
        gp_regressor = pyro.condition(self.model, data={"y": y})
        hmc_kernel = NUTS(gp_regressor, jit_compile=True)
        mcmc = MCMC(hmc_kernel, num_samples=self.num_samples,
                    warmup_steps=self.warmup_steps,
                    num_chains=self.num_chains)
        mcmc.run(x=X)
        self.samples = mcmc.get_samples()

    def predict(self, X):
        predictive = Predictive(self.model, self.samples)
        preds = predictive(X)["y"].float()
        return preds.mean(axis=0)
        