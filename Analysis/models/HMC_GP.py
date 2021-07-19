import torch
torch.set_default_dtype(torch.float64)

import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from pyro.infer import Predictive
import pyro.contrib.gp.kernels as kernels
import pyro.contrib.gp as gp


class HMC_GP():
    def __init__(self, num_samples=10, warmup_steps=50, num_chains=1):
        self.num_samples = num_samples
        self.warmup_steps = warmup_steps
        self.num_chains = num_chains
        self.mcmc = None

    def get_predictions(self, kernel, x, mu):
        K = kernel.forward(x, x) + torch.eye(x.size()[0]) * 1e-6
        L = torch.linalg.cholesky(K)
        return mu + torch.mv(L, torch.ones(x.size()[0]))

    def get_kernel(self, p):
        k1 = kernels.RBF(1, variance=p["s1"], lengthscale=p["l1"])
        k2 = kernels.Product(kernels.RBF(1, variance=p["s2"], lengthscale=p["l2"]),
                             kernels.Periodic(1, variance=torch.tensor([1.]), lengthscale=p["lp2"], period=p["p2"]))
        k3 = kernels.RationalQuadratic(1, variance=p["s3"], lengthscale=p["l3"], scale_mixture=p["m3"])
        # k4 = kernels.WhiteNoise(1, variance=p["v4"])

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

        kernel = self.get_kernel(p)
        
        # with pyro.plate("etas", x.size()[0]):
        #     eta = pyro.sample("eta", dist.Normal(0, 1))
        
        mu = pyro.sample("mu", dist.Normal(0, 1))
        noise = pyro.sample("noise", dist.LogNormal(0, 1))

        y = pyro.sample("y", dist.Normal(self.get_predictions(kernel, x, mu), noise))
            
        return y

    def fit(self, X, y):
        gp_regressor = pyro.condition(self.model, data={"y": y})
        hmc_kernel = NUTS(gp_regressor, jit_compile=True)
        self.mcmc = MCMC(hmc_kernel, num_samples=self.num_samples,
                         warmup_steps=self.warmup_steps,
                         num_chains=self.num_chains)
        self.mcmc.run(x=X)
    

    def posterior_predictive(self, x):
        N = self.num_samples * self.num_chains
        y_hats = torch.empty((N, x.size()[0]))

        for i in range(N):
            samples = self.get_samples()
            p = {k: samples[k][i] for k in samples.keys() if k not in ["mu", "noise"]}

            y = self.get_predictions(self.get_kernel(p), x, samples['mu'][i])
                
            gpr = gp.models.GPRegression(x, y, self.get_kernel(p))
            mean, cov = gpr(x, full_cov=True)
                
            y_hats[i] = dist.MultivariateNormal(mean, cov + torch.eye(x.size()[0])*1e-6).sample()

        return torch.mean(y_hats, axis=0)

    def predict(self, X):
        if self.mcmc is None:
            Exception("The predict method must be called after the fit method")

        samples = self.get_samples()
        predictive = Predictive(self.model, samples)
        preds = predictive(X)["y"].float()
        return preds.mean(axis=0)

    def get_samples(self):
        return self.mcmc.get_samples()

    def summary(self):
        return self.mcmc.summary()
        