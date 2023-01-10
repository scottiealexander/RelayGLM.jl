# RelayGLM.jl

A Julia package for fitting GLM models to data from pairs of synaptically connected neurons in order to predict which pre-synaptic spikes were relayed (i.e. elicited a spike in the post-synaptic neuron). This package is heavily used by [this codebase](https://github.com/scottiealexander/relayglm_paper) to create the figures for [this paper](https://www.eneuro.org/content/9/4/ENEURO.0088-22.2022.long).

## Install

Launch Julia and then:

```julia
using Pkg
Pkg.add(PackageSpec(url="https://github.com/scottiealexander/SpkCore.jl.git"))
Pkg.add(PackageSpec(url="https://github.com/scottiealexander/RelayGLM.jl.git"))
```

**NOTE**: the unregistered dependency, `SpkCore.jl`, needs to be manually added before `RelayGLM.jl` can be installed (for now, future updates to the `Pkg` system will likely remove this requirement).

## To use with PairsDB.jl

Launch Julia and then:

```julia
# install PairsDB
using Pkg
Pkg.add(PackageSpec(url="https://github.com/scottiealexander/PairsDB.jl.git"))

# an example for testing
using PairsDB

# load the metadata required to access the data
db = get_database("(?:contrast|area|grating)");

# load data for pair id 204
ret, lgn, _, _ = get_data(db, id=204);

# init a container to hold our predictors (just one in this example)
ps = PredictorSet();

# construct a "spec" for a predictor / filter
# (the predictor matrix is constructed lazily when needed)
ps[:retina] = Predictor(ret, ret, CosineBasis(length=60, offset=2, nbasis=8, b=10, ortho=false, bin_size=0.001));

# indicator for relayed spikes
response = wasrelayed(ret, lgn);

# wrapper object to hold everything together
glm = GLM(ps, response);

# fit the model
result = cross_validate(RRI, Binomial, Logistic, glm, nfold=10, shuffle_design=true);

# plot the learned filter, assuming you have PyPlot installed...
using PyPlot

plot(get_coef(result, :retina))

```

See also:
* [PairsDB.jl](https://github.com/scottiealexander/PairsDB.jl.git)
* [relayglm_paper](https://github.com/scottiealexander/relayglm_paper)
