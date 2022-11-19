# Learning differentiable solvers for systems with hard constraints

We provide an implementation of our hard-constrained method for the non-linear Burgers' PDE. 

To run
- Hard constrained
```
python src/burgers/pytorch_deeponet.py --Nbasis 50 --batch 3 --max_iterations 80 --nsamples 600 --nsamples_residual 300 --log_every_n_steps 10 --ngpus 6
```

- Soft-constrained
```
python src/burgers/pytorch_deeponet.py --Nbasis 1 --batch 50 --log_every_n_steps 10 --ngpus 6
```
