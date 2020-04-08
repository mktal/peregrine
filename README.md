# Peregrine 
## A General Purpose Optimizer for Fast Machine Learning
by Xiaocheng Tang [https://mktal.github.io/]


This code implements in C/C++ a fast second-order sparse training algorithm that is [shown](https://link.springer.com/article/10.1007/s10107-016-0997-3) to be order of magnitude faster than other first-order methods like (stochastic) gradient descent. 
The algorithm provides a more effective learning scheme through a sequence of quadratic approximations with Hessian information. 
This code can be easily extended to, i.e., *distributed settings* or training *neural nets*, with python libraries like Numpy, [Apache Spark](http://spark.apache.org) or [TensoFlow](https://www.tensorflow.org/). Please see `examples` for more details.

This project was presented in 2016 ICML workshop [Optimization Methods for the Next Generation of Machine Learning](http://optml.lehigh.edu/events/icml2016/).

## Getting Started
How to run the code locally:

```bash
pip install ./peregrine/
# train sparse logistic regression
cd peregrine/peregrine/examples && python single_node.py
```

## Citation
* Katya Scheinberg and Xiaocheng Tang, _**Practical Inexact Proximal Quasi-Newton Method with Global Complexity Analysis**_, Mathematical Programming Series A, 160(1), 495–529., 2016. 

```
@article{Scheinberg:2016wj,
  author = {Scheinberg, Katya and Tang, Xiaocheng},
  title = {{Practical inexact proximal quasi-Newton method with global complexity analysis}},
  journal = {Mathematical Programming},
  year = {2016},
  volume = {160},
  number = {1},
  pages = {495--529}
}
```
