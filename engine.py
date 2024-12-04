import random as rd
from network import Value

class Neuron:

  def __init__(self, nin):
    self.w = [Value(rd.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(rd.uniform(-1,1))


  def __call__(self,x):
    # w * x + b

    # summing weights and adding bias
    # starting with self.b in the zip
    tot = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    out = tot.tanh()

    return out

  def parameters(self):
    return self.w + [self.b]

class Layer:

  def __init__(self, nin, nout):
    self.neurons = [Neuron(nin) for _ in range(nout)]
  
  def __call__(self,x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs
  
  def parameters(self):
    out = []
    for neuron in self.neurons:
      out.extend(neuron.parameters())
    return out
  
class MLP:

  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(sz)-1)]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def parameters(self):
    out = []
    for layer in self.layers:
      out.extend(layer.parameters())
    return out
    