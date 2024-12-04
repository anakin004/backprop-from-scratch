import math

class Value:

    ''' value node, for biases, weights, and results'''
    ''' contains mathematical operations to interact with other value nodes '''
    ''' additionally contains backprop with topological graph'''

    def __init__(self,data, _children=(), _op='', label = ''):
        self._backward = lambda: None
        self.grad = 0.0
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    # repr used when we print object
    def __repr__(self):
        return f"Value(data={self.data})" 
    
    def __radd__(self, other):
        return self + other
    
    def __add__(self, other):

        # if other is a constant then we wrap it in Value obj
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self,other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward = _backward

        return out

    def __sub__(self, other):
        return self + (-other)
    
    # wrapper for negation, we can simply avoid having to write
    # script for sub by using what we already have
    def __neg__(self):
        return self * -1
    
    def __mul__(self, other):

        # if other is constant we wrap it
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self,other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    # python will try to default if __mul__ does not work
    def __rmul__(self, other): # other * self, case if we have 2 * Val obj
        return self * other
    

    def __truediv__(self, other): # divide 
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1
    
    def __rtruediv__(self, other): # divide but flipped:
        other = other if isinstance(other, Value) else Value(other)
        return other * self**-1
    

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supports int/float powers"
        out = Value(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * ( self.data ** (other-1) ) * out.grad

        out._backward = _backward

        return out 

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data 
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def backward(self):

        topo = []
        visited = set()
        
        #builing topological graph for backprop,
        #we only append the parent node until all children are processed
        #this allows us to visit all nodes sequentially in reverse 
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        self.grad = 1.0
        build_topo(self)


        for node in reversed(topo):
            node._backward()
        

