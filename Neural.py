import numpy as np
class Parameter() :
    def __init__(self,shape) :
        self.shape=shape
        np.random.seed(42)
        self.grad=np.zeros(self.shape)
        self.size=self.grad.size
        self.data=np.random.randn(self.size).reshape(self.shape)

class Generic() : 
    def __init__(self) :
        self.list_params=[] # Liste des paramètres
        self.save=None # Objet pour sauver des infos dans le forward
    def forward(self,x) : 
        # calcul du forward, x=x_s est le vecteur des donnees d'entrees
        # Cette fonction rend  y=x_{s+1} qui est le vecteur des donnees de sortie
        self.save=np.copy(x)
        y=None
        return y
    def backward(self,hat_y) :  
        # retropropagation du gradient sur la couche, 
        #hat_y est le vecteur du gradient de x_{s+1}
        #Cette fonction rend :
        #hat_x, le vecteur du gradient de x_s
        #hat_p, le vecteur du gradient par rapport aux paramètres, ils sont rangés dans p.grad pour tout p dans list_params
        hat_x=None
        return hat_x


class Arctan(Generic) :
    # y = arctan(x), sans paramètre
    def __init__(self) :
        super().__init__()
    def forward(self,x) :
        self.save=np.copy(x)
        return np.arctan(x)
    def backward(self,hat_y) :
        if self.save is None :   # pas encore de forward : rien à rétropropager
            return None
        return hat_y/(1.+self.save**2)


class RELU(Generic) :
    # y = max(x,0), sans paramètre
    def __init__(self) :
        super().__init__()
    def forward(self,x) :
        self.save=np.copy(x)
        return np.maximum(x,0)
    def backward(self,hat_y) :
        return hat_y*(self.save>0)


class ABS(Generic) :
    # y = |x|, sans paramètre
    def __init__(self) :
        super().__init__()
    def forward(self,x) :
        self.save=np.copy(x)
        return np.abs(x)
    def backward(self,hat_y) :
        return hat_y*np.sign(self.save)


class Sigmoid(Generic) :
    # Fonction d'activation x -> 1/(1+exp(-x)), sans paramètre
    def __init__(self) :
        super().__init__()
    def forward(self,x) :
        self.save=np.copy(x)
        return 1./(1.+np.exp(-x))
    def backward(self,hat_y) :
        s=1./(1.+np.exp(-self.save))
        return hat_y*s*(1.-s)


class Dense(Generic) :
    # Couche affine y = W x + b, avec x de forme (n_in, n_samples)
    def __init__(self,n_in,n_out) :
        super().__init__()
        self.W=Parameter((n_out,n_in))
        self.b=Parameter((n_out,1))
        self.list_params=[self.W,self.b]
    def forward(self,x) :
        self.save=np.copy(x)
        return self.W.data@x+self.b.data
    def backward(self,hat_y) :
        self.W.grad=hat_y@self.save.T
        self.b.grad=np.sum(hat_y,axis=1,keepdims=True)
        return self.W.data.T@hat_y


class Loss_L2(Generic) :
    # Perte J(x) = 1/2 ||x - y||^2 où y sont les données cibles
    def __init__(self,y) :
        super().__init__()
        self.y=y
    def forward(self,x) :
        self.save=np.copy(x)
        return 0.5*np.sum((x-self.y)**2)
    def backward(self,hat_y=1.) :
        return hat_y*(self.save-self.y)


class Sequential(Generic) :
    # Enchaînement de couches (éventuellement elles-mêmes Sequential)
    def __init__(self,list_layers) :
        super().__init__()
        self.list_layers=list_layers
        self.list_params=[p for L in list_layers for p in L.list_params]
    def forward(self,x) :
        z=np.copy(x)
        for L in self.list_layers :
            z=L.forward(z)
        return z
    def backward(self,hat_y) :
        for L in reversed(self.list_layers) :
            hat_y=L.backward(hat_y)
        return hat_y
