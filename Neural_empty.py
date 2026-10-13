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
