import taichi as ti

@ti.data_oriented
class Fluid:
    def apparent_viscosity(self):
        pass
    

@ti.data_oriented
class NewtonianFluid(Fluid):
    def __init__(self,nu):
        self.nu=nu

    @ti.func
    def apparent_viscosity(self,shear_rate,rho):
        return self.nu
    


@ti.data_oriented
class PowerLawFluid(Fluid):
    def __init__(self,m,n):
        self.m=m #  flow consistency index
        self.n=n # flow behavior index

    @ti.func
    def apparent_viscosity(self,shear_rate,rho):
        nu=self.m*shear_rate**(self.n-1.)
        return nu