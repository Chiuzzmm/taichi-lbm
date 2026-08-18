import taichi as ti
import taichi.math as tm


@ti.data_oriented
class Pseudopotential():
    def __init__(self,params: dict):
        default_params={
            'a':None,
            'b':None,
            'R':1.0,
            'G':1.0,
        }
        params = {**default_params, **params}
        self.a=params['a']
        self.b=params['b']
        self.R=params['R']
        self.G=params['G']
        self.rho_cr=params['rho_cr']
        self.pressure_cr=params['pressure_cr']
        self.temperature_cr=params['temperature_cr']
        self.show_info()

    @ti.func
    def EOS(self):
        pass

    @ti.func
    def get_psi(self,dens,T):
        p_EOS=self.EOS(dens,T)
        psi=tm.sqrt(abs(6./self.G*(p_EOS-dens/3.0)))
        return psi

    def show_info(self):
        print("="*20)
        print("EOS_pars")
        print(f"  a: {self.a}, b={self.b}")
        print(f"  rho_cr: {self.rho_cr}, p_cr={self.pressure_cr},T_cr={self.temperature_cr}")

@ti.data_oriented
class SC_psi():
    def __init__(self,rho0):
        self.rho0=rho0

    @ti.func
    def get_psi(self,dens,T):
        return self.rho0*(1.0-tm.exp(-dens/self.rho0))


@ti.data_oriented
class vdW_psi(Pseudopotential):
    def __init__(self,params: dict):
        super().__init__(params)

    @ti.func
    def EOS(self,dens,T):
        p=dens*self.R*T/(1.0-self.b*dens)-self.a*dens**2.
        return p


@ti.data_oriented
class RK_psi(Pseudopotential):
    def __init__(self,params: dict):
        super().__init__(params)

    @ti.func
    def EOS(self,dens,T):
        p=dens*self.R*T/(1.0-self.b*dens)-self.a*dens**2./tm.sqrt(T)/(1.+self.b*dens)
        return p


@ti.data_oriented
class PR_psi(Pseudopotential):
    def __init__(self, params:dict):
        super().__init__(params)
        self.omega=params['omega']

    @ti.func
    def EOS(self,dens,T):
        alpha=(1+(0.37464+1.54226*self.omega-0.269922*self.omega**2)*(1-tm.sqrt(T/self.temperature_cr)))**2
        p=dens*self.R*T/(1.0-self.b*dens)-self.a*alpha*dens**2/(1.0+2.0*self.b*dens-(self.b*dens)**2)
        return p


@ti.data_oriented
class CS_psi(Pseudopotential):
    def __init__(self, params:dict):
        super().__init__(params)

    @ti.func
    def EOS(self,dens,T):
        brho4=self.b*dens/4.
        aaa=1.+brho4+brho4**2-brho4**3
        bbb=(1-brho4)**3
        p=dens*self.R*T*aaa/bbb-self.a*dens**2
        return p