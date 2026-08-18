import taichi as ti



@ti.data_oriented
class ShanChenModel:
    def __init__(self,lb_field:ti.template(),group):
        self.group=group
        self.wc = ti.Vector.field(2, float, shape=(9))
        self.psi_field=ti.field(float,shape=(lb_field.NX+2,lb_field.NY+2,lb_field.num_components[None]),offset=(-1,-1,0))

        for k in range(9):
            c = ti.Vector([lb_field.c[k, 0], lb_field.c[k, 1]])
            self.wc[k] = lb_field.weights[k] * c
                
    @ti.kernel
    def update_psi(self):
        pass


    @ti.kernel
    def cau_SCforce(self):
        pass
    


@ti.data_oriented
class ShanChenForceC1(ShanChenModel):
    def __init__(self,params: dict):
        default_params={
            'lb_field':None,
            'g_coh':None,
            'group':None,
            'psi':None
        }
        params = {**default_params, **params}

        super().__init__(params['lb_field'],params['group'])
        self.g_coh = params['g_coh'] #interaction strength
        self.psi=params['psi']

    @ti.kernel
    def update_psi(self,lb_field:ti.template()):
        for m in range(self.group.count[None]):
            i,j = self.group.group[m]
            for component in range(lb_field.num_components[None]):
                self.psi_field[i,j,component]=self.psi.get_psi(lb_field.rho[i,j,component],lb_field.T[i,j])


    @ti.kernel
    def cau_SCforce(self,lb_field:ti.template()):
        #reset
        lb_field.SCforce.fill([.0,.0])
        for m in range(self.group.count[None]):
            i,j = self.group.group[m]
            F = ti.Vector([0.0, 0.0])
            for k in ti.static(range(1,9)):
                c = ti.Vector([lb_field.c[k,0], lb_field.c[k,1]])
                x2 = i + c.x  # 允许 x2 为 -1 或 NX
                y2 = j + c.y
                F+=self.wc[k]*self.psi_field[x2,y2,0]
            F*=(-self.g_coh*self.psi_field[i,j,0])
            lb_field.SCforce[i,j,0]+=F


@ti.data_oriented
class ShanChenForceC2(ShanChenModel):
    def __init__(self,params: dict):
        default_params={
            'lb_field':None,
            'g_coh':None,
            'gadh':None,
            'group':None,
            'psi_list':None,
            'fluid_strategy':None
        }
        params = {**default_params, **params}

        super().__init__(params['lb_field'],params['group'])
        self.g_coh= params['g_coh']
        self.g_adh= params['gadh']
        self.psi1=params['psi_list'][0]
        self.psi2=params['psi_list'][1]
        self.fluid_strategy=params['fluid_strategy']

    @ti.kernel
    def update_psi(self,lb_field:ti.template()):
        for m in range(self.group.count[None]):
            i,j = self.group.group[m]

            self.psi_field[i,j,0]=self.psi1.get_psi(lb_field.rho[i,j,0],lb_field.T[i,j])
            self.psi_field[i,j,1]=self.psi2.get_psi(lb_field.rho[i,j,1],lb_field.T[i,j])



    @ti.kernel
    def cau_SCforce(self,lb_field:ti.template()): # type: ignore
        #reset
        lb_field.SCforce.fill([.0,.0])
        for m in range(self.group.count[None]):
            i,j = self.group.group[m]
            for component1 in range(lb_field.num_components[None]):
                F_coh = ti.Vector([0.0, 0.0])
                F_adh=ti.Vector([0.0, 0.0])
                for k in ti.static(range(1,9)):
                    c=ti.Vector([lb_field.c[k,0],lb_field.c[k,1]])
                    x2 = i + c.x
                    y2 = j + c.y
                    if self.fluid_strategy(x2,y2):
                        for component2 in range(lb_field.num_components[None]):
                            if component2!=component1:
                                F_coh+=self.wc[k]*self.psi_field[x2,y2,component2]*self.g_coh[component1,component2]
                    else:
                        F_adh+=self.wc[k]*self.g_adh[component1]#*self.psi_field[x2,y2,component1]

                F_total = (-self.psi_field[i,j,component1]) * (F_coh + F_adh)
                lb_field.SCforce[i, j, component1] = F_total