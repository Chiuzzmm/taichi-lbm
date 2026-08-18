
import taichi as ti


@ti.data_oriented
class LBField:
    def __init__(self,name, NX, NY,num_components):
        self.name=name
        self.NX=NX
        self.NY=NY
        self.num_components=ti.field(int,shape=())
        self.num_components[None]=num_components

        print("="*20)
        print("init LBField")
        print(f"  NX (网格数): {self.NX}")
        print(f"  NY (网格数): {self.NY}")
        print(f"  组分数量 : {self.num_components[None]}")


        # 分布函数
        self.f = ti.Vector.field(9, float, shape=(NX, NY,num_components)) #populations (old)
        self.f2 = ti.Vector.field(9, float, shape=(NX, NY,num_components)) #populations (new)
        self.SCforce=ti.Vector.field(2, float, shape=(NX, NY,num_components))
        

        # 宏观量
        self.rho = ti.field(float, shape=(NX, NY,num_components)) #density of components
        self.pressure=ti.field(float, shape=(NX, NY,num_components))  #pressure
        self.body_force=ti.Vector.field(2,float,shape=(NX, NY,num_components)) #force populations
        self.shear_rate=ti.field(float, shape=(NX, NY,num_components)) 

        self.vel = ti.Vector.field(2, float, shape=(NX, NY)) # fluid velocity
        self.total_rho=ti.field(float, shape=(NX, NY)) # density
        self.total_pressure=ti.field(float, shape=(NX, NY))
        self.T = ti.field( float, shape=(NX, NY)) #temperature

        

        self.time = ti.field(ti.i32, shape=())

        # D2Q9模型参数
        self.NPOP = 9 
        self.weights = ti.types.vector(9, float)(4, 1, 1, 1, 1, 1 / 4, 1 / 4, 1 / 4, 1 / 4) / 9.0 
        self.c = ti.types.matrix(9, 2, int)([0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1])
        
        #boundary info
        self.gravity_force=ti.Vector([0.0,0.0]) #gravity

        self.neighbor=ti.Matrix.field(n=9,m=2,dtype=int,shape=(self.NX,self.NY))

        self.mask=ti.field(int,shape=(self.NX,self.NY)) # 掩码：1-流体, -1-固体

        #多物理场耦合
        self.sc_field=None

        #unit conversion
        self.Ct=1.0 #time
        self.Cl=1.0 #length
        self.C_rho=1.0 #density
        self.Cu=1.0 #velocity
        self.C_pressure=1.0
        self.C_force=1.0 # force conversion
        self.C_torque=1.0 
        self.C_temperature=1.0 # temperature conversion

        self.Cnu=1.0 #Kinematic viscosity 
        self.shear_viscosity_LB=ti.field(float, shape=(num_components,)) 
        self.bulk_viscosity_LB=ti.field(float, shape=(num_components,)) 

        self.C_T=1.0


        self.mask.fill(1)
        self.SCforce.fill([0.0,0.0])

    def init_conversion(self,params: dict):
        default_params={
            'Cl':None,
            'Ct':None,
            'C_rho':None,
            'C_pressure':None,
            'C_temperature':None
        }
        params = {**default_params, **params}
        
        self.Ct=params['Ct']
        self.Cl=params['Cl']
        self.Cu=self.Cl/self.Ct
        self.Cnu=self.Cl**2/self.Ct
        self.C_rho=params['C_rho']
        self.C_pressure=params['C_pressure']
        self.C_temperature=params['C_temperature']

        self.C_force=self.Cl**3*self.C_rho/self.Ct**2 
        self.C_torque=self.C_force*self.Cl
        
        

        print("="*20)
        print("init conversion")
        print(f"  Cl : {self.Cl}")
        print(f"  Ct : {self.Ct}")
        print(f"  C_rho : {self.C_rho}")
        
    def set_gravity(self,g):
        self.gravity_force=g

    def get_Cforce(self):
        return self.C_force
    
    def get_Ctorque(self):
        return self.C_torque
    
    def get_Cu(self):
        return self.Cu
    

    @ti.kernel
    def init_LBM(self,collsion:ti.template(),group:ti.template()):
        for m in range(group.count[None]):
            i,j=group.group[m]
            for component in range(self.num_components[None]):
                self.f[i,j,component]=self.f2[i,j,component]=collsion.f_eq(self.c,self.weights,self.vel[i,j],self.rho[i,j,component])
        print("init LBM")

    @ti.kernel
    def neighbor_classify(self):
        #group identify
        for ix,iy in self.mask:
            if self.mask[ix,iy]!=-1:
                for k in ti.static(range(self.NPOP)):
                    ix2=ix-self.c[k,0]
                    iy2=iy-self.c[k,1]
                    # if ix2<0 or ix2>self.NX-1 or iy2<0 or iy2>self.NY-1 or self.mask[ix2,iy2]==-1:
                    if 0 <= ix2 < self.NX and 0 <= iy2 < self.NY and self.mask[ix2, iy2] != -1:
                        self.neighbor[ix,iy][k,0]=ix2
                        self.neighbor[ix,iy][k,1]=iy2
                        
                    else:
                        self.neighbor[ix,iy][k,0]=-1
                        self.neighbor[ix,iy][k,1]=-1


@ti.data_oriented
class MacroscopicEngine:
    def __init__(self,group):
        self.group=group

    def time_updata(self,lb_field:ti.template()):
        lb_field.time[None]+=1

    @ti.kernel
    def density(self,lb_field:ti.template()):
        lb_field.total_rho.fill(.0)
        lb_field.rho.fill(.0)

        for m in range(self.group.count[None]):
            i,j=self.group.group[m]
            for component in  range(lb_field.num_components[None]):
                # compute the density and uncorrected velocity
                for k in ti.static(range(lb_field.NPOP)):
                    lb_field.rho[i,j,component]+=lb_field.f[i,j,component][k]
                lb_field.total_rho[i,j]+=lb_field.rho[i,j,component]

    @ti.kernel
    def pressure0(self,lb_field:ti.template()):
        lb_field.pressure.fill(.0)
        for m in range(self.group.count[None]):
            i,j=self.group.group[m]
            lb_field.pressure[i, j,0]+= (lb_field.rho[i, j,0])/3.0

            lb_field.total_pressure[i, j]=lb_field.pressure[ i, j,0] 

    @ti.kernel
    def pressure1(self,lb_field:ti.template()):
        lb_field.pressure.fill(.0)
        for m in range(self.group.count[None]):
            i,j=self.group.group[m]
            lb_field.pressure[i, j,0] += (lb_field.rho[i, j,0] + 0.5 * lb_field.sc_field.g_coh * lb_field.sc_field.psi_field[i,j,0] ** 2) / 3.0

            lb_field.total_pressure[i, j]=lb_field.pressure[i, j,0]


    @ti.kernel
    def pressure2(self,lb_field:ti.template()):
        lb_field.pressure.fill(.0)
        for m in range(self.group.count[None]):
            i,j=self.group.group[m]

            for component1 in  range(lb_field.num_components[None]):
                lb_field.pressure[i, j,component1] += (lb_field.rho[i, j,component1] + 0.5 * lb_field.sc_field.g_coh[component1, component1] * lb_field.sc_field.psi_field[i,j,component1] ** 2) / 3.0

            p_ij=0.0
            p_ii=0.0
            for component1 in  range(lb_field.num_components[None]):
                p_ii+=(lb_field.pressure[i,j,component1])
                for component2 in range(component1 + 1, lb_field.num_components[None]):
                    if component1!=component2:
                        p_ij += lb_field.sc_field.g_coh[component1, component2] * lb_field.sc_field.psi_field[i,j,component1] * lb_field.sc_field.psi_field[i,j,component2]
                
            lb_field.total_pressure[i, j] = p_ii + p_ij / 6.0 


    def pressure(self,lb_field:ti.template()):
        if lb_field.sc_field==None:
            self.pressure0(lb_field)
        else:
            if lb_field.num_components[None]==1:
                self.pressure1(lb_field)
            else:
                self.pressure2(lb_field)
        

    @ti.kernel
    def force_density(self,lb_field:ti.template()):
        for m in range(self.group.count[None]):
            i,j=self.group.group[m]
            for component in  range(lb_field.num_components[None]):
                lb_field.body_force[i,j,component]=(lb_field.SCforce[i,j,component]+lb_field.rho[i,j,component]/lb_field.total_rho[i,j]*lb_field.gravity_force)


    @ti.kernel
    def velocity(self,lb_field:ti.template()):
        lb_field.vel.fill([.0,.0])
        for m in range(self.group.count[None]):
            i, j = self.group.group[m]
            for component in  range(lb_field.num_components[None]):
                vel_temp = ti.Vector([0.0, 0.0])
                for k in ti.static(range(lb_field.NPOP)):
                    vel_temp.x += lb_field.f[i, j,component][k] * lb_field.c[k, 0]
                    vel_temp.y += lb_field.f[i, j,component][k] * lb_field.c[k, 1]
                
                vel_temp+=0.5 * lb_field.body_force[i,j,component]
                lb_field.vel[i,j]+=vel_temp
            
            lb_field.vel[i, j] /= lb_field.total_rho[i, j]
