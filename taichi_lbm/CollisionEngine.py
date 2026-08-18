import taichi as ti
import numpy as np
import taichi.math as tm


class Collision:
    def __init__(self,group,fluid_model):
        self.group=group
        self.fluid_model=fluid_model
        self.tau_min=0.5000001
        self.tau_max=4.0


    @ti.func
    def f_eq(self,c,w,vel,rho):
        eu=c @ vel
        uv=tm.dot(vel,vel)
        return w*rho*(1+3*eu+4.5*eu*eu-1.5*uv)

    @ti.func
    def f_force(self,c,w,F,vel):
        cF=c @ F
        cu=c @vel
        uF=tm.dot(vel,F)
        return w*(3*cF+9*cF*cu-3*uF)
    

    @ti.func
    def tau_max_min(self,tau):
        if tau<self.tau_min:
            tau=self.tau_min
        elif tau>self.tau_max:
            tau=self.tau_max
        return tau

@ti.data_oriented
class BGKCollision(Collision):
    def __init__(self,params:dict):
        default_params={
            'group':None,
            'fluid_model':None,
            'NX':None,
            'NY':None,
            'num_components':None,
            'shearviscosity':None
        }
        params = {**default_params, **params}
        
        group=params['group']
        fluid_model=params['fluid_model']
        NX=params['NX']
        NY=params['NY']
        num_components=params['num_components']
        shearviscosity=params['shearviscosity']

        super().__init__(group,fluid_model)
        self.tau=ti.field(float, shape=(NX, NY,num_components)) # relaxation time
        self.nu=ti.field(float, shape=(num_components)) 

        for c in range(num_components):
            self.nu[c]=shearviscosity[c]

        self.initTau()

    @ti.kernel
    def initTau(self):
        for idx in range(self.group.count[None]):
            i,j = self.group.group[idx]
            for component in range(self.nu.shape[0]):  
                self.tau[i,j,component]=self.nu[component]*3.+0.5

    @ti.kernel
    def relaxation_time(self,lb_field:ti.template()):
        for idx in range(self.group.count[None]):
            i,j = self.group.group[idx]
            for component in range(lb_field.num_components[None]):  
                tau=self.fluid_model.apparent_viscosity(lb_field.shear_rate[i,j,component],lb_field.rho[i,j,component])*3.+0.5
                self.tau[i,j,component]=self.tau_max_min(tau)

    @ti.kernel
    def shear_rate(self,lb_field:ti.template()):
        for idx in range(self.group.count[None]):
            i,j = self.group.group[idx]
            for component in range(lb_field.num_components[None]):
                fneq=lb_field.f[i,j,component]-self.f_eq(lb_field.c,lb_field.weights,lb_field.vel[i,j],lb_field.rho[i,j,component])
                cof=-1.5/lb_field.rho[i,j,component]/self.tau[i,j,component]

                Sxx=(fneq[1]+fneq[3]+fneq[5]+fneq[6]+fneq[7]+fneq[8])*cof
                Syy=(fneq[2]+fneq[4]+fneq[5]+fneq[6]+fneq[7]+fneq[8])*cof
                Sxy=(fneq[5]-fneq[6]+fneq[7]-fneq[8])*cof
                
                lb_field.shear_rate[i,j,component]=tm.sqrt(2*(Sxx**2+Syy**2+2*Sxy**2))
      


    @ti.kernel#LBM solve
    def apply(self,lb_field:ti.template()):
        for idx in range(self.group.count[None]):
            i,j = self.group.group[idx]
            for component in range(lb_field.num_components[None]):  
                omega=1./self.tau[i,j,component]

                feqeq=self.f_eq(lb_field.c,lb_field.weights,lb_field.vel[i,j],lb_field.rho[i,j,component])
                force_ij=self.f_force(lb_field.c,lb_field.weights,lb_field.body_force[i,j,component],lb_field.vel[i,j])

                collision_operator =-omega*(lb_field.f[i,j,component]-feqeq)
                
                force_term=(1.0-0.5*omega)*force_ij

                lb_field.f2[i,j,component]=lb_field.f[i,j,component]+collision_operator+force_term



@ti.data_oriented
class TRTCollision(Collision):
    def __init__(self,params:dict):
        default_params={
            'group':None,
            'fluid_model':None,
            'NX':None,
            'NY':None,
            'num_components':None,
            'shearviscosity':None,
            'magic':None
        }
        params = {**default_params, **params}
        
        group=params['group']
        fluid_model=params['fluid_model']
        NX=params['NX']
        NY=params['NY']
        num_components=params['num_components']
        shearviscosity=params['shearviscosity']
        magic=params['magic']

    
        super().__init__(group,fluid_model)
        self.magic=ti.field(float, shape=(num_components,)) 
        self.nu=ti.field(float, shape=(num_components)) 
        self.tau_sym=ti.field(float, shape=(NX, NY,num_components)) 
        self.tau_antisym=ti.field(float, shape=(NX, NY,num_components)) 

        for component in range(num_components):
            self.magic[component]=magic[component]
            self.nu[component]=shearviscosity[component]

        self.initTau()

    @ti.kernel
    def initTau(self):
        for idx in range(self.group.count[None]):
            i,j = self.group.group[idx]
            for component in range(self.nu.shape[0]):  
                self.tau_sym[i,j,component]=self.nu[component]*3.+0.5
                self.tau_antisym[i,j,component]=self.magic[component]/(self.tau_sym[i,j,component]-0.5)+0.5

    @ti.kernel
    def relaxation_time(self,lb_field:ti.template()):
        for idx in range(self.group.count[None]):
            i,j = self.group.group[idx]
            for component in range(lb_field.num_components[None]):  
                tau=self.fluid_model.apparent_viscosity(lb_field.shear_rate[i,j,component],lb_field.rho[i,j,component])*3.+0.5
                self.tau_sym[i,j,component]=self.tau_max_min(tau)
                self.tau_antisym[i,j,component]=self.magic[component]/(self.tau_sym[i,j,component]-0.5)+0.5

    @ti.kernel
    def shear_rate(self,lb_field:ti.template()):
        for idx in range(self.group.count[None]):
            i,j = self.group.group[idx]
            for component in range(lb_field.num_components[None]):
                fneq=lb_field.f[i,j,component]-self.f_eq(lb_field.c,lb_field.weights,lb_field.vel[i,j],lb_field.rho[i,j,component])
                cof=-1.5/lb_field.rho[i,j,component]/self.tau_sym[i,j,component]

                Sxx=(fneq[1]+fneq[3]+fneq[5]+fneq[6]+fneq[7]+fneq[8])*cof
                Syy=(fneq[2]+fneq[4]+fneq[5]+fneq[6]+fneq[7]+fneq[8])*cof
                Sxy=(fneq[5]-fneq[6]+fneq[7]-fneq[8])*cof
                
                lb_field.shear_rate[i,j,component]=tm.sqrt(2*(Sxx**2+Syy**2+2*Sxy**2))


    @ti.func
    def f_sym(self,f):
        f_sym=ti.Vector([0.0 for _ in range(9)])
        f_sym[0]=f[0]
        f_sym[1]=(f[1]+f[3])/2
        f_sym[2]=(f[2]+f[4])/2
        f_sym[3]=f_sym[1]
        f_sym[4]=f_sym[2]
        f_sym[5]=(f[5]+f[7])/2
        f_sym[6]=(f[6]+f[8])/2
        f_sym[7]=f_sym[5]
        f_sym[8]=f_sym[6]
        return f_sym
    
    @ti.func
    def feq_sym(self,feqeq):
        feq_sym=ti.Vector([0.0 for _ in range(9)])
        feq_sym[0]=feqeq[0]
        feq_sym[1]=(feqeq[1]+feqeq[3])/2
        feq_sym[2]=(feqeq[2]+feqeq[4])/2
        feq_sym[3]=feq_sym[1]
        feq_sym[4]=feq_sym[2]
        feq_sym[5]=(feqeq[5]+feqeq[7])/2
        feq_sym[6]=(feqeq[6]+feqeq[8])/2
        feq_sym[7]=feq_sym[5]
        feq_sym[8]=feq_sym[6]
        return feq_sym
    
    @ti.func
    def force_sym(self,force):
        force_sym=ti.Vector([0.0 for _ in range(9)])
        force_sym[0]=force[0]
        force_sym[1]=(force[1]+force[3])/2
        force_sym[2]=(force[2]+force[4])/2
        force_sym[3]=force_sym[1]
        force_sym[4]=force_sym[2]
        force_sym[5]=(force[5]+force[7])/2
        force_sym[6]=(force[6]+force[8])/2
        force_sym[7]=force_sym[5]
        force_sym[8]=force_sym[6]
        return force_sym
    
    @ti.kernel#LBM solve
    def apply(self,lb_field:ti.template()):
        for idx in range(self.group.count[None]):
            i,j = self.group.group[idx]
            for component in range(lb_field.num_components[None]):
                omega_sym=1./self.tau_sym[i,j,component]
                omega_antisym=1./self.tau_antisym[i,j,component]

                feqeq=self.f_eq(lb_field.c,lb_field.weights,lb_field.vel[i,j],lb_field.rho[i,j,component])
                force_ij=self.f_force(lb_field.c,lb_field.weights,lb_field.body_force[i,j,component],lb_field.vel[i,j])

                #symmetrical and antisymmetrical particle distribution functions
                f_sym=self.f_sym(lb_field.f[i,j,component])
                feq_sym=self.feq_sym(feqeq)
                force_sym=self.force_sym(force_ij)

                f_antisym=lb_field.f[i,j,component]-f_sym
                feq_antisym=feqeq-feq_sym
                force_antisym=force_ij-force_sym

                collision_operator =-omega_sym * (f_sym-feq_sym)-omega_antisym*(f_antisym-feq_antisym)
                force_term=(1.0-0.5*omega_sym)*force_sym+(1.0-0.5*omega_antisym)*force_antisym
                lb_field.f2[i,j,component]=lb_field.f[i,j,component]+collision_operator+force_term

@ti.data_oriented
class MRTCollision(Collision):
    def __init__(self,params:dict):
        default_params={
            'group':None,
            'fluid_model':None,
            'NX':None,
            'NY':None,
            'num_components':None,
            'shearviscosity':None,
            'bulkviscosity':None
        }
        params = {**default_params, **params}
        
        group=params['group']
        fluid_model=params['fluid_model']
        NX=params['NX']
        NY=params['NY']
        num_components=params['num_components']
        shearviscosity=params['shearviscosity']
        bulkviscosity=params['bulkviscosity']

        super().__init__(group,fluid_model)

        self.M=ti.field(float,shape=(9,9))
        self.M_inverse=ti.field(float,shape=(9,9))
        M_np =np.array([
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [-4.0, -1.0, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0, 2.0],
                [4.0, -2.0, -2.0, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0],
                [0.0, -2.0, 0.0, 2.0, 0.0, 1.0, -1.0,-1.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0],
                [0.0, 0.0, -2.0, 0.0, 2.0, 1.0, 1.0, -1.0, -1.0],
                [0.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0]])
        M_inv_np=np.linalg.inv(M_np)
        for i in ti.static(range(9)):
            for j in ti.static(range(9)):
                self.M[i,j]=M_np[i,j]
                self.M_inverse[i,j]=M_inv_np[i,j]


        self.nu=ti.field(float, shape=(num_components)) 
        self.nu_bulk=ti.field(float, shape=(num_components)) 
        self.diag=ti.Vector.field(9, float, shape=(NX, NY,num_components))


        for component in range(num_components):
            self.nu[component]=shearviscosity[component]
            self.nu_bulk[component]=bulkviscosity[component]

        self.initTau()

    @ti.kernel
    def initTau(self):
        m=ti.Vector([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
        self.diag.fill(m)
        for idx in range(self.group.count[None]):
            i,j = self.group.group[idx]
            for component in range(self.nu.shape[0]):  
                tau=self.nu[component]*3.+0.5
                tau=self.tau_max_min(tau)
                self.diag[i,j,component][7]=self.diag[i,j,component][8]=1./tau

                tau_bulk=self.nu_bulk[component]*3.+0.5
                tau_bulk=self.tau_max_min(tau_bulk)
                self.diag[i,j,component][1]=1./tau_bulk

                self.diag[i,j,component][0] = 0.0  # 质量
                self.diag[i,j,component][3] = 0.0  # x动量
                self.diag[i,j,component][5] = 0.0  # y动量



    @ti.kernel
    def relaxation_time(self,lb_field:ti.template()):
        for idx in range(self.group.count[None]):
            i,j = self.group.group[idx]
            for component in range(lb_field.num_components[None]):  
                tau=self.fluid_model.apparent_viscosity(lb_field.shear_rate[i,j,component],lb_field.rho[i,j,component])*3.+0.5
                tau=self.tau_max_min(tau)
                self.diag[i,j,component][7]=self.diag[i,j,component][8]=1./tau
   

    @ti.kernel
    def shear_rate(self,lb_field:ti.template()):
        for idx in range(self.group.count[None]):
            i,j = self.group.group[idx]
            for component in range(lb_field.num_components[None]):
                fneq=lb_field.f[i,j,component]-self.f_eq(lb_field.c,lb_field.weights,lb_field.vel[i,j],lb_field.rho[i,j,component])
                
                MSM=ti.Matrix([[0.0 for _ in range(9)] for _ in range(9)])
                mneq=ti.Vector([0.0 for _ in range(9)])
                # for ii in range(9):
                #     for jj in range(9):
                #         for k in range(9):
                #             MSM[ii,jj]+=self.M_inverse[ii,k]*self.M[k,jj]*self.diag[i,j,component][k]


                # for ii in range(9):
                #     for jj in range(9):
                #         mneq[ii]+=MSM[ii,jj]*fneq[jj]
           
                m_neq =self.mf(fneq)
                S_m_neq = ti.Vector([self.diag[i,j,component][k] * m_neq[k] for k in range(9)])
                for ii in range(9):
                    for jj in range(9):
                        mneq[ii]+=self.M_inverse[ii,jj]*S_m_neq[jj]

                

                cof=-1.5/lb_field.rho[i,j,component]
                Sxx=(mneq[1]+mneq[3]+mneq[5]+mneq[6]+mneq[7]+mneq[8])*cof
                Syy=(mneq[2]+mneq[4]+mneq[5]+mneq[6]+mneq[7]+mneq[8])*cof
                Sxy=(mneq[5]-mneq[6]+mneq[7]-mneq[8])*cof

                # m=self.mf(lb_field.f[i,j,component])
                # pxx=m[7]
                # pxy=m[8]
                # pxxneq=pxx-lb_field.rho[i,j,component]*(lb_field.vel[i,j].x**2-lb_field.vel[i,j].y**2)
                # pxyneq=pxy-lb_field.rho[i,j,component]*lb_field.vel[i,j].x*lb_field.vel[i,j].y
                
                # cofxx=-1.5/lb_field.rho[i,j,component]**self.diag[i,j,component][7]
                # cofxy=-1.5/lb_field.rho[i,j,component]**self.diag[i,j,component][8]
                
                # Sxx=cofxx*pxxneq
                # Syy=-Sxx
                # Sxy=cofxy*pxyneq
                
                lb_field.shear_rate[i,j,component]=tm.sqrt(2*(Sxx**2+Syy**2+2*Sxy**2))

    @ti.func
    def mf(self,f):
        m=ti.Vector([0.0 for _ in range(9)])
        for ii in ti.static(range(9)):
            for jj in ti.static(range(9)):
                m[ii]+=self.M[ii,jj]*f[jj]
        return m
    

    @ti.func
    def m_eq(self,vel,rho):
        jx=vel.x
        jy=vel.y

        meq=ti.Vector([0.0 for _ in range(9)])

        meq[0]=1.0
        meq[1]=-2.0+3.0*(jx**2+jy**2)
        meq[2]=1.0-3.0*(jx**2+jy**2)
        meq[3]=jx
        meq[4]=-jx
        meq[5]=jy
        meq[6]=-jy
        meq[7]=(jx**2-jy**2)
        meq[8]=jx*jy

        meq*=rho
        return meq

    @ti.func
    def m_force(self,F,vel):
        mforce=ti.Vector([0.0 for _ in range(9)])
        Fu=tm.dot(F,vel)

        mforce[1]=6*Fu
        mforce[2]=-6*Fu
        mforce[3]=F[0]
        mforce[4]=-F[0]
        mforce[5]=F[1]
        mforce[6]=-F[1]
        mforce[7]=2*(F[0]*vel.x-F[1]*vel.y)
        mforce[8]=F[1]*vel.x+F[0]*vel.y

        return mforce
    
    @ti.kernel#LBM solve
    def apply(self,lb_field:ti.template()):
        
        for idx in range(self.group.count[None]):
            i,j = self.group.group[idx]
            for component in range(lb_field.num_components[None]):
                
                a=ti.Vector([0.0 for _ in range(9)])
                b=ti.Vector([0.0 for _ in range(9)])
                fpop2=ti.Vector([0.0 for _ in range(9)])

                #Transform to moment space
                m=self.mf(lb_field.f[i,j,component])
                meq=self.m_eq(lb_field.vel[i,j],lb_field.rho[i,j,component]) #Compute equilibrium moments
                mf=self.m_force(lb_field.body_force[i,j,component],lb_field.vel[i,j]) #Guo Forcing

                for ii in ti.static(range(9)):
                    a[ii]=self.diag[i,j,component][ii]*(m[ii]-meq[ii])
                    b[ii]=(1.0-self.diag[i,j,component][ii]/2.0)*(mf[ii])

                m2=m-a+b#Collide

                #Transform to population space
                for ii in ti.static(range(9)):
                    for jj in ti.static(range(9)):
                        fpop2[ii]+=self.M_inverse[ii,jj]*m2[jj]

                lb_field.f2[i,j,component]=fpop2

@ti.data_oriented
class HuangMRTCollision(MRTCollision):
    def __init__(self,num_components,group,k1,epslion,g_coh):
        super().__init__(num_components,group)
        self.k1=ti.field(float, shape=(num_components,)) 
        self.k2=ti.field(float, shape=(num_components,)) 
        self.g_coh=ti.field(float,shape=(num_components,))

        for component in range(self.num_components[None]):
            self.k1[component]=k1[component]
            self.k2[component]=epslion[component]/(-8.)-k1[component]
            self.g_coh[component]=g_coh[component,component]

    @ti.func
    def m_Q(self,F,psi,component,g_coh):
        Qm=ti.Vector([0.0 for _ in range(9)])
        F2=tm.dot(F,F)
        a=3*(self.k1[component]+2*self.k2[component])*F2
        b=g_coh*psi**2
        
        Qm[1]=a/b
        Qm[2]=-Qm[1]
        Qm[7]=self.k1[component]*(F[0]**2-F[1]**2)/b
        Qm[8]=self.k1[component]*F[0]*F[1]/b

        return Qm
    
    @ti.kernel
    def apply(self,lb_field:ti.template(),sc_field:ti.template()):
        for idx in range(self.group.count[None]):
            i,j = self.group.group[idx]

            for component in range(lb_field.num_components[None]):
                m=ti.Vector([0.0 for _ in range(9)])
                a=ti.Vector([0.0 for _ in range(9)])
                b=ti.Vector([0.0 for _ in range(9)])
                c=ti.Vector([0.0 for _ in range(9)])
                fpop2=ti.Vector([0.0 for _ in range(9)])

                #Transform to moment space
                for ii in ti.static(range(9)):
                    for jj in ti.static(range(9)):
                        ti.atomic_add(m[ii],self.M[ii,jj]*lb_field.f[i,j,component][jj])

                meq=self.m_eq(lb_field.vel[i,j],lb_field.rho[i,j,component]) #Compute equilibrium moments
                mf=self.m_force(lb_field.body_force[i,j,component],lb_field.vel[i,j]) #Guo Forcing
                
                psi=sc_field.psi_field[i,j,component]
                mQ=self.m_Q(lb_field.body_force[i,j,component],psi,component,self.g_coh[component]) # huang

                for ii in ti.static(range(9)):
                    a[ii]=self.diag[component,ii]*(m[ii]-meq[ii])
                    b[ii]=(1.0-self.diag[component,ii]/2.0)*(mf[ii])
                    c[ii]=self.diag[component,ii]*mQ[ii]

                m2=m-a+b+c#Collide

                #Transform to population space
                for ii in ti.static(range(9)):
                    for jj in ti.static(range(9)):
                        ti.atomic_add(fpop2[ii],self.M_inverse[ii,jj]*m2[jj])

                lb_field.f2[i,j,component]=fpop2
