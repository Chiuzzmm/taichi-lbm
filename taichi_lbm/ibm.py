import taichi as ti
import taichi.math as tm
import numpy as np
import math


@ti.dataclass
class IBMNodes:
    pos: ti.math.vec2
    vel: ti.math.vec2
    vel_desired: ti.math.vec2
    force: ti.math.vec2
    force_temp: ti.math.vec2
    mask_insiderDomain:int

@ti.dataclass
class Sphere():
    id: int
    pos: ti.math.vec2
    vel: ti.math.vec2
    force: ti.math.vec2
    torque: float
    radius: float
    angle: float
    NumIB:int


@ti.data_oriented
class SphereIB():
    def __init__(self,Max_Number,Max_NumIB):
        self.Max_Number=Max_Number
        self.num=ti.field(int,shape=())
        self.Sphere=Sphere.field(shape=(Max_Number))

        self.Max_NumIB=Max_NumIB
        self.NUMIB_cof=1
        self.IBNodes=IBMNodes.field(shape=(Max_Number,Max_NumIB))

        self.IBNodeInsiderGroup=ti.Vector.field(2,int,shape=(Max_Number*Max_NumIB))
        self.IBInsiderCount=ti.field(int,shape=())


    def init_Balls(self, num, id_np, pos_np, vel_np,force_np,torque_np, radius_np, angle_np):
        self.num[None] = num
        for  m in range(self.num[None]):
            self.Sphere[m].id = id_np[m]
            self.Sphere[m].pos = pos_np[m]
            self.Sphere[m].vel = vel_np[m]
            self.Sphere[m].force = force_np[m]
            self.Sphere[m].torque = torque_np[m]
            self.Sphere[m].radius = radius_np[m]
            self.Sphere[m].angle = angle_np[m]
        print(f"init {num} balls")



    @ti.kernel
    def init_IB_nodes(self):
        num=0
        for m in range(self.num[None]):
            self.Sphere[m].NumIB= int(ti.round(tm.pi * 2 * self.Sphere[m].radius / self.NUMIB_cof))
            for n in range(self.Sphere[m].NumIB):
                angle = 2.0 * math.pi * n / self.Sphere[m].NumIB
                self.IBNodes[m,n].pos.x=self.Sphere[m].pos.x+self.Sphere[m].radius*tm.cos(angle)
                self.IBNodes[m,n].pos.y=self.Sphere[m].pos.y+self.Sphere[m].radius*tm.sin(angle)
            num+=self.Sphere[m].NumIB
        print(f"init {self.num[None]} balls with {num} IB nodes")

    @ti.func
    def computer_IB_nodes_pos(self,m,n):
        pos=ti.Vector([0.0,0.0])
        angle = 2.0 * math.pi * n / self.Sphere[m].NumIB
        xref=self.Sphere[m].radius*tm.cos(angle) 
        yref=self.Sphere[m].radius*tm.sin(angle)

        pos[0]=self.Sphere[m].pos.x+xref*tm.cos(self.Sphere[m].angle)-yref*tm.sin(self.Sphere[m].angle)
        pos[1]=self.Sphere[m].pos.y+xref*tm.sin(self.Sphere[m].angle)+yref*tm.cos(self.Sphere[m].angle)
        return pos
    
    @ti.kernel
    def reset_force(self):
        for m,n in self.IBNodes:
            self.IBNodes[m,n].force=ti.Vector([0.0,0.0])
    

@ti.data_oriented
class IBDEMCouplerEngine():
    def __init__(self,lb_field:ti.template()):
        self.NX=lb_field.NX
        self.NY=lb_field.NY

    @ti.func
    def CheckPos(self, pos, diameter):
        posflag = 2  #cross
        x_min = pos.x - diameter
        x_max = pos.x + diameter
        y_min = pos.y - diameter
        y_max = pos.y + diameter
        nx = self.NX - 1
        ny = self.NY - 1

        #inside
        if x_min >= 0 and x_max <= nx and y_min >= 0 and y_max <= ny:
            posflag = 1
        #outside
        elif x_max < 0 or x_min > nx or y_max < 0 or y_min > ny:
            posflag = 0
        return posflag


    @ti.func
    def InDomain(self,pos):
        flag=0
        if pos.x>0 and pos.x<self.NX-1 and pos.y>0 and pos.y<self.NY-1:
            flag=1
        return flag
    


    @ti.kernel
    def SphereIBUpdate(self,sphere_field:ti.template()):

        sphere_field.IBNodeInsiderGroup.fill([0,0])
        sphere_field.IBInsiderCount[None]=0
        ballNum=sphere_field.num[None]
        for m in range(ballNum):
            posflag=self.CheckPos(sphere_field.Sphere[m].pos,2*sphere_field.Sphere[m].radius)
            if posflag==1:#inside
                for n in range(sphere_field.Sphere[m].NumIB):
                    sphere_field.IBNodes[m,n].pos=sphere_field.computer_IB_nodes_pos(m,n)
                    sphere_field.IBNodes[m,n].vel_desired=sphere_field.Sphere[m].vel
                    idx=ti.atomic_add(sphere_field.IBInsiderCount[None],1)
                    sphere_field.IBNodeInsiderGroup[idx]=ti.Vector([m,n])

            elif posflag==2:#cross
                for n in range(sphere_field.Sphere[m].NumIB):
                    sphere_field.IBNodes[m,n].pos=sphere_field.computer_IB_nodes_pos(m,n)
                    sphere_field.IBNodes[m,n].vel_desired=sphere_field.Sphere[m].vel
                    if self.InDomain(sphere_field.IBNodes[m,n].pos):
                        idx=ti.atomic_add(sphere_field.IBInsiderCount[None],1)
                        sphere_field.IBNodeInsiderGroup[idx]=ti.Vector([m,n])
                
    def SphereInfoUpdata(self,sphere_field:ti.template(),pos_np,vel_np,angle_np):
        ballNum=sphere_field.num[None]
        for m in range(ballNum):
            sphere_field.Sphere[m].pos=pos_np[m]
            sphere_field.Sphere[m].vel=vel_np[m]
            sphere_field.Sphere[m].angle=angle_np[m]

        self.SphereIBUpdate(sphere_field)

    def GetSphereForce(self,sphere_field:ti.template()):
        force_data = np.array(sphere_field.Sphere.force.to_numpy() ,dtype=np.float64)
        force_matrix = force_data[:sphere_field.num[None], :]
        return force_matrix

    def GetSphereTorque(self,sphere_field:ti.template()):
        torque_data = np.array(sphere_field.Sphere.torque.to_numpy(),dtype=np.float64)
        num_spheres = sphere_field.num[None]
        torque_matrix = torque_data[:num_spheres]
        return torque_matrix




@ti.data_oriented
class IBEngine():
    def __init__(self,MDFIteration):
        self.MDFIteration=MDFIteration

    @ti.kernel#IB solve
    def ComputeParticleForce(self,sphere_field:ti.template()):
        IBInsiderCount=sphere_field.IBInsiderCount[None]
        for idx in range(IBInsiderCount):
            m,n=sphere_field.IBNodeInsiderGroup[idx]
            #reset force
            sphere_field.Sphere[m].torque=0.0
            sphere_field.Sphere[m].force=ti.Vector([0.0,0.0])

            fx,fy=sphere_field.IBNodes[m,n].force
            rx,ry=sphere_field.IBNodes[m,n].pos-sphere_field.Sphere[m].pos

            sphere_field.Sphere[m].torque+=fx*ry-fy*rx
            sphere_field.Sphere[m].force.x+=fx
            sphere_field.Sphere[m].force.y+=fy

    @ti.func
    def BottomAndTop(self,pos):
        posArray=ti.Vector([0,0,0,0])
        posArray[0]=ti.cast(tm.floor(pos.x),int)
        posArray[1]=ti.cast(tm.floor(pos.y),int)
        posArray[2]=ti.cast(tm.ceil(pos.x),int)
        posArray[3]=ti.cast(tm.ceil(pos.y),int)
        return posArray
    
    @ti.func
    def weight(self,pos,iX,iY):
        return (1.0-ti.abs(pos.x-iX))*(1.0-ti.abs(pos.y-iY))
    

    @ti.kernel#IB solve
    def InterpolateBodyVelocities(self,sphere_field:ti.template(),lb_filed:ti.template()):
        IBInsiderCount=sphere_field.IBInsiderCount[None]
        for idx in range(IBInsiderCount):
            m,n=sphere_field.IBNodeInsiderGroup[idx]
            #reset vel
            sphere_field.IBNodes[m,n].vel=ti.Vector([0.0,0.0])

            # Identify the lowest fluid lattice node in interpolation range (see spreading).
            xbottom,ybottom,xtop,ytop=self.BottomAndTop(sphere_field.IBNodes[m,n].pos)

            for iX, iY in ti.ndrange((xbottom, xtop + 1), (ybottom, ytop + 1)): 
                # Compute distance between object node and fluid lattice node.
			    # Compute interpolation weights for x- and y-direction based on the distance.
                weight=self.weight(sphere_field.IBNodes[m,n].pos,iX,iY)

                # Compute node velocities.
                sphere_field.IBNodes[m,n].vel+=lb_filed.vel[iX,iY]*weight

            # Compute the Lagrangian correction force
            ForceCorrection=2*(sphere_field.IBNodes[m,n].vel_desired-sphere_field.IBNodes[m,n].vel)
            sphere_field.IBNodes[m,n].force_temp=ForceCorrection
            sphere_field.IBNodes[m,n].force+=ForceCorrection

    @ti.kernel#IB solve
    def SpreadBodyForces(self,sphere_field:ti.template(),lb_filed:ti.template()):
        #spread forces
        IBInsiderCount=sphere_field.IBInsiderCount[None]
        for idx in range(IBInsiderCount):
            m,n=sphere_field.IBNodeInsiderGroup[idx]

            xbottom,ybottom,xtop,ytop=self.BottomAndTop(sphere_field.IBNodes[m,n].pos)

            for iX, iY in ti.ndrange((xbottom, xtop + 1), (ybottom, ytop + 1)): 
                # Compute interpolation weights for x- and y-direction based on the distance.
                weight=self.weight(sphere_field.IBNodes[m,n].pos,iX,iY)
                #Compute lattice force.
                f_temp=ti.Vector([0.0,0.0])
                f_temp+=sphere_field.IBNodes[m,n].force_temp*weight
                #Correct previous Eulerian velocity
                r2=lb_filed.total_rho[iX,iY]*2.0
                lb_filed.vel[iX,iY]+=f_temp/r2
                for component in  range(lb_filed.num_components[None]):
                    lb_filed.body_force[iX,iY,component]+=(f_temp*lb_filed.rho[iX,iY,component]/lb_filed.total_rho[iX,iY])



    def MultiDirectForcingLoop(self,sphere_field:ti.template(),lb_filed:ti.template()):
        #reset force
        sphere_field.reset_force()
        for m in range(self.MDFIteration):
            self.InterpolateBodyVelocities(sphere_field=sphere_field,lb_filed=lb_filed)
            self.SpreadBodyForces(sphere_field=sphere_field,lb_filed=lb_filed)
