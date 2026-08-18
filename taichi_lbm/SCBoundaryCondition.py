import taichi as ti
from .BoundaryCondition import PlaneBoundary,BoundaryCondition
import taichi.math as tm
from PIL import Image
import numpy as np



@ti.data_oriented
class OpenBoundaryPsi(PlaneBoundary):
    def __init__(self, spec, direction):
        super().__init__(spec, direction)
    
    def apply(self):
        pass



@ti.data_oriented
class OpenNeumannPsi(OpenBoundaryPsi):
    def __init__(self, spec, direction):
        super().__init__(spec, direction)


    @ti.kernel
    def apply(self, lb_field:ti.template()): 
        for m in range(self.group.count[None]):
            i,j = self.group.group[m]

            ix2= i-lb_field.c[self.direction,0]
            iy2= j-lb_field.c[self.direction,1]

            for component in range(lb_field.num_components[None]):
                lb_field.sc_field.psi_field[i,j,component]=lb_field.sc_field.psi_field[ix2,iy2,component]

@ti.data_oriented
class OpenConvective1orderPsi(OpenBoundaryPsi):
    def __init__(self, spec, direction):
        super().__init__(spec, direction)

    @ti.kernel
    def apply(self,lb_field:ti.template()):
        U=ti.Vector([0.,0.])
        for m in range(self.group.count[None]):
            i,j = self.group.group[m]
            ix2= i-lb_field.c[self.direction,0]
            iy2= j-lb_field.c[self.direction,1]
            U+=lb_field.vel[ix2,iy2]

        U/=self.group.count[None]
        Uave=tm.sqrt(tm.dot(U,U))
        lambdda=1.*Uave

        for m in range(self.group.count[None]):
            i,j = self.group.group[m]
            ix2= i-lb_field.c[self.direction,0]
            iy2= j-lb_field.c[self.direction,1]

            for component in range(lb_field.num_components[None]):
                lb_field.sc_field.psi_field[i,j,component]=(lb_field.sc_field.psi_field[i,j,component]+lambdda*lb_field.sc_field.psi_field[ix2,iy2,component])/(1.+lambdda)



@ti.data_oriented
class OpenConvective2orderPsi(OpenBoundaryPsi):
    def __init__(self, spec, direction):
        super().__init__(spec, direction)

    @ti.kernel
    def apply(self,lb_field:ti.template()):
        U=ti.Vector([0.,0.])
        for m in range(self.group.count[None]):
            i,j = self.group.group[m]
            ix2= i-lb_field.c[self.direction,0]
            iy2= j-lb_field.c[self.direction,1]
            U+=lb_field.vel[ix2,iy2]

        U/=self.group.count[None]
        Uave=tm.sqrt(tm.dot(U,U))
        lambdda=1.*Uave

        for m in range(self.group.count[None]):
            i,j = self.group.group[m]
            ix2= i-lb_field.c[self.direction,0]
            iy2= j-lb_field.c[self.direction,1]
            ix3= i-lb_field.c[self.direction,0]*2
            iy3= j-lb_field.c[self.direction,1]*2


            for component in range(lb_field.num_components[None]):
                lb_field.sc_field.psi_field[i,j,component]=(lb_field.sc_field.psi_field[i,j,component]+2*lambdda*lb_field.sc_field.psi_field[ix2,iy2,component]-0.5*lambdda*lb_field.sc_field.psi_field[ix3,iy3,component])/(1.+1.5*lambdda)
    

@ti.data_oriented
class OpenExtrapolationPsi(OpenBoundaryPsi):
    def __init__(self, spec, direction):
        super().__init__(spec, direction)

    @ti.kernel
    def apply(self,lb_field:ti.template()):
        for m in range(self.group.count[None]):
            i,j = self.group.group[m]

            ix2= i-lb_field.c[self.direction,0]
            iy2= j-lb_field.c[self.direction,1]
            ix3= ix2-lb_field.c[self.direction,0]
            iy3= iy2-lb_field.c[self.direction,1]

            for component in range(lb_field.num_components[None]):
                lb_field.sc_field.psi_field[i,j,component]=2*lb_field.sc_field.psi_field[ix2,iy2,component]-lb_field.sc_field.psi_field[ix3,iy3,component]


@ti.data_oriented
class FixedPsi(BoundaryCondition):
    def __init__(self, spec, psi_value):
        super().__init__(spec)
        self.psi_value = psi_value

    @ti.kernel
    def apply(self, lb_field: ti.template()):
        for m in range(self.group.count[None]):
            i, j = self.group.group[m]
            lb_field.sc_field.psi_field[i, j, 0] = self.psi_value


@ti.data_oriented
class PeriodicPsi_LR():

    @ti.kernel
    def apply(self,lb_field:ti.template()):
        for y in range(-1,lb_field.NY + 1):
            for component in range(lb_field.num_components[None]):
                lb_field.sc_field.psi_field[-1, y, component] = lb_field.sc_field.psi_field[lb_field.NX-1, y, component]   # 左虚=右实
                lb_field.sc_field.psi_field[lb_field.NX, y, component] = lb_field.sc_field.psi_field[0, y, component]  # 右虚=左实

@ti.data_oriented
class PeriodicPsi_BT():

    @ti.kernel
    def apply(self,lb_field:ti.template()):
        for x in range(-1,lb_field.NX + 1):
            for component in range(lb_field.num_components[None]):
                lb_field.sc_field.psi_field[x, -1, component] = lb_field.sc_field.psi_field[x, lb_field.NY-1, component]    # 下虚=上实
                lb_field.sc_field.psi_field[x, lb_field.NY, component] = lb_field.sc_field.psi_field[x, 0, component]  # 上虚=下实




@ti.data_oriented
class BoundaryEngineSC:
    def __init__(self):
        self.boundary_conditions = {}

    def add_boundary_condition(self,name,bc):
        self.boundary_conditions[name]=bc
        # print(f"  add boundary PSI {name}, size: {bc.group.count[None]}")

    def apply_boundary_conditions(self,lb_field:ti.template()):
        for bc in self.boundary_conditions.values():
            bc.apply(lb_field)


    @ti.kernel
    def png_cau(self,bc:ti.template(), img: ti.template(), color: ti.template()):
        for m in range(bc.group.count[None]):
            i, j = bc.group.group[m]
            img[i, j] = color


    def writing_boundary(self, lb_field: ti.template()):
        img = ti.Vector.field(3, dtype=ti.f32, shape=(lb_field.NX+2, lb_field.NY+2),offset=(-1,-1))
        colors = [
            # ti.Vector([0.0, 0.0, 0.0]),
            ti.Vector([52 / 255, 152 / 255, 219 / 255]),
            ti.Vector([1.0, 0.0, 0.0]),
            ti.Vector([143 / 255, 24 / 255, 172 / 255]),
            ti.Vector([255 / 255, 190 / 255, 53 / 255]),
            ti.Vector([1.0, 0.0, 0.0]),
        ]
        num=0
        for bc in self.boundary_conditions.values():
            self.png_cau(bc,img, colors[num])  # 为每个边界条件分配颜色
            num+=1
        img_np = img.to_numpy()
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        img_pil.save('boundaryPSI.png')
        print("writing boundary PSI finish")

