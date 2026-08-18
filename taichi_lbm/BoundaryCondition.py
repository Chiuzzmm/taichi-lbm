from PIL import Image
import taichi as ti
import taichi.math as tm
import numpy as np



@ti.data_oriented
class MaskAndGroup:
    def __init__(self,NX,NY):
        self.group=ti.Vector.field(2,int,shape=(NX*NY,))
        self.count=ti.field(int,shape=())

@ti.data_oriented
class BoundarySpec:
    """边界条件描述符，定义如何识别边界节点"""
    def __init__(self, geometry_fn, direction=None,value_fn=None):
        self.geometry_fn =geometry_fn  # 几何判断函数
        self.direction = direction      # 可选方向约束
        self.value_fn =value_fn

@ti.data_oriented
class BoundaryClassifier:
    """边界分类器, 根据Spec自动生成节点组"""
    def __init__(self,shape):
        self.shape=shape

    def get_group(self,spec:BoundarySpec):
        group_copy=MaskAndGroup(self.shape.shape[0],self.shape.shape[1])
        self._classify_nodes(spec,group_copy)
        return group_copy

    @ti.kernel
    def _classify_nodes(self, spec: ti.template(),group:ti.template()):
        # 在Taichi内核中执行分类
        for i,j in self.shape:
            if spec.geometry_fn(i, j):
                idx = ti.atomic_add(group.count[None], 1)
                group.group[idx]=ti.Vector([i,j])

@ti.data_oriented
class BoundaryCondition:
    def __init__(self, spec: BoundarySpec):
        self.spec = spec
        self.group = None  # 延迟初始化

    def precompute(self, classifier: BoundaryClassifier):
        """预计算边界组"""
        self.group = classifier.get_group(self.spec)

    @ti.kernel
    def apply(self,lb_field: ti.template()):
        """需子类实现"""
        pass


@ti.data_oriented
class PlaneBoundary(BoundaryCondition):
    def __init__(self,spec:BoundarySpec,direction):
        super().__init__(spec)
        self.direction = direction
        # 根据方向确定需要设置的分布函数分量
        if direction==1:
            self.unknow = [3, 6, 7]
        elif direction==3:
            self.unknow = [1, 5, 8]
        elif direction==4:
            self.unknow=[2,5,6]
        elif direction==2:
            self.unknow=[4,7,8]
        elif direction==None:
            self.unknow=None

@ti.data_oriented
class VelocityBoundary(PlaneBoundary):
    def __init__(self, spec: BoundarySpec, velocity_value,direction):
        super().__init__(spec,direction)
        self.velocity_value = velocity_value

@ti.data_oriented
class VelocityBB(VelocityBoundary):
    def __init__(self, spec, velocity_value, direction):
        super().__init__(spec, velocity_value, direction)

    @ti.kernel
    def apply(self, lb_field: ti.template()):
        for m in range(self.group.count[None]):
            i, j = self.group.group[m]
            ix2= i-lb_field.c[self.direction,0]
            iy2= j-lb_field.c[self.direction,1]
            # ix3= i-lb_field.c[self.direction,0]*2
            # iy3= j-lb_field.c[self.direction,1]*2

            for component in range(lb_field.num_components[None]):
                # rhow=1.0
                rhow=lb_field.rho[i, j,component]
                # rhow=1.5 * lb_field.rho[i, j,component] - 0.5 * lb_field.rho[ix2,iy2,component]
                # rhow=2*lb_field.rho[ix2,iy2,component]-lb_field.rho[ix3,iy3,component]
                # rhow=3*lb_field.rho[i,j,component]-3*lb_field.rho[ix2,iy2,component]+lb_field.rho[ix3,iy3,component]
                for d in ti.static(self.unknow):
                    cv = lb_field.c[d,0] * self.velocity_value.x + lb_field.c[d,1] * self.velocity_value.y
                    lb_field.f[i, j,component][d] += 6 * lb_field.weights[d] * rhow * cv
                    
@ti.data_oriented
class NEEMMethod():
    @ti.func
    def f_eq(self,c,w,vel,rho):
        eu=c @ vel
        uv=tm.dot(vel,vel)
        return w*rho*(1+3*eu+4.5*eu*eu-1.5*uv)
    
    @ti.func    
    def NEEM(self,c,w,vel1,rho1,vel2,rho2,f2):
        feqeq_b=self.f_eq(c,w,vel1,rho1)
        feqeq_f=self.f_eq(c,w,vel2,rho2)
        return feqeq_b+f2-feqeq_f

    @ti.func
    def rhow(self,direction,vel,f):
        rhow=0.0
        if direction==2:
            rhow=1./(1.+vel.y)*(f[0]+f[1]+f[3]+2*(f[2]+f[5]+f[6]))
        elif direction==4:
            rhow=1./(1.-vel.y)*(f[0]+f[1]+f[3]+2*(f[4]+f[7]+f[8]))
        elif direction==1:
            rhow=1./(1.+vel.x)*(f[0]+f[2]+f[4]+2*(f[1]+f[5]+f[8]))
        elif direction==3:
            rhow=1./(1.-vel.x)*(f[0]+f[2]+f[4]+2*(f[3]+f[6]+f[7]))
        return rhow


@ti.data_oriented
class VelocityEquilibrium(VelocityBoundary,NEEMMethod):
    def __init__(self, spec, velocity_value, direction):
        super().__init__(spec, velocity_value, direction)
    
    @ti.kernel
    def apply(self, lb_field: ti.template()):
        for m in range(self.group.count[None]):
            i, j = self.group.group[m]

            lb_field.vel[i,j]=self.velocity_value
            for component in range(lb_field.num_components[None]):
                lb_field.rho[i,j,component]=self.rhow(self.direction,self.velocity_value,lb_field.f[i,j,component])

            for component in range(lb_field.num_components[None]):
                lb_field.f[i,j,component]=self.f_eq(lb_field.c,lb_field.weights,lb_field.vel[i,j],lb_field.rho[i,j,component])

@ti.data_oriented
class VelocityNEEM(VelocityBoundary,NEEMMethod):
    def __init__(self, spec, velocity_value, direction):
        super().__init__(spec, velocity_value, direction)

    @ti.kernel
    def apply(self, lb_field: ti.template()):
        for m in range(self.group.count[None]):
            i, j = self.group.group[m]
            ix2= i-lb_field.c[self.direction,0]
            iy2= j-lb_field.c[self.direction,1]

            lb_field.vel[i,j]=self.velocity_value
            for component in range(lb_field.num_components[None]):
                lb_field.rho[i,j,component]=self.rhow(self.direction,self.velocity_value,lb_field.f[i,j,component])

            for component in range(lb_field.num_components[None]):
                lb_field.f[i,j,component]=self.NEEM(lb_field.c,lb_field.weights,lb_field.vel[i,j],lb_field.rho[i, j,component],lb_field.vel[ix2,iy2],lb_field.rho[ix2,iy2,component],lb_field.f[ix2,iy2,component])



@ti.data_oriented
class PressureBoundary(PlaneBoundary):
    def __init__(self,  spec: BoundarySpec, rho_value, direction):
        super().__init__(spec,direction)
        self.rho_value = rho_value


@ti.data_oriented
class PressureABC(PressureBoundary):
    def __init__(self, spec, rho_value, direction):
        super().__init__(spec, rho_value, direction)

    @ti.func
    def ABC(self,c,w,f,rho_w,uw):
        cu=c@uw
        uw2=tm.dot(uw,uw)
        return  -f+2*w*rho_w*(1.0+4.5*cu*cu-1.5*uw2)
    
    @ti.kernel
    def apply(self, lb_field: ti.template()):
        
        for m in range(self.group.count[None]):
            i, j = self.group.group[m]
            ix2= i-lb_field.c[self.direction,0]
            iy2= j-lb_field.c[self.direction,1]
            
            # 计算外推速度
            uw = 1.5 * lb_field.vel[i, j] - 0.5 * lb_field.vel[ix2, iy2]
            lb_field.vel[i, j] = uw
            lb_field.total_rho[i, j] = self.rho_value
            for component in range(lb_field.num_components[None]):
                # 调用ABC方法计算分布函数
                ftemp = self.ABC(lb_field.c, lb_field.weights, lb_field.f[i, j,component], self.rho_value, uw)
                # 更新特定方向的分布函数分量
                for d in ti.static(self.unknow):
                    lb_field.f[i, j,component][d] = ftemp[d]



@ti.data_oriented
class PressureNEEM(PressureBoundary,NEEMMethod):
    def __init__(self, spec, rho_value, direction):
        super().__init__(spec, rho_value, direction)

    
    @ti.kernel
    def apply(self, lb_field: ti.template()):
        for m in range(self.group.count[None]):
            i, j = self.group.group[m]
            ix2= i-lb_field.c[self.direction,0]
            iy2= j-lb_field.c[self.direction,1]

            lb_field.vel[i,j]=lb_field.vel[ix2,iy2]
            lb_field.total_rho[i,j]=self.rho_value
            for component in range(lb_field.num_components[None]):
                for d in ti.static(self.unknow):
                    lb_field.f[i,j,component][d]=self.NEEM(lb_field.c,lb_field.weights,lb_field.vel[i,j],lb_field.rho[i, j,component],lb_field.vel[ix2,iy2],lb_field.rho[ix2,iy2,component],lb_field.f[ix2,iy2,component])[d]




@ti.data_oriented
class BounceBackWall(BoundaryCondition):
    def __init__(self,spec: BoundarySpec):
        super().__init__(spec)
        # self.bounce_map = {
        #     0:0,
        # 1: 3, 3: 1,
        # 2: 4, 4: 2,
        # 5: 7, 7: 5,
        # 6: 8, 8: 6
        # }
        self.bounce_map=ti.Vector([0, 3, 4, 1, 2, 7, 8, 5, 6])

    @ti.kernel
    def apply(self, lb_field:ti.template()): # type: ignore
        for m in range(self.group.count[None]):
            i,j=self.group.group[m]
            for k in ti.static(range(9)):
                ix2=lb_field.neighbor[i,j][k,0]
                iy2=lb_field.neighbor[i,j][k,1]
                if ix2!=-1:
                    #stream in boundary
                    for component in range(lb_field.num_components[None]):
                        lb_field.f[i,j,component][k]=lb_field.f2[ix2,iy2,component][k]
                else:
                    #bounce back on the static wall 
                    ipop=self.bounce_map[k]
                    for component in range(lb_field.num_components[None]):
                        lb_field.f[i,j,component][k]=lb_field.f2[i,j,component][ipop] #f or f2?




@ti.data_oriented
class OpenBoundary(PlaneBoundary):
    def __init__(self, spec:BoundarySpec,direction):
        super().__init__(spec,direction)


    def apply(self):
        pass

    @ti.func
    def density(self,lb_field:ti.template(),i:int,j:int):
        lb_field.total_rho[i,j]=0.0
        for component in  range(lb_field.num_components[None]):
            lb_field.rho[i,j,component]=0.0
            for k in range(9):
                lb_field.rho[i,j,component]+=lb_field.f[i,j,component][k]
            lb_field.total_rho[i,j]+=lb_field.rho[i,j,component]

    @ti.func
    def force_density(self,lb_field:ti.template(),i:int,j:int):
        for component in  range(lb_field.num_components[None]):
            lb_field.body_force[i,j,component]=(lb_field.SCforce[i,j,component]+lb_field.rho[i,j,component]/lb_field.total_rho[i,j]*lb_field.gravity_force)

    @ti.func
    def vel(self,lb_field:ti.template(),i:int,j:int):
        lb_field.vel[i,j]=ti.Vector([0.0,0.0])
        for component in  range(lb_field.num_components[None]):
            vel_temp = ti.Vector([0.0, 0.0])
            for k in ti.static(range(lb_field.NPOP)):
                vel_temp.x += lb_field.f[i, j,component][k] * lb_field.c[k, 0]
                vel_temp.y += lb_field.f[i, j,component][k] * lb_field.c[k, 1]
            
            vel_temp+=0.5 * lb_field.body_force[i,j,component]
            lb_field.vel[i,j]+=vel_temp
        lb_field.vel[i, j] /= lb_field.total_rho[i, j]


    @ti.func
    def SCforce(self,lb_field:ti.template(),i:int,j:int):
        #reset
        lb_field.SCforce[i,j,0]=0.0
        F = ti.Vector([0.0, 0.0])
        for k in ti.static(range(1,9)):
            c=ti.Vector([lb_field.c[k,0],lb_field.c[k,1]])
            x2 = i + c.x
            y2 = j + c.y
            F+=lb_field.weights[k] * c*lb_field.sc_field.psi_field[x2,y2,0]
        F*=(-lb_field.sc_field.g_coh*lb_field.sc_field.psi_field[i,j,0])
        lb_field.SCforce[i,j,0]+=F


@ti.data_oriented
class OpenNeumann(OpenBoundary):
    def __init__(self, spec, direction):
        super().__init__(spec, direction)


    @ti.kernel
    def apply(self, lb_field:ti.template()): 
        for m in range(self.group.count[None]):
            i,j = self.group.group[m]
            ix2= i-lb_field.c[self.direction,0]
            iy2= j-lb_field.c[self.direction,1]

            for component in range(lb_field.num_components[None]):
                for d in ti.static(self.unknow):
                    lb_field.f[i, j, component][d] = lb_field.f[ix2, iy2, component][d]

            #denisty and velcity
            self.density(lb_field,i,j)
            self.vel(lb_field,i,j)


@ti.data_oriented
class OpenConvective1order(OpenBoundary):
    def __init__(self, spec, direction):
        super().__init__(spec, direction)

    @ti.kernel
    def apply(self, lb_field:ti.template()):
        for m in range(self.group.count[None]):
            i,j = self.group.group[m]
            ix2= i-lb_field.c[self.direction,0]
            iy2= j-lb_field.c[self.direction,1]

            U=lb_field.vel[ix2,iy2]
            Uave=tm.sqrt(tm.dot(U,U))
            lambdda=1.*Uave
            
            for component in range(lb_field.num_components[None]):
                for d in ti.static(self.unknow):
                    lb_field.f[i,j,component][d]=(lb_field.f[i,j,component][d]+lambdda*lb_field.f[ix2,iy2,component][d])/(1.+lambdda)

            #denisty and velcity
            self.density(lb_field,i,j)
            self.vel(lb_field,i,j)
            

@ti.data_oriented
class OpenConvective2order(OpenBoundary):
    def __init__(self, spec, direction):
        super().__init__(spec, direction)

    @ti.kernel
    def apply(self, lb_field:ti.template()): 
        for iteration in range(3):
            for m in range(self.group.count[None]):
                i,j = self.group.group[m]
                ix2= i-lb_field.c[self.direction,0]
                iy2= j-lb_field.c[self.direction,1]
                ix3= i-lb_field.c[self.direction,0]*2
                iy3= j-lb_field.c[self.direction,1]*2

                self.SCforce(lb_field,ix2,iy2)
                self.force_density(lb_field,ix2,iy2)
                self.vel(lb_field,ix2,iy2)
                U=lb_field.vel[ix2,iy2]
                Uave=tm.sqrt(tm.dot(U,U))
                lambdda=1.*Uave

                for component in range(lb_field.num_components[None]):
                    for d in ti.static(self.unknow):
                        lb_field.f[i,j,component][d]=(lb_field.f[i,j,component][d]+2*lambdda*lb_field.f[ix2,iy2,component][d]-0.5*lambdda*lb_field.f[ix3,iy3,component][d])/(1.+1.5*lambdda)

                #denisty and velcity
                self.density(lb_field,i,j)
                lb_field.sc_field.psi_field[i,j,0]=lb_field.sc_field.psi.get_psi(lb_field.rho[i,j,0],lb_field.T[i,j])



@ti.data_oriented
class OpenExtrapolation(OpenBoundary):
    def __init__(self, spec, direction):
        super().__init__(spec, direction)


    @ti.kernel
    def apply(self, lb_field:ti.template()): 
        for m in range(self.group.count[None]):
            i,j = self.group.group[m]
            ix2= i-lb_field.c[self.direction,0]
            iy2= j-lb_field.c[self.direction,1]
            ix3= ix2-lb_field.c[self.direction,0]
            iy3= iy2-lb_field.c[self.direction,1]

            for component in range(lb_field.num_components[None]):
                for d in ti.static(self.unknow):
                    lb_field.f[i,j,component][d]=2*lb_field.f[ix2,iy2,component][d]-lb_field.f[ix3,iy3,component][d]

            #denisty and velcity
            # self.density(lb_field,i,j)
            # self.vel(lb_field,i,j)



@ti.data_oriented
class PeriodicAllBoundary(BoundaryCondition):
    def __init__(self, spec: BoundarySpec):
        super().__init__(spec)
        
    @ti.kernel
    def apply(self, lb_field:ti.template()): 
        for m in range(self.group.count[None]):
            i, j = self.group.group[m]
            for k in ti.static(range(lb_field.NPOP)):
                x2 = (i - lb_field.c[k, 0] + lb_field.NX) % lb_field.NX 
                y2 = (j - lb_field.c[k, 1] + lb_field.NY) % lb_field.NY
                for component in range(lb_field.num_components[None]):
                    lb_field.f[i, j, component][k] = lb_field.f2[x2, y2, component][k]


@ti.data_oriented
class InsideBoundary(BoundaryCondition):
    def __init__(self,spec: BoundarySpec):
        super().__init__(spec)

    @ti.kernel
    def apply(self, lb_field:ti.template()): 
        
        for m in range(self.group.count[None]):
            i,j =self.group.group[m]
            for k in ti.static(range(lb_field.NPOP)):
                ix2=lb_field.neighbor[i,j][k,0]
                iy2=lb_field.neighbor[i,j][k,1]
                for component in  range(lb_field.num_components[None]):
                    lb_field.f[i,j,component][k]=lb_field.f2[ix2,iy2,component][k]


@ti.data_oriented
class FluidBoundary(BoundaryCondition):
    def __init__(self, spec:BoundarySpec):
        super().__init__(spec)



@ti.data_oriented
class BoundaryEngine:
    def __init__(self):
        self.boundary_conditions = {}
    
    @ti.kernel
    def Mask_rectangle_identify(self,lb_field:ti.template(),xmin:float,xmax:float,ymin:float,ymax:float):
        for i,j in lb_field.mask:
            if i<xmax and i>xmin:
                if j<ymax and j>ymin:
                    lb_field.mask[i,j]=-1

    @ti.kernel
    def Mask_cricle_identify(self,lb_field:ti.template(),x:float,y:float,r:float):
        for ix,iy in lb_field.mask:
            if (ix-x)**2+(iy-y)**2<r**2:
                lb_field.mask[ix,iy]=-1

    def add_boundary_condition(self,name,bc):
        self.boundary_conditions[name]=bc
        print(f"  add boundary {name}, size: {bc.group.count[None]}")

    def apply_boundary_conditions(self,lb_field:ti.template()):
        for bc in self.boundary_conditions.values():
            bc.apply(lb_field)

    
    @ti.kernel
    def png_cau(self,bc:ti.template(), img: ti.template(), color: ti.template()):
        for m in range(bc.group.count[None]):
            i, j = bc.group.group[m]
            img[i, j] = color


    def writing_boundary(self, lb_field: ti.template()):
        img = ti.Vector.field(3, dtype=ti.f32, shape=(lb_field.NX, lb_field.NY))
        colors = [
            ti.Vector([0.0, 0.0, 0.0]),
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
        img_pil.save('boundary.png')
        print("writing boundary finish")



