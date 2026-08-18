import sys
import os
# 添加包的路径
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import taichi as ti
import taichi_lbm
import numpy as np
from matplotlib import cm
import time
import matplotlib.pyplot as plt


ti.init(arch=ti.gpu)

#mesh 
scale=1e-3
domainx=30*scale
domainy=8*scale
NX_LB =int(30*20)
NY_LB =int(8*20)
Cl=domainx/NX_LB

Length_XX1=10*scale
Length_YY1=8*scale

p0=[domainx/8*3,Length_YY1/2]


# flow boundary condition
Umax=1e-2
pressure_lnlet=10

# flow info
num_components=1
shear_viscosity=1e-6
bulk_viscosity=2.5e-6
rho_flow=1000#it.fish.get('rho_ball')#1000 # density of fluid

cs=0.578 # sound speed 
Ma=Umax/cs*2 #The larger the Ma, the larger the time step

#conversion coefficient
Ux_LB=Ma*cs# vel of LB
Uy_LB=0.0
ULB=ti.Vector([Ux_LB,Uy_LB])
Cu=Umax/Ux_LB #vel conversion (main)
Ct=Cl/Cu #time conversion
C_rho=rho_flow/1 # density conversion (main)

#TRT
Magic=[1/4]


#==============================================
name="test"
lb_field=taichi_lbm.LBField(name,NX_LB,NY_LB,num_components)

#unit 
LB_params={
    'Cl':Cl,
    'Ct':Ct,
    'C_rho':C_rho,
    'C_pressure':1.
}
lb_field.init_conversion(LB_params)

#==============================================
boundary_engine=taichi_lbm.BoundaryEngine()
boundary_classifier=taichi_lbm.BoundaryClassifier(ti.field(float,shape=(NX_LB,NY_LB)))


boundary_engine.Mask_cricle_identify(lb_field,p0[0]/Cl-0.5,p0[1]/Cl-0.5,Length_YY1/10/Cl-0.5)

@ti.func
def fluid_boundary(i,j):
    return lb_field.mask[i,j]==1

fluid=taichi_lbm.BoundarySpec(geometry_fn=fluid_boundary)
fluid_bc=taichi_lbm.FluidBoundary(spec=fluid)
fluid_bc.precompute(classifier=boundary_classifier)
boundary_engine.add_boundary_condition("fluid",fluid_bc)



periodic=taichi_lbm.BoundarySpec(geometry_fn=fluid_boundary)
periodic_bc=taichi_lbm.PeriodicAllBoundary(spec=periodic)
periodic_bc.precompute(classifier=boundary_classifier)
boundary_engine.add_boundary_condition("periodic",periodic_bc)


@ti.func
def wall_boundary(i,j):
    flag=0
    if lb_field.mask[i,j]==1 :
        for k in ti.static(range(lb_field.NPOP)):
            ix2=i-lb_field.c[k,0]
            iy2=j-lb_field.c[k,1]
            if lb_field.mask[ix2,iy2]==-1 or i==0 or i==NX_LB-1:
                flag=1
    return flag

wall=taichi_lbm.BoundarySpec(geometry_fn=wall_boundary)
wall_bc=taichi_lbm.BounceBackWall(spec=wall)
wall_bc.precompute(classifier=boundary_classifier)
boundary_engine.add_boundary_condition("wall",wall_bc)


@ti.func
def inlet_boundary(i, j):
    flag=0
    if lb_field.mask[i,j]==1 and i==0 :
        flag=1
    return flag  

inlet = taichi_lbm.BoundarySpec(geometry_fn=inlet_boundary)
velocity_bc=taichi_lbm.VelocityBB(spec=inlet,velocity_value=ULB,direction=3)
velocity_bc.precompute(classifier=boundary_classifier)
boundary_engine.add_boundary_condition("inlet",velocity_bc)

@ti.func
def outlet_boundary(i,j):
    flag=0
    if lb_field.mask[i,j]==1 and i==NX_LB-1 :
        flag=1
    return flag 

outlet=taichi_lbm.BoundarySpec(geometry_fn=outlet_boundary)
pressure_bc=taichi_lbm.OpenNeumann(spec=outlet,direction=1)
pressure_bc.precompute(classifier=boundary_classifier)
boundary_engine.add_boundary_condition("outlet",pressure_bc)


lb_field.neighbor_classify()
boundary_engine.writing_boundary(lb_field)
#==============================================
macroscopic_engine=taichi_lbm.MacroscopicEngine(fluid_bc.group)
#==============================================

params={
    'group':fluid_bc.group,
    'fluid_model':taichi_lbm.NewtonianFluid(nu=shear_viscosity/lb_field.Cnu),
    'NX':NX_LB,
    'NY':NY_LB,
    'num_components':num_components,
    'shearviscosity':[shear_viscosity/lb_field.Cnu],
    'magic':Magic,
    'bulkviscosity':[bulk_viscosity/lb_field.Cnu]
}
# collision_engine=taichi_lbm.BGKCollision(params)
# collision_engine=taichi_lbm.TRTCollision(params)
collision_engine=taichi_lbm.MRTCollision(params)

# #==============================================
post_processing_engine=taichi_lbm.PostProcessingEngine(0)



@ti.kernel 
def init_hydro(vel:ti.types.vector(2, ti.f32),pressure_lnlet:float):
    if pressure_lnlet==0.0:
        for m in range(fluid_bc.group.count[None]):
            ix,iy=fluid_bc.group.group[m]
            lb_field.vel[ix,iy]=vel
            for component in range(lb_field.num_components[None]):
                lb_field.rho[ix,iy,component]=1.0/ lb_field.num_components[None]
    else:
        rho_inlet=1+pressure_lnlet*3/lb_field.C_pressure
        for m in range(fluid_bc.group.count[None]):
            ix,iy=fluid_bc.group.group[m]
            k=(1.0-rho_inlet)/lb_field.NX
            lb_field.vel[ix,iy]=ti.Vector([.0,.0])
            for component in range(lb_field.num_components[None]):
                lb_field.rho[ix,iy,component]=(k*ix+rho_inlet)/ lb_field.num_components[None]
    print("init hydro")
init_hydro(ULB,0.0)

lb_field.init_LBM(collision_engine,fluid_bc.group)
#==============================================


# ==============================================solve & show
def lbm_solve():
    # LBM SOLVE
    macroscopic_engine.density(lb_field)
    macroscopic_engine.pressure(lb_field)
    macroscopic_engine.force_density(lb_field)
    macroscopic_engine.velocity(lb_field)
    
    collision_engine.apply(lb_field)
    
    boundary_engine.apply_boundary_conditions(lb_field)
    
    pass

def post():
    # pressure = cm.Blues(post_processing_engine.post_denstiy(lb_field))
    vel_img = cm.plasma(post_processing_engine.post_vel(lb_field))
    # img1 = np.concatenate((pressure, vel_img), axis=1)
    return vel_img

showmode=2 #1=while # 0=iterations
start_time = time.time()
gui = ti.GUI(name, (NX_LB,NY_LB)) 

if showmode==2:
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for i in range(10):
            for j in range(50):
                lbm_solve()
            img=post()
            gui.set_image(img)
            gui.show()
else:
    video_manager = ti.tools.VideoManager(output_dir="./results", framerate=24, automatic_build=False)
    for i in range(50):
        for j in range(100):
            lbm_solve()
        img=post()
        gui.set_image(img)
        gui.show()
        print(f'\rDelta p= {lb_field.total_pressure[100,100]-lb_field.total_pressure[0,100]}', end='')

        filename="test"+'%d' % i
        # savefilename = f'2C_unMix_{i:05d}.png'   # create filename with suffix png
        # gui.show(savefilename)
        # post_processing_engine.writeVTK(filename,lb_field)
        # end_time = time.time()
        # elapsed_time = (end_time - start_time)
        # print({elapsed_time})


        video_manager.write_frame(img)
    print('Exporting .mp4 and .gif videos...')
    video_manager.make_video(gif=True, mp4=False)
    print(f'GIF video is saved to {video_manager.get_output_filename(".gif")}')


# 在计算结束后显示绘图窗口
if showmode == 2:
    plt.ioff()  # 关闭交互模式
    plt.show()  # 显示绘图窗口并进入事件循环


