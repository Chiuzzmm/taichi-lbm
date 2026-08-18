import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import taichi as ti
import taichi_lbm
import numpy as np
from matplotlib import cm
import time



ti.init(arch=ti.gpu)
showmode=2 #1=while # 0=iterations

#mesh 
NX_LB =int(256)
NY_LB =int(256)
Cl=1e-4

# flow boundary condition
Re=10
Umax=5e-3

# flow info
num_components=1
rho_flow=1000#it.fish.get('rho_ball')#1000 # density of fluid

cs=0.578 # sound speed 
Ma=Umax/cs*1.0 #The larger the Ma, the larger the time step

#conversion coefficient
Ux_LB=Ma*cs# vel of LB
Uy_LB=0.0
ULB=ti.Vector([Ux_LB,Uy_LB])
Cu=Umax/Ux_LB #vel conversion (main)
Ct=Cl/Cu #time conversion
C_rho=rho_flow/1 # density conversion (main)


power_n=1.0
power_K=(Umax/Cu)**(2.-power_n)*(NX_LB-1.)**power_n/Re
print("power_K",power_K)

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



@ti.func
def fluid_boundary(i,j):
    return lb_field.mask[i,j]==1

fluid=taichi_lbm.BoundarySpec(geometry_fn=fluid_boundary)
fluid_bc=taichi_lbm.FluidBoundary(spec=fluid)
fluid_bc.precompute(classifier=boundary_classifier)
boundary_engine.add_boundary_condition("fluid",fluid_bc)

@ti.func
def inside_boundary(i,j):
    flag=0
    if lb_field.mask[i,j]==1 :
        for k in ti.static(range(lb_field.NPOP)):
            ix2=i-lb_field.c[k,0]
            iy2=j-lb_field.c[k,1]
            if ix2>0 and ix2<lb_field.NX-1 and iy2>0 and iy2<lb_field.NY-1 and lb_field.mask[ix2,iy2]==1:
                flag=1
    return flag

inside=taichi_lbm.BoundarySpec(geometry_fn=inside_boundary)
inside_bc=taichi_lbm.InsideBoundary(spec=inside)
inside_bc.precompute(classifier=boundary_classifier)
boundary_engine.add_boundary_condition("inside",inside_bc)

@ti.func
def wall_boundary(i,j):
    flag=0
    if lb_field.mask[i,j]==1 :
        for k in ti.static(range(lb_field.NPOP)):
            ix2=i-lb_field.c[k,0]
            iy2=j-lb_field.c[k,1]
            if ix2<0 or ix2>lb_field.NX-1 or iy2<0 or iy2>lb_field.NY-1 or lb_field.mask[ix2,iy2]==-1:
                flag=1
    return flag

wall=taichi_lbm.BoundarySpec(geometry_fn=wall_boundary)
wall_bc=taichi_lbm.BounceBackWall(spec=wall)
wall_bc.precompute(classifier=boundary_classifier)
boundary_engine.add_boundary_condition("wall",wall_bc)


@ti.func
def inlet_boundary(i, j):
    flag=0
    if j==NY_LB-1:
        flag=1
    return flag  

inlet = taichi_lbm.BoundarySpec(geometry_fn=inlet_boundary)
top_bc=taichi_lbm.VelocityBB(spec=inlet,velocity_value=ULB,direction=2)
top_bc.precompute(classifier=boundary_classifier)
boundary_engine.add_boundary_condition("inlet",top_bc)


lb_field.neighbor_classify()
boundary_engine.writing_boundary(lb_field)
#==============================================
macroscopic_engine=taichi_lbm.MacroscopicEngine(fluid_bc.group)
#==============================================

params={
    'group':fluid_bc.group,
    # 'fluid_model':taichi_lbm.NewtonianFluid(1e-4/lb_field.Cnu),
    'fluid_model':taichi_lbm.PowerLawFluid(power_K,power_n),
    'NX':NX_LB,
    'NY':NY_LB,
    'num_components':num_components,
    'shearviscosity':[power_K],
    'magic':[1/4.],
    'bulkviscosity':[power_K]
}
# collision_engine=taichi_lbm.BGKCollision(params)
# collision_engine=taichi_lbm.TRTCollision(params)
collision_engine=taichi_lbm.MRTCollision(params)
# print(collision_engine.diag[127,127,0][8])

#==============================================
post_processing_engine=taichi_lbm.PostProcessingEngine(0)


@ti.kernel 
def init_hydro():
    for m in range(fluid_bc.group.count[None]):
        ix,iy=fluid_bc.group.group[m]
        lb_field.vel[ix,iy]=ti.Vector([0.,0.])
        for component in range(lb_field.num_components[None]):
            lb_field.rho[ix,iy,component]=1.0/ lb_field.num_components[None]
    print("init hydro")
init_hydro()

lb_field.init_LBM(collision_engine,fluid_bc.group)


# ==============================================solve & show
def lbm_solve():
    # LBM SOLVE
    macroscopic_engine.density(lb_field)
    macroscopic_engine.pressure(lb_field)
    macroscopic_engine.force_density(lb_field)
    macroscopic_engine.velocity(lb_field)
    
    collision_engine.shear_rate(lb_field)
    collision_engine.relaxation_time(lb_field)
    collision_engine.apply(lb_field)
    
    boundary_engine.apply_boundary_conditions(lb_field)
    
    pass

def post():
    # pressure = cm.Blues(post_processing_engine.post_pressure(lb_field))
    vel_img = cm.plasma(post_processing_engine.post_vel(lb_field))
    # img1 = np.concatenate((pressure, vel_img), axis=1)
    return vel_img



oldvel=ti.Vector.field(2,float,shape=(NX_LB,NY_LB))
@ti.kernel
def error()-> ti.f32:
    errornow=0.0
    for i,j in lb_field.vel:
        veldif=lb_field.vel[i,j]-oldvel[i,j]
        errornow+=ti.math.dot(veldif,veldif)

        oldvel[i,j]=lb_field.vel[i,j]
    return  errornow
    


start_time = time.time()
gui = ti.GUI(name, (NX_LB,NY_LB)) 

if showmode==1:
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for i in range(10):
            for j in range(20):
                lbm_solve()
            img=post()
            gui.set_image(img)
            gui.show()
            # print(collision_engine.diag[240,250,0])

elif showmode==2:
    video_manager = ti.tools.VideoManager(output_dir="./results", framerate=24, automatic_build=False)
    for i in range(50):
        for j in range(100):
            lbm_solve()
        img=post()
        gui.set_image(img)
        gui.show()

        filename="test"+'%d' % i
        # savefilename = f'2C_unMix_{i:05d}.png'   # create filename with suffix png
        # gui.show(savefilename)
        post_processing_engine.writeVTK(fname= filename,lb_field=lb_field)
        # end_time = time.time(), elapsed_time = (end_time - start_time)
        # print({elapsed_time})

        video_manager.write_frame(img)
    print('Exporting .mp4 and .gif videos...')
    video_manager.make_video(gif=True, mp4=False)
    print(f'GIF video is saved to {video_manager.get_output_filename(".gif")}')

elif showmode==3:
    i=0
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for j in range(10):
            for k in range(1000):
                lbm_solve()
            img=post()
            gui.set_image(img)
            gui.show()
            print(f"u={lb_field.vel[126,126]/ULB.x}")

        i+=1
        filename="Newtoniann=1.52"
        post_processing_engine.writeVTK(fname= filename,lb_field=lb_field)
        
        errornow=error()
        print(f"i={i}e={errornow}")
        if errornow<1e-8:
            gui.close()
            break



