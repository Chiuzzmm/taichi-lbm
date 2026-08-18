import sys
import os

# 添加包的路径
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import taichi as ti
import taichi_lbm
import numpy as np
from matplotlib import cm
import time
import taichi.math as tm

print("\n" * 50)
ti.init(arch=ti.gpu)

#mesh 
scale=1e-3
domainx=1*scale
domainy=1*scale
NX_LB =int(20*10)+1
NY_LB =int(20*10)+1
Cl=domainx/(NX_LB-1)


# flow boundary condition

Umax=1e-3

# flow info
num_components=1
shear_viscosity=1e-6
bulk_viscosity=1e-6
rho0=1

cs=0.578 # sound speed 
Ma=Umax/cs*1.5 #The larger the Ma, the larger the time step

#conversion coefficient
Ux_LB=Ma*cs# vel of LB
Uy_LB=0.0
ULB=ti.Vector([Ux_LB,Uy_LB])
Cu=Umax/Ux_LB #vel conversion (main)
Ct=Cl/Cu #time conversion
C_rho=rho0/1 # density conversion (main)

# SC EOS
rho_density = np.array([
    [0.0214769757706125, 3.85149307207969],
    [0.0240768092050442, 3.73852047172410],
    [0.0270279723451117, 3.62436648394743],
    [0.0303866231380030, 3.50890765731552],
    [0.0342200986869335, 3.39200335859288],
    [0.0386097872910226, 3.27349228093092],
    [0.0436548957054659, 3.15318788636729],
    [0.0494775794531934, 3.03087245867260],
    [0.0562299442008887, 2.90628907559641],
    [0.0641039116291098, 2.77913062559753],
    [0.0733453816895005, 2.64902429160983],
    [0.0842752289872210, 2.51550900183698],
    [0.0973215091582950, 2.37800139655330],
    [0.113071008931171, 2.23574210655470],
    [0.132356291264673, 2.08770614656931],
    [0.156413030238316, 1.93244248895799],
    [0.187191599699299, 1.76775878467311],
    [0.228058754735636, 1.59001518629020]
])
G=np.array([-8, -7.8, -7.6, -7.4, -7.2, -7, -6.8, -6.6, -6.4, -6.2, -6, -5.8, -5.6, -5.4, -5.2, -5, -4.8, -4.6])


rho_cr=tm.log(2.0)
idx=17
g_coh=G[idx]
rho_liq=rho_density[idx,1]
rho_gas=rho_density[idx,0]
print(f"G={g_coh}")
print(f"density ratio={rho_liq/rho_gas}")

#huang MRT pars
k1=np.array([0.15]) #sufrce tension
epslion=np.array([1.0]) #density ratio




#==============================================
name="test"
lb_field=taichi_lbm.LBField(name,NX_LB,NY_LB,num_components)
#unit 
LB_params={
    'Cl':Cl,
    'Ct':Ct,
    'C_rho':C_rho,
    'C_pressure':1,
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

stream_bc=taichi_lbm.PeriodicAllBoundary(spec=fluid)
stream_bc.precompute(classifier=boundary_classifier)
boundary_engine.add_boundary_condition("stream",stream_bc)


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
    'bulkviscosity':[bulk_viscosity/lb_field.Cnu]
}

collision_engine=taichi_lbm.MRTCollision(params)

# collision_engine=taichi_lbm.HuangMRTCollision(num_components,fluid_bc.group,k1,epslion,np.array([[g_coh]]))


#=============================================
sc_pars={
    'lb_field':lb_field,
    'g_coh':g_coh,
    'group':fluid_bc.group,
    # 'psi':taichi_lbm.CS_psi(params=CS_params),
    'psi':taichi_lbm.SC_psi(rho0=1.0)
}
lb_field.sc_field = taichi_lbm.ShanChenForceC1(sc_pars)

boundary_engine_SC=taichi_lbm.BoundaryEngineSC()
boundary_classifier_psi=taichi_lbm.BoundaryClassifier(ti.field(float,shape=(NX_LB+2,NY_LB+2),offset=(-1,-1)))


boundary_engine_SC.add_boundary_condition("periodicLR",taichi_lbm.PeriodicPsi_LR())
boundary_engine_SC.add_boundary_condition("periodicBT",taichi_lbm.PeriodicPsi_BT())

#==============================================


@ti.kernel
def init_hydro():
    lb_field.vel.fill([.0,.0])
    for m in range(fluid_bc.group.count[None]):
        ix,iy=fluid_bc.group.group[m]

        # R=50**2
        # r=(ix-lb_field.NX/2)**2+(iy-lb_field.NY/2)**2
        # if r<R:
        #     lb_field.rho[ix,iy,0]=rho_liq
        #     lb_field.sc_field.psi_field[ix,iy,0]=rho_liq
        # else:
        #     lb_field.rho[ix,iy,0]=rho_gas
        #     lb_field.sc_field.psi_field[ix,iy,0]=rho_gas

        rho=rho_cr+0.1*ti.random()
        lb_field.rho[ix,iy,0]=rho
        rho_psi=lb_field.sc_field.psi.get_psi(rho,1)
        lb_field.sc_field.psi_field[ix,iy,0]=rho_psi

    for x in range(-1,lb_field.NX+1):
        lb_field.sc_field.psi_field[x, -1, 0] = lb_field.sc_field.psi_field[x, lb_field.NY-1, 0]
        lb_field.sc_field.psi_field[x, lb_field.NY, 0] = lb_field.sc_field.psi_field[x, 0, 0]
    
    for y in range(-1,lb_field.NY + 1):
        lb_field.sc_field.psi_field[-1, y, 0] = lb_field.sc_field.psi_field[lb_field.NX-1, y, 0]   
        lb_field.sc_field.psi_field[lb_field.NX, y, 0] = lb_field.sc_field.psi_field[0, y, 0] 

    lb_field.T.fill(1)
init_hydro()

lb_field.init_LBM(collision_engine,fluid_bc.group)


#==============================================
post_processing_engine=taichi_lbm.PostProcessingEngine(0)

# ==============================================solve & show
def lbm_solve():
    # LBM SOLVE
    macroscopic_engine.density(lb_field)
    macroscopic_engine.pressure(lb_field)

    lb_field.sc_field.update_psi(lb_field)
    boundary_engine_SC.apply_boundary_conditions(lb_field)
    lb_field.sc_field.cau_SCforce(lb_field)

    macroscopic_engine.force_density(lb_field)
    macroscopic_engine.velocity(lb_field)
    collision_engine.apply(lb_field)
    boundary_engine.apply_boundary_conditions(lb_field)

    
    macroscopic_engine.time_updata(lb_field)

def post():
    pressure = cm.Blues(post_processing_engine.post_pressure(lb_field))
    vel_img = cm.plasma(post_processing_engine.post_vel(lb_field))
    img1 = np.concatenate((pressure, vel_img), axis=1)
    return img1


start_time = time.time()
gui = ti.GUI(name, (1*NX_LB,2*NY_LB)) 
result_dir = "./results"
video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)



showmode=2 #1=while # 0=iterations
if showmode==1:
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for j in range(10):
            lbm_solve()
        img=post()
        # print(f"\rdroplet density 0= {lb_field.rho[100, 100, 0]:.3f}, air density 0= {lb_field.rho[0, 100, 0]:.3f}, ratio={lb_field.rho[100, 100, 0]/lb_field.rho[0, 100, 0]:.3f}", end='')
        gui.set_image(img)
        gui.show()
        time.sleep(0.1)
else:
    for i in range(50):
        for j in range(100):
            lbm_solve()
        img=post()
        gui.set_image(img)
        gui.show()
        filename="test"+'%d' % i
        # savefilename = f'2C_unMix_{i:05d}.png'   # create filename with suffix png
        # gui.show(savefilename)

        video_manager.write_frame(img)


    print('Exporting .mp4 and .gif videos...')
    video_manager.make_video(gif=True, mp4=False)
    print(f'GIF video is saved to {video_manager.get_output_filename(".gif")}')
    # macroscopic_engine.writeVTK(filename,lb_field)
    # end_time = time.time()
    # elapsed_time = (end_time - start_time)
    # print({elapsed_time})
