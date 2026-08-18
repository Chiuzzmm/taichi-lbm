# taichi-lbm – Lattice Boltzmann Solver Powered by Taichi

## Overview
taichi-lbm is a fluid simulator based on the Lattice Boltzmann Method (LBM) and the high-performance Taichi computing framework.  
Born from doctoral research, the project has been systematically refactored and open-sourced to provide a  simulation toolchain for complex flow problems.  
**Note:** The code is still under active development—some features are incomplete and bugs may exist.

- **Multi-component flows:** simultaneous simulation of multiple fluid species  
- **Rich boundary conditions:** velocity, pressure, bounce-back, periodic, and more  
- **Non-Newtonian fluids:** power-law and other non-Newtonian rheology models  
- **Multiple collision kernels:** BGK, TRT, MRT, etc.  
- **High performance:** GPU acceleration via Taichi  

## Quick Start
### Prerequisites
- Python
- Taichi

### Basic Example
1. Copy any example from the `cases` folder to the same level as the `taichi_lbm` directory and run it.  
2. Or modify the code in the example and then execute.

```python
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
```


## Core Architecture
### Data Structures
#### `LBField` – Central Field Container
`LBField` provides a unified data-management and access interface for the entire LBM simulation and is the fundamental building block of taichi-lbm.  
Leveraging Taichi’s GPU memory management, all field variables can be accessed in parallel on the GPU, support multi-component flows, and allow modular boundary treatment.

```python
lb_field = taichi_lbm.LBField(name, NX_LB, NY_LB, num_components)
```

`name`: simulation name  
`NX_LB, NY_LB`: number of lattice nodes in x and y directions  
`num_components`: number of fluid components (single-component = 1, multi-component > 1)

Contains the density field `lb_field.rho`, velocity field `lb_field.vel`, distribution-function field `lb_field.f`, node-type mask `lb_field.mask`, and a unit-conversion system.

Currently `LBField` only supports the `D2Q9` model; further decoupling and modularization are planned.

Usage example:
```python
# create field object
lb_field = taichi_lbm.LBField("channel_flow", 300, 100, 1)

# configure unit system
lb_field.init_conversion({
    'Cl': 1e-4,      # lattice spacing 0.1 mm
    'Ct': 1e-5,      # time step 10 μs
    'C_rho': 1000,   # water density 1000 kg/m³
})

# initialize flow field
lb_field.init_LBM(collision_model, fluid_region)
```

#### `MaskAndGroup` – Lattice Classification & Grouping

`MaskAndGroup` is the core module in taichi-lbm for efficiently classifying and grouping lattice nodes.  
Through intelligent node tagging and grouping, it underpins boundary-condition application, physical-quantity evaluation, and parallel-performance optimization.

Together with the `lb_field.mask` attribute it enforces a unified tagging scheme:

```python
# node-type tags
MASK_FLUID = 1     # fluid node
MASK_SOLID = -1    # solid node
```

Grouping data are automatically created during boundary partitioning and contain the lattice indices together with their total counts.

### Boundary-Condition System
taichi-lbm’s boundary-condition system is built on a modular design that implements flexible boundary handling through a **descriptor–classifier–condition** three-layer architecture.

```
BoundarySpec (descriptor)  
    ↓ defines boundary geometry & properties  
BoundaryClassifier (classifier)  
    ↓ automatically identifies boundary nodes  
BoundaryCondition (condition)  
    ↓ applies the actual boundary treatment  
```

#### Boundary Descriptor (`BoundarySpec`)
A boundary descriptor specifies the **geometric features** and **physical attributes** of a boundary; it does **not** contain any concrete numerical treatment.

```python
@ti.func
def inlet_boundary(i, j):
    """Define inlet-boundary geometry"""
    flag = 0
    if lb_field.mask[i, j] == 1 and i == 0:  # left edge & fluid node
        flag = 1
    return flag

# create boundary descriptor
inlet = taichi_lbm.BoundarySpec(geometry_fn=inlet_boundary)
```

`BoundaryEngine` additionally supplies two ready-to-use helpers for solid-boundary setup: `Mask_rectangle_identify` and `Mask_circle_identify`.


#### Boundary Classifier (`BoundaryClassifier`)
The classifier automatically scans the computational domain according to the descriptor, identifies boundary nodes and groups them.

```python
# create classifier (based on field shape)
boundary_classifier = taichi_lbm.BoundaryClassifier(
    ti.field(float, shape=(NX_LB, NY_LB))
)

# pre-compute boundary group
boundary_group = boundary_classifier.get_group(inlet_spec)
print(f"Identified {boundary_group.count[None]} boundary nodes")
```

#### Supported Boundary Types
- Velocity boundaries: `VelocityBB`, `VelocityEquilibrium`, `VelocityNEEM`  
- Pressure boundaries: `PressureABC`, `PressureNEEM`  
- Solid walls: `BounceBackWall`  
- Periodic boundaries: `PeriodicAllBoundary`  
- Open boundaries: `OpenNeumann`, `OpenExtrapolation`, `OpenConvective*`  
- Internal boundaries: `InsideBoundary`


#### BoundaryEngine – Boundary Orchestrator
The boundary engine coordinates the application of all boundary conditions.

```python
# create engine
boundary_engine = taichi_lbm.BoundaryEngine()

# register conditions
boundary_engine.add_boundary_condition("inlet",  velocity_bc)
boundary_engine.add_boundary_condition("outlet", pressure_bc)
boundary_engine.add_boundary_condition("walls",  bounceback_bc)

# apply them
boundary_engine.apply_boundary_conditions(lb_field)
```

Visual inspection is supported:
```python
# generate boundary map
boundary_engine.writing_boundary(lb_field)
# outputs: boundary.png (colors denote different boundary types)
```

Complete workflow for creating a bounded domain:

```python
# init engine & classifier
boundary_engine   = taichi_lbm.BoundaryEngine()
boundary_classifier = taichi_lbm.BoundaryClassifier(ti.field(float, shape=(NX_LB, NY_LB)))

# descriptor
@ti.func
def inlet_boundary(i, j):
    flag = 0
    if lb_field.mask[i, j] == 1 and i == 0:
        flag = 1
    return flag

inlet = taichi_lbm.BoundarySpec(geometry_fn=inlet_boundary)
velocity_bc = taichi_lbm.VelocityBB(spec=inlet, velocity_value=ULB, direction=3)

# pre-compute node set
velocity_bc.precompute(classifier=boundary_classifier)

# register
boundary_engine.add_boundary_condition("inlet", velocity_bc)

# apply inside LBM loop
boundary_engine.apply_boundary_conditions(lb_field)
```


### Collision Models
taichi-lbm provides a spectrum of collision models, from the basic BGK to advanced MRT, covering simple flows to complex multiphase problems.

```
Collision (base)
├── BGKCollision   (single-relaxation-time)
├── TRTCollision   (two-relaxation-time)
├── MRTCollision   (multiple-relaxation-time)
└── HuangMRTCollision (multiphase MRT)
```

#### `Collision` – Base functionality
Implements core routines such as equilibrium distribution `f_eq()` and forcing term `f_force()`.  
For numerical stability, `tau_max_min()` enforces upper/lower bounds on relaxation times—mainly used for non-Newtonian fluids.

#### `BGKCollision` – Single-relaxation-time model
The simplest collision model; all modes share the same relaxation time.  
Computationally cheap and adequate for simple flows.

Required parameters:
```python
bgk_params = {
    'group'          : fluid_group,
    'fluid_model'    : newtonian_fluid,
    'NX'             : NX_LB,
    'NY'             : NY_LB,
    'num_components' : 1,
    'shearviscosity' : [shear_viscosity_LB]  # kinematic viscosity (LB units)
}

collision_engine = taichi_lbm.BGKCollision(bgk_params)
```

#### `TRTCollision` – Two-relaxation-time model
Symmetric and anti-symmetric modes are relaxed separately, giving better stability and a tunable *Magic* parameter for performance tuning.

Required parameters:
```python
trt_params = {
    'group'          : fluid_group,
    'fluid_model'    : newtonian_fluid,
    'NX'             : NX_LB,
    'NY'             : NY_LB,
    'num_components' : 1,
    'shearviscosity' : [shear_viscosity_LB],
    'magic'          : [0.25]  # TRT Magic parameter
}

collision_engine = taichi_lbm.TRTCollision(trt_params)
```

#### `MRTCollision` – Multiple-relaxation-time model
Each physical moment can be assigned its own relaxation time; bulk viscosity is controlled independently.

Required parameters:
```python
mrt_params = {
    'group'          : fluid_group,
    'fluid_model'    : newtonian_fluid,
    'NX'             : NX_LB,
    'NY'             : NY_LB,
    'num_components' : 1,
    'shearviscosity' : [shear_viscosity_LB],
    'bulkviscosity'  : [bulk_viscosity_LB]  # bulk viscosity
}

collision_engine = taichi_lbm.MRTCollision(mrt_params)
```

The moment space is constructed with the Gram–Schmidt procedure:

```python
# D2Q9 MRT transformation matrix
self.M = ti.field(float, shape=(9, 9))
M_matrix = np.array([
    [ 1,  1,  1,  1,  1,  1,  1,  1,  1],  # mass
    [-4, -1, -1, -1, -1,  2,  2,  2,  2],  # energy
    [ 4, -2, -2, -2, -2,  1,  1,  1,  1],  # energy squared
    [ 0,  1,  0, -1,  0,  1, -1, -1,  1],  # x-momentum
    [ 0, -2,  0,  2,  0,  1, -1, -1,  1],  # x-momentum energy
    [ 0,  0,  1,  0, -1,  1,  1, -1, -1],  # y-momentum
    [ 0,  0, -2,  0,  2,  1,  1, -1, -1],  # y-momentum energy
    [ 0,  1, -1,  1, -1,  0,  0,  0,  0],  # stress tensor
    [ 0,  0,  0,  0,  0,  1, -1,  1, -1]   # stress tensor
])
```


#### `HuangMRTCollision` – Multiphase MRT Model
> R. Huang, H. Wu, *Third-order analysis of pseudopotential lattice Boltzmann model for multiphase flow*, J. Comput. Phys. 327 (2016) 121–139.

An extra term **S Qₘ** is added in the MRT collision step:

$$
\overline{\mathbf{m}} = \mathbf{m} - \mathbf{S}\,(\mathbf{m}-\mathbf{m}^{\text{eq}})
+ \delta_t\!\left(\mathbf{I}-\frac{\mathbf{S}}{2}\right)\mathbf{F}_m
+ \mathbf{S}\,\mathbf{Q}_m
$$

where **m**, **S**, **m**<sup>eq</sup>, **F**<sub>m</sub> have the usual MRT meanings, while

$$
\mathbf{Q}_m =
\begin{bmatrix}
0 \\[4pt]
3(k_1+2k_2)\dfrac{|\mathbf{F}|^2}{G\psi^2} \\[8pt]
-3(k_1+2k_2)\dfrac{|\mathbf{F}|^2}{G\psi^2} \\[8pt]
0 \\ 0 \\ 0 \\ 0 \\[4pt]
k_1\dfrac{F_x^2-F_y^2}{G\psi^2} \\[6pt]
k_1\dfrac{F_x F_y}{G\psi^2}
\end{bmatrix}
\quad\text{and}\quad
\varepsilon = -8(k_1+k_2).
$$

By tuning **ε** and **k₁** independently, the liquid–gas density ratio and the surface tension can be adjusted.

*Currently this model does **not** support non-Newtonian fluids.*


### Macroscopic Engine
#### `MacroscopicEngine` – Density, Velocity & Pressure Evaluation

`MacroscopicEngine` computes macroscopic quantities (density, velocity, pressure, …) from the particle distribution functions.  
This step is mandatory every LBM time-step and converts microscopic distributions into continuous macroscopic fields.

### Fluid Models
Newtonian and power-law non-Newtonian fluids are currently supported.

```python
# Newtonian fluid
fluid_model = taichi_lbm.NewtonianFluid(nu=0.1)

# Power-law non-Newtonian fluid
fluid_model = taichi_lbm.PowerLawFluid(m=0.1, n=0.8)  # m: consistency index, n: power-law exponent
```

### Shan-Chen Model – Multi-Component Flows

A pseudopotential approach is implemented to capture gas–liquid interfaces and phase separation.

#### `SCBoundaryCondition` – Boundary Handling
The pseudopotential field is extended by one lattice layer beyond the `LBField` domain:

```python
psi_field = ti.field(float,
                     shape=(lb_field.NX+2, lb_field.NY+2, lb_field.num_components[None]),
                     offset=(-1, -1, 0))
```

Usage is identical to ordinary boundaries.

Currently supported types:
- Periodic: `PeriodicPsi_LR`, `PeriodicPsi_BT`  
- Open: `OpenBoundaryPsi`, `OpenNeumannPsi`, `OpenConvective1orderPsi`, `OpenConvective2orderPsi`, `OpenExtrapolationPsi`  
- Fixed value: `FixedPsi`

#### Equations of State (EOS)

The pseudopotential ψ couples an EOS to the LBM.  Several classical EOS are provided:

- SC: `SC_psi`  
- van der Waals: `vdW_psi`  
- Real-gas EOS: `RK_psi`, `PR_psi`, `CS_psi`

Minimal setup:

```python
sc_pars = {
    'lb_field' : lb_field,
    'g_coh'    : g_coh,
    'group'    : fluid_bc.group,
    'psi'      : taichi_lbm.SC_psi(rho0=1.0)
}
lb_field.sc_field = taichi_lbm.ShanChenForceC1(sc_pars)
```


## Quick-Start Examples

#### Circular Cylinder
Velocity inlet on the left, open outlet on the right, periodic top & bottom.  
A solid circular cylinder is immersed in the domain.

![](/results/Circular%20cylinder.gif)

#### Contraction Flow
Velocity inlet on the left, pressure outlet on the right, no-slip top wall.  
The channel contains a localized contraction.

![](/results/contraction%20flow.gif)

#### Power-Law Fluid
Lid-driven cavity: no-slip on left, right and bottom walls; moving lid on top.  
Non-Newtonian fluid modeled with a power-law rheology.

![](/results/Power%20Law%20Fluid.gif)

#### Bubble / Two-Phase Separation
Fully periodic domain.  
Initial density field is randomized; subsequent phase separation produces distinct gas and liquid regions.

![](/results/two-phase%20separation.gif)

