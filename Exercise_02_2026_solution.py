#!/usr/bin/env python
# coding: utf-8

# # Exercise 4: Methane in Rotsee
# 
# Learning goals: 
# - set up a one-dimensional diffusion-reaction model for methane in a lake
# - understand, describe and comment on the methane concentration evolution in Rotsee and similar lake systems 
# 
# ## Dynamics of methane in Rotsee
# 
# Lakes are an overlooked source of methane (CH<sub>4</sub>) to the atmosphere and could represent a significant fraction of the total natural CH<sub>4</sub> emissions. In Rotsee, a small eutrophic lake near Lucerne, the CH<sub>4</sub> released from the sediments is oxidized during its way through the water column. However, the surface of the lake is still oversaturated in CH<sub>4</sub> compared to the atmosphere. Therefore, CH<sub>4</sub> is released to the atmosphere. In this exercise you will model the CH<sub>4</sub> concentration evolution in the water column with an 1-dimensional model, taking into account the vertical diffusivity and the aerobic CH<sub>4</sub> oxidation.
# 

# Rotsee is a small lake near Lucerne with a maximum depth of 16 m. The lake undergoes seasonal mixing in winter (December to April), and during this period, we assume the same value the turbulent diffusivity ($K_{Z}$) at all depths. From May to November the lake exhibits a stable stratification, with a strong chemocline in the metalimnion. Vertical mixing is strong in the epilimnion (0-7 m), weak in the metalimnion (7-9 m), and again stronger in the hypolimnion (9-16 m). Thus $K_{Z}$ is depth-dependent. 
# 
# During the stratified period, CH<sub>4</sub> released from the lake sediment accumulates in the hypolimnion. As an example, the figure below shows observed CH<sub>4</sub> concentrations in Rotsee in May (white squares) June (grey squares) and July (black squares). CH<sub>4</sub> concentrations in the hypolimnion strongly increased with time, while those in the epilimnion remained constant at a low level of about 1 $\mu$mol m<sup>-3</sup>. In this exercise you will simulate this behaviour of the CH<sub>4</sub> concentrations, starting from the mixed condition with low methane concentrations at the end of the mixing period in spring.

# ![CH4](img/CH4_Rotsee.png)

# ## Mathematical model
# 
# The behavior of any dissolved substance in the water column can be described by using a one-dimensional (vertical, all the variables are averaged over horizontal cross sections) model that solves a partial differential equation (PDE). In the PDE we can incorporate terms describing the diffusive spread of the substance, the advective motion of the water column, any exchange with the sediment pore water, transformation processes that may occur and possible inflows/outflows (sources/sinks). 
# The following simplified PDE describes the methane concentration in Rotsee taking into consideration only the vertical diffusive transport and aerobic CH4 oxidation:
# 
# $$
# \frac{\partial C_{L}(t)}{\partial t} = \frac{1}{A(z)} \frac{\partial}{\partial z} \biggl(A(z)K_{Z}\frac{\partial C_{L}(t)}{\partial z}\biggr) + r_{CL}C_{L}(t) - \frac{1}{A(z)}\frac{dA}{dz}F_{Sed}
# $$
# 
# where $C_{L}$ is the concentration of the methane in the lake, $A(z)$ the horizontal area of the lake at each depth, $K_{Z}$ is the coefficient of turbulent diffusion, $r_{CL}$ is the first-order CH<sub>4</sub> oxidation rate (negative value), and $F_{Sed}$ is the flux of methane from the sediment. The first term on the right hand side describes the diffusive transport, the second the oxidation, and the third the flux from the sediment. The exchange with the atmosphere is neglected in this example.

# ## Numerical solution
# 
# This partial differential equation can be solved using the FiPy PDE solver, as described in the lecture. However, in this case, only one state variable is simulated.
# First the required packages are imported:

# In[6]:


from fipy import (CellVariable, Grid1D, TransientTerm, DiffusionTerm, 
    PowerLawConvectionTerm, FaceVariable)
from fipy.terms import ImplicitSourceTerm
import numpy as np
import matplotlib.pyplot as plt


# Then the numerical grid both in space and time is defined:

# In[8]:


# Grid definition

Zmax = 16  # Lake depth
N = 17  # number of nodes
dz = Zmax / N
mesh = Grid1D (nx=N, dx=dz)

tmax = 200  # maximum simulation time [d]
M = 201  # number of timesteps
t = np.linspace(0.0, tmax, M)
dt = tmax / (M - 1)


# We assume that the turbulent diffusivity is changing with depth as follows
# - epilimnion (0-7 m depth) $K_{Z}$ = 100 m<sup>2</sup> d<sup>-1</sup>
# - metalimnion (7-9 m depth) $K_{Z}$ = 0.05 m<sup>2</sup> d<sup>-1</sup>
# - hypolimnion (9-16 m depth)  $K_{Z}$ = 0.1 m<sup>2</sup> d<sup>-1</sup> 
# 
# Define diffusivity as a function of z:

# In[10]:


# Define a function for Kz

z = mesh.x

Kz_epi = 100  # turbulent diffusivity in the epilimnion
Kz_meta = 0.05  # turbulent diffusivity in the metalimnion
Kz_hypo = 0.1  # turbulent diffusivity in the hypolimnion
z_epi = 7  # lower boundary of epilimnion
z_meta = 9  # lower boundary of metalimnion
Kz = Kz_epi*(z < z_epi) + Kz_meta*(z >= z_epi)*(z < z_meta) + Kz_hypo*(z >= z_meta)
Kzvar = CellVariable(mesh=mesh, value=Kz)


# Now define the remaining model parameters:
# - Area as a function of depth
# - Methane oxiadation rate
# - Methane source from sediment
# 
# Assume that the lake area decreases with depth as A(z) = 4.8 $\cdot{}$ 10<sup>5</sup> – 3 $\cdot{}$ 10<sup>4</sup> z. Use a first order oxidation rate $r_{CL}$ = - 0.05 d<sup>-1</sup> (only in the epilimnion, as there is no oxygen in the hypolimnion!), and a flux from the sediment (source term at all depths) of 5 mmol m<sup>-2</sup> d<sup>-1</sup>.
# 
# Also convert depth-dependendent model parameters to FiPy CellVariables.

# In[12]:


# Define model parameters
Az = np.maximum(-30000 * z + 4.8e5, 0)  # Can't be less than 0
Avar = CellVariable(mesh=mesh, value=Az)
dAdz = -30000 # area gradient with depth
k_ox = -0.00*(z < z_epi) # methane oxidation rate
k_ox_var = CellVariable(mesh=mesh, value=k_ox)
F_sed = 5  # sediment flux (mmol/m2/d)


# Next, define the state variable for methane, and its initial and boundary conditions (no flux at surface and bottom).

# In[14]:


# Define state variable, initial and boundary conditions
CH4 = CellVariable(mesh=mesh, name="CH4", value=0.01, hasOld=True)

# no flux boundary conditions for the surface (left) and bottom (right)
CH4_grad_surf = 0.0
CH4.faceGrad.constrain(CH4_grad_surf, mesh.facesLeft)
CH4_grad_bot = 0.0
CH4.faceGrad.constrain(CH4_grad_bot, mesh.facesRight)


# Define the partial differential equation to be solved and its coefficients.

# In[16]:


# Define the PDE with its coefficients

diffCoeff = Avar * Kzvar # coefficient in diffusive term

CH4_equation = (
    TransientTerm(coeff=Avar, var=CH4)
    == DiffusionTerm(coeff=diffCoeff, var=CH4)
     + ImplicitSourceTerm(coeff=k_ox_var*Avar, var=CH4)
     - F_sed*dAdz
)


# Run the solver.
# 

# In[18]:


# Pre-allocate memory for solution
sol_CH4 = np.zeros((N, M))
sol_CH4[:, 0] = CH4.value

# Solve the differential equation

CH4.updateOld()
for n in range(1,M):
    CH4_equation.solve(var=CH4, dt=dt)
    sol_CH4[:, n] = CH4.value


# Plot CH<sub>4</sub> vs depth every 30 days in the same plot. Do the results qualitatively agree with the observations presented in the figure above?

# In[20]:


# -----------------------Plots-----------------------------
# plot the concentration of methane as a function of both time
# and space as a contour plot with the function surf in a new figure.
plt.figure()
plt.contourf(t, z, sol_CH4)
plt.gca().invert_yaxis()
plt.ylabel('Depth [m]')
plt.xlabel('Time [d]')
plt.colorbar().set_label('CH$_4$ concentration [mmol/m$^3$]')
plt.show()

# plot vertical concentrations of methane every 30 days as a function of
# depth into a new figure.
plt.figure()
# Move origin to top left corner
ax = plt.subplot(1, 1, 1)
ax.invert_yaxis()
plt.ylabel('Depth [m]')
plt.xlabel('CH$_4$ concentration [mmol/m$^3$]')
for n in range(0, int(len(t) / 30)):
    plt.plot(sol_CH4[:, n * 30], z)
plt.legend(['0 days', '30 days', '60 days', '90 days', '120 days', '150 days', '180 days', '210 days'])



# Take a piece of paper and draw by hand the expected vertical profile of the <b>final</b> concentration of methane in the lake for the following cases:
# - Oxidation rate increased by a factor of 10
# - Methane flux increased by a factor of 10
# - Diffusivity in the hypolimnion increased by a factor of 10
# - Diffusivity in the metalimnion increased by a factor of 10
# - Diffusivity in the epilimnion decreased by a factor of 10.
# 
# Modify the model parameters in the code individually as listed above and see whether the results of the simulations agree with your expectations.
# 

# In[ ]:





# In[ ]:





# In[ ]:




