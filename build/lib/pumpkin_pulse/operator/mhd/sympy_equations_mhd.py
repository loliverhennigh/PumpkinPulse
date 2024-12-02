import sympy as sp
import subprocess

def append_equation_to_latex(eq, name, latex_contents):
    latex_contents += r"""
\subsection{""" + name + r"""}
\centering
\begin{equation}
\centering
""" + sp.latex(eq) + r"""
\end{equation}
"""
    # Remove all strings like \left(x, y, z, t\right)
    latex_contents = latex_contents.replace(r'{\left(x,y,z,t \right)}', '')
    return latex_contents


# Define the symbols
x, y, z, t = sp.symbols('x y z t')
species_mass = sp.symbols('m_s')
charge = sp.symbols('q_s')
gamma = sp.symbols('gamma')
u_0 = sp.symbols('u_0')
c = sp.symbols('c')

# Make a functions
density = sp.Function('rho')(x, y, z, t)
vel_x = sp.Function('vx')(x, y, z, t)
vel_y = sp.Function('vy')(x, y, z, t)
vel_z = sp.Function('vz')(x, y, z, t)
pressure = sp.Function('p')(x, y, z, t)
mag_x = sp.Function('bx')(x, y, z, t)
mag_y = sp.Function('by')(x, y, z, t)
mag_z = sp.Function('bz')(x, y, z, t)
electric_x = sp.Function('ex')(x, y, z, t)
electric_y = sp.Function('ey')(x, y, z, t)
electric_z = sp.Function('ez')(x, y, z, t)

# Make conservative variables
mom_x = vel_x * density
mom_y = vel_y * density
mom_z = vel_z * density
energy = (pressure / (gamma - 1)) + 0.5 * density * (vel_x**2 + vel_y**2 + vel_z**2)

# Make Equations
# Continuity equation
continuity_eq = sp.Eq(sp.diff(density, t) + sp.diff(mom_x, x) + sp.diff(mom_y, y) + sp.diff(mom_z, z), 0)

# Momentum equation
momentum_eq_x = sp.Eq(
    (
        sp.diff(mom_x, t)
        + sp.diff(density * vel_x * vel_x + pressure, x)
        + sp.diff(density * vel_x * vel_y, y)
        + sp.diff(density * vel_x * vel_z, z)
    ),
    (
        density * charge / species_mass * (electric_x + vel_y * mag_z - vel_z * mag_y)
    )
)
momentum_eq_y = sp.Eq(
    (
        sp.diff(mom_y, t)
        + sp.diff(density * vel_y * vel_x, x)
        + sp.diff(density * vel_y * vel_y + pressure, y)
        + sp.diff(density * vel_y * vel_z, z)
    ),
    (
        density * charge / species_mass * (electric_y + vel_z * mag_x - vel_x * mag_z)
    )
)
momentum_eq_z = sp.Eq(
    (
        sp.diff(mom_z, t)
        + sp.diff(density * vel_z * vel_x, x)
        + sp.diff(density * vel_z * vel_y, y)
        + sp.diff(density * vel_z * vel_z + pressure, z)
    ),
    (
        density * charge / species_mass * (electric_z + vel_x * mag_y - vel_y * mag_x)
    )
)

# Energy equation
energy_eq = sp.Eq(
    (
       sp.diff(energy, t)
       + sp.diff(vel_x * (energy + pressure), x)
       + sp.diff(vel_y * (energy + pressure), y)
       + sp.diff(vel_z * (energy + pressure), z)
    ),
    (
        density * charge / species_mass * (vel_x * electric_x + vel_y * electric_y + vel_z * electric_z)
    )
)

# Solve for velocity equation
density_dt_eq = sp.solve(continuity_eq, sp.diff(density, t))[0]
density_dt_eq = sp.simplify(density_dt_eq)
density_dt_eq = sp.Eq(sp.diff(density, t), density_dt_eq)

# Solve for velocity x equation
velocity_x_dt_eq = sp.solve(momentum_eq_x, sp.diff(vel_x, t))[0]
velocity_x_dt_eq = velocity_x_dt_eq.subs(sp.diff(density, t), density_dt_eq.lhs)
velocity_x_dt_eq = sp.simplify(velocity_x_dt_eq)
velocity_x_dt_eq = sp.Eq(sp.diff(vel_x, t), velocity_x_dt_eq)

# Solve for velocity y equation
velocity_y_dt_eq = sp.solve(momentum_eq_y, sp.diff(vel_y, t))[0]
velocity_y_dt_eq = velocity_y_dt_eq.subs(sp.diff(density, t), density_dt_eq.lhs)
velocity_y_dt_eq = sp.simplify(velocity_y_dt_eq)
velocity_y_dt_eq = sp.Eq(sp.diff(vel_y, t), velocity_y_dt_eq)

# Solve for velocity z equation
velocity_z_dt_eq = sp.solve(momentum_eq_z, sp.diff(vel_z, t))[0]
velocity_z_dt_eq = velocity_z_dt_eq.subs(sp.diff(density, t), density_dt_eq.lhs)
velocity_z_dt_eq = sp.simplify(velocity_z_dt_eq)
velocity_z_dt_eq = sp.Eq(sp.diff(vel_z, t), velocity_z_dt_eq)

# Solve for pressure equation
pressure_dt_eq = sp.solve(energy_eq, sp.diff(pressure, t))[0]
pressure_dt_eq = pressure_dt_eq.subs(sp.diff(density, t), density_dt_eq.lhs)
pressure_dt_eq = pressure_dt_eq.subs(sp.diff(vel_x, t), velocity_x_dt_eq.lhs)
pressure_dt_eq = pressure_dt_eq.subs(sp.diff(vel_y, t), velocity_y_dt_eq.lhs)
pressure_dt_eq = pressure_dt_eq.subs(sp.diff(vel_z, t), velocity_z_dt_eq.lhs)
pressure_dt_eq = sp.simplify(pressure_dt_eq)
pressure_dt_eq = sp.Eq(sp.diff(pressure, t), pressure_dt_eq)

# Write the equations to tex file
latex_contents = r"""
\documentclass{article}
\usepackage{amsmath}
\usepackage{geometry}
\geometry{
    paperwidth=25in,   % Custom width
    paperheight=15in,  % Custom height
    left=0.1in,
    right=0.1in,
    top=0.1in,
    bottom=0.1in,
}
\begin{document}
\section{Conservation Equations}

"""
latex_contents = append_equation_to_latex(continuity_eq, 'Continuity Equation', latex_contents)
latex_contents = append_equation_to_latex(momentum_eq_x, 'Momentum Equation X', latex_contents)
latex_contents = append_equation_to_latex(momentum_eq_y, 'Momentum Equation Y', latex_contents)
latex_contents = append_equation_to_latex(momentum_eq_z, 'Momentum Equation Z', latex_contents)
latex_contents = append_equation_to_latex(energy_eq, 'Energy Equation', latex_contents)
latex_contents = latex_contents + r"""
\subsection{Primitive Variable Equations}
"""
latex_contents = append_equation_to_latex(density_dt_eq, 'Density Equation', latex_contents)
latex_contents = append_equation_to_latex(velocity_x_dt_eq, 'Velocity X Equation', latex_contents)
latex_contents = append_equation_to_latex(velocity_y_dt_eq, 'Velocity Y Equation', latex_contents)
latex_contents = append_equation_to_latex(velocity_z_dt_eq, 'Velocity Z Equation', latex_contents)
latex_contents = append_equation_to_latex(pressure_dt_eq, 'Pressure Equation', latex_contents)
latex_contents = latex_contents + r"""
\end{document}
"""
with open('conservation_equations.tex', 'w') as f:
    f.write(latex_contents)
subprocess.call(['pdflatex', 'conservation_equations.tex'])

# Magnetic reconnection equation
b_0 = 0.1
lambda_0 = 0.5
n_0 = 1.0
b_x = 0.0
b_y = b_0 * sp.tanh(y / lambda_0)
b_z = 0.0
j_x = - (b_0 / lambda_0) * sp.cosh(y / lambda_0)**(-2)
j_y = 0.0
j_z = 0.0
n = n_0 * ((1.0 / 5.0) + (1.0 / sp.cosh(y / lambda_0)) ** 2.0)
p = (b_0 / 12.0) * n
v_x = j_x / (n * charge)
v_y = j_y / (n * charge)
v_z = j_z / (n * charge)

# Check equations



# Solve the equation
#solution = sp.solve(eq, phi_1_1_1)
#solution = sp.simplify(solution[0])
#print(solution)

