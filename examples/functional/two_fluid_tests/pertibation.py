import sympy as sp

x = sp.symbols('x')
y = sp.symbols('y')
z = sp.symbols('z')
l = sp.symbols('l')
phi = sp.cos(2 * sp.pi * x / l) * sp.cos(2 * sp.pi * y / l)

# pert_b_z = ez X grad(phi)
ex = 0
ey = 0
ez = 1
grad_phi_x = sp.diff(phi, x)
grad_phi_y = sp.diff(phi, y)
grad_phi_z = sp.diff(phi, z)

pert_b_x = (ey * grad_phi_z - ez * grad_phi_y)
pert_b_y = -(ex * grad_phi_z - ez * grad_phi_x)
pert_b_z = (ex * grad_phi_y - ey * grad_phi_x)

print(pert_b_x)
print(pert_b_y)
print(pert_b_z)

# Check div(b) = 0
div_b = sp.diff(pert_b_x, x) + sp.diff(pert_b_y, y) + sp.diff(pert_b_z, z)
print(div_b)
