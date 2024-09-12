import sympy as sp

x = sp.symbols('x')
y = sp.symbols('y')
z = sp.symbols('z')
gamma = sp.symbols('gamma')
b_0 = sp.symbols('b_0')
lambda_0 = sp.symbols('lambda_0')
n_0 = sp.symbols('n_0')
charge = sp.symbols('charge')
pressure = sp.Function('pressure')(x, y, z)

# Make j
j_x = 0
j_y = 0
j_z = - (b_0 / lambda_0) * ((1.0 / sp.cosh((y + 2.0 * sp.pi) / lambda_0)) ** 2.0 - (1.0 / sp.cosh((y - 2.0 * sp.pi) / lambda_0)) ** 2.0)

# Make n
n = n_0 * ((1.0 / 5.0) + ((1.0 / sp.cosh((y + 2.0 * sp.pi) / lambda_0)) ** 2.0) + ((1.0 / sp.cosh((y - 2.0 * sp.pi) / lambda_0)) ** 2.0))

# Get velocity
v_x = j_x / (n * charge)
v_y = j_y / (n * charge)
v_z = j_z / (n * charge)

# Make energy
e = (pressure / (gamma - 1.0)) + (0.5 * n * (v_x ** 2.0 + v_y ** 2.0 + v_z ** 2.0))

# Make equation
eq = sp.Eq(sp.diff((e + pressure) * v_x, x) + sp.diff((e + pressure) * v_y, y) + sp.diff((e + pressure) * v_z, z), 0)

# Substitute in the pressure
#eq = eq.subs(pressure, (b_0 / 12.0) * n)

# solve for dp/dz
dp_dz = sp.solve(eq, sp.diff(pressure, z))
print(dp_dz)
#print(eq)

# Simplify
#eq = sp.simplify(eq)
#
##
#print(eq)
