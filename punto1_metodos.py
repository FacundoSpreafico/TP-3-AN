
import numpy as np
import sympy as sp

def newton_taylor_2nd_func(f, f1, f2, x0, tol=1e-8, max_iter=100):
    xn = x0
    for n in range(max_iter):
        f_val = f(xn)
        f1_val = f1(xn)
        f2_val = f2(xn)
        den = 2*f1_val**2 - f_val*f2_val
        if den == 0:
            return None, n, False
        xn_next = xn - (2*f_val*f1_val) / den
        if abs(xn_next - xn) < tol:
            return xn_next, n + 1, True
        xn = xn_next
    return xn, max_iter, False

def bisection_method(f, a, b, tol=1e-8, max_iter=100):
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) y f(b) deben tener signos opuestos")
    for i in range(max_iter):
        c = (a + b) / 2.0
        if abs(f(c)) < tol or (b - a) / 2 < tol:
            return c, i + 1
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return c, max_iter

def hybrid_method(f, f1, f2, a, b, tol=1e-8, max_iter=100):
    c, iter_bis = bisection_method(f, a, b, tol, max_iter//2)
    raiz, iter_newton, converged = newton_taylor_2nd_func(f, f1, f2, c, tol, max_iter//2)
    return raiz, iter_bis + iter_newton, converged

# Definimos la función f(x) = cos(x) - x y sus derivadas
x = sp.Symbol('x')
f_expr = sp.cos(x) - x
f = sp.lambdify(x, f_expr, 'numpy')
f1 = sp.lambdify(x, f_expr.diff(x), 'numpy')
f2 = sp.lambdify(x, f_expr.diff(x, 2), 'numpy')

# Ejecutamos los métodos
raiz_taylor, iters_taylor, ok_taylor = newton_taylor_2nd_func(f, f1, f2, 0.5)
raiz_hybrid, iters_hybrid, ok_hybrid = hybrid_method(f, f1, f2, 0, 1)

# Imprimimos resultados
print("Método Newton-Taylor 2° orden:")
print("Raíz =", raiz_taylor)
print("Iteraciones =", iters_taylor)
print("¿Convergió?", ok_taylor)

print("\nMétodo Híbrido (Bisección + Newton-Taylor):")
print("Raíz =", raiz_hybrid)
print("Iteraciones =", iters_hybrid)
print("¿Convergió?", ok_hybrid)
