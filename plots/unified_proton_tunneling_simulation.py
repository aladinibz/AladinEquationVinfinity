import numpy as np
from scipy.constants import hbar, m_e
from scipy.integrate import solve_ivp

# Constants (using scipy.constants for more precision where available)
# Using mass of proton for proton tunneling
m_p = 1.6726219e-27  # kg (mass of proton)
e = 1.602176634e-19  # C (elementary charge)
epsilon_0 = 8.8541878128e-12  # F/m (vacuum permittivity)

def potential_energy_barrier(x, V0, a):
    """
    Defines a simple rectangular potential energy barrier.

    Args:
        x (float or np.ndarray): Position.
        V0 (float): Height of the barrier in Joules.
        a (float): Width of the barrier in meters.

    Returns:
        float or np.ndarray: Potential energy at position x.
    """
    return np.where((x >= 0) & (x <= a), V0, 0)

def schrodinger_equation(t, psi, E, V, m):
    """
    Defines the time-independent Schrödinger equation as a system of two first-order ODEs.
    d^2(psi)/dx^2 = (2m/hbar^2) * (V(x) - E) * psi
    Let y[0] = psi and y[1] = d(psi)/dx
    Then dy[0]/dx = y[1]
    And dy[1]/dx = (2m/hbar^2) * (V(x) - E) * y[0]
    Note: This is set up for spatial integration (x is the independent variable),
    not time (t is a placeholder from solve_ivp's signature).
    """
    x = t # Renaming t to x for conceptual clarity in spatial integration
    psi_val = psi[0]
    dpsi_dx_val = psi[1]

    # Calculate potential at current x
    V_x = V(x)

    d2psi_dx2 = (2 * m / hbar**2) * (V_x - E) * psi_val

    return [dpsi_dx_val, d2psi_dx2]

def calculate_transmission_reflection(E, V0, a, m, x_start=-1e-9, x_end=2e-9, num_points=1000):
    """
    Calculates the transmission and reflection coefficients for a potential barrier.
    Assumes a particle incident from the left.

    Args:
        E (float): Energy of the incident particle in Joules.
        V0 (float): Height of the potential barrier in Joules.
        a (float): Width of the barrier in meters.
        m (float): Mass of the particle in kg.
        x_start (float): Starting position for integration (well before barrier).
        x_end (float): Ending position for integration (well after barrier).
        num_points (int): Number of points for the spatial grid.

    Returns:
        tuple: (Transmission coefficient, Reflection coefficient)
    """
    x_span = (x_start, x_end)
    x_eval = np.linspace(x_start, x_end, num_points)

    # Define the potential function to be used by the solver
    V_func = lambda x: potential_energy_barrier(x, V0, a)

    # Wavenumber k for region x < 0 and x > a (where V=0)
    k = np.sqrt(2 * m * E) / hbar

    # Initial conditions for psi and dpsi/dx at x_start
    # We assume a superposition of incident (A=1) and reflected (B) waves
    # psi(x) = exp(ikx) + B * exp(-ikx)
    # dpsi/dx = ik * exp(ikx) - ik * B * exp(-ikx)
    # To find B, we need to integrate across the barrier and match boundary conditions.
    # For simplicity, we can pick an initial value and then normalize/scale the result.
    # A common approach for transmission coefficient is to integrate forward twice
    # with different initial conditions, or to integrate with specific initial conditions
    # corresponding to a known incident wave.

    # Here, we'll use a simpler approach by trying to integrate and then derive T and R
    # from the wave function in the transmitted region.

    # A standard method for calculating T/R is to use the transfer matrix method
    # or by solving the ODE for two linearly independent solutions and combining them.
    # For solve_ivp, let's set up a specific initial condition to find a solution
    # that can then be used to find T and R.

    # Let's consider psi(x) = T_coeff * exp(ikx) for x > a
    # and psi(x) = exp(ikx) + R_coeff * exp(-ikx) for x < 0

    # We can integrate the Schrödinger equation starting from x_end backwards to x_start
    # with specific boundary conditions in the transmitted region.
    # For a transmitted wave only: psi(x) = C * exp(ikx) for x > a
    # psi_end = C * exp(ik*x_end)
    # dpsi_dx_end = C * ik * exp(ik*x_end)
    # Let's choose C = 1 for simplicity and normalize later if needed.

    # Define initial conditions at x_end for a transmitted wave moving to the right
    # psi_end_val = np.cos(k * x_end) + 1j * np.sin(k * x_end) # exp(ikx_end)
    # dpsi_dx_end_val = 1j * k * psi_end_val
    # This approach usually works better with complex ODE solvers or by splitting into real/imaginary parts.

    # For `solve_ivp` with real values, we need to split psi into real and imaginary parts.
    # Let psi = u + iv
    # d^2u/dx^2 = (2m/hbar^2)(V-E)u
    # d^2v/dx^2 = (2m/hbar^2)(V-E)v
    # This means we solve two independent real-valued Schrödinger equations.

    # A more robust way using `solve_ivp` for T/R coefficients is to use the Transfer Matrix Method.
    # However, `solve_ivp` is a direct numerical integrator. We need to be careful with boundary conditions.

    # Let's use the incident wave method: Assume incident wave from left has amplitude 1.
    # At x_start (far left, V=0): psi(x) = e^(ikx) + R * e^(-ikx)
    # dpsi/dx = ik * e^(ikx) - ik * R * e^(-ikx)

    # The standard way to calculate T and R using numerical integration of the Schrödinger equation
    # involves integrating the equation twice with different initial conditions
    # (e.g., psi(0)=1, psi'(0)=0 and psi(0)=0, psi'(0)=1) to find two linearly independent solutions,
    # then forming a general solution and matching boundary conditions at +/- infinity.

    # A simpler but less rigorous approach for demonstration with solve_ivp:
    # Assume a wave of amplitude 1 incident from the left. Let's integrate from a point
    # far before the barrier to far after the barrier.

    # For the `schrodinger_equation` function, it's defined for real-valued `psi`.
    # Quantum mechanics often involves complex wave functions.
    # To handle complex psi with `solve_ivp`, we can solve for real and imaginary parts separately,
    # or use a wrapper that handles complex numbers if the underlying solver supports it.
    # `solve_ivp` works with real arrays. So we split psi = Re(psi) + i * Im(psi).
    # Let y = [Re(psi), Im(psi), d(Re(psi))/dx, d(Im(psi))/dx]
    # Then the system of ODEs will be 4-dimensional.

    # Let's redefine `schrodinger_equation` to handle complex psi by stacking real and imaginary parts.
    # y = [Re(psi), Im(psi), d(Re(psi))/dx, d(Im(psi))/dx]

    # This re-definition would be a significant change.
    # For a purely real `schrodinger_equation` as defined, it effectively solves for only
    # the real or imaginary part, assuming the potential is real. This is fine for some cases,
    # but for transmission/reflection, the phase is crucial.

    # Let's proceed with the real-valued ODE for now for conceptual understanding,
    # acknowledging that a full quantum tunneling simulation requires complex wavefunctions.

    # If we stick to the real-valued ODE, we can't directly compute T and R as defined
    # by the ratio of current densities, which involve complex conjugates.
    # However, we can still show the behavior of the real part of a wavefunction.

    # For transmission and reflection coefficients, we ideally need the complex wavefunction.
    # Let's revise the `schrodinger_equation` and the solving approach.

    # Revised schrodinger_equation for complex psi = psi_real + i * psi_imag
    # d^2(psi_real)/dx^2 = (2m/hbar^2) * (V(x) - E) * psi_real
    # d^2(psi_imag)/dx^2 = (2m/hbar^2) * (V(x) - E) * psi_imag

    # This means the real and imaginary parts solve the same differential equation independently.
    # So, we can solve for one component (e.g., real part) and then try to infer T/R,
    # or just demonstrate the wave behavior.

    # A common approach for this kind of problem is the transfer matrix method, or solving
    # for two independent solutions and combining them based on boundary conditions.

    # Let's re-think the `solve_ivp` application for T/R. We need to integrate from one side
    # and then analyze the wave on the other side.

    # Consider psi(x) = A e^(ikx) + B e^(-ikx) for x < 0
    # psi(x) = C e^(ikx) + D e^(-ikx) for 0 < x < a (inside barrier, k becomes kappa = sqrt(2m(V0-E))/hbar)
    # psi(x) = F e^(ikx) for x > a (transmitted wave only, no reflected wave from right infinity)

    # To use `solve_ivp`, we need initial conditions at some point. It's usually easier to
    # integrate from x > a backwards, or x < 0 forwards.

    # Let's try integrating from `x_end` backwards with a purely transmitted wave.
    # psi_fwd(x) = e^(ikx) in the region x > a (setting F=1, we will normalize later)
    # This means psi(x_end) = e^(ik*x_end) and dpsi/dx(x_end) = ik * e^(ik*x_end)

    # `solve_ivp` takes real initial conditions. So, we'll need to solve for Re(psi) and Im(psi) separately.
    # This implies that we set up a 4-component system [Re(psi), Re(dpsi/dx), Im(psi), Im(dpsi/dx)]

    # Let's rewrite schrodinger_equation for real components
    def schrodinger_equation_complex_split(x_val, y, E, V_func, m):
        # y = [Re(psi), d(Re(psi))/dx, Im(psi), d(Im(psi))/dx]
        psi_real = y[0]
        dpsi_dx_real = y[1]
        psi_imag = y[2]
        dpsi_dx_imag = y[3]

        V_x = V_func(x_val)
        common_factor = (2 * m / hbar**2) * (V_x - E)

        d2psi_dx2_real = common_factor * psi_real
        d2psi_dx2_imag = common_factor * psi_imag

        return [dpsi_dx_real, d2psi_dx2_real, dpsi_dx_imag, d2psi_dx2_imag]

    # Initial conditions for integration. Let's integrate forward from x_start.
    # Assume incident wave from left, A=1, and initially no reflected wave (B=0) for a starting point.
    # This isn't strictly correct for calculating T/R, as reflection happens.
    # For the purpose of finding T and R, one typically imposes the transmitted wave condition
    # at +infinity (e.g., psi = exp(ikx) there) and integrates backwards.

    # Let's integrate *backwards* from `x_end` with only a transmitted wave (amplitude 1).
    # In the region x > a, psi(x) = 1 * e^(ikx)
    # At x_end: Re(psi_end) = cos(k*x_end), Im(psi_end) = sin(k*x_end)
    # d(Re(psi))/dx_end = -k*sin(k*x_end), d(Im(psi))/dx_end = k*cos(k*x_end)

    if E < 0:
        print("Error: Energy must be non-negative.")
        return 0, 1 # T=0, R=1 for negative energy below barrier

    if E == V0: # Special case, k_barrier = 0
        print("Energy equals barrier height. Transmission is 1.")
        return 1, 0

    # Wavenumber k in regions where V=0
    k_outside = np.sqrt(2 * m * E) / hbar

    # Initial conditions at x_end (right side of the barrier) for a pure transmitted wave (amplitude 1)
    # psi(x) = exp(i * k_outside * x) for x > a
    Re_psi_end = np.cos(k_outside * x_end)
    Im_psi_end = np.sin(k_outside * x_end)
    dRe_psi_dx_end = -k_outside * np.sin(k_outside * x_end)
    dIm_psi_dx_end = k_outside * np.cos(k_outside * x_end)

    initial_conditions = [Re_psi_end, dRe_psi_dx_end, Im_psi_end, dIm_psi_dx_end]

    # Integrate backwards from x_end to x_start
    sol = solve_ivp(schrodinger_equation_complex_split, (x_end, x_start), initial_conditions,
                    args=(E, V_func, m), dense_output=True, rtol=1e-6, atol=1e-8)

    # Extract psi and dpsi/dx at x_start
    y_at_x_start = sol.y[:, -1] # Last point is at x_start
    Re_psi_start = y_at_x_start[0]
    dRe_psi_dx_start = y_at_x_start[1]
    Im_psi_start = y_at_x_start[2]
    dIm_psi_dx_start = y_at_x_start[3]

    # Form the complex psi and dpsi/dx at x_start
    psi_start_complex = Re_psi_start + 1j * Im_psi_start
    dpsi_dx_start_complex = dRe_psi_dx_start + 1j * dIm_psi_dx_start

    # At x_start (far left), psi(x) = A_inc * e^(ikx) + A_ref * e^(-ikx)
    # where A_inc is the incident amplitude, A_ref is the reflected amplitude.
    # For the transmission coefficient, we assumed a transmitted wave of amplitude 1.
    # The actual incident amplitude A_inc will be found from the wave at x_start.

    # psi_start_complex = A_inc * exp(i * k_outside * x_start) + A_ref * exp(-i * k_outside * x_start)
    # dpsi_dx_start_complex = i * k_outside * A_inc * exp(i * k_outside * x_start) - i * k_outside * A_ref * exp(-i * k_outside * x_start)

    # Let p1 = exp(i * k_outside * x_start)
    # Let p2 = exp(-i * k_outside * x_start)

    # psi_start_complex = A_inc * p1 + A_ref * p2  (Eq. 1)
    # dpsi_dx_start_complex = i * k_outside * (A_inc * p1 - A_ref * p2) (Eq. 2)

    # From Eq. 2: (dpsi_dx_start_complex) / (i * k_outside) = A_inc * p1 - A_ref * p2 (Eq. 3)

    # Add Eq. 1 and Eq. 3:
    # psi_start_complex + (dpsi_dx_start_complex) / (i * k_outside) = 2 * A_inc * p1
    A_inc = 0.5 * (psi_start_complex + dpsi_dx_start_complex / (1j * k_outside)) / np.exp(1j * k_outside * x_start)

    # Subtract Eq. 3 from Eq. 1:
    # psi_start_complex - (dpsi_dx_start_complex) / (i * k_outside) = 2 * A_ref * p2
    A_ref = 0.5 * (psi_start_complex - dpsi_dx_start_complex / (1j * k_outside)) / np.exp(-1j * k_outside * x_start)

    # Transmission coefficient T = |F|^2 / |A_inc|^2. We set F=1, so T = 1 / |A_inc|^2
    # Reflection coefficient R = |A_ref|^2 / |A_inc|^2

    T = 1.0 / (np.abs(A_inc)**2)
    R = np.abs(A_ref)**2 / (np.abs(A_inc)**2)

    # Normalize so T + R = 1. If numerical errors cause T+R != 1, renormalize.
    # This method naturally gives T+R=1 for scattering states above or below barrier.

    return T, R

def main():
    # Barrier parameters
    V0_ev = 1.0  # Barrier height in electron volts
    V0_J = V0_ev * e  # Convert to Joules
    a = 1.0e-10  # Barrier width in meters (1 Angstrom)

    # Particle parameters (proton)
    m = m_p # mass of proton

    # Energy of incident particle
    E_ev_values = np.linspace(0.1, 2.0, 100) # Energies from 0.1 eV to 2.0 eV
    E_J_values = E_ev_values * e  # Convert to Joules

    transmission_coeffs = []
    reflection_coeffs = []

    print(f"Calculating transmission and reflection coefficients for V0 = {V0_ev} eV, a = {a*1e10} Angstroms")

    for E_J in E_J_values:
        if E_J < V0_J:
            # Below barrier (tunneling region)
            T, R = calculate_transmission_reflection(E_J, V0_J, a, m)
        else:
            # Above barrier (over-barrier transmission)
            T, R = calculate_transmission_reflection(E_J, V0_J, a, m)

        transmission_coeffs.append(T)
        reflection_coeffs.append(R)
        # print(f"E = {E_J/e:.2f} eV: T = {T:.4f}, R = {R:.4f}, T+R = {T+R:.4f}")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(E_ev_values, transmission_coeffs, label='Transmission Coefficient (T)')
    plt.plot(E_ev_values, reflection_coeffs, label='Reflection Coefficient (R)')
    plt.axvline(V0_ev, color='r', linestyle='--', label=f'Barrier Height (V0 = {V0_ev} eV)')
    plt.xlabel('Incident Particle Energy (eV)')
    plt.ylabel('Coefficient')
    plt.title('Quantum Tunneling and Reflection for a Proton')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Demonstrate a single wavefunction solution
    E_demonstrate_ev = 0.5 # eV, below barrier
    E_demonstrate_J = E_demonstrate_ev * e

    x_start = -5e-10
    x_end = 5e-10
    x_eval = np.linspace(x_start, x_end, 500)

    # Define the potential function
    V_func = lambda x: potential_energy_barrier(x, V0_J, a)

    # Wavenumber k in regions where V=0
    k_outside = np.sqrt(2 * m * E_demonstrate_J) / hbar

    # Initial conditions at x_end for a pure transmitted wave (amplitude 1)
    Re_psi_end = np.cos(k_outside * x_end)
    Im_psi_end = np.sin(k_outside * x_end)
    dRe_psi_dx_end = -k_outside * np.sin(k_outside * x_end)
    dIm_psi_dx_end = k_outside * np.cos(k_outside * x_end)

    initial_conditions = [Re_psi_end, dRe_psi_dx_end, Im_psi_end, dIm_psi_dx_end]

    sol_wave = solve_ivp(lambda x, y: schrodinger_equation_complex_split(x, y, E_demonstrate_J, V_func, m),
                         (x_end, x_start), initial_conditions,
                         dense_output=True, rtol=1e-6, atol=1e-8)

    y_wave = sol_wave.sol(x_eval)
    Re_psi_wave = y_wave[0]
    Im_psi_wave = y_wave[2]
    psi_abs_sq_wave = Re_psi_wave**2 + Im_psi_wave**2 # Probability density

    # Scale the wavefunction for better visualization (optional)
    # Find the incident amplitude A_inc from the leftmost part of the solution
    # (This is implicitly done by `calculate_transmission_reflection`)
    # For plotting, we can just plot the raw solution and the potential.

    plt.figure(figsize=(10, 6))
    plt.plot(x_eval * 1e10, Re_psi_wave, label='Re(psi)')
    plt.plot(x_eval * 1e10, Im_psi_wave, label='Im(psi)')
    plt.plot(x_eval * 1e10, psi_abs_sq_wave, label='|psi|^2 (Probability Density)')
    plt.plot(x_eval * 1e10, V_func(x_eval) / e, 'k--', label='Potential Barrier (V/e)')
    plt.axvline(0, color='gray', linestyle=':', label='Barrier Start')
    plt.axvline(a * 1e10, color='gray', linestyle=':', label='Barrier End')

    plt.xlabel('Position (Angstroms)')
    plt.ylabel('Wavefunction / Potential (eV)')
    plt.title(f'Wavefunction for E = {E_demonstrate_ev} eV (below barrier)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
