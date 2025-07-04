"""
aircraft_12dof_sim.py
Simple 12-state rigid-body aircraft dynamics demo.

States (index):
 0 u     body-x velocity (m/s)
 1 v     body-y velocity (m/s)
 2 w     body-z velocity (m/s)
 3 p     roll rate (rad/s)
 4 q     pitch rate (rad/s)
 5 r     yaw  rate (rad/s)
 6 phi   roll angle (rad)
 7 theta pitch angle (rad)
 8 psi   yaw  angle (rad)
 9 x_e   inertial east position (m)
10 y_n   inertial north position (m)
11 z_d   inertial down position (m)

The script integrates the equations of motion with a fixed-step RK4 solver.
Replace `forces_zero` and `moments_zero` with your aerodynamic / propulsion
models or control laws.
"""
import numpy as np

def gravity_force_body(mass, phi, theta, psi, g=9.81):
    """Return gravity force vector in the body frame."""
    # Gravity in NED frame = [0, 0, mg]
    g_ned = np.array([0.0, 0.0, mass * g])

    # Inertial (NED) to body frame rotation
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    R_i2b = np.array([
        [cth * cpsi, cth * spsi, -sth],
        [sphi*sth*cpsi - cphi*spsi, sphi*sth*spsi + cphi*cpsi, sphi*cth],
        [cphi*sth*cpsi + sphi*spsi, cphi*sth*spsi - sphi*cpsi, cphi*cth]
    ])

    return -R_i2b @ g_ned  # negative because gravity pulls down

def forces_with_controls(t, state, params):
    u, v, w, p, q, r, phi, theta, psi, *_ = state
    mass = params["mass"]

    # Gravity in body frame
    Fg = gravity_force_body(mass, phi, theta, psi)

    # Simplified thrust along body-x (from throttle)
    throttle = params.get("throttle", 0.0)
    Tmax = params.get("Tmax", 2000.0)  # N
    Fx = Tmax * throttle
    Ft = np.array([Fx, 0, 0])

    
    # Airspeed magnitude
    V = np.linalg.norm([u, v, w])

    # Angle of attack
    alpha = np.arctan2(w, u)

    # Aerodynamic parameters
    rho = params.get("rho", 1.225)  # air density at sea level kg/m^3
    S = params.get("S", 16.2)       # wing area in m^2

    # Lift and drag coefficients (simple linear model)
    CL0 = params.get("CL0", 0.2)          # lift coef at alpha=0
    CLalpha = params.get("CLalpha", 5.5)  # per radian increase in alpha
    CD0 = params.get("CD0", 0.02)         # drag at at L=0
    k = params.get("k", 0.07)             # drag factor
    
    CL = CL0 + CLalpha * alpha            # compute lift coefficient
    CD = CD0 + k * CL**2                  # compute drag coefficient
    
    # Lift and Drag magnitudes
    qbar = 0.5 * rho * V**2               
    L = qbar * S * CL
    D = qbar * S * CD
    
    # Lift direction approx: perpendicular to velocity in body-x/w plane (positive z direction)
    # Drag direction: opposite velocity vector
    
    # Normalize wind vector
    v_hat = np.array([u, v, w]) / V
    
    # Drag vector (opposes velocity)
    F_drag = -D * v_hat
    
    # Lift vector (approximate lift perpendicular to velocity and lateral axis)
    # For simplicity, approximate lift direction as perpendicular to velocity in x-z plane:
    lift_dir = np.array([-np.sin(alpha), 0.0, np.cos(alpha)])  # perpendicular in x-z plane
    F_lift = L * lift_dir
    
    # Total aerodynamic force
    F_aero = F_lift + F_drag
    
    # Total force in body frame
    F_total = Fg + Ft + F_aero
    
    return F_total

def moments_with_controls(t, state, params):
    # Map control surface deflections to moments (simple linear model)
    delta_e = params.get("delta_e", 0.0)  # Elevator
    delta_a = params.get("delta_a", 0.0)  # Aileron
    delta_r = params.get("delta_r", 0.0)  # Rudder

    Me = 1000.0 * delta_e  # Pitch moment
    Ma = 800.0  * delta_a  # Roll moment
    Mr = 500.0  * delta_r  # Yaw moment

    return np.array([Ma, Me, Mr])

def aircraft_eom(t, state, forces_func, moments_func, params):
    """Return time derivative of 12-state vector for a rigid aircraft."""
    u, v, w, p, q, r, phi, theta, psi, x_e, y_n, z_d = state
    m      = params["mass"]
    Ix, Iy, Iz = params["I"]

    # External forces & moments in body axes
    X, Y, Z = forces_func(t, state, params)
    L, M, N = moments_func(t, state, params)

    # Translational dynamics (Newton, body frame)
    du = r*v - q*w + X/m
    dv = p*w - r*u + Y/m
    dw = q*u - p*v + Z/m

    # Rotational dynamics (Euler, body frame)
    dp = (L + (Iy - Iz)*q*r) / Ix
    dq = (M + (Iz - Ix)*p*r) / Iy
    dr = (N + (Ix - Iy)*p*q) / Iz

    # Euler-angle kinematics
    tan_theta = np.tan(theta)
    sec_theta = 1.0 / np.cos(theta)

    dphi   = p + tan_theta*(q*np.sin(phi) + r*np.cos(phi))
    dtheta = q*np.cos(phi)  - r*np.sin(phi)
    dpsi   = (q*np.sin(phi) + r*np.cos(phi)) * sec_theta

    # Body-to-inertial (NED) rotation matrix
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    R_b2i = np.array([
        [cth*cpsi, sphi*sth*cpsi - cphi*spsi, cphi*sth*cpsi + sphi*spsi],
        [cth*spsi, sphi*sth*spsi + cphi*cpsi, cphi*sth*spsi - sphi*cpsi],
        [-sth,     sphi*cth,                  cphi*cth]
    ])

    vel_body = np.array([u, v, w])
    dx_e, dy_n, dz_d = R_b2i @ vel_body

    return np.array([
        du, dv, dw,
        dp, dq, dr,
        dphi, dtheta, dpsi,
        dx_e, dy_n, dz_d
    ])

def rk4_step(fun, t, y, dt, *args):
    """Classic fourth-order Runge-Kutta step."""
    k1 = fun(t,         y,               *args)
    k2 = fun(t + dt/2,  y + dt*k1/2,     *args)
    k3 = fun(t + dt/2,  y + dt*k2/2,     *args)
    k4 = fun(t + dt,    y + dt*k3,       *args)
    return y + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0

def simulate(state0, t_final, dt, params,
             forces_func=forces_with_controls, moments_func=moments_with_controls):
    """Integrate from t=0 to t_final. Returns (time, states)."""
    n_steps = int(np.ceil(t_final/dt)) + 1
    states  = np.zeros((n_steps, len(state0)))
    times   = np.zeros(n_steps)

    state = state0.copy()
    t = 0.0
    for i in range(n_steps):
        times[i]  = t
        states[i] = state
        state = rk4_step(aircraft_eom, t, state, dt,
                         forces_func, moments_func, params)
        t += dt
    return times, states

if __name__ == "__main__":
    # Example: free-fall point mass starting at 1000 m altitude.
    params = {
        "mass": 1200.0,
        "I": (800.0, 1200.0, 1000.0),
        "throttle": 0.6,
        "delta_e": -0.05,  # small pitch-up
        "delta_a":  0.00,
        "delta_r":  0.00,
        "Tmax": 2000.0
    }

    # Initial state (rest, level, 1000 m altitude)
    state0 = np.zeros(12)
    state0[11] = -1000.0  # z_d (down is positive)

    t, y = simulate(state0, t_final=10.0, dt=0.01, params=params)

    # Print altitude every second
    for i in range(0, len(t), 100):
        print(f"time={t[i]:4.1f} s, altitude={-y[i,11]:7.1f} m")
