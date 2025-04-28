import numpy as np
import qutip as qt
from scipy import linalg as la
from scipy.integrate import simpson
from scipy.ndimage import gaussian_filter1d

# Set the default options for QuTiP
options_={'atol': 1e-17,'rtol': 1e-15}

# basic functions
def setup_hamiltonian(eps,Delta,epsi,v):
    """
    Creates the hamiltonian of a qubit coupled to an impurity.

    Parameters:
    ----------
    eps : float
        Magnetic field of the qubit along the z-axis.
    Delta : float
        Magnetic field of the qubit along the x-axis.
    epsi : float
        Frequency of the impurity.
    v : float
        Coupling beteween qubit and impurity.

    returns:
    --------
    H : Qobj
        Hamiltonian of the system.
    """
    # Defining operators on the qubit and impurity spaces
    szq = qt.tensor(qt.sigmaz(),qt.qeye(2))
    syq = qt.tensor(qt.sigmay(),qt.qeye(2))
    sxq = qt.tensor(qt.sigmax(),qt.qeye(2))
    szi = qt.tensor(qt.qeye(2),qt.sigmaz())

    # Hamiltoniana
    HA=-eps/2*szq-Delta/2*sxq
    HI=-epsi/2*szi
    HAI=-v/2*szq*szi
    H=HA+HAI+HI

    return H

def setup_hamiltonian_ancilla(eps,Delta,epsi,v):

    # Spazio tri-partito: 1 Ancilla, 2 Qubit, 3 Impurità
    szq = qt.tensor(qt.qeye(2),qt.sigmaz(),qt.qeye(2))
    sza = qt.tensor(qt.sigmaz(),qt.qeye(2),qt.qeye(2))
    sxq = qt.tensor(qt.qeye(2),qt.sigmax(),qt.qeye(2))
    szi = qt.tensor(qt.qeye(2),qt.qeye(2),qt.sigmaz())
    
    # Hamiltoniana
    HA=-eps/2*szq-Delta/2*sxq
    HI=-epsi/2*szi
    HAI=-v/2*szq*szi
    H=HA+HAI+HI

    return H

def PartialTrace(rho, n=0):
    """
    Computes the partial trace of a 3D array of density matrices.

    Parameters:
    ----------
    rho : ndarray
        A 3D NumPy array of shape (N, d, d), representing N density matrices of size dxd.
    n : int, optional
        Specifies which subsystem to trace out:
        - n=0: Trace out the second subsystem (default).
        - n=1: Trace out the first subsystem.

    Returns:
    --------
    partial_rho : ndarray
        A 3D NumPy array of the reduced density matrices after tracing out the specified subsystem.

    Raises:
    -------
    ValueError
        If n is not 0 or 1.
    """
    if n == 0:
        partial_rho = rho[:, ::2, ::2] + rho[:, 1::2, 1::2]
    elif n == 1:
        partial_rho = rho[:, :2, :2] + rho[:, 2:, 2:]
    else:
        raise ValueError("Invalid value for n. Expected 0 or 1.")
    return partial_rho

def TraceDistance(rho1, rho2):
    """
    Computes the trace distance between two sets of density matrices.

    Parameters:
    ----------
    rho1 : ndarray
        A 3D NumPy array of shape (N, d, d), representing N density matrices of size dxd.
    rho2 : ndarray
        A 3D NumPy array of shape (N, d, d), representing N density matrices of size dxd.

    Returns:
    --------
    trace_distance : ndarray
        A 1D NumPy array of length N, containing the trace distances between corresponding
        density matrices in rho1 and rho2.
    """
    # Compute the difference in diagonal elements
    d1 = np.real(rho1[:, 0, 0] - rho2[:, 0, 0])
    # Compute the absolute difference in off-diagonal elements
    d2 = np.abs(rho1[:, 0, 1] - rho2[:, 0, 1])
    # Return the trace distance
    return np.sqrt(d1**2 + d2**2)

def Von_Neumann(rhopt):
    """
    Computes the Von Neumann entropy for a set of density matrices.

    Parameters:
    ----------
    rhopt : ndarray
        A 3D NumPy array of shape (N, d, d), representing N density matrices of size dxd.

    Returns:
    --------
    entropy : ndarray
        A 1D NumPy array of length N, containing the Von Neumann entropy for each density matrix.
    """
    # Preallocate the entropy array for efficiency
    entropy = np.zeros(len(rhopt))

    for t in range(len(rhopt)):
        # Compute eigenvalues of the density matrix
        eigenvalues = np.real(la.eigvalsh(rhopt[t]))
        # Replace non-positive eigenvalues with a small positive value to avoid log(0)
        eigenvalues = np.clip(eigenvalues, 1e-40, None)
        # Compute the entropy for the current density matrix
        entropy[t] = -np.sum(eigenvalues * np.log(eigenvalues))
       
    return entropy

def clean_density_matrices(rho, tol=1e-15):
    """
    Cleans a 3D array of density matrices:
    - Ensures numerical Hermiticity
    - Suppresses real and imaginary parts below a given threshold
    - Returns real matrices if residual imaginary parts are negligible

    Parameters:
    ----------
    rho : ndarray
        3D NumPy array of shape (N, d, d), representing N density matrices of size dxd.
    tol : float
        Tolerance below which a component is considered numerically zero.

    Returns:
    --------
    rho_cleaned : ndarray
        "Cleaned" 3D array with the same dimensions as `rho`.
    """
    # Enforce Hermiticity: (rho + rho^\dagger)/2
    rho_herm = 0.5 * (rho + np.conjugate(rho.transpose(0, 2, 1)))

    # Suppress spurious components below the threshold
    real_clean = np.where(np.abs(rho_herm.real) < tol, 0, rho_herm.real)
    imag_clean = np.where(np.abs(rho_herm.imag) < tol, 0, rho_herm.imag)
    rho_cleaned = real_clean + 1j * imag_clean

    # Force real matrices if imaginary parts are negligible
    return np.real_if_close(rho_cleaned, tol=1000)  # tol=1000 → ignores imaginary parts < ~1e-12

# dynamical simulation
def dynamics(rhoA0, deltap, deltap0, eps, Delta, epsi, v, times):
    """
    Simulates the dynamics of a qubit coupled to an impurity.

    Parameters:
    ----------
    rhoA0 : Qobj
        Initial density matrix of the qubit.
    deltap : float
        Asymmetry parameter for the rates.
    deltap0 : float
        Initial population imbalance of the impurity.
    eps : float
        Magnetic field of the qubit along the z-axis.
    Delta : float
        Magnetic field of the qubit along the x-axis.
    epsi : float
        Frequency of the impurity.
    v : float
        Coupling between the qubit and impurity.
    times : array-like
        Time points for the simulation.

    Returns:
    --------
    result_array : ndarray
        A 3D NumPy array containing the density matrices at each time step.
    """
    # Setup the Hamiltonian
    H = setup_hamiltonian(eps, Delta, epsi, v)

    # Initial condition for the impurity
    gamma = 1
    gm = gamma / 2 * (1 + deltap)
    gp = gamma / 2 * (1 - deltap)
    p0 = 0.5 + deltap0 * 0.5
    p1 = 1 - p0
    rhoI0 = p0 * qt.ket2dm(qt.basis(2, 0)) + p1 * qt.ket2dm(qt.basis(2, 1))

    # Calculate angles for eigenbasis
    theta0 = np.arctan2(Delta, (eps + v))
    theta1 = np.arctan2(Delta, (eps - v))
    c = np.cos((theta0 - theta1) / 2)

    # Define eigenvectors analytically
    c0, s0 = np.cos(theta0 / 2), np.sin(theta0 / 2)
    c1, s1 = np.cos(theta1 / 2), np.sin(theta1 / 2)
    eig_vec = [
        c0 * qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) + s0 * qt.tensor(qt.basis(2, 1), qt.basis(2, 0)),
        c1 * qt.tensor(qt.basis(2, 0), qt.basis(2, 1)) + s1 * qt.tensor(qt.basis(2, 1), qt.basis(2, 1)),
        -s0 * qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) + c0 * qt.tensor(qt.basis(2, 1), qt.basis(2, 0)),
        -s1 * qt.tensor(qt.basis(2, 0), qt.basis(2, 1)) + c1 * qt.tensor(qt.basis(2, 1), qt.basis(2, 1)),
    ]

    # Define the jump operator
    T = eig_vec[0] * eig_vec[1].dag() + eig_vec[2] * eig_vec[3].dag()

    # Collapse operators with rates and cosine factors
    sqrt_gm, sqrt_gp = np.sqrt(gm), np.sqrt(gp)
    c_ops = [c * sqrt_gm * T, c * sqrt_gp * T.dag()]

    # Initial condition for the full system (qubit + impurity)
    rho0 = qt.tensor(rhoA0, rhoI0)

    # Solve the master equation
    result = qt.mesolve(H, rho0, times, c_ops, options=options_)

    # Convert results to a 3D NumPy array and clean density matrices
    result_array = np.array([state.full() for state in result.states])
    result_array = clean_density_matrices(result_array)

    return result_array

def dynamics_ancilla(rhoA0, deltap, deltap0, eps, Delta, epsi, v, times):
    """
    Simulates the dynamics of a quantum system consisting of a qubit, ancilla, and impurity 
    using the Lindblad master equation.
    Parameters:
    -----------
    rhoA0 : qutip.Qobj
        Initial density matrix of the ancilla system.
    deltap : float
        Asymmetry parameter for the decay rates of the impurity.
    deltap0 : float
        Initial population imbalance of the impurity state.
    eps : float
        Energy splitting of the qubit.
    Delta : float
        Coupling strength between the qubit and impurity.
    epsi : float
        Energy splitting of the impurity.
    v : float
        Coupling strength between the ancilla and the qubit-impurity system.
    times : array-like
        Array of time points for which the dynamics are simulated.
    Returns:
    --------
    result_array : numpy.ndarray
        A 3D NumPy array where each element corresponds to the density matrix 
        of the combined system (qubit + ancilla + impurity) at a specific time.
    Notes:
    ------
    - The Hamiltonian of the system is constructed using the `setup_hamiltonian_ancilla` function.
    - The initial state of the impurity is determined by the temperature and decay rates.
    - Jump operators are defined based on the eigenvectors of the qubit-impurity system and 
      extended to include the ancilla.
    - The Lindblad master equation is solved using QuTiP's `mesolve` function.
    - The result is converted into a 3D NumPy array for further analysis.
    """
    
    # Setup the Hamiltonian for the system with ancilla
    H = setup_hamiltonian_ancilla(eps, Delta, epsi, v)
    
    # Initialize impurity state based on temperature and rates
    g = 1  # Total decay rate
    gm = g / 2 * (1 + deltap)  # Rate for |0⟩ → |1⟩
    gp = g / 2 * (1 - deltap)  # Rate for |1⟩ → |0⟩
    p0 = 0.5 + deltap0 * 0.5  # Initial population of |0⟩
    p1 = 1 - p0  # Initial population of |1⟩
    rhoI0 = p0 * qt.ket2dm(qt.basis(2, 0)) + p1 * qt.ket2dm(qt.basis(2, 1))

    # Combine initial states of qubit+ancilla and impurity
    rho0 = qt.tensor(rhoA0, rhoI0)

    # Define jump operators for the dynamics
    theta0 = np.arctan2(Delta, (eps + v))
    theta1 = np.arctan2(Delta, (eps - v))
    c = np.cos((theta0 - theta1) / 2)  # Cosine factor for rates

    # Eigenvectors for the qubit-impurity system
    c0, c1 = np.cos(theta0 / 2), np.cos(theta1 / 2)
    s0, s1 = np.sin(theta0 / 2), np.sin(theta1 / 2)
    eig_vec = [
        c0 * qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) + s0 * qt.tensor(qt.basis(2, 1), qt.basis(2, 0)),
        c1 * qt.tensor(qt.basis(2, 0), qt.basis(2, 1)) + s1 * qt.tensor(qt.basis(2, 1), qt.basis(2, 1)),
        -s0 * qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) + c0 * qt.tensor(qt.basis(2, 1), qt.basis(2, 0)),
        -s1 * qt.tensor(qt.basis(2, 0), qt.basis(2, 1)) + c1 * qt.tensor(qt.basis(2, 1), qt.basis(2, 1)),
    ]

    # Transition operator in the qubit-impurity space
    T = eig_vec[0] * eig_vec[1].dag() + eig_vec[2] * eig_vec[3].dag()
    # Extend the transition operator to include the ancilla
    T_ancilla = qt.tensor(qt.qeye(2), T)

    # Collapse operators with rates and cosine factor
    c_ops = [c * np.sqrt(gm) * T_ancilla, c * np.sqrt(gp) * T_ancilla.dag()]
    
    # Solve the master equation for the dynamics
    result = qt.mesolve(H, rho0, times, c_ops, options=options_)

    # Convert the result to a 3D NumPy array
    result_array = np.array([state.full() for state in result.states])

    return result_array

# Non-Markovianity measures
def N_BLP(times, eps, Delta, epsi, v, deltap0, deltap, test=False, theta=0, phi=0):
    """
    Computes the non-Markovianity measure N_BLP based on the Breuer-Laine-Piilo (BLP) criterion.

    Parameters:
    ----------
    times : array-like
        Array of time points for the simulation.
    eps : float
        Magnetic field of the qubit along the z-axis.
    Delta : float
        Magnetic field of the qubit along the x-axis.
    epsi : float
        Frequency of the impurity.
    v : float
        Coupling between the qubit and impurity.
    deltap0 : float
        Initial population imbalance of the impurity.
    deltap : float
        Asymmetry parameter for the rates.
    test : bool, optional
        If True, prints intermediate results for debugging. Default is False.
    theta : float, optional
        Polar angle for the initial state of the qubit. Default is 0.
    phi : float, optional
        Azimuthal angle for the initial state of the qubit. Default is 0.

    Returns:
    --------
    NBLP : float
        The non-Markovianity measure N_BLP.
    trdist : ndarray
        Trace distance between the reduced density matrices at each time step.
    st_p : ndarray
        Positive part of the time derivative of the trace distance.
    rhoQp : ndarray
        Reduced density matrices for the first initial state.
    rhoQm : ndarray
        Reduced density matrices for the second initial state.

    Notes:
    ------
    - The BLP measure quantifies non-Markovianity by integrating the positive part of the time derivative
      of the trace distance between two quantum states.
    - The initial states are parameterized by the angles `theta` and `phi`.
    """
    
    # Define initial states parameterized by theta and phi
    psi1 = np.cos(theta / 2) * qt.basis(2, 0) + np.exp(1j * phi) * np.sin(theta / 2) * qt.basis(2, 1)
    psi2 = -np.sin(theta / 2) * qt.basis(2, 0) + np.exp(1j * phi) * np.cos(theta / 2) * qt.basis(2, 1)
    rho0p = qt.ket2dm(psi1)
    rho0m = qt.ket2dm(psi2)

    # Simulate dynamics for the two initial states
    rho1 = dynamics(rho0p, deltap, deltap0, eps, Delta, epsi, v, times)
    rho2 = dynamics(rho0m, deltap, deltap0, eps, Delta, epsi, v, times)

    # Compute partial trace over the impurity to get reduced density matrices
    rhoQp = PartialTrace(rho1)
    rhoQm = PartialTrace(rho2)

    # Compute trace distance between the reduced density matrices
    trdist = TraceDistance(rhoQp, rhoQm)

    # Compute the positive part of the time derivative of the trace distance
    st = np.gradient(trdist, times, edge_order=2)
    st_p = np.maximum(st, 0)  # Keep only the positive part

    # Integrate the positive derivative to compute N_BLP
    NBLP = simpson(y=st_p, x=times)

    return NBLP, trdist, st_p, rhoQp, rhoQm

def N_LFS(times, eps, Delta, epsi, v, deltap0, deltap, bell_state=0):
    """
    Computes the non-Markovianity measure N_LFS based on the Luo-Fu-Song (LFS) criterion.

    Parameters:
    ----------
    times : array-like
        Array of time points for the simulation.
    eps : float
        Magnetic field of the qubit along the z-axis.
    Delta : float
        Magnetic field of the qubit along the x-axis.
    epsi : float
        Frequency of the impurity.
    v : float
        Coupling between the qubit and impurity.
    deltap0 : float
        Initial population imbalance of the impurity.
    deltap : float
        Asymmetry parameter for the rates.
    bell_state : int, optional
        Index of the Bell state to use as the initial state of the ancilla system.
        - 0: |Ψ+⟩ = (|10⟩ + |01⟩) / √2
        - 1: |Φ+⟩ = (|00⟩ + |11⟩) / √2
        - 2: |Ψ−⟩ = (|10⟩ − |01⟩) / √2
        - 3: |Φ−⟩ = (|00⟩ − |11⟩) / √2
        Default is 0.

    Returns:
    --------
    NLFS : float
        The non-Markovianity measure N_LFS.
    I : ndarray
        Mutual information between the system and ancilla at each time step.
    dI : ndarray
        Time derivative of the mutual information.
    dI_positiva : ndarray
        Positive part of the time derivative of the mutual information.

    Notes:
    ------
    - The LFS measure quantifies non-Markovianity by integrating the positive part of the time derivative
      of the mutual information between the system and ancilla.
    """
    # Define the initial Bell state of the ancilla system
    bell_states = [
        (qt.tensor(qt.basis(2, 1), qt.basis(2, 0)) + qt.tensor(qt.basis(2, 0), qt.basis(2, 1))).unit(),
        (qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) + qt.tensor(qt.basis(2, 1), qt.basis(2, 1))).unit(),
        (qt.tensor(qt.basis(2, 1), qt.basis(2, 0)) - qt.tensor(qt.basis(2, 0), qt.basis(2, 1))).unit(),
        (qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) - qt.tensor(qt.basis(2, 1), qt.basis(2, 1))).unit(),
    ]
    if bell_state not in range(4):
        raise ValueError("Invalid bell_state. Expected 0, 1, 2, or 3.")
    rho0A = qt.ket2dm(bell_states[bell_state])

    # Simulate the dynamics of the system
    rhot = dynamics_ancilla(rho0A, deltap, deltap0, eps, Delta, epsi, v, times)

    # Compute reduced density matrices
    rho_sa = PartialTrace(rhot)  # System + Ancilla
    rho_s = PartialTrace(rho_sa, n=1)  # System
    rho_a = PartialTrace(rho_sa)  # Ancilla

    # Compute Von Neumann entropies
    entropy_sa = Von_Neumann(rho_sa)
    entropy_a = Von_Neumann(rho_a)
    entropy_s = Von_Neumann(rho_s)

    # Compute mutual information
    I = entropy_s + entropy_a - entropy_sa

    # Compute the positive part of the time derivative of mutual information
    dI = np.gradient(I, times, edge_order=2)
    dI_positive = np.maximum(dI, 0)

    # Integrate the positive derivative to compute N_LFS
    NLFS = simpson(y=dI_positive, x=times)

    return NLFS, I, dI, dI_positive
