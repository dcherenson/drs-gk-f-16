# DRS gatekeeper: F-16 case study

This repository contains the public F-16 case study for [Distributionally
Robust Safety Under Arbitrary Uncertainties: A Safety Filtering
Approach](https://drs-gk.github.io/). It runs a full JSBSim F-16 simulation,
tracks a saved low-altitude trajectory through the Black Canyon of the
Gunnison, and compares a nominal controller with DRS gatekeeper.

The nominal controller is a trajectory-tracking PID. gatekeeper evaluates
a nominal/backup switch using an empirical uncertainty model and a fast
polynomial surrogate while JSBSim remains the environment used for the live
aircraft rollout.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first.
The default installation includes CUDA 13 acceleration on Linux x86_64 and
aarch64. GPU execution requires an NVIDIA GPU with compute capability 7.5 or
newer and driver version 580 or newer. CUDA libraries are installed with the
Python dependencies. See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu).
macOS and Windows installations use CPU JAX. The bundled aircraft, terrain,
nominal trajectory, and compact model artifacts provide everything needed for
a run after installation.

```bash
git clone https://github.com/dcherenson/drs-gk-f-16.git
cd drs-gk-f-16
uv python install 3.13
uv sync
uv run python run.py
```

The default command runs the nominal flight followed by a flight supervised
by gatekeeper, with the PID controller and video recording enabled. To use a
conventional virtual environment instead:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python run.py
```

The runtime code is in `src/`. It uses JSBSim, NumPy, and JAX.
Plotting and recording use the dependencies declared by `pyproject.toml`.
JAX selects the available accelerator automatically. The runner prints the
selected backend and device at startup. A CPU run can be selected explicitly:

```bash
JAX_PLATFORMS=cpu uv run python run.py
```

The demo has been validated on macOS with Python 3.13 and CPU JAX.
Interactive flight requires a desktop session with OpenGL 3.3 support.
The numerical `--headless` mode does not initialize OpenGL.

### Runner options

`run.py` keeps the public command surface small:

```text
--headless                 Run numerical simulation and save plots/summaries
                           without opening GL or recording video.
--case {both,nominal,gatekeeper}
                           Run both flights, only the nominal flight, or only
                           the gatekeeper flight.
--max-steps N              Stop each selected flight after at most N control
                           steps. Useful for short runs.
--no-record                Disable video recording in the rendered workflow.
```

For example, a short numerical smoke run is:

```bash
uv run python run.py --headless --case both --max-steps 30
```

The single editable configuration module is
[`src/config.py`](src/config.py). Its settings are grouped as
`DEMO`, `ENVIRONMENT`, `PID`, `GATEKEEPER`, `BACKUP`, `UNCERTAINTY`, and
`VISUALIZATION`.

For example, `ENVIRONMENT["wind_sigma"] = 10.0` increases gust strength,
`GATEKEEPER["beta"] = 0.5` changes the ambiguity margin, and
`DEMO["nominal_end_fraction"] = 0.2` uses a longer portion of the saved path.
The initial-state fields in `ENVIRONMENT` use the saved reference when set to
`None`. Setting a value overrides that initial condition. These edits use the
same saved learned models and do not trigger training.

On the validation Mac using CPU JAX, gatekeeper reached 95% of the reference
path in 701 steps, used backup for 67 steps, and maintained at least 47.9 ft of
terrain clearance. The nominal flight collided with terrain. These are
observations for seed 1340 and the first 10% of the saved trajectory. The
complete numerical PID comparison took about 1 second for nominal flight and
98 seconds for gatekeeper, including initialization and first compilation.
Interactive rendering and recording add work. Runtime varies by hardware. The
complete recorded live comparison took about 34 seconds for nominal flight and
133 seconds for gatekeeper on the same machine.

### Outputs

Each selected case writes its artifacts under `output/`:

```text
output/
├── nominal/
│   ├── flight.mp4
│   ├── flight.gif
│   ├── trajectory.png
│   ├── summary.json
│   └── trajectory.npz
├── gatekeeper/
│   ├── flight.mp4
│   ├── flight.gif
│   ├── trajectory.png
│   ├── summary.json
│   └── trajectory.npz
├── comparison.png
└── comparison.json
```

Headless runs write the numerical plots and summaries without opening a GL
window or recording video. `--no-record` is available when running a rendered
case and leaves the numerical outputs enabled.

The checked-in model files are:

```text
src/assets/models/nominal_coeff_weights.npz
src/assets/models/uncertainty.npz
src/assets/models/nominal_trajectory.npz
src/assets/terrain/black-canyon.tif
```

The compact uncertainty/model NPZ stores 29,997 empirical context/residual
rows as numerical arrays. Runtime model persistence uses portable numerical
arrays and metadata loaded directly with NumPy.

## F-16 environment

The aircraft is simulated in local North-East coordinates over the bundled
10 m Black Canyon DEM. Horizontal positions are in feet in the local DEM frame.
altitude $`h`$ is also in feet relative to the DEM entrance elevation. Angles
are in radians unless a command or threshold is explicitly labelled in degrees.
The live altitude tape displays feet above mean sea level (MSL).

The core aircraft state is

```math
x_t = [p^N_t,p^E_t,h_t,u_t,v_t,w_t,p_t,q_t,r_t,\phi_t,\theta_t,\psi_t]^\top,
```

where $`u,v,w`$ are body velocities and $`p,q,r`$ are body angular rates.
gatekeeper's packed state appends JSBSim normal accelerations $`n_y,n_z`$ as
two additional telemetry values alongside the core aircraft state. The action is

```math
a_t = [\delta_{a,t},\delta_{e,t},\delta_{r,t},\delta_{T,t}]^\top,
```

in the order **aileron, elevator, rudder, throttle**. The surface commands are
normalized to `[-1, 1]` and throttle is normalized to `[0, 1]`.
Flight angles $`\alpha,\beta`$ and surface commands $`\delta`$ are distinct from
gatekeeper’s risk parameters with the same Greek symbols.

### Rigid-body rollout model

JSBSim propagates the live aircraft. The equations below describe the learned
surrogate used by gatekeeper:

```math
x_{t+1}=x_t+\Delta t\,\dot x_t,\qquad\Delta t=1/30\,\mathrm{s}.
```

For the equations below, $`c_\xi=\cos\xi`$, $`s_\xi=\sin\xi`$, and
$`g=32.174\,\mathrm{ft/s^2}`$. The displayed equations are the unclipped
rigid-body form. The implementation applies finite feature, state, and
derivative clips where needed to keep the numerical rollout finite.

The local position and body-velocity equations are

```math
\begin{aligned}
\dot p^N_t={}&u_t(c_{\theta_t}c_{\psi_t})
 +v_t(s_{\phi_t}s_{\theta_t}c_{\psi_t}-c_{\phi_t}s_{\psi_t})\\
 &+w_t(c_{\phi_t}s_{\theta_t}c_{\psi_t}+s_{\phi_t}s_{\psi_t}),\\[2pt]
\dot p^E_t={}&u_t(c_{\theta_t}s_{\psi_t})
 +v_t(s_{\phi_t}s_{\theta_t}s_{\psi_t}+c_{\phi_t}c_{\psi_t})\\
 &+w_t(c_{\phi_t}s_{\theta_t}s_{\psi_t}-s_{\phi_t}c_{\psi_t}),\\[2pt]
\dot h_t={}&u_ts_{\theta_t}-v_ts_{\phi_t}c_{\theta_t}-w_tc_{\phi_t}c_{\theta_t},\\[2pt]
\dot u_t={}&X_t+r_tv_t-q_tw_t-gs_{\theta_t},\\
\dot v_t={}&Y_t+p_tw_t-r_tu_t+g s_{\phi_t}c_{\theta_t},\\
\dot w_t={}&Z_t+q_tu_t-p_tv_t+g c_{\phi_t}c_{\theta_t}.
\end{aligned}
```

The angular rates and Euler angles follow

```math
\begin{aligned}
\dot{\boldsymbol\omega}_t
  &= I^{-1}\left(\begin{bmatrix}L_t\\M_t\\N_t\end{bmatrix}
       -\boldsymbol\omega_t\times(I\boldsymbol\omega_t)\right),
  &\boldsymbol\omega_t&=[p_t,q_t,r_t]^\top,\\
\dot\phi_t&=p_t+q_t\sin\phi_t\tan\theta_t+r_t\cos\phi_t\tan\theta_t,\\
\dot\theta_t&=q_t\cos\phi_t-r_t\sin\phi_t,\\
\dot\psi_t&=\frac{q_t\sin\phi_t+r_t\cos\phi_t}{\cos\theta_t}.
\end{aligned}
```

The live JSBSim environment advances four internal ticks at
$`1/120\,\mathrm{s}`$ for each control step. The surrogate used by gatekeeper
rollouts uses explicit Euler at
$`\Delta t=1/30\,\mathrm{s}`$.

### Polynomial surrogate

The surrogate predicts the six aerodynamic coefficient channels

```math
\mathbf C_{\mathrm{nom}}
=[C_X,C_Y,C_Z,C_L,C_M,C_N]^\top
```

from a polynomial expansion of the feature vector, in this exact order:

```math
\nu_t=[\alpha_t,\beta_t,\mathrm{Mach}_t,p_t,q_t,r_t,
\delta_{e,t},\delta_{a,t},\delta_{r,t}]^\top.
```

Throttle is handled by the separate thrust term and is not one of these nine
coefficient features. With dynamic pressure
$`\bar q_t=\tfrac12\rho V_t^2`$, wing area $`S`$, span $`b`$, mean chord
$`\bar c`$, and fixed surrogate mass $`m`$, the channels used by the rigid-body
map are

```math
\begin{aligned}
X_t&=\bar q_t\frac{S}{m}C_{X,t}+T(\delta_{T,t}),
&Y_t&=\bar q_t\frac{S}{m}C_{Y,t},
&Z_t&=\bar q_t\frac{S}{m}C_{Z,t},\\
L_t&=\bar q_t S b C_{L,t},
&M_t&=\bar q_t S\bar c C_{M,t},
&N_t&=\bar q_t S b C_{N,t}.
\end{aligned}
```

Here $`X,Y,Z`$ are mass-normalized body-axis acceleration channels after the
$`\bar qS/m`$ scaling. $`T(\delta_T)`$ is the fitted
throttle-to-acceleration polynomial. The surrogate uses fixed air density,
mass, and inertia values.

The stored degree-three polynomial map is

```math
\mathbf C_{\mathrm{nom},t}=\varphi(\nu_t)^\top W+B.
```

### Identification and empirical uncertainty

The uncertainty model follows a similar empirical residual-sampling approach
to the [racecar implementation](https://github.com/joonlee16/drs-gk-racecar)
from the same paper, using nearby recorded contexts to sample model errors.

The nominal coefficient map is fitted from simulator-derived samples. Runtime
residuals are sampled from nearby empirical contexts. If $`\mathbf w_t`$ is a
sampled coefficient residual, the perturbed coefficients are

```math
\mathbf C_{\mathrm{pert},t}
 = \mathbf C_{\mathrm{nom},t}+\mathbf w_t.
```

The force and moment equations above then use
$`\mathbf C_{\mathrm{pert},t}`$ once. In particular, the residual is not added
a second time inside the dimensional force calculation.

The online sampler uses the packed 17-feature context in this order:

```text
[alpha, beta, mach, p, q, r,
 delta_t, delta_e, delta_a, delta_r,
 prev_delta_t, prev_delta_e, prev_delta_a, prev_delta_r,
 qbar, canyon_width, canyon_width_grad]
```

`delta_t` denotes throttle in the artifact feature names. The current canyon
width is used as a ±200 ft gate. The candidate pool is capped at 512 rows and
the 20 nearest valid contexts are used for categorical-distance sampling.
Runtime sampling uses nearby empirical contexts from the saved rows under this
gate and categorical-distance rule.

![F-16 coefficient model identification and uncertainty](https://github.com/user-attachments/assets/cd2c8906-3985-49d6-aea7-b588946076ce)

### Reference trajectory and terrain

The saved nominal trajectory is derived from the optimization workflow in
[Peter Sharpe's `air-racing-optimization` repository](https://github.com/peterdsharpe/air-racing-optimization).
The terrain is a USGS 10 m digital elevation model acquired through
[OpenTopography](https://www.opentopography.org/) for the Black Canyon of the
Gunnison. The source clip uses the latitude/longitude bounding box
`(38.52, 38.62, -107.78, -107.65)` in `(south, north, west, east)` order.
Those derived artifacts are already included in this repository.

## DRS gatekeeper

gatekeeper wraps the nominal policy with a backup policy. At each update it
tests candidate switch offsets $`s\in\{0,1,\ldots,24\}`$ over an empirical
rollout horizon. Offset zero means that the backup is used immediately.

The F-16 demo parameters are:

| parameter | value |
| --- | ---: |
| candidate offsets $`M`$ | 25 (`0` through `24`) |
| rollout horizon $`T`$ | 100 control steps |
| empirical samples per candidate $`N`$ | 256 |
| safety tolerance $`\epsilon`$ | 0.05 |
| decision tolerance $`\delta`$ | 0.1 |
| ambiguity radius $`\beta`$ | 1.0 |
| backup failure tolerance $`\alpha`$ | 0.0 |
| Wasserstein order $`p`$ | 1 |
| Lipschitz mode | fixed |
| fixed Lipschitz constant $`L_H`$ | 50 |
| gatekeeper update interval | every 10 control steps |

The rollout safety margin is terrain clearance with a 10 ft buffer:

```math
h_{\mathrm{terrain}}(x_t)
 = h_t-h_{\mathrm{terrain}}(p^N_t,p^E_t)-10\,\mathrm{ft}.
```

The terminal backup-set check requires speed below 600 kt and nonnegative
vertical rate. Its useful mathematical margin is therefore

```math
h_{\mathrm{terminal}}(x_T)
 = \min\left(100,\;600(1.68781)-V_T,\;\dot h_T\right),
```

where speed and vertical rate are evaluated in ft/s. The constant 100 cap
preserves the original terminal score based on speed and vertical rate.

For each candidate, the minimum rollout margin $`H`$ is counted as a failure
when $`H\leq L_H\beta^{1/p}`$. If $`k`$ of $`N`$ samples fail, the upper confidence
bound and acceptance threshold are

```math
\rho=1-(1-\delta)^{1/M},\qquad
\bar q=\operatorname{Beta}^{-1}(1-\rho;k+1,N-k),\qquad
\bar q\leq\frac{\epsilon-\alpha}{1-\alpha}.
```

For $`k=N`$, the bound is 1. The latest feasible switching offset is selected.
If no new candidate passes, gatekeeper retains the previous commitment and
uses backup once its switching time is reached.

When selected, the backup controller targets 300 kt and an altitude 200 ft
above the maximum canyon-top profile. These are demo settings in
`src/config.py`.

## Citation

```bibtex
@article{cherenson2026drs_gatekeeper,
  author  = {Cherenson, Daniel M. and Lee, Haejoon and Kim, Taekyung and Panagou, Dimitra},
  title   = {{Distributionally Robust Safety Under Arbitrary Uncertainties: A Safety Filtering Approach}},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026},
  volume  = {11},
  number  = {11},
  pages   = {13199--13206},
  doi     = {10.1109/LRA.2026.3737038},
  url     = {https://ieeexplore.ieee.org/document/11704677}
}
```
