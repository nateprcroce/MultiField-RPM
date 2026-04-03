import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap

# System parameters
L = 100                 # Length of one side of 2D lattice
R = 2.0                 # STDEV of random field distribution
N = L * L               # number of spins
quality = 400           # time steps
n = 7                   # number of fields H

# Save Screenshots of Each State
save = False            # Only set True if you want to save picture of each state (ie. to form video animation)

# H Field Generation Parameters
lowBound = -2           # lower bound of H values generated
saveVal = 1             # value H fields will converge to record RPM (this is also our upper bound of H values generated)
slopeChange = 3         # number of times slope will change for each H field generated
stdev = 2               # Standard deviation of randomization of time between slope changes (keep between 2 (very random) and 10 (less random))
const_chance = 1/4      # chance that H field will be constant

# Constant bond strengths
J = 1.0
J_right = np.full((L, L), J)
J_left  = np.full((L, L), J)
J_up    = np.full((L, L), J)
J_down  = np.full((L, L), J)

# End of Parameters ----------------------------------------------------------------------------------------------------------------

# Quenched disorder
np.random.seed(42) # Set seed for reproducibility
f = np.random.normal(0, R, size=(L, L)) # Generate random base field values at each site

# Neighbors for 2D lattice
neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
spin_cmap = ListedColormap(['red', 'blue'])  # 0 → red, 1 → blue

# Compute local field for site (i,j)
def local_field(i, j, spins, H_fields, n):
    surr = 0.0
    # right neighbor
    surr += J_right[i, j] * spins[i, (j + 1) % L]
    # left neighbor
    surr += J_left[i, j] * spins[i, (j - 1) % L]
    # down neighbor
    surr += J_down[i, j] * spins[(i + 1) % L, j]
    # up neighbor
    surr += J_up[i, j] * spins[(i - 1) % L, j]

    idx = (i * L + j) % n
    H_ext = H_fields[idx]
    return surr + f[i, j] + H_ext

# Zero-temperature relaxation function
def relax(spins, H_fields, n):
    changed = True
    while changed:
        changed = False
        for i in range(L):
            for j in range(L):
                h = local_field(i, j, spins, H_fields, n)
                if spins[i, j] != np.sign(h) and h != 0:
                    spins[i, j] = np.sign(h)
                    changed = True
    return spins

# Function that checks if the two states where we expect RPM exhibit RPM
def are_spin_configs_identical(config1, config2, count):
    if config1.shape != config2.shape:
        return False  # avoid broadcasting issues

    diff = config1 - config2
    filename = f"diff_output_" + str(count) + ".txt"
    np.savetxt(filename, diff, fmt="%d")
    print(f"Saved difference array to {filename}")

    return np.all(diff == 0)

# Function that randomly generates n H fields
def H_path_randomize(keyVal, t1, t2, slopeChange):
    t = 0
    H_path = []
    currVal = random.uniform(lowBound, keyVal)  # Determines starting value of H Path

    for i in range(0,slopeChange):
        if (t1 - (quality//40) - t) > 0:
            # tDur = int(quality // (2*(slopeChange+1)))
            expectedTDur = quality//(2*(slopeChange+1))
            tDur = round(random.gauss(expectedTDur, expectedTDur / stdev))
            while (t + tDur > t1 - (quality//40) or tDur <= 0):
                tDur = round(random.gauss(expectedTDur, expectedTDur / stdev))
            rand = random.randint(0, int(1/const_chance - 1))
            if (rand == int(1/const_chance - 1)):
                nextVal = currVal
            else:
                nextVal = random.uniform(lowBound, keyVal)
            H_path = np.concatenate([H_path,np.linspace(currVal, nextVal, tDur + 1)[1:]])
            currVal = nextVal
            t += tDur

    # Increase from currVal to keyVal, will hit keyVal at timestep t1
    H_path = np.concatenate([H_path, np.linspace(currVal, keyVal, t1 - t + 1)[1:]])
    currVal = keyVal
    t = t1

    for i in range(0, slopeChange):
        if (t2 - (quality//40) - t) > 0:
            # tDur = int(quality / (2*(slopeChange+1)))
            expectedTDur = quality // (2 * (slopeChange + 1))
            tDur = round(random.gauss(expectedTDur, expectedTDur / stdev))
            while (t + tDur > t2 - (quality//40) or tDur <= 0):
                tDur = round(random.gauss(expectedTDur, expectedTDur / stdev))
            rand = random.randint(0, 3)
            if (rand == 3):
                nextVal = currVal
            else:
                nextVal = random.uniform(lowBound, keyVal)
            H_path = np.concatenate([H_path, np.linspace(currVal, nextVal, tDur + 1)[1:]])
            currVal = nextVal
            t += tDur

    # Increase from currVal to keyVal, will hit keyVal at timestep t2
    H_path = np.concatenate([H_path, np.linspace(currVal, keyVal, t2 - t + 1)[1:]])
    return H_path

# Set up plots
plt.ion()

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(1, 2)  # 3 rows: H(t), M(t), He vs Me

ax_time = fig.add_subplot(gs[:, 0])
ax_spin = fig.add_subplot(gs[:, 1])

# Setup for H field vs time series plot
ax_time.set_title("H Fields over Time")
ax_time.set_xlabel("Time Step")
ax_time.set_ylabel("Value")
ax_time.set_xlim(0, quality)  # will auto-expand
ax_time.set_ylim(lowBound, saveVal + 1) # dynamic y-limits

# Create plot lines for each H field to be plotted
line_Hfields = []
colors = plt.cm.coolwarm(np.linspace(0, 1, n))   # or choose any colormap
for k in range(n):                             # set up all the H vs time plot lines
    line, = ax_time.plot([], [], label=f"H{k}", color=colors[k])
    line_Hfields.append(line)

# Include legend if less than 10 fields
if (n < 10):
    ax_time.legend(loc="upper left")


# Setup for spin configuration plot
ax_spin.set_title("Spin Configuration")
ax_spin.axis('off')
ax_spin.set_aspect('equal')

# Spin grid
x, y = np.meshgrid(np.arange(L), np.arange(L))
spins = -np.ones((L, L))  # initial spin state
spin_display_array = (spins + 1) // 2  # -1 → 0, +1 → 1
im_spin = ax_spin.imshow(spin_display_array, cmap=spin_cmap, interpolation='nearest', origin='upper')

# Storage
H_vals = [[] for i in range(n)]  # For each field Hk
magnetization = []
time_vals = []
saved_spin_configs = [] # where saved spin configs are saved to test RPM
count = 0

H_paths = np.array([H_path_randomize(saveVal, quality//2, quality, slopeChange) for i in range(n)])
H_prev = []     # Ensure H is increasing before snapshot is taken

# Main loop
for t in range(quality):
    H_fields = [H_paths[k][t] for k in range(n)]
    spins = relax(spins, H_fields, n)
    M = np.sum(spins) / N
    magnetization.append(M)
    time_vals.append(t)

    # Update 2D time plot
    for k in range(n):
        H_vals[k].append(H_fields[k])
        line_Hfields[k].set_data(time_vals, H_vals[k])

    # Update spin plot
    im_spin.set_data(spins)
    im_spin.set_clim(0, 1)
    ax_spin.axis('off')
    ax_spin.set_title(f"Spin Configuration\nM = {M:.3f}")

    # We save the spin configuration if we're at one of the two time steps we're checking for RPM
    for i in H_fields:
        if (i == saveVal and H_prev != H_fields):
            # Store spin configuration in local memory
            saved_spin_configs.append(spins.copy())  # use .copy() to avoid future mutation

    # Saves screenshot of each frame if desired
    if save == True:
        plot_path = f"Loop_Timelapse_{t}.png"
        plt.savefig(plot_path, dpi=300)
        print(f"Saved: {plot_path}")

    H_prev = H_fields
    plt.pause(0.01)

# Check if the two spin configurations we saved are identical for RPM
identical = are_spin_configs_identical(saved_spin_configs[0], saved_spin_configs[1], 0)
print("Spin Configurations are Identical: ", identical)

# Finalize
plt.ioff()
plt.show()