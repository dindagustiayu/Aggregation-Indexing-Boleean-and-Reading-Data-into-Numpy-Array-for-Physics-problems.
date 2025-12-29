# Reading Data Files into a Numpy arrays and Graph Plotting with Matplotlib Bar and Scatter Plots
- This work aims to applying the main function for reading text files and statistically summarizing the values into a Numpy array.
- Important Numpy data including: `genfromtxt`, `min`, `max`, `sign`, `sum`, `std`, and others in solving physics problems by Python 3 and Jupyter Notebook.

- # P6.1 - Maxwell-Boltzmann Distribution
Write a function to the plot the Maxwell-Boltzmann Distribution of molecular speed for a gas of particles of a given mass at a given temperature, indicating the modal speed ($v_{*}$), mean ($\langle v \rangle$), and root mean square (rms, $\langle v^{2} \rangle^{1/2}$) speeds with vertical lines.

Call this function for the atomic gasses Helium (m = 4u), and Argon (m = 40u) at 300 K.

__Hints__: The modal speed is the maximum of the probability distribution and can be found $df/dv$. The mean and rms speeds can be obtained, respectively, from the integrals.

$\langle v\rangle = \int_{0}^{\infty} vf(v)$ and $\langle v^{2} \rangle = \int_{0}^{\infty} v^{2}f(v) dv$.

The following expression for the different types of average speed can be derived:

- $v_{*} = \sqrt{\frac{2k_{B}T}{m}}$ (mode)
- $\langle v \rangle = \sqrt{\frac{8k_{B}T}{\pi m}}$ (mean)
- $\langle v^{2} \rangle^{1/2} = \sqrt{\frac{3k_{B}T}{m}}$ (rms speed).

- ![Maxwell-Boltzmann Distribution of He and Ar](https://github.com/dindagustiayu/Aggregation-Indexing-Boleean-and-Reading-Data-into-Numpy-Array-for-Physics-problems./releases%20Distribution%20of%20He%20and%https://github.com/dindagustiayu/Aggregation-Indexing-Boleean-and-Reading-Data-into-Numpy-Array-for-Physics-problems./releases).

- # P6.2 Maxwell Boltzmann Distribution
The Maxwell-Boltzmann equation, the basis of the kinetic thoery of gases, defines the distribution of speeds for a gas at a certain temperature. The Maxwell-Boltzmann distribuiton can be used to determine the distribution of the kinetic energy is identical to the distribution of the speeds for a certain gas at any temperature, _T_:

$f(v)=4\pi V^{2}(\frac{m}{2\pi k_{B}T})^{3/2}e^{\frac{-mv^{2}}{2k_{B}T}}$

__The key Functions and Argument__:

- `def`: starts the function
- `return`: sends back the computed value when the function is called
- `for` loop: iterate over multiple items
- `zip`: combines multiple list into tuples
- `https://github.com/dindagustiayu/Aggregation-Indexing-Boleean-and-Reading-Data-into-Numpy-Array-for-Physics-problems./releases`: generate evenly spaced numbers (start/beginning range, stop/end of range, num/number of points)
- `https://github.com/dindagustiayu/Aggregation-Indexing-Boleean-and-Reading-Data-into-Numpy-Array-for-Physics-problems./releases`: control global plot settings(fonts, sizes, styles)
- `https://github.com/dindagustiayu/Aggregation-Indexing-Boleean-and-Reading-Data-into-Numpy-Array-for-Physics-problems./releases`: show labels for each curve

Calculated speed distribution of particles in He, Ne, Ar, and Xe, with temperature at 300 K.

Reference:
- [The Maxwell-Boltzmann](https://github.com/dindagustiayu/Aggregation-Indexing-Boleean-and-Reading-Data-into-Numpy-Array-for-Physics-problems./releases~https://github.com/dindagustiayu/Aggregation-Indexing-Boleean-and-Reading-Data-into-Numpy-Array-for-Physics-problems./releases)
- [P.33 Atomic Mass Unit](https://github.com/dindagustiayu/Aggregation-Indexing-Boleean-and-Reading-Data-into-Numpy-Array-for-Physics-problems./releases)

![Probability Density of He, Ne, Ar and Xe at T = 300 K](https://github.com/dindagustiayu/Aggregation-Indexing-Boleean-and-Reading-Data-into-Numpy-Array-for-Physics-problems./releases%20Density%20of%20He%2C%20Ne%2C%20Ar%2C%20and%20Xe%20at%20T%20%3D%20300%https://github.com/dindagustiayu/Aggregation-Indexing-Boleean-and-Reading-Data-into-Numpy-Array-for-Physics-problems./releases).

# P6.3 - The Lennard-Jones Interatomic Potential
The Lennard-Jones potential is given by the following equation:

$V(r) = 4\epsilon[(\frac{\sigma}{r})^{12}-(\frac{\sigma}{r})^{6}]$

or sometimes expressed as:

$V(r)=\frac{A}{r^{12}}-\frac{B}{r^{6}}$

reference: [Lennard-Jones Potential](https://github.com/dindagustiayu/Aggregation-Indexing-Boleean-and-Reading-Data-into-Numpy-Array-for-Physics-problems./releases(Physical_and_Theoretical_Chemistry)/Physical_Properties_of_Matter/Atomic_and_Molecular_Properties/Intermolecular_Forces/Specific_Interactions/Lennard-Jones_Potential)

__The key argument__:

- `plt` and `pyplot`: matplotlib's functionality
- `https://github.com/dindagustiayu/Aggregation-Indexing-Boleean-and-Reading-Data-into-Numpy-Array-for-Physics-problems./releases` and `https://github.com/dindagustiayu/Aggregation-Indexing-Boleean-and-Reading-Data-into-Numpy-Array-for-Physics-problems./releases`: set the plot limits manually
- `linestyle`, `linewidth`, `color`, `marker`: matplotlib properties
- `zorder`: controls the drawing order
- `def`: intriduces a function definition
- `return`: species the output of the function, without `return`, a function gives back `None` by default.


(a) The $\epsilon$ and $\sigma$ for Xenon are found to be 1.77 kJ/mol and 4.10 Angstroms, respectively is the separation distance at which the potential is zero: $V(\sigma)=0$. Determine the van der Waals radius for the Xenon atom. Simply plotting _V(r)_ on a grid _r_ values will not yield a very satisfaction graph.

(b) Calculate the intermolecular potential between two Argon (Ar) atoms separated by a distance of 4.0 $\mathring{\text{A}}$ (use $\epsilon=0.997$ kJ/mol and $\sigma=3.40 \mathring{\text{A}}$).

![Lennard-Jones Potential for Xenon](https://github.com/dindagustiayu/Aggregation-Indexing-Boleean-and-Reading-Data-into-Numpy-Array-for-Physics-problems./releases).

# P6.4 - Radioactive list
A radioactive material with an original mass, $N_{0}$, after a time _t_.

- $N(t)=N_{0}e^{\frac{-t}{\tau}}$

(a) Make a while loop which fills two lists: One with spaced time-points _t_, and with values of _N(t)_ at these time-points. The loop should run until the remaining amount of materials is below 50% of the original. Start in _t = 0 s_, and use time-steps of 60s, a mass $N_{0}=4.5$ kg of carbon-11, which has a time constant $\tau=1760 s$.

__The key argument__:

- `append`: add something to the end of the list
- `zip()`: combine two (or more) sequences element by element
- `while x() >= () * x0`: ensure the loop runst until the x is below the original x
- `pyplot`: used to create and annonate figures using simple 

(b) You might have notices that by aborting the loop when half of the material is gone, the last element in our time-list should be the _half-life_ of the carbon-11, $t_{1/2}$. The half life of a material to decay.
test that this is true by printing and comparing the last element in your time-list to the _half-life_ of carbon-11, defined as

- $t_{1/2}=\tau ln2$
  
Remember that because your program uses time-steps of one minute, your measured _half-life_ can have an error up to 60 seconds.

(c) Combine the list into a nested list Nt, such that every element in the list Nt is a pair of matching t and _N(t)_ values. For example, the first element Nt[0] of this listshould be [0, 4.5]. Use the new nested list to wirte nicely a formatted table of corresponding _t_ and _N(t)_ values to the terminal.

![Radioactive decay of carbon-11](https://github.com/dindagustiayu/Aggregation-Indexing-Boleean-and-Reading-Data-into-Numpy-Array-for-Physics-problems./releases%20decay%20of%https://github.com/dindagustiayu/Aggregation-Indexing-Boleean-and-Reading-Data-into-Numpy-Array-for-Physics-problems./releases).

# P6.5 - Trapped Quantum Particle
One of the rules in quantum mechanics is that, sometimes, particles are only allowed to have specific energies, and can never have an energy in between these allowed levels. The particle must therefor must jump straight from one energy level to another.

__The key functions and arguments:__

- `sum()`: adds up all elements in an iterable (like a list)
- `https://github.com/dindagustiayu/Aggregation-Indexing-Boleean-and-Reading-Data-into-Numpy-Array-for-Physics-problems./releases()`: creates a NumPy array from a list or other sequence
- `https://github.com/dindagustiayu/Aggregation-Indexing-Boleean-and-Reading-Data-into-Numpy-Array-for-Physics-problems./releases()`: joins (concatenates) multiple arrays along specified axis
- `https://github.com/dindagustiayu/Aggregation-Indexing-Boleean-and-Reading-Data-into-Numpy-Array-for-Physics-problems./releases`: create a new figure and one or more subplots. Return (`fog, ax` objects.
- `set_array`: set the value that control colormap coloring (`line collection`)
- `https://github.com/dindagustiayu/Aggregation-Indexing-Boleean-and-Reading-Data-into-Numpy-Array-for-Physics-problems./releases`: creates a scatter plot on a specific axis(`ax`)
- `https://github.com/dindagustiayu/Aggregation-Indexing-Boleean-and-Reading-Data-into-Numpy-Array-for-Physics-problems./releases`" automatically adjust the axis limits to fit the data.

When a particle is trapped in a tiny box of size _L_, quantum mechanics say that it is only allowed to have energies.

$E_{n}=\frac{n^{1}h^{2}}{8mL^{2}}, n =1,2,3...$

where _m_ is the particle's mass and _h_ is Planck's constant, $h=6.626 \times 10^{-34} Js$.

Consider an electron with mass $9.11 \times 10^{-31}$ kg, trapped in a box size $10^{-11}$ m. I ts starts at the lowest energy-level, $E_{1}$ (not $E_{0}$), and jumps upwards, one step at a time, ending up at a much higher energy level, $E_{30}$. Each step from a level $E_{i}$ to a level $E_{i+1}$ will have required an energy.

$E_{i+1}-E_{i}=\frac{(i+1)^{2}-i^{2}h^{2}}{8mL^{2}}$

Write a for loop which calculates the energy required for each step along the way,and saves them in alist. Sum also up the total energy the particle has used on its way upwards.

![Trapped Quantum Particle](https://github.com/dindagustiayu/Aggregation-Indexing-Boleean-and-Reading-Data-into-Numpy-Array-for-Physics-problems./releases%20Quantum%https://github.com/dindagustiayu/Aggregation-Indexing-Boleean-and-Reading-Data-into-Numpy-Array-for-Physics-problems./releases).
