import numpy as np
import re, sys, math
from ase import Atoms
from ase.io import read, write
import matplotlib.pyplot as plt

""" this program plots the g(r) for pairs of atomic species, as described in
	http://www.compsoc.man.ac.uk/~lucky/Democritus/Theory/rdf.html
	written by Claudio Padilha, Fri 25 Aug 2017 @ York University
	* use python 3.3 at least to run this code* """

def organize (a):
	"""this function receives an Atoms data structure (ASE libraries),
	organizes the atoms in it according to their atomic symbols,
	and returns another Atoms data structure which is organized"""
	symb = []
	b = Atoms ()
	b.cell = a.cell
	b.pbc = [True, True, True]
	for j in a:
	    if str(j.symbol) not in symb:
	    	symb.append(str(j.symbol))

	symb = sorted(symb)

	for i in symb:
		for j in a:
			if str(j.symbol) == i:
				b.append (j)

	return b

def dist (a, b=Atoms('X', positions=[(0,0,0)])):
	"""this function calculates the distance that the atom a is
	from atom b. Default for atom b is X species at origin"""
	return ((a.x-b.x) ** 2 + (a.y-b.y) ** 2 + (a.z-b.z) ** 2) ** 0.5

re.UNICODE
cp2k_outfile = sys.argv[1]
xyz_outfile = sys.argv[2]
save = sys.argv[3]

# open cp2k outfile to find the last unit cell
# (assuming you run a cell relaxation)
with open(cp2k_outfile, 'r') as f:
    content = f.read()
    # first we look for the string just before the data
    i = content.index('GEOMETRY OPTIMIZATION COMPLETED')
    # this is the block of text where the information is
    data = content[i+276:i+518]

# now we build a list of 3 lists to store the cell components
cell = []

# split the text which contain the vector components into lines
lines = re.split("\n+", data)
for line in lines:
	# split each line by spaces
	entries = re.split("\s+",line)
	# store the components of each vector
	temp = []
	# add each list (vector) into the main list (cell)
	for i in range (5, 8):
		temp.append(entries[i])
	cell.append(temp)

# turn the cell into a numpy array of floats
cell = np.array(cell, dtype=float)

# read the xyz file from the relaxation run
# ase read always reads the last configuration (index=-1 by default)
at = read(xyz_outfile, format='xyz')

# set the cell we obtained previously
at.cell = cell

# set it periodic in all 3 dimensions
at.pbc=[True,True,True]

# sort atomic species
at = organize(at)

# find smallest vector of unit cell
mod = lambda x : (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** 0.5
mod_cell = []
for vec in at.cell:
	mod_cell.append(mod(vec))

# index of max vector of cell
max_v_i = mod_cell.index(np.max(mod_cell))

# collect unique symbols of atoms
symb = sorted(list(set(at.get_chemical_symbols())))

# here we collect all distances in a n x n matrix, where entry
# (i,j) is a list of distances between atoms of species i and j
d = [[[] for j in range(len(symb))] for i in range(len(symb))]

# Here the magic happens. Generate all possible neighboring cells,
# and calculate the distances between each atom of the original
# cell and the neighboring cells (including a copy of the original).
for a1 in at:
	for i in range (-1, 2):
		for j in range (-1, 2):
			for k in range (-1, 2):
				# tp is at copied and translated
				tp = at.copy()
				tp.translate(i*at.cell[0]+j*at.cell[1]+k*at.cell[2])
				for a2 in tp:
					# print(dist(a1,a2))
					d[symb.index(a1.symbol)][symb.index(a2.symbol)].append(dist(a1,a2))

# number of bins in the 'histogram' of g(r)
n = 60

# create big figure that will contain subplots
fig = plt.figure()

# create multiplot for all available pairs of species
# active plot is number k
k = 1
for i in range(len(symb)):
	for j in range(len(symb)):
		# create subplot
		sf = fig.add_subplot(len(symb),len(symb),k)

		# here we will store the counts for each slice (r, r + dr)
		h = np.zeros(n)

		# thickness of each slice
		dr = mod_cell[max_v_i] / n

		# copy the vector of distances into a numpy float array
		temp = np.array(d[i][j], dtype='float')

		# count the occurrences 
		h[0] = 0
		for l in range(1, n):
			h[l] = len(temp[(temp > l * dr) & (temp < (l + 1) * dr)]) / (4 * math.pi * dr * (l * dr) ** 2)

		# this is the plot itself
		sf.plot(np.linspace(0, mod_cell[max_v_i],n), h, color='black')
		
		# we put all xticks equal and get rid of yticks
		sf.set_xticks(np.arange(0,11,2.5))
		sf.set_yticklabels([])
		sf.tick_params(axis='y', which='both', left='off')
		
		# this puts tick labels and axis label along x axis only at the bottom subplots
		if (i < len(symb) - 1):
			sf.set_xticklabels([])
		else:
			sf.set_xlabel('distance ($\AA$)')
	
		# this puts axis label along y axis only at the left most subplots
		if (j == 0):
			sf.set_ylabel('g(r)')

		# put the kind of pair we are plotting the rdf for eachplot
		sf.set_title(symb[i] + ' - ' + symb[j])

		# next active plot
		k += 1

# make plot look nice and show it
plt.tight_layout()
plt.show()
