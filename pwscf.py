import os, copy, numpy
from labutil.objects import TextFile, ExternalCode, File, Param, Struc, ase2struc, Dir, Kpoints, PseudoPotential
from labutil.util import prepare_dir, run_command
from ase.spacegroup import crystal
from ase.io import write
from ase.build import bulk, make_supercell, add_vacuum, stack
from ase import Atoms

class PWscf_inparam(Param):
    """
    Data class containing parameters for a Quantum Espresso PWSCF calculation
    it does not include info on the cell itself, since that will be taken from a Struc object
    """
    pass


def qe_value_map(value):
    """
    Function used to interpret correctly values for different
    fields in a Quantum Espresso input file (i.e., if the user
    specifies the string '1.0d-4', the quotes must be removed
    when we write it to the actual input file)
    :param: a string
    :return: formatted string to be used in QE input file
    """
    if isinstance(value, bool):
        if value:
            return '.true.'
        else:
            return '.false.'
    elif isinstance(value, (float, numpy.float)) or isinstance(value, (int, numpy.int)):
        return str(value)
    elif isinstance(value, str):
        return "'{}'".format(value)
    else:
        print("Strange value ", value)
        print("type ", type(value))
        raise ValueError

def write_pwscf_input(runpath, params, struc, kpoints, pseudopots, name = 'pwscf.in', constraint=None):
    """Make input param string for PW"""
    # automatically fill in missing values
    print("MAKING FILE")
    pcont = copy.deepcopy(params.content)
    pcont['SYSTEM']['ntyp'] = struc.n_species
    pcont['SYSTEM']['nat'] = struc.n_atoms
    pcont['SYSTEM']['ibrav'] = 0
    # Write the main input block
    inptxt = ''
    for namelist in ['CONTROL', 'SYSTEM', 'ELECTRONS', 'IONS', 'CELL']:
        inptxt += '&{}\n'.format(namelist)
        for key, value in pcont[namelist].items():
            inptxt += '    {} = {}\n'.format(key, qe_value_map(value))
        inptxt += '/ \n'
    # write the K_POINTS block
    if kpoints.content['option'] == 'automatic':
        inptxt += 'K_POINTS {automatic}\n'
    inptxt += ' {:d} {:d} {:d}'.format(*kpoints.content['gridsize'])
    if kpoints.content['offset']:
        inptxt += '  1 1 1\n'
    else:
        inptxt += '  0 0 0\n'

    # write the ATOMIC_SPECIES block
    inptxt += 'ATOMIC_SPECIES\n'
    for elem, spec in struc.species.items():
        inptxt += '  {} {} {}\n'.format(elem, spec['mass'], pseudopots[elem].content['name'])

    # Write the CELL_PARAMETERS block
    inptxt += 'CELL_PARAMETERS {angstrom}\n'
    for vector in struc.content['cell']:
        inptxt += ' {} {} {}\n'.format(*vector)

    # Write the ATOMIC_POSITIONS in crystal coords
    inptxt += 'ATOMIC_POSITIONS {angstrom}\n'
    for index, positions in enumerate(struc.content['positions']):
        inptxt += '  {} {:1.5f} {:1.5f} {:1.5f}'.format(positions[0], *positions[1])
        if constraint and constraint.content['atoms'] and str(index) in constraint.content['atoms']:
            inptxt += ' {} {} {} \n'.format(*constraint.content['atoms'][str(index)])
        else:
            inptxt += '\n'
    
    out_path = os.path.join(runpath, name+'.pwi')
    infile = TextFile(path=out_path, text=inptxt)
    infile.write()
    print(f"Writing {out_path}")
    return infile


def run_qe_pwscf(struc, runpath, pseudopots, params, kpoints, constraint=None, ncpu=1, nkpool=1):
    pwscf_code = ExternalCode({'path': os.environ['QE_PW_COMMAND']})
    prepare_dir(runpath.path)
    infile = write_pwscf_input(params=params, struc=struc, kpoints=kpoints, runpath=runpath,
                               pseudopots=pseudopots, constraint=constraint)
    outfile = File({'path': os.path.join(runpath.path, 'pwscf.out')})
    pwscf_command = "mpirun -np {} {} -nk {} < {} > {}".format(ncpu, pwscf_code.path, nkpool, infile.path, outfile.path)
    run_command(pwscf_command)
    return outfile


def parse_qe_pwscf_output(outfile):
    with open(outfile.path, 'r') as outf:
        for line in outf:
            if line.lower().startswith('     pwscf'):
                walltime = line.split()[-3] + line.split()[-2]
            if line.lower().startswith('     total force'):
                total_force = float(line.split()[3]) * (13.605698066 / 0.529177249)
            if line.lower().startswith('!    total energy'):
                total_energy = float(line.split()[-2]) * 13.605698066
            if line.lower().startswith('          total   stress'):
                pressure = float(line.split()[-1])
            if 'number of k points' in line.lower():
                unique_k =line.split()[4]
    result = {'energy': total_energy, 'force': total_force, 'pressure': pressure, 'unique ks':unique_k}
    return result

'''
METHODS ADDED 
'''

def make_struc_doped(nxy=1, nz = 2, alat=3.78, blat=3.88, clat=11.68, vacuum=0, cleave_plane='NO',
                         separation=0, slab = True):
    """
    Creates the crystal structure using ASE and saves to a cif file. Constructs a root2xroot2 YBCO structure
    with 1/8 Y --> Ca doping
    nxy, nz: unit cell dimensions follow  nxy *2root 2, nxy* 2root 2, nz 
    alat, blat, clat: conventianal (NOT root2) lattice parameters
    vacuum: vacuum spacing between slabs
    cleave_plane: Not yet implemented
    separation: not yet implemented
    slab: if true add a CuO capping layer
    :return: structure object converted from ase
    
    Slab will be 'capped' with a CuO layer (will not make sense in bulk) """
        
    if slab == False: #start with bulk contsants
        lattice = numpy.array([[-alat,blat,0],[alat,blat,0],[0,0,clat]])

        symbols = ['Cu', 'Cu', 'O', 'O',
                   'O','O','Ba', 'Ba',
                   'Cu', 'Cu', 'O', 'O', 'O', 'O',
                   'Y', 'Y',
                   'O', 'O', 'O', 'O' ,'Cu', 'Cu',
                   'Ba', 'Ba', 'O', 'O']
        sc_pos = [[0,0,0], #Cu
                  [0.5,0.5,0], #Cu
                  [0.25,0.25,0], #O
                  [0.75,0.75,0], #O
                  [0,0,0.15918], #O
                  [0.5,0.5,0.15918], #O
                  [0.5, 0, 0.18061], #Ba
                  [0, 0.5, 0.18061], #Ba
                  [0,0,0.35332], #Cu
                  [0.5,0.5,0.35332], #Cu
                  [0.25,0.25,0.37835], #O
                  [0.75,0.75,0.37835], #O
                  [0.25,0.75,0.37935], #O
                  [0.75,0.25,0.37935], #O
                  [0.5,0,0.5], #Y
                  [0,0.5,0.5], #Y
                  [0.25,0.25,0.62065], #O
                  [0.75,0.75,0.62065], #O
                  [0.25,0.75,0.62165], #O
                  [0.75,0.25,0.62165], #O
                  [0,0,0.64668], #Cu
                  [0.5,0.5,0.64668], #Cu
                  [0.5, 0, 0.81939], #Ba
                  [0, 0.5, 0.81939], #Ba
                  [0,0,0.84082], #O
                  [0.5,0.5,0.84082] #O
                 ]
        YBCO = Atoms(symbols=symbols, scaled_positions=sc_pos, cell=lattice)
        
        #make an x/y supercell of 2 and dope 1/8 Y --> Ca
        multiplier = numpy.identity(3)
        multiplier[0,0]=2
        multiplier[1,1]=2
        supercell = make_supercell(YBCO, multiplier)

        temp_sym = supercell.get_chemical_symbols()
        temp_sym[15] = 'Ca'
        supercell.set_chemical_symbols(temp_sym)
        
    else: #use relax bulk results
        lattice = numpy.array([[-7.587021933,7.807135371,0],[7.587021933,7.807135371,0],[0,0,11.698429422]])
        alat = -7.587021933
        blat = 7.807135371
        clat = 11.698429422

        pos = [
            [ 3.78343845e+00,  9.75891921e+00, -1.67587300e-04],
            [ 1.00725138e-02,  9.75891921e+00, -1.67587300e-04],
            [ 4.23017900e-04,  1.30382920e-03, -1.64595500e-04],
            [ 4.23017900e-04,  3.90226386e+00, -1.64595500e-04],
            [ 3.79308795e+00,  3.90226386e+00, -1.64595500e-04],
            [-3.79393398e+00,  7.80843920e+00, -1.64595500e-04],
            [ 3.79327612e+00,  7.80723808e+00, -1.57934000e-04],
            [ 2.34848800e-04,  1.17106004e+01, -1.57934000e-04],
            [-3.79374581e+00,  3.90346498e+00, -1.57934000e-04],
            [ 2.34848800e-04,  7.80723808e+00, -1.57934000e-04],
            [-3.76572710e-03,  5.85492076e+00, -1.49792700e-04],
            [ 3.79727669e+00,  5.85492076e+00, -1.49792700e-04],
            [-3.76572710e-03,  1.36629177e+01, -1.49792700e-04],
            [-3.78974524e+00,  5.85578229e+00, -1.49792700e-04],
            [-3.78837203e+00,  9.75891921e+00, -1.13868000e-04],
            [-5.13893340e-03,  1.95178384e+00, -1.13868000e-04],
            [ 3.79681994e+00,  3.90599078e+00,  1.87712964e+00],
            [-3.30897490e-03,  3.90599078e+00,  1.87712964e+00],
            [-3.30897490e-03, -2.42309020e-03,  1.87712964e+00],
            [-3.79020199e+00,  7.80471228e+00,  1.87712964e+00],
            [ 3.79367597e+00,  7.80631532e+00,  1.88198895e+00],
            [-3.79334596e+00,  3.90438774e+00,  1.88198895e+00],
            [-1.65001400e-04,  7.80631532e+00,  1.88198895e+00],
            [-1.65001400e-04,  1.17115231e+01,  1.88198895e+00],
            [ 1.89675548e+00,  9.75891921e+00,  2.09648480e+00],
            [ 5.69026645e+00,  5.85535153e+00,  2.10018595e+00],
            [-1.89675548e+00,  5.85535153e+00,  2.10018595e+00],
            [-1.90582564e+00,  9.75891921e+00,  2.10377245e+00],
            [-1.88768533e+00,  1.95178384e+00,  2.10377245e+00],
            [ 1.89675548e+00,  5.84739926e+00,  2.10670874e+00],
            [-5.69026645e+00,  5.86330380e+00,  2.10670874e+00],
            [ 1.89675548e+00,  1.95178384e+00,  2.13600676e+00],
            [-3.79277906e+00,  3.90345510e+00,  4.09769996e+00],
            [ 3.79424287e+00,  7.80724796e+00,  4.09769996e+00],
            [-7.31908500e-04,  7.80724796e+00,  4.09769996e+00],
            [-7.31908500e-04,  1.17105905e+01,  4.09769996e+00],
            [-3.79691828e+00,  7.81270396e+00,  4.10085907e+00],
            [ 3.79010366e+00,  3.89799910e+00,  4.10085907e+00],
            [ 3.40730960e-03,  5.56858670e-03,  4.10085907e+00],
            [ 3.40730960e-03,  3.89799910e+00,  4.10085907e+00],
            [-5.48362496e-02,  1.95178384e+00,  4.34900178e+00],
            [-3.73867472e+00,  9.75891921e+00,  4.34900178e+00],
            [-5.69026645e+00,  7.74921333e+00,  4.38013791e+00],
            [ 1.89675548e+00,  3.96148973e+00,  4.38013791e+00],
            [ 3.79329307e+00,  9.75891921e+00,  4.40433153e+00],
            [ 2.17894200e-04,  9.75891921e+00,  4.40433153e+00],
            [-8.00205360e-03,  5.85523771e+00,  4.40490696e+00],
            [ 3.80151302e+00,  5.85523771e+00,  4.40490696e+00],
            [-3.78550891e+00,  5.85546534e+00,  4.40490696e+00],
            [-8.00205360e-03,  1.36626007e+01,  4.40490696e+00],
            [ 1.89675548e+00,  1.17107839e+01,  4.43561367e+00],
            [ 1.89675548e+00,  7.80705449e+00,  4.43561367e+00],
            [-1.89773859e+00,  7.80037845e+00,  4.43774161e+00],
            [-1.89577238e+00,  3.91032461e+00,  4.43774161e+00],
            [-1.89773859e+00,  1.17174600e+01,  4.43774161e+00],
            [ 5.69124955e+00,  7.80037845e+00,  4.43774161e+00],
            [ 1.89675548e+00,  1.95178384e+00,  5.84911393e+00],
            [-1.91567103e+00,  9.75891921e+00,  5.84914643e+00],
            [-1.87783993e+00,  1.95178384e+00,  5.84914643e+00],
            [ 1.89675548e+00,  9.75891921e+00,  5.84921696e+00],
            [ 5.69026645e+00,  5.85535153e+00,  5.84922278e+00],
            [-1.89675548e+00,  5.85535153e+00,  5.84922278e+00],
            [ 1.89675548e+00,  5.83603639e+00,  5.84929230e+00],
            [-5.69026645e+00,  5.87466667e+00,  5.84929230e+00],
            [-1.89578402e+00,  3.91046728e+00,  7.26463087e+00],
            [ 5.69123791e+00,  7.80023577e+00,  7.26463087e+00],
            [-1.89772695e+00,  1.17176027e+01,  7.26463087e+00],
            [-1.89772695e+00,  7.80023577e+00,  7.26463087e+00],
            [ 1.89675548e+00,  7.80726351e+00,  7.26709917e+00],
            [ 1.89675548e+00,  1.17105749e+01,  7.26709917e+00],
            [-7.89885530e-03,  1.36626216e+01,  7.29027403e+00],
            [-7.89885530e-03,  5.85521679e+00,  7.29027403e+00],
            [ 3.80140982e+00,  5.85521679e+00,  7.29027403e+00],
            [-3.78561211e+00,  5.85548626e+00,  7.29027403e+00],
            [ 3.79348859e+00,  9.75891921e+00,  7.29058728e+00],
            [ 2.23762000e-05,  9.75891921e+00,  7.29058728e+00],
            [ 1.89675548e+00,  3.96227918e+00,  7.32234835e+00],
            [-5.69026645e+00,  7.74842388e+00,  7.32234835e+00],
            [-5.41733603e-02,  1.95178384e+00,  7.34616111e+00],
            [-3.73933761e+00,  9.75891921e+00,  7.34616111e+00],
            [-3.79693495e+00,  7.81272687e+00,  7.59786957e+00],
            [ 3.79008699e+00,  3.89797618e+00,  7.59786957e+00],
            [ 3.42397940e-03,  5.59150060e-03,  7.59786957e+00],
            [ 3.42397940e-03,  3.89797618e+00,  7.59786957e+00],
            [-7.41994100e-04,  1.17105987e+01,  7.60094916e+00],
            [-3.79276897e+00,  3.90346331e+00,  7.60094916e+00],
            [ 3.79425296e+00,  7.80723975e+00,  7.60094916e+00],
            [-7.41994100e-04,  7.80723975e+00,  7.60094916e+00],
            [ 1.89675548e+00,  1.95178384e+00,  9.56201848e+00],
            [-5.69026645e+00,  5.86325706e+00,  9.59137593e+00],
            [ 1.89675548e+00,  5.84744600e+00,  9.59137593e+00],
            [-1.88769978e+00,  1.95178384e+00,  9.59410138e+00],
            [-1.90581118e+00,  9.75891921e+00,  9.59410138e+00],
            [-1.89675548e+00,  5.85535153e+00,  9.59777641e+00],
            [ 5.69026645e+00,  5.85535153e+00,  9.59777641e+00],
            [ 1.89675548e+00,  9.75891921e+00,  9.60148070e+00],
            [ 3.79376074e+00,  7.80636347e+00,  9.81621494e+00],
            [-3.79326119e+00,  3.90433959e+00,  9.81621494e+00],
            [-2.49774700e-04,  7.80636347e+00,  9.81621494e+00],
            [-2.49774700e-04,  1.17114750e+01,  9.81621494e+00],
            [ 3.79689188e+00,  3.90592218e+00,  9.82105992e+00],
            [-3.38091530e-03,  3.90592218e+00,  9.82105992e+00],
            [-3.38091530e-03, -2.35449600e-03,  9.82105992e+00],
            [-3.79013005e+00,  7.80478087e+00,  9.82105992e+00]]
        sym = ['O',
               'O',
               'Cu',
               'Cu',
               'Cu',
               'Cu',
               'Cu',
               'Cu',
               'Cu',
               'Cu',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'Ba',
               'Ba',
               'Ba',
               'Ba',
               'Ba',
               'Ba',
               'Ba',
               'Ba',
               'Cu',
               'Cu',
               'Cu',
               'Cu',
               'Cu',
               'Cu',
               'Cu',
               'Cu',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'Ca',
               'Y',
               'Y',
               'Y',
               'Y',
               'Y',
               'Y',
               'Y',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'Cu',
               'Cu',
               'Cu',
               'Cu',
               'Cu',
               'Cu',
               'Cu',
               'Cu',
               'Ba',
               'Ba',
               'Ba',
               'Ba',
               'Ba',
               'Ba',
               'Ba',
               'Ba',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O',
               'O']
        supercell = Atoms(symbols=sym, positions=pos, 
                     cell=lattice)
        
    write('initial.cif', supercell)
    
    #make the supercell of the 2rt(2) doped supercell
    multiplier = numpy.identity(3)
    multiplier[2,2]=nz
    supercell = make_supercell(supercell, multiplier)
    
    write('supercell.cif',supercell)
    
    if slab:
        #make the position of the 'capping' layer
        Cu_layer = Atoms(symbols = ['O','O','Cu','Cu','Cu','Cu',
                                    'Cu','Cu','Cu','Cu','O','O','O','O','O','O'],
                         positions = [[ 3.78343845e+00,  9.75891921e+00, 0],
                                      [ 1.00725138e-02,  9.75891921e+00, 0],
                                    [ 4.23017900e-04,  1.30382920e-03, 0],
                                    [ 4.23017900e-04,  3.90226386e+00, 0],
                                    [ 3.79308795e+00,  3.90226386e+00, 0],
                                    [-3.79393398e+00,  7.80843920e+00, 0],
                                    [ 3.79327612e+00,  7.80723808e+00, 0],
                                    [ 2.34848800e-04,  1.17106004e+01, 0],
                                    [-3.79374581e+00,  3.90346498e+00, 0],
                                    [ 2.34848800e-04,  7.80723808e+00, 0],
                                    [-3.76572710e-03,  5.85492076e+00, 0],
                                    [ 3.79727669e+00,  5.85492076e+00, 0],
                                    [-3.76572710e-03,  1.36629177e+01, 0],
                                    [-3.78974524e+00,  5.85578229e+00, 0],
                                    [-3.78837203e+00,  9.75891921e+00, 0],
                                    [-5.13893340e-03,  1.95178384e+00, 0]],
                       cell = numpy.array([[alat,blat,0],[-alat,blat,0],[0,0,0.1]]))
        #cap the unit cell
        supercell = stack(supercell, Cu_layer)
        print(supercell.cell)
        
        write('capped.cif',supercell)
        
        #add vacuum
        cell = supercell.get_cell()
        cell[2][2]+=20
        supercell.set_cell(cell)
        write('vac.cif',supercell)
    
    #output to cif
    name = f'dYBCO_rt2_{nxy}{nxy}{nz}_{vacuum}vac_{cleave_plane}cleave_{separation}sep'
    write(f'{name}.cif', supercell)
    structure = Struc(ase2struc(supercell))
    
    return [structure, name]


def make_struc_undoped(nxy=1, nz = 2, alat=3.82, blat=3.89, clat=11.68, vacuum=0, cleave_plane='NO',
                         separation=0, slab = True):
    """
    Creates the crystal structure using ASE and saves to a cif file. Constructs a root2xroot2 YBCO structure
    nxy, nz: unit cell dimensions follow  nxy *root 2, nxy* root 2, nz 
    alat, blat, clat: conventianal (NOT root2) lattice parameters
    vacuum: vacuum spacing between slabs
    cleave_plane: "BaO", 'CuO', "Y", or 'NO' for no cleave plane
    separation: separation of the cleave
    :return: structure object converted from ase
    
    Structure will have CuO chains on the top and the bottom of the unit cell
    """
    lattice = numpy.array([[alat,0,0],[0,blat,0],[0,0,clat]])
    
    symbols = ['Cu','O','O','Ba','Cu','O','O','Y','O','O','Cu','Ba','O']
    sc_pos = [[0.0000000000,0.0000000000, 0], #Cu
              [0.000000,1.963076,0.000000], #O
              [0.000000,0.000000,1.882091], #O
              [1.922334,1.963076,2.135460], #Ba
              [0.000000,0.000000,4.177560], #Cu
              [0.000000,1.963076,4.473472], #O
              [1.922334,0.000000,4.485271], #O
              [1.922334,1.963076,5.911832], #Y
              [1.922334,0.000000,7.338392], #O
              [0.000000,1.963076,7.350192], #O
              [0.000000,0.000000,7.646103], #Cu
              [1.922334,1.963076,9.688204], #Ba
              [0.000000,0.000000,9.941573] #O
             ] 
    YBCO = Atoms(symbols=symbols, positions=sc_pos, cell=lattice)
    
    #make a supercell in the z direction
    multiplier = numpy.identity(3)
    multiplier[2,2]=nz
    supercell = make_supercell(YBCO, multiplier)
    
    if slab:
        #make the position of the 'capping' layer
        Cu_layer = Atoms(symbols = ['Cu', 'O'], 
                         positions = [[0.0000000000,0.0000000000, 0],
                                      [0.000000,1.963076,0.000000]], 
                       cell = numpy.array([[alat,0,0],[0,blat,0],[0,0,2.135460]]))
        
        Cu_layer = make_supercell(Cu_layer, multiplier)

        #cap the unit cell
        supercell = stack(supercell, Cu_layer)
    
        #add vacuum
        add_vacuum(supercell, vacuum)
    
    #find the plane to cleave on (closest to the middle)
    if cleave_plane == 'CuO':
        split = int(nz/2)*13 + 2
        
    elif cleave_plane == "BaO":
        split = int(nz/2)*13 + 4
        
    elif cleave_plane == "Y":
        split = int(nz/2)*13 + 8
    
    if cleave_plane != "NO":
        temp_pos = supercell.get_positions()
        temp_pos[split:,2] += separation #add separation in z to all atoms after cleave plane
        supercell.set_positions(temp_pos)
        temp_cell = supercell.get_cell()
        temp_cell[2][2] += separation #add separation to cell height so vacuum is unchanged
        supercell.set_cell(temp_cell)
        
    #make supercell in x/y
    multiplier = numpy.identity(3)
    multiplier[1,1]=nxy
    multiplier[0,0]=nxy
    supercell = make_supercell(supercell, multiplier)
        
    #output ot a cif
    name = f'YBCO_conv_{nxy}{nxy}{nz}_{vacuum}vac_{cleave_plane}cleave_{separation}sep'
    write(f'{name}.cif', supercell)
    structure = Struc(ase2struc(supercell))
    
    return [structure, name]

def make_cleave_struc_undoped(lattice, symbols, sc_pos, nz, cleave_plane='NO',
                         separation=0):
    """
    Creates the crystal structure using ASE and saves to a cif file. Constructs a root2xroot2 YBCO structure
    nxy, nz: unit cell dimensions follow  nxy *root 2, nxy* root 2, nz 
    alat, blat, clat: conventianal (NOT root2) lattice parameters
    vacuum: vacuum spacing between slabs
    cleave_plane: "BaO", 'CuO', "Y", or 'NO' for no cleave plane
    separation: separation of the cleave
    :return: structure object converted from ase
    
    Structure will have CuO chains on the top and the bottom of the unit cell
    """
    
    supercell = Atoms(symbols=symbols, positions=sc_pos, cell=lattice)
  
    
    #find the plane to cleave on (closest to the middle)
    if cleave_plane == 'CuO':
        split = int(nz/2)*13 + 2
        
    elif cleave_plane == "BaO":
        split = int(nz/2)*13 + 4
        
    elif cleave_plane == "Y":
        split = int(nz/2)*13 + 8
    
    if cleave_plane != "NO":
        temp_pos = supercell.get_positions()
        temp_pos[split:,2] += separation #add separation in z to all atoms after cleave plane
        supercell.set_positions(temp_pos)
        temp_cell = supercell.get_cell()
        temp_cell[2][2] += separation #add separation to cell height so vacuum is unchanged
        supercell.set_cell(temp_cell)
        
    #output ot a cif
    name = f'YBCO_{cleave_plane}_cleave_{separation}_sep'
    #write(f'{name}.cif', supercell)
    structure = Struc(ase2struc(supercell))
    
    return [structure, name]

def make_cleave_struc_doped(lattice, symbols, sc_pos, nz, cleave_plane='NO',
                         separation=0):
    """
    Creates the crystal structure using ASE and saves to a cif file. Constructs a root2xroot2 YBCO structure
    nxy, nz: unit cell dimensions follow  nxy *root 2, nxy* root 2, nz 
    alat, blat, clat: conventianal (NOT root2) lattice parameters
    vacuum: vacuum spacing between slabs
    cleave_plane: "BaO", 'CuO', "Y", or 'NO' for no cleave plane
    separation: separation of the cleave
    :return: structure object converted from ase
    
    Structure will have CuO chains on the top and the bottom of the unit cell
    """
    
    supercell = Atoms(symbols=symbols, positions=sc_pos, cell=lattice)
  
    
    #find the plane to cleave on (closest to the middle)
    if cleave_plane == 'CuO':
        split = int(nz/2)*104 + 16
        
    elif cleave_plane == "BaO":
        split = int(nz/2)*104 + 32
        
    elif cleave_plane == "Y":
        split = int(nz/2)*104 + 64
    
    if cleave_plane != "NO":
        temp_pos = supercell.get_positions()
        temp_pos[split:,2] += separation #add separation in z to all atoms after cleave plane
        supercell.set_positions(temp_pos)
        temp_cell = supercell.get_cell()
        temp_cell[2][2] += separation #add separation to cell height so vacuum is unchanged
        supercell.set_cell(temp_cell)
        
    #output ot a cif
    name = f'YBCO_{cleave_plane}_cleave_{separation}_sep'
    #write(f'{name}.cif', supercell)
    structure = Struc(ase2struc(supercell))
    
    return [structure, name]

def write_inputs(ecut = 60, nkxy = 8, nkz = 1, struc = None, dirname = None, calc = 'vc-relax'):
    '''
    Generate input files based on an input structure
    '''
    pseudopots = {'Y': PseudoPotential(ptype='uspp', element='Y', functional='PBE', name='Y.pbe-nsp-van.UPF'),
                  'Ba': PseudoPotential(ptype='uspp', element='Ba', functional='PBE', name='Ba.pbe-nsp-van.UPF'),
                  'Cu': PseudoPotential(ptype='uspp', element='Cu', functional='PBE', name='Cu.pbe-n-van_ak.UPF'),
                  'O': PseudoPotential(ptype='uspp', element='O', functional='PBE', name='O.pbe-van_ak.UPF'),
                 'Ca': PseudoPotential(ptype='uspp', element='Ca', functional='PBE', name='Ca.pbe-nsp-van.UPF')}
    kpts = Kpoints(gridsize=[nkxy, nkxy, nkz], option='automatic', offset=False)
    #runpath = Dir(path=os.path.join('n/$SCRATCH/hoffman_lab/2021_AP275', 
    #                                dirname))
    runpath = Dir(path=os.path.join(os.environ['WORKDIR'], "Input_files"))
    input_params = PWscf_inparam({
        'CONTROL': {
            'calculation': calc,
            'pseudo_dir': './pseudo',
            'outdir': './outdir',
            'tstress': True,
            'tprnfor': True,
            'disk_io': 'none',
            'nstep' : 300,
            'etot_conv_thr' : 1.0e-4,
            'forc_conv_thr' : 1.0e-3
        },
        'SYSTEM': {
            'ecutwfc': ecut,
            'ecutrho': ecut * 12,
            'occupations': 'smearing',
            'smearing': 'gaussian',
            'degauss': 0.01
             },
        'ELECTRONS': {
            'diagonalization': 'david',
            'electron_maxstep': 120,
            'mixing_mode': 'local-TF',
            'mixing_beta': 0.5,
            'mixing_ndim': 10,
            'conv_thr': 1e-8
        },
        'IONS': {
            'ion_dynamics': 'bfgs'
        },
        'CELL': {
            'cell_dynamics': 'bfgs',
            'press_conv_thr':5
        },
        })
        
    pwscf_code = ExternalCode({'path': os.environ['QE_PW_COMMAND']})
    prepare_dir(runpath.path)
    infile = write_pwscf_input(params=input_params, struc=struc, kpoints=kpts, runpath=runpath.path,
                               pseudopots=pseudopots, name = dirname, constraint=None)
    return infile