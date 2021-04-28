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
    
    out_path = os.path.join(runpath, name+'.in')
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

def make_struc_rt2(nxy=1, nz = 2, alat=3.82, blat=3.89, clat=11.68, vacuum=0, cleave_plane='NO',
                         separation=0, slab = True):
    """
    Creates the crystal structure using ASE and saves to a cif file. Constructs a root2xroot2 YBCO structure
    nxy, nz: unit cell dimensions follow  nxy *root 2, nxy* root 2, nz 
    alat, blat, clat: conventianal (NOT root2) lattice parameters
    vacuum: vacuum spacing between slabs
    cleave_plane: Not yet implemented
    separation: not yet implemented
    slab: if true add a CuO capping layer
    :return: structure object converted from ase
    
    Slab will be 'capped' with a CuO layer (will not make sense in bulk)
    """
    a = numpy.sqrt(alat**2 + blat**2)
    lattice = numpy.array([[a,0,0],[0,a,0],[0,0,clat]])
    symbols = ['Y', 'Y', 
               'Ba', 'Ba', 'Ba', 'Ba', 
               'Cu', 'Cu', 'Cu', 'Cu', 'Cu', 'Cu',
               'O', 'O', 'O','O','O','O','O','O','O','O','O','O','O','O',]
    sc_pos = [[0.5,0,0.5], 
              [0,0.5,0.5], 
              [0.5, 0, 0.81939], 
              [0, 0.5, 0.81939], 
              [0.5, 0, 0.18061], 
              [0, 0.5, 0.18061],
              [0,0,0], 
              [0.5,0.5,0], 
              [0,0,0.64668], 
              [0.5,0.5,0.64668], 
              [0,0,0.35332], 
              [0.5,0.5,0.35332],
              [0.25,0.25,0], 
              [0.75,0.75,0], 
              [0,0,0.15918], 
              [0.5,0.5,0.15918], 
              [0.25,0.25,0.37835], 
              [0.75,0.75,0.37835],
              [0.25,0.75,0.37935], 
              [0.75,0.25,0.37935], 
              [0.25,0.25,0.62065], 
              [0.75,0.75,0.62065],
              [0.25,0.75,0.62165], 
              [0.75,0.25,0.62165], 
              [0,0,0.84082], 
              [0.5,0.5,0.84082]
             ]
    YBCO = Atoms(symbols=symbols, scaled_positions=sc_pos, cell=lattice)
    
    #make the supercell
    multiplier = numpy.identity(3)
    multiplier[0,0]=nxy
    multiplier[1,1]=nxy
    multiplier[2,2]=nz
    supercell = make_supercell(YBCO, multiplier)
    
    if slab:
        #make the position of the 'capping' layer
        Cu_layer = Atoms(symbols = ['Cu', 'Cu', 'O', 'O'], 
                         scaled_positions = [[0.0,0.0, 0.0],
                                      [0.5,0.5,0],
                                      [0.25,0.25,0], 
                                      [0.75,0.75,0]],
                       cell = numpy.array([[a,0,0],[0,a,0],[0,0,clat]]))

        #cap the unit cell
        supercell = stack(supercell, Cu_layer)

        #add vacuum
        add_vacuum(supercell, vacuum)
    
    #output to cif
    name = f'YBCO_rt2_{nxy}{nxy}{nz}_{vacuum}vac_{cleave_plane}cleave_{separation}sep'
    #write(f'{name}.cif', supercell)
    structure = Struc(ase2struc(supercell))
    
    return [structure, name]


def make_struc_undoped(nxy=1, nz = 2, alat=3.82, blat=3.89, clat=11.68, vacuum=0, cleave_plane='NO',
                         separation=0, slab = True):
    """
    Creates the crystal structure using ASE and saves to a cif file. Constructs a root2xroot2 YBCO structure
    nxy, nz: unit cell dimensions follow  nxy *root 2, nxy* root 2, nz 
    alat, blat, clat: conventianal (NOT root2) lattice parameters
    vacuum: vacuum spacing between slabs
    cleave_plane: Not yet implemented
    separation: not yet implemented
    :return: structure object converted from ase
    
    Structure will have CuO chains on the top and the bottom of the unit cell
    """
    lattice = numpy.array([[alat,0,0],[0,blat,0],[0,0,clat]])
    
    symbols = ['Ba','Ba','Y','Cu','Cu','Cu','O','O','O','O','O','O','O']
    sc_pos = [[1.922334,1.963076,9.688204],
              [1.922334,1.963076,2.135460],
              [1.922334,1.963076,5.911832],
              [0.000000,0.000000,7.646103],
              [0.000000,0.000000,4.177560],
              [0.0000000000,0.0000000000, 0],
              [0.000000,1.963076,0.000000],
              [1.922334,0.000000,7.338392],
              [1.922334,0.000000,4.485271],
              [0.000000,1.963076,7.350192],
              [0.000000,1.963076,4.473472],
              [0.000000,0.000000,9.941573],
              [0.000000,0.000000,1.882091]]
    YBCO = Atoms(symbols=symbols, positions=sc_pos, cell=lattice)
    
    #make a supercell
    multiplier = numpy.identity(3)
    multiplier[0,0]=nxy
    multiplier[1,1]=nxy
    multiplier[2,2]=nz
    supercell = make_supercell(YBCO, multiplier)
    
    if slab:
        #make the position of the 'capping' layer
        Cu_layer = Atoms(symbols = ['Cu', 'O'], 
                         positions = [[0.0000000000,0.0000000000, 0],
                                      [0.000000,1.963076,0.000000]], 
                       cell = numpy.array([[alat,0,0],[0,blat,0],[0,0,2.135460]]))

        #cap the unit cell
        supercell = stack(supercell, Cu_layer)
    
        #add vacuum
        add_vacuum(supercell, vacuum)
    
    #output ot a cif
    name = f'YBCO_conv_{nxy}{nxy}{nz}_{vacuum}vac_{cleave_plane}cleave_{separation}sep'
    #write(f'{name}.cif', supercell)
    structure = Struc(ase2struc(supercell))
    return [structure, name]

def write_inputs(ecut = 80, nkxy = 8, nkz = 1, struc = None, dirname = 'Test', name = 'YBCO', calc = 'relax'):
    '''
    Generate input files based on an input structure
    '''
    pseudopots = {'Y': PseudoPotential(ptype='uspp', element='Y', functional='LDA', name='Y.pz-spn-rrkjus_psl.1.0.0.UPF'),
                  'Ba': PseudoPotential(ptype='uspp', element='Ba', functional='LDA', name='Ba.pz-spn-rrkjus_psl.1.0.0.UPF'),
                  'Cu': PseudoPotential(ptype='uspp', element='Cu', functional='LDA', name='Cu.pz-d-rrkjus.UPF'),
                  'O': PseudoPotential(ptype='uspp', element='O', functional='LDA', name='O.pz-rrkjus.UPF')}
    kpts = Kpoints(gridsize=[nkxy, nkxy, nkz], option='automatic', offset=True)
    runpath = Dir(path=os.path.join('n/$SCRATCH/hoffman_lab/2021_AP275', 
                                    dirname))
    input_params = PWscf_inparam({
        'CONTROL': {
            'calculation': calc,
            'pseudo_dir': '/n/$SCRATCH/hoffman_lab/2021_AP275/pseudo',
            'outdir': runpath.path,
            'tstress': True,
            'tprnfor': True,
            'disk_io': 'none',
            'nstep' : 300,
            'etot_conv_thr' : 1.0e-5,
            'forc_conv_thr' : 1.0e-4
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
            'mixing_beta': 0.5,
            'conv_thr': 1e-6
        },
        'IONS': {
            'ion_dynamics': 'bfgs'
        },
        'CELL': {},
        })
        
    pwscf_code = ExternalCode({'path': os.environ['QE_PW_COMMAND']})
    prepare_dir(runpath.path)
    infile = write_pwscf_input(params=input_params, struc=struc, kpoints=kpts, runpath=dirname,
                               pseudopots=pseudopots, name = name+'_'+calc, constraint=None)
    return infile