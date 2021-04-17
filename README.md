# AP275

Pseudopotentials are stored in PP folder.

1. Start with a relaxed calculation (in Project.ipynb), output in YBCO_first_relax.pwo
2. Do the ecut_test (where were the files generated?)
3. Do a z test, relaxing after every additional unit cell.

Current questions:

Have the input files and cifs in the Z_Test_LL folder to do the the Z tests in 3. 
- Should the undoped unit cell be conventional instead of rt(2)xrt(2), which will include more atoms?
- Why does the 1x1x1 slab say it has 108 atoms in vesta, when it should only have 26 atoms (as seen is the structure parameters of vesta)