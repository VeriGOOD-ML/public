from os import listdir
from os.path import isfile, join

design_list = ["05", "06"]

for design in design_list:
    mypath = "./design" + design + "/invs_script"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for tcl_file in onlyfiles:
        tcl_file = mypath + "/" + tcl_file
        with open(tcl_file) as f:
            content = f.read().splitlines()
        f.close()

        line = "# This script was written and developed by ABKGroup students at UCSD. "
        line += "However, the underlying commands and reports are copyrighted by Cadence.\n"
        line += "# We thank Cadence for granting permission to share our research to help "
        line += "promote and foster the next generation of innovators.\n"

        f = open(tcl_file, "w")
        f.write(line)
        for line in content:
            f.write(line + "\n")
        f.close()





