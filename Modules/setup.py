import os,sys

def getHomeDir():
    return os.path.dirname(os.getcwd())

def writeFile(text,fileDir):
    with open(fileDir,"w") as file:
        file.write(text)

def writeHomeDir(homeDir):
    writeFile(homeDir,os.path.join(homeDir,"Modules","MAIN_DIR.txt"))
    if os.path.exists(os.path.join(homeDir,"Notebooks")):
        writeFile(homeDir,os.path.join(homeDir,"Notebooks","MAIN_DIR.txt"))

if __name__ == "__main__":

    homeDir = getHomeDir()
    writeHomeDir(homeDir)
    
    if sys.version_info[0]==3:
        os.system("python3 setupcython.py build_ext --inplace")
    else:
        os.system("python setupcython.py build_ext --inplace")