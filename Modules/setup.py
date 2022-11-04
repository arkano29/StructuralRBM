import os

def getHomeDir():
    return os.path.dirname(os.getcwd())

def writeFile(text,fileDir):
    with open(fileDir,"w") as file:
        file.write(text)

def writeHomeDir(homeDir):
    writeFile(homeDir,os.path.join(homeDir,"Modules","MAIN_DIR.txt"))
    writeFile(homeDir,os.path.join(homeDir,"Notebooks","MAIN_DIR.txt"))

if __name__ == "__main__":

    homeDir = getHomeDir()
    writeHomeDir(homeDir)

    os.system("python3 setupcython.py build_ext --inplace")
