# Based on https://github.com/dcmocanu/sparse-evolutionary-artificial-neural-networks/blob/master/SET-RBM-Sparse-Python-Data-Structures/sparseoperations.pyx

# compile this file with: "cythonize -a -i sparseupdate.pyx" or "python setupcython.py build_ext --inplace" or "python3 setupcython.py build_ext --inplace"
cimport numpy as np

def LL_W_updates_Cython(double wDecay, double lr, np.ndarray[np.float32_t,ndim=2] DV, np.ndarray[np.float32_t,ndim=2] DH, np.ndarray[np.float32_t,ndim=2] MV, np.ndarray[np.float32_t,ndim=2] MH, np.ndarray[int,ndim=1] vis_loc,np.ndarray[int,ndim=1] hid_loc,np.ndarray[np.float32_t,ndim=1] out):
    cdef:
        size_t i,j
        double s1,s2
    for i in range(vis_loc.shape[0]):
        s1=0
        s2=0
        for j in range(DV.shape[0]):
            s1+=DV[j,vis_loc[i]]*DH[j, hid_loc[i]]
            s2+=MV[j,vis_loc[i]]*MH[j, hid_loc[i]]
        out[i]=(s1/DV.shape[0]-s2/DV.shape[0])#-wDecay*out[i]

def discriminative_W_updates_Cython(double wDecay, double lr, np.ndarray[np.float32_t,ndim=3] INPUT, np.ndarray[np.float32_t,ndim=2] SUM, np.ndarray[int,ndim=1] vis_loc,np.ndarray[int,ndim=1] hid_loc,np.ndarray[np.float32_t,ndim=1] out):
    cdef:
        size_t i,j
        double s
    for i in range(vis_loc.shape[0]):
        s=0
        for j in range(INPUT.shape[0]):
            s+=INPUT[j,vis_loc[i],hid_loc[i]]*SUM[j, hid_loc[i]]
        out[i]=s/INPUT.shape[0]#-wDecay*out[i]