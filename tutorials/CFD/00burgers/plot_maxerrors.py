import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

# ROM intrusive errors
# error_Linf_intrusive = np.load("./ITHACAoutput/ErrorsLinf/errLinfUIntrusive.npy")
error_Linf_intrusive_NMLSPGCentral = np.load("./ITHACAoutput/ErrorsLinf/errLinfUNMLSPG.npy")
# error_Linf_intrusive_NMLSPGTrue = np.load("./ITHACAoutput/ErrorsLinf/errLinfUNMLSPGTrue.npy")
# error_consistency = np.load("./ITHACAoutput/ErrorsLinf/errConsistency.npy")
# non intrusive errors
# error_Linf_nonintrusive= np.load("./ITHACAoutput/ErrorsLinf/errLinfUnonIntrusive.npy")
# error_Linf_nonintrusive_covae = np.load("./Autoencoders/ConvolutionalAe/errLinfUconvAeNonIntrusive.npy")

# projection errors
# error_Linf_ROM_projection = np.load("./ITHACAoutput/ErrorsLinf/errLinfUtrueProjectionROM.npy")
error_Linf_CAE_projection = np.load("./Autoencoders/ConvolutionalAe/errLinfUconvAeProjection.npy")


# plt.semilogy(np.arange(error_Linf_intrusive.shape[0])[1:]/1000, error_Linf_intrusive[1:], label="intrusive",  linewidth=4)
# plt.semilogy(np.arange(error_Linf_intrusive.shape[0])[1:]/1000, error_Linf_nonintrusive[1:], label="non-intrusive",  linewidth=4)
# plt.semilogy(np.arange(error_Linf_intrusive.shape[0])[1:]/1000, error_Linf_nonintrusive_covae[1:], label="non-intrusive convolutional AE",  linewidth=4)
plt.semilogy(np.arange(error_Linf_CAE_projection.shape[0])[1:]/1000, error_Linf_CAE_projection[1:], label="projection error CAE",  linewidth=2)
# plt.semilogy(np.arange(error_Linf_CAE_projection.shape[0])[1:]/1000, error_Linf_ROM_projection[1:], label="projection error ROM",  linewidth=2)
plt.semilogy(np.arange(error_Linf_CAE_projection.shape[0])[1:]/1000, error_Linf_intrusive_NMLSPGCentral[1:], label="NM-LSPG-Central",  linewidth=2)
# plt.semilogy(np.arange(error_Linf_CAE_projection.shape[0])[1:]/1000, error_Linf_intrusive_NMLSPGTrue[1:], label="NM-LSPG-TrueJacobian",  linewidth=2)
# plt.semilogy(np.arange(error_consistency.shape[0])[1:]/1000, error_consistency[1:], label="non-intrus-cae-lstm consistency",  linewidth=2)

plt.legend()
plt.ylim([1e-4,1e-0])
plt.grid(True, which="both")
plt.xlabel("time instants [s]")
plt.ylabel(" log10 relative Linf error")
plt.title("Error of test sample for reduced Burgers' PDE\n reduced dimension is 4")
plt.show()