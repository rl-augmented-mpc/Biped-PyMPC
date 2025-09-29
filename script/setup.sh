#!/bin/bash

gdown 'https://drive.google.com/uc?id=1Dnh_pZS7uwSCxQTlOMHU14IbNbh0fRAA'
gdown 'https://drive.google.com/uc?id=1cwX_FOIrJd-xI8VhOBva9MTjElH4ptdN'
gdown 'https://drive.google.com/uc?id=1CM26E7BECiCx-liNmJA3g3Q6MnMq4Teq'
gdown 'https://drive.google.com/uc?id=1FTZ6L6LauDwEC9P93x8bxIcp_H1T90w_'
gdown 'https://drive.google.com/uc?id=1Hlduw-bUFtVQPkMQM66UL3xsxgf1AMy5'

# move files
mv libqp_former.so biped_pympc/cusadi/src/cusadi_functions
mv libsparse_pdipm_multiple_iterations.so biped_pympc/cusadi/src/cusadi_functions
mv mpc_multiple_iter_5_solver_240v_140eq_160ineq.casadi biped_pympc/cusadi/src/casadi_functions
mv mpc_multiple_iter_20_solver_240v_140eq_160ineq.casadi biped_pympc/cusadi/src/casadi_functions
mv srbd_qp_mat.casadi biped_pympc/cusadi/src/casadi_functions