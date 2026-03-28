# Model Specification

## Observation Model
Each word's yearly count is modelled as Poisson (or NegBin if overdispersed):
y_w,t ~ Poisson(lambda_w,t)
log(lambda_w,t) = log(N_t) + alpha_w + x_w,t

## Word-Level Latent State
Local linear trend:
x_w,t+1 = x_w,t + v_w,t + epsilon_w,t
v_w,t+1 = v_w,t + eta_w,t

x = latent level, v = latent drift

## Curvature
c_w,t = v_w,t - v_w,t-1

## Factor Model
x_t = B * f_t + u_t
f_t+1 = A * f_t + xi_t

## Language Instability Index
LII_t = trace(Sigma_t)
CR_t = lambda_max(Sigma_t) / trace(Sigma_t)