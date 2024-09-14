import fasttext

model_path = './models/subject_aerospace.model'
model = fasttext.load_model(model_path)

# Engineering,_Aerospace
text= 'A combined experimental and numerical study was carried out to understand the development of vortices on the U.S. Army/Navy basic finner configuration at high angles of incidence. The experiments were carried out at the Florida A&M University and Florida State University low-speed wind tunnel, and numerical simulations were performed at U.S. Army Research Laboratory using the U.S. Department of Defense Kestrel solver. The results show a good agreement of basic aerodynamic characteristics between the experimental data and numerical simulations at low angles of incidence over the range of flow conditions; however, at high angles of incidence, the predicted aerodynamic coefficients from simulations were relatively lower. Both experimental and numerical simulation results show that asymmetric vortices develop on the cone cylinder and interact with the fins downstream at high angles of incidence. The primary vortices eventually lift off from the body, resulting in multiple secondary shear-layer vortices. The vortex liftoff location moves upstream with an increase in angles of incidence. The liftoff location of the two primary vortices can result in a change of side-force magnitude and direction. At supersonic speeds, the characteristics of the two primary vortices in terms of their size, strength, and the location on the body are very different as compared to those at subsonic speeds.'

model_predict = model.predict(text, k=-1, threshold=0.5)
print(model_predict)
