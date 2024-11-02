# Step 1: Install necessary system packages
!apt-get install -y pkg-config libcairo2-dev

# Step 2: Install Python libraries
!pip install streamlit pyngrok

code = """

import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Define the system of differential equations for a 3-compartment PK model
def three_compartment_model(y, t, CL, V1, Vp1, Vp2, Q1, Q2, Scr, loading_infusion_rate, infusion_rate, time_infusion_start, time_infusion_end, intrathecal_dose_time, intrathecal_dose):
    A1, A2, A3 = y
    CL_adjusted = CL * (1 + ((Scr - 26) * 0.0097))  # Adjust CL by serum creatinine (Scr)

    # Determine if loading or continuous infusion applies
    infusion = 0
    if 0 <= t <= 0.5:
        infusion = loading_infusion_rate  # Loading dose over first 0.5 hours
    elif time_infusion_start <= t <= time_infusion_end:
        infusion = infusion_rate  # Remaining dose over the specified duration

    # Intrathecal injection
    if intrathecal_dose_time <= t <= intrathecal_dose_time + 0.1:
        A1 += intrathecal_dose

    # Central compartment (elimination + distribution)
    dA1_dt = infusion - (CL_adjusted / V1) * A1 - (Q1 / V1) * (A1 - A2) - (Q2 / V1) * (A1 - A3)

    # Peripheral compartment 1 (fast distribution)
    dA2_dt = (Q1 / Vp1) * (A1 - A2)

    # Peripheral compartment 2 (slow distribution)
    dA3_dt = (Q2 / Vp2) * (A1 - A3)

    return [dA1_dt, dA2_dt, dA3_dt]

# Function to simulate PK with loading and infusion dose
def simulate_patient(dose_per_m2, BSA, CL_mean, V1_mean, Vp1_mean, Vp2_mean, Q1_mean, Q2_mean, Scr, CV_CL, CV_V1, age):
    # Introduce inter-individual variability (IIV) using log-normal distributions
    CL = np.random.lognormal(np.log(CL_mean), CV_CL)
    V1 = np.random.lognormal(np.log(V1_mean), CV_V1)

    # Calculate total dose based on BSA
    total_dose = dose_per_m2 * BSA  # mg

    # 10% loading dose over 0.5 hours
    loading_dose = total_dose * 0.1
    loading_duration = 0.5  # hours
    loading_infusion_rate = loading_dose / loading_duration  # mg/h

    # Remaining 90% dose infused over 23.5 hours
    remaining_dose = total_dose * 0.9
    infusion_duration = 23.5  # hours
    infusion_rate = remaining_dose / infusion_duration  # mg/h

    # Intrathecal dose based on age
    intrathecal_dose = 6 if age < 10 else 12.5  # mg

    # Time points
    time_points = np.linspace(0, 48, 500)  # simulate for 48 hours

    # Initial conditions
    A1_0 = 0  # Start with no drug in central compartment
    A2_0 = 0
    A3_0 = 0
    initial_conditions = [A1_0, A2_0, A3_0]

    # Simulate the PK model
    result = odeint(
        three_compartment_model, initial_conditions, time_points,
        args=(CL, V1, Vp1_mean, Vp2_mean, Q1_mean, Q2_mean, Scr,
              loading_infusion
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Define the system of differential equations for a 3-compartment PK model
def three_compartment_model(y, t, CL, V1, Vp1, Vp2, Q1, Q2, Scr, loading_infusion_rate, infusion_rate, time_infusion_start, time_infusion_end, intrathecal_dose_time, intrathecal_dose):
    A1, A2, A3 = y
    CL_adjusted = CL * (1 + ((Scr - 26) * 0.0097))  # Adjust CL by serum creatinine (Scr)

    # Determine if loading or continuous infusion applies
    infusion = 0
    if 0 <= t <= 0.5:
        infusion = loading_infusion_rate  # Loading dose over first 0.5 hours
    elif time_infusion_start <= t <= time_infusion_end:
        infusion = infusion_rate  # Remaining dose over the specified duration

    # Intrathecal injection
    if intrathecal_dose_time <= t <= intrathecal_dose_time + 0.1:
        A1 += intrathecal_dose

    # Central compartment (elimination + distribution)
    dA1_dt = infusion - (CL_adjusted / V1) * A1 - (Q1 / V1) * (A1 - A2) - (Q2 / V1) * (A1 - A3)

    # Peripheral compartment 1 (fast distribution)
    dA2_dt = (Q1 / Vp1) * (A1 - A2)

    # Peripheral compartment 2 (slow distribution)
    dA3_dt = (Q2 / Vp2) * (A1 - A3)

    return [dA1_dt, dA2_dt, dA3_dt]

# Function to simulate PK with loading and infusion dose
def simulate_patient(dose_per_m2, BSA, CL_mean, V1_mean, Vp1_mean, Vp2_mean, Q1_mean, Q2_mean, Scr, CV_CL, CV_V1, age):
    # Introduce inter-individual variability (IIV) using log-normal distributions
    CL = np.random.lognormal(np.log(CL_mean), CV_CL)
    V1 = np.random.lognormal(np.log(V1_mean), CV_V1)

    # Calculate total dose based on BSA
    total_dose = dose_per_m2 * BSA  # mg

    # 10% loading dose over 0.5 hours
    loading_dose = total_dose * 0.1
    loading_duration = 0.5  # hours
    loading_infusion_rate = loading_dose / loading_duration  # mg/h

    # Remaining 90% dose infused over 23.5 hours
    remaining_dose = total_dose * 0.9
    infusion_duration = 23.5  # hours
    infusion_rate = remaining_dose / infusion_duration  # mg/h

    # Intrathecal dose based on age
    intrathecal_dose = 6 if age < 10 else 12.5  # mg

    # Time points
    time_points = np.linspace(0, 48, 500)  # simulate for 48 hours

    # Initial conditions
    A1_0 = 0  # Start with no drug in central compartment
    A2_0 = 0
    A3_0 = 0
    initial_conditions = [A1_0, A2_0, A3_0]

    # Simulate the PK model
    result = odeint(
        three_compartment_model, initial_conditions, time_points,
        args=(CL, V1, Vp1_mean, Vp2_mean, Q1_mean, Q2_mean, Scr,
              loading_infusion_rate, infusion_rate, 0.5, 24.0, 1.0, intrathecal_dose)
    )

    # Convert to concentrations (mg/L)
    C1 = result[:, 0] / V1

    # Create data for CSV output with infusion rate information
    pk_data = []
    for idx, time in enumerate(time_points):
        current_infusion_rate = loading_infusion_rate if time <= 0.5 else (infusion_rate if 0.5 < time <= 24.0 else 0)
        pk_data.append([time, C1[idx], current_infusion_rate])

    return pk_data

# Population PK parameters
CL_mean = 6.9
V1_mean = 20.5
Vp1_mean = 42.1
Vp2_mean = 3.28
Q1_mean = 0.258
Q2_mean = 0.224
Scr = 26.0
CV_CL = 0.175
CV_V1 = 0.1

# Dosing and patient data
dose_per_m2_5gm = 5000

n_patients = 1000

# Generate random BSA and other data for each patient
ages = np.random.uniform(0.75, 15.2, n_patients)
weights = np.random.uniform(4.5, 113.0, n_patients)
heights = np.random.randint(67,175,n_patients)
serum_Creatinine = np.random.uniform(8.0,135.0, n_patients)
albumin_levels = np.random.uniform(3.5, 5.2, n_patients)  # Albumin (g/dL)
total_protein_levels = np.random.uniform(6.0, 8.3, n_patients)  # Total protein (g/dL)
total_bilirubin_levels = np.random.uniform(0.1, 1.2, n_patients)  # Total bilirubin (mg/dL)
direct_bilirubin_levels = np.random.uniform(0.0, 0.3, n_patients)  # Direct bilirubin (mg/dL)
ast_levels = np.random.uniform(10, 40, n_patients)  # AST (U/L)
alt_levels = np.random.uniform(7, 56, n_patients)  # ALT (U/L)

# List of concomitant medications
medications = ["Dasatinib", "Imatinib", "Omeprazole", "Sulphonamides", "NSAIDs", "Penicillin"]

# Store results for all patients
pk_data_1 = []

# Simulate for each patient
for i in range(n_patients):
    age = ages[i]
    weight = weights[i]
    height = heights[i]
    scr = serum_Creatinine[i]
    albumin = albumin_levels[i]
    total_protein = total_protein_levels[i]
    total_bilirubin = total_bilirubin_levels[i]
    direct_bilirubin = direct_bilirubin_levels[i]
    ast = ast_levels[i]
    alt = alt_levels[i]

    # Calculate BSA (m²)
    bsa = np.sqrt((height * weight) / 3600)

    # Randomly assign concomitant medication
    concomitant_medication = np.random.choice(medications)

    # Run the PK simulation
    patient_pk_data_1 = simulate_patient(dose_per_m2_5gm, bsa, CL_mean, V1_mean, Vp1_mean, Vp2_mean, Q1_mean, Q2_mean, scr, CV_CL, CV_V1, age)

    # Append data with patient information
    for time, concentration, infusion_rate in patient_pk_data_1:
        pk_data_1.append([i+1, age, weight, height, scr, albumin, total_protein, total_bilirubin, direct_bilirubin, ast, alt, bsa, dose_per_m2_5gm, dose_per_m2_5gm * bsa, concomitant_medication, time, concentration, infusion_rate])

# Dosing and patient data
dose_per_m2_4gm = 4000
n_patients = 1000

# Generate random BSA and other data for each patient
ages = np.random.uniform(0.75, 15.2, n_patients)
weights = np.random.uniform(4.5, 113.0, n_patients)
heights = np.random.randint(67,175,n_patients)
serum_Creatinine = np.random.uniform(8.0,135.0, n_patients)
albumin_levels = np.random.uniform(3.5, 5.2, n_patients)  # Albumin (g/dL)
total_protein_levels = np.random.uniform(6.0, 8.3, n_patients)  # Total protein (g/dL)
total_bilirubin_levels = np.random.uniform(0.1, 1.2, n_patients)  # Total bilirubin (mg/dL)
direct_bilirubin_levels = np.random.uniform(0.0, 0.3, n_patients)  # Direct bilirubin (mg/dL)
ast_levels = np.random.uniform(10, 40, n_patients)  # AST (U/L)
alt_levels = np.random.uniform(7, 56, n_patients)  # ALT (U/L)

# List of concomitant medications
medications = ["Dasatinib", "Imatinib", "Omeprazole", "Sulphonamides", "NSAIDs", "Penicillin"]

# Store results for all patients
pk_data_2 = []

# Simulate for each patient
for i in range(n_patients):
    age = ages[i]
    weight = weights[i]
    height = heights[i]
    scr = serum_Creatinine[i]
    albumin = albumin_levels[i]
    total_protein = total_protein_levels[i]
    total_bilirubin = total_bilirubin_levels[i]
    direct_bilirubin = direct_bilirubin_levels[i]
    ast = ast_levels[i]
    alt = alt_levels[i]

    # Calculate BSA (m²)
    bsa = np.sqrt((height * weight) / 3600)

    # Randomly assign concomitant medication
    concomitant_medication = np.random.choice(medications)

    # Run the PK simulation
    patient_pk_data_2 = simulate_patient(dose_per_m2_4gm, bsa, CL_mean, V1_mean, Vp1_mean, Vp2_mean, Q1_mean, Q2_mean, scr, CV_CL, CV_V1, age)

    # Append data with patient information
    for time, concentration, infusion_rate in patient_pk_data_2:
        pk_data_2.append([1001+i, age, weight, height, scr, albumin, total_protein, total_bilirubin, direct_bilirubin, ast, alt, bsa, dose_per_m2_4gm, dose_per_m2_4gm * bsa, concomitant_medication, time, concentration, infusion_rate])

# Dosing and patient data
dose_per_m2_3gm = 3000
n_patients = 1000

# Generate random BSA and other data for each patient
ages = np.random.uniform(0.75, 15.2, n_patients)
weights = np.random.uniform(4.5, 113.0, n_patients)
heights = np.random.randint(67,175,n_patients)
serum_Creatinine = np.random.uniform(8.0,135.0, n_patients)
albumin_levels = np.random.uniform(3.5, 5.2, n_patients)  # Albumin (g/dL)
total_protein_levels = np.random.uniform(6.0, 8.3, n_patients)  # Total protein (g/dL)
total_bilirubin_levels = np.random.uniform(0.1, 1.2, n_patients)  # Total bilirubin (mg/dL)
direct_bilirubin_levels = np.random.uniform(0.0, 0.3, n_patients)  # Direct bilirubin (mg/dL)
ast_levels = np.random.uniform(10, 40, n_patients)  # AST (U/L)
alt_levels = np.random.uniform(7, 56, n_patients)  # ALT (U/L)

# List of concomitant medications
medications = ["Dasatinib", "Imatinib", "Omeprazole", "Sulphonamides", "NSAIDs", "Penicillin"]

# Store results for all patients
pk_data_3 = []

# Simulate for each patient
for i in range(n_patients):
    age = ages[i]
    weight = weights[i]
    height = heights[i]
    scr = serum_Creatinine[i]
    albumin = albumin_levels[i]
    total_protein = total_protein_levels[i]
    total_bilirubin = total_bilirubin_levels[i]
    direct_bilirubin = direct_bilirubin_levels[i]
    ast = ast_levels[i]
    alt = alt_levels[i]

    # Calculate BSA (m²)
    bsa = np.sqrt((height * weight) / 3600)

    # Randomly assign concomitant medication
    concomitant_medication = np.random.choice(medications)

    # Run the PK simulation
    patient_pk_data_3 = simulate_patient(dose_per_m2_3gm, bsa, CL_mean, V1_mean, Vp1_mean, Vp2_mean, Q1_mean, Q2_mean, scr, CV_CL, CV_V1, age)

    # Append data with patient information
    for time, concentration, infusion_rate in patient_pk_data_3:
        pk_data_3.append([2001+i, age, weight, height, scr, albumin, total_protein, total_bilirubin, direct_bilirubin, ast, alt, bsa, dose_per_m2_3gm, dose_per_m2_3gm * bsa, concomitant_medication, time, concentration, infusion_rate])

# Dosing and patient data
dose_per_m2_2gm = 2000
n_patients = 1000

# Generate random BSA and other data for each patient
ages = np.random.uniform(0.75, 15.2, n_patients)
weights = np.random.uniform(4.5, 113.0, n_patients)
heights = np.random.randint(67,175,n_patients)
serum_Creatinine = np.random.uniform(8.0,135.0, n_patients)
albumin_levels = np.random.uniform(3.5, 5.2, n_patients)  # Albumin (g/dL)
total_protein_levels = np.random.uniform(6.0, 8.3, n_patients)  # Total protein (g/dL)
total_bilirubin_levels = np.random.uniform(0.1, 1.2, n_patients)  # Total bilirubin (mg/dL)
direct_bilirubin_levels = np.random.uniform(0.0, 0.3, n_patients)  # Direct bilirubin (mg/dL)
ast_levels = np.random.uniform(10, 40, n_patients)  # AST (U/L)
alt_levels = np.random.uniform(7, 56, n_patients)  # ALT (U/L)

# List of concomitant medications
medications = ["Dasatinib", "Imatinib", "Omeprazole", "Sulphonamides", "NSAIDs", "Penicillin"]

# Store results for all patients
pk_data_4 = []

# Simulate for each patient
for i in range(n_patients):
    age = ages[i]
    weight = weights[i]
    height = heights[i]
    scr = serum_Creatinine[i]
    albumin = albumin_levels[i]
    total_protein = total_protein_levels[i]
    total_bilirubin = total_bilirubin_levels[i]
    direct_bilirubin = direct_bilirubin_levels[i]
    ast = ast_levels[i]
    alt = alt_levels[i]

    # Calculate BSA (m²)
    bsa = np.sqrt((height * weight) / 3600)

    # Randomly assign concomitant medication
    concomitant_medication = np.random.choice(medications)

    # Run the PK simulation
    patient_pk_data_4 = simulate_patient(dose_per_m2_2gm, bsa, CL_mean, V1_mean, Vp1_mean, Vp2_mean, Q1_mean, Q2_mean, scr, CV_CL, CV_V1, age)

    # Append data with patient information
    for time, concentration, infusion_rate in patient_pk_data_4:
        pk_data_4.append([3001+i, age, weight, height, scr, albumin, total_protein, total_bilirubin, direct_bilirubin, ast, alt, bsa, dose_per_m2_2gm, dose_per_m2_2gm * bsa, concomitant_medication, time, concentration, infusion_rate])

# Dosing and patient data
dose_per_m2_1gm = 1000
n_patients = 1000

# Generate random BSA and other data for each patient
ages = np.random.uniform(0.75, 15.2, n_patients)
weights = np.random.uniform(4.5, 113.0, n_patients)
heights = np.random.randint(67,175,n_patients)
serum_Creatinine = np.random.uniform(8.0,135.0, n_patients)
albumin_levels = np.random.uniform(3.5, 5.2, n_patients)  # Albumin (g/dL)
total_protein_levels = np.random.uniform(6.0, 8.3, n_patients)  # Total protein (g/dL)
total_bilirubin_levels = np.random.uniform(0.1, 1.2, n_patients)  # Total bilirubin (mg/dL)
direct_bilirubin_levels = np.random.uniform(0.0, 0.3, n_patients)  # Direct bilirubin (mg/dL)
ast_levels = np.random.uniform(10, 40, n_patients)  # AST (U/L)
alt_levels = np.random.uniform(7, 56, n_patients)  # ALT (U/L)

# List of concomitant medications
medications = ["Dasatinib", "Imatinib", "Omeprazole", "Sulphonamides", "NSAIDs", "Penicillin"]

# Store results for all patients
pk_data_5 = []

# Simulate for each patient
id = 4001+i
for i in range(n_patients):
    age = ages[i]
    weight = weights[i]
    height = heights[i]
    scr = serum_Creatinine[i]
    albumin = albumin_levels[i]
    total_protein = total_protein_levels[i]
    total_bilirubin = total_bilirubin_levels[i]
    direct_bilirubin = direct_bilirubin_levels[i]
    ast = ast_levels[i]
    alt = alt_levels[i]

    # Calculate BSA (m²)
    bsa = np.sqrt((height * weight) / 3600)

    # Randomly assign concomitant medication
    concomitant_medication = np.random.choice(medications)

    # Run the PK simulation
    patient_pk_data_5 = simulate_patient(dose_per_m2_1gm, bsa, CL_mean, V1_mean, Vp1_mean, Vp2_mean, Q1_mean, Q2_mean, scr, CV_CL, CV_V1, age)

    # Append data with patient information
    for time, concentration, infusion_rate in patient_pk_data_5:
        pk_data_5.append([4001+i, age, weight, height, scr, albumin, total_protein, total_bilirubin, direct_bilirubin, ast, alt, bsa, dose_per_m2_1gm, dose_per_m2_1gm * bsa, concomitant_medication, time, concentration, infusion_rate])


# Convert to DataFrame and save as CSV
pk_df1 = pd.DataFrame(pk_data_1, columns=['ID', 'Age', 'Weight', 'Height', 'Scr', 'Albumin', 'Total_Protein', 'Total_Bilirubin', 'Direct_Bilirubin', 'AST', 'ALT', 'BSA', 'Dose_g_m2', 'Total_Dose_mg', 'Concomitant_Medication', 'Time', 'Concentration', 'Infusion_Rate'])
pk_df2 = pd.DataFrame(pk_data_2, columns=['ID', 'Age', 'Weight', 'Height', 'Scr', 'Albumin', 'Total_Protein', 'Total_Bilirubin', 'Direct_Bilirubin', 'AST', 'ALT', 'BSA', 'Dose_g_m2', 'Total_Dose_mg', 'Concomitant_Medication', 'Time', 'Concentration', 'Infusion_Rate'])
pk_df3 = pd.DataFrame(pk_data_3, columns=['ID', 'Age', 'Weight', 'Height', 'Scr', 'Albumin', 'Total_Protein', 'Total_Bilirubin', 'Direct_Bilirubin', 'AST', 'ALT', 'BSA', 'Dose_g_m2', 'Total_Dose_mg', 'Concomitant_Medication', 'Time', 'Concentration', 'Infusion_Rate'])
pk_df4 = pd.DataFrame(pk_data_4, columns=['ID', 'Age', 'Weight', 'Height', 'Scr', 'Albumin', 'Total_Protein', 'Total_Bilirubin', 'Direct_Bilirubin', 'AST', 'ALT', 'BSA', 'Dose_g_m2', 'Total_Dose_mg', 'Concomitant_Medication', 'Time', 'Concentration', 'Infusion_Rate'])
pk_df5 = pd.DataFrame(pk_data_5, columns=['ID', 'Age', 'Weight', 'Height', 'Scr', 'Albumin', 'Total_Protein', 'Total_Bilirubin', 'Direct_Bilirubin', 'AST', 'ALT', 'BSA', 'Dose_g_m2', 'Total_Dose_mg', 'Concomitant_Medication', 'Time', 'Concentration', 'Infusion_Rate'])

pk_df = pd.concat([pk_df1, pk_df2, pk_df3, pk_df4, pk_df5], ignore_index=True) # Corrected line: Use pd.concat and fix the typo

################# Ryan Model ################


# Two Compartmental Model with dosing function
def two_compartment_model_with_dose(y, t, CL, Vc, Vp, Q, BSA, dose_g_per_m2):
    Ac, Ap = y
    dose_rate = dosing_function(t, BSA, dose_g_per_m2)  # Infusion rate at time t
    dAc_dt = dose_rate - (CL / Vc) * Ac - (Q / Vc) * Ac + (Q / Vp) * Ap
    dAp_dt = (Q / Vc) * Ac - (Q / Vp) * Ap
    return [dAc_dt, dAp_dt]

# Allometric scaling based on body weight
def allometric_scaling(WT, theta_CL_std, theta_Vc_std, theta_Vp_std, theta_Q_std, omega_CL, omega_Vc, omega_Vp, omega_Q):
    CL = theta_CL_std * (WT / 70)**0.75 * np.exp(np.random.normal(0, omega_CL))
    Vc = theta_Vc_std * (WT / 70)**1.0 * np.exp(np.random.normal(0, omega_Vc))
    Vp = theta_Vp_std * (WT / 70)**1.0 * np.exp(np.random.normal(0, omega_Vp))
    Q = theta_Q_std * (WT / 70)**0.75 * np.exp(np.random.normal(0, omega_Q))
    return CL, Vc, Vp, Q

# Dosing function
def dosing_function(t, BSA, dose_g_per_m2):
    if t <= 0.33:
        return (200 * dose_g_per_m2 / 4.0) * BSA / 0.33  # Loading dose rate (mg/h)
    elif t <= 24:
        return (3800 * dose_g_per_m2 / 4.0) * BSA / (24 - 0.33)  # Maintenance dose rate (mg/h)
    else:
        return 0  # No dose after 24 hours

# Parameters
theta_CL_std = 11.0
theta_Vc_std = 63.4
theta_Vp_std = 13.6
theta_Q_std = 0.13

# Inter-individual variability (IIV)
omega_CL = 10.7 / 100
omega_Vc = 13.2 / 100
omega_Vp = 0.0
omega_Q = 0.0

# Simulation setup
np.random.seed(42)
WTs = np.random.uniform(4.5, 11.9, 1000)  # Random weights
HTs = np.random.uniform(51, 78, 1000)  # Random heights
Ages = np.random.uniform(2.9, 12.9, 1000) 
time = np.linspace(0, 72, 500)  # Time points over 72 hours
Ac_all_patients = np.zeros((1000, len(time)))  # Store central compartment data

# Initialize data collection for CSV
pk_data_rayan = []

# Simulate PK for each virtual patient
for i in range(1000):
    WT = WTs[i]
    HT = HTs[i]
    Age = Ages[i]

    # Calculate Body Surface Area (BSA)
    BSA = 0.007184 * (WT**0.425) * (HT**0.725)
    
    # Randomized methotrexate dose per body surface area (range: 2.0 – 4.5 g/m²)
    dose_g_per_m2 = np.random.uniform(2.0, 4.5)

    # Calculate total dose based on BSA
    total_dose = dose_g_per_m2 * BSA  # mg
    # Allometric scaling for the patient
    CL, Vc, Vp, Q = allometric_scaling(WT, theta_CL_std, theta_Vc_std, theta_Vp_std, theta_Q_std, omega_CL, omega_Vc, omega_Vp, omega_Q)



    # Initial conditions for drug amounts in compartments
    Ac0 = 0  # Central compartment
    Ap0 = 0  # Peripheral compartment
    y0 = [Ac0, Ap0]

    # Solve the ODEs for the current patient
    solution = odeint(two_compartment_model_with_dose, y0, time, args=(CL, Vc, Vp, Q, BSA, dose_g_per_m2))
    Ac = solution[:, 0]  # Amount in the central compartment

    # Store results in concentration form (mg/L) for the central compartment
    Ac_concentration = Ac / Vc
    Ac_all_patients[i, :] = Ac_concentration

    # Append individual time-point data to pk_data for CSV export
    for j, t in enumerate(time):
        infusion_rate = dosing_function(t, BSA, dose_g_per_m2)  # Get infusion rate at this time
        pk_data_rayan.append([5001+i,Age, WT, HT, BSA, dose_g_per_m2,dose_g_per_m2 * BSA, t, Ac_concentration[j], infusion_rate])

pk_df_rayan = pd.DataFrame(pk_data_rayan, columns=['ID','Age', 'Weight', 'Height', 'BSA', 'Dose_g_m2','Total_Dose_mg', 'Time', 'Concentration', 'Infusion_Rate'])
##################### Combined Dataset ########3
simulated_data = pd.concat([pk_df, pk_df_rayan], ignore_index=True)

################## Our Model #######################
# Define features and target
features = simulated_data[['Weight', 'Height', 'BSA', 'Dose_g_m2', 'Age']]  # Add relevant features
target = simulated_data['Concentration']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# List to store results
results = {}

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
results['Linear Regression'] = {
    'MSE': mean_squared_error(y_test, y_pred_lr),
    'R2 Score': r2_score(y_test, y_pred_lr)
}
st.title("Serum Concentration Predictor")

# Function to calculate BSA
def calculate_bsa(weight, height):
    return 0.007184 * (weight ** 0.425) * (height ** 0.725)

# Function to predict concentration based on user input
def predict_concentration(weight, height, dose):
    bsa = calculate_bsa(weight, height)  # Calculate BSA
    input_data = pd.DataFrame([[weight, height, bsa, dose]], columns=['Weight', 'Height', 'BSA', 'Dose_g_m2'])
    return lr.predict(input_data)[0]
    
    
st.header("Input Patient Data")
height = st.number_input("Enter Height (cm)", min_value=0.0, step=0.1)
weight = st.number_input("Enter Weight (kg)", min_value=0.0, step=0.1)
dose = st.number_input("Enter Dose (mg)", min_value=0.0, step=0.1)

if st.button("Predict"):
    prediction = predict_serum_concentration(height, weight, dose)
    st.subheader("Predicted Serum Concentration")
    st.write(f"{prediction:.2f} units")
"""

# Write the code to a .py file
with open("app.py", "w") as file:
    file.write(code)
