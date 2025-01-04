import streamlit as st # type: ignore
from striprtf.striprtf import rtf_to_text # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore
import os


# Define the folder containing the dataframes
DATA_FOLDER = "Spectra/"
DISPLAY_FOLDER = "Meta_Data/"
apptitle = "SQuAD Visual Inspector"
st.set_page_config(page_title=apptitle, layout='wide',initial_sidebar_state='collapsed')
padding_top = 0
_,col2,_ = st.columns([3,6,2.25])
col2.title("SQuAD Anomaly Visual Inspector")

#####################################################################################
# READ THE VANDEN BERK COMPOSITE AND DEFINE WAVELENGTH WINDOW
#####################################################################################
wavelength = np.arange(1300,3001,2)
with open('Vanden_Berk.rtf', 'r') as file:
    rtf_content = file.read()

# Convert RTF to plain text
text_content = rtf_to_text(rtf_content)

# Initialize arrays
array1 = []
array2 = []

# Process the text
for line in text_content.splitlines():
    # Split by whitespace and convert to floats
    numbers = line.split()
    if len(numbers) >= 2:
        array1.append(float(numbers[0]))
        array2.append(float(numbers[1]))

# Convert lists to numpy arrays
VB_Wavelength = np.array(array1)[500:2200]
VB_Flux = np.array(array2)[500:2200]/max(np.array(array2)[500:2200])

######################################################################################
# ADD SIDEBAR TO SELECT THE ANOMALY DATASET
######################################################################################


selected_group = st.sidebar.selectbox(
    'Select the Anomaly Group',
    ('C IV Peakers', 'Excess Si IV', 'Si IV Deficient',
     'Blue BALs', 'Flat BALs', 'Red BALs', 'FeLoBALs',
     'Moderately Reddened', 'Heavily Reddened', 'Plateau-Shaped')
)

# Load the list of CSV files in the folder
data_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]

selected_file = {'C IV Peakers':'CIV.csv','Excess Si IV':'SiIV_Excess.csv',
                 'Si IV Deficient':'SiIV_Deficient.csv','Blue BALs':'Blue_BALs.csv',
                 'Flat BALs':'Flat_BALs.csv','Red BALs':'Red_BALs.csv',
                 'FeLoBALs':'FeLoBALs.csv','Moderately Reddened':'Moderately_Red.csv',
                 'Heavily Reddened':'Heavy_Reddened.csv','Plateau-Shaped':'Humped.csv'}

df = pd.read_csv(os.path.join(DATA_FOLDER, selected_file[selected_group]))
df_display = pd.read_csv(os.path.join(DISPLAY_FOLDER, selected_file[selected_group]))

######################################################################################
# QUASAR LINE DATA BOX
######################################################################################

st.title("Quasar Line Data")
with st.expander('View Metadata',expanded=True):
    with st.container():
        col1, col2 = st.columns([3,1.3])
        with col1:
            st.dataframe(df_display)
            #st.dataframe(df)

        with col2:
        # Allow user to select a row for plotting
            row_index = st.number_input("Select a row to plot", min_value=0, max_value=len(df)-1, step=1)
            st.subheader('Wu-Shen Catalog Data')
            with st.container():
                col11, col12 = st.columns([1,1])
                with col11:
                    st.metric('S iIV Equivalent Width',np.round(df.iloc[row_index].SiIV,decimals=2))
                    st.metric('C IV Equivalent Width',np.round(df.iloc[row_index].CIV,decimals=2))
                    st.metric('He II Equivalent Width',np.round(df.iloc[row_index].HeII,decimals=2))
                with col12:
                    st.metric('C III Equivalent Width',np.round(df.iloc[row_index].CIII,decimals=2))
                    st.metric('Mg II Equivalent Width',np.round(df.iloc[row_index].MgII,decimals=2))
                    st.metric('BAL PROBABILITY',np.round(df.iloc[row_index].BAL_PROB,decimals=1))

######################################################################################
# QUASAR SPECTRUM BOX
######################################################################################

############################# PLOTTING THE SPECTRUM ##################################
st.title('Quasar Spectrum')
with st.container():
    col1, col2 = st.columns([1,1])
    with col1:
        vb_scale = st.slider('Scale Vanden Berk Composite:',min_value=0.0,max_value=5.0,
                                value=1.0,step=0.01)
        row_data = df.iloc[row_index]
        wavelength = np.arange(1300,3001,2)
        fig = go.Figure()
        # Plot the selected row spectrum
        fig.add_trace(go.Scatter(x=wavelength, y=df.iloc[row_index][4:855], mode='lines', name=f"{selected_group} : {row_index+1}"))
        # Plot the Vanden Berk
        fig.add_trace(go.Scatter(x=VB_Wavelength, y=VB_Flux*vb_scale, mode='lines', name="Vanden Berk"))

        st.plotly_chart(fig,use_container_width=True)

#################### PLOTTING PCA POINTER ###########################################
    with col2:

        st.subheader("Cluster Visualization")
        #plot_BAL_clusters(pca_BAL, model_BAL)
        df_Coords = pd.read_csv('/Users/arihanttiwari/Documents/VI_App/PCA_Coords.csv')

        import pickle

        # Load variables (adjust paths if necessary)
        with open('variables.pkl', 'rb') as f:
            data = pickle.load(f)

        temp_BAL = data['temp_BAL']
        pca_BAL = data['pca_BAL']
        model_BAL = data['model_BAL']

        
        # Get the PLATE, MJD, FIBERID for the selected row
        selected_row = df.iloc[row_index]
        plate, mjd, fiberid = selected_row['PLATE'], selected_row['MJD'], selected_row['FIBERID']

        # Find the corresponding X, Y in df_Coords
        highlight_point = df_Coords[(df_Coords['PLATE'] == plate) & 
                                    (df_Coords['MJD'] == mjd) & 
                                    (df_Coords['FIBERID'] == fiberid)]

        # Prepare data for PCA clusters
        x = np.array(pca_BAL.iloc[:, 0])
        y = np.array(pca_BAL.iloc[:, 1] * -1)  # Invert PCA 2 axis for consistency
        group = temp_BAL['Cluster']
        cluster_centers = model_BAL.cluster_centers_

        # Define color dictionary for clusters
        cdict = {2: 'gray', 0: 'goldenrod', 1: 'teal', 4: 'white'}

        # Create an interactive scatter plot
        fig = go.Figure()

        # Plot each cluster
        for g in np.unique(group):
            ix = np.where(group == g)
            if g != 4:  # Regular clusters
                fig.add_trace(go.Scatter(
                    x=x[ix],
                    y=y[ix],
                    mode='markers',
                    marker=dict(size=5, color=cdict[g], opacity=0.5),
                    name=f'Cluster {g + 1}'
                ))
            else:  # Anomalies
                fig.add_trace(go.Scatter(
                    x=x[ix],
                    y=y[ix],
                    mode='markers',
                    marker=dict(size=5, color=cdict[g], opacity = 0.8),
                    name='Anomaly'
                ))

        # Plot the cluster centroids
        #fig.add_trace(go.Scatter(
        #    x=cluster_centers[:, 0],
        #    y=cluster_centers[:, 1] * -1,  # Invert PCA 2 axis
        #    mode='markers',
        #    marker=dict(size=10, color='red', symbol='x'),
        #    name='Cluster Centroid'
        #))

        # Highlight the selected point
        if not highlight_point.empty:
            x_star = highlight_point['X'].values[0]
            y_star = highlight_point['Y'].values[0]*-1
            fig.add_trace(go.Scatter(
                x=[x_star],
                y=[y_star],
                mode='markers',
                marker=dict(size=20, color='red', symbol='star'),
                name='Selected Point'
            ))

        # Update layout for titles, axis labels, and background
        fig.update_layout(
            xaxis_title='PCA 1 Coefficient',
            yaxis_title='PCA 2 Coefficient',
            legend=dict(font=dict(size=12)),
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(family='Times New Roman', color='white', size=14)
        )

        # Display the plot
        st.plotly_chart(fig)


######################################################################################
# QUASAR HISTOGRAM BOX
######################################################################################

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
tot = pd.read_csv('UV_Line_Data.csv')
abk = df_display[['PLATE','MJD','FIBERID']]
abk = abk.merge(tot,on=['PLATE','MJD','FIBERID'],how='inner')

def Clip_Data(data, cut=99, flag=0):
    # Drop all non-numerical values such as -inf, inf, and nan, and also 0
    data = data[~np.isnan(data)]  # Remove NaN values
    data = data[np.isfinite(data)]  # Remove -inf and inf values
    data = data[data != 0]  # Remove 0 values

    # Take the absolute value of the remaining data
    data = abs(data)

    # Set threshold to the given percentile
    threshold = np.percentile(data, cut)
    
    # Clip values above the threshold
    if flag == 0:
        clipped_data = data[data < threshold]
    else:
        clipped_data = data
    
    return clipped_data

# Streamlit App Interface
st.title('Line Data Histograms')

with st.container():
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        st.subheader("C IV EW")
        fig1, ax1 = plt.subplots()
        sns.histplot(Clip_Data(tot['CIV_EW'],98), bins=20, kde=True, ax=ax1, color='blue')
        plt.axvline(abk.iloc[row_index]['CIV_EW'],lw=5,ls='--')
        ax1.set_title('CIV_EW Histogram')
        ax1.set_xlabel('CIV_EW')
        ax1.set_ylabel('Frequency')
        st.pyplot(fig1)

    with col2:
        st.subheader("SIIV_EW")
        fig2, ax2 = plt.subplots()
        sns.histplot(Clip_Data(tot['SiIV_EW'],98), bins=20, kde=True, ax=ax2, color='green')
        plt.axvline(abk.iloc[row_index]['SiIV_EW'],lw=5,ls='--')
        ax2.set_title('SIIV_EW Histogram')
        ax2.set_xlabel('SIIV_EW')
        ax2.set_ylabel('Frequency')
        st.pyplot(fig2)
    
    with col3:
        st.subheader("C III EW")
        fig1, ax1 = plt.subplots()
        sns.histplot(Clip_Data(tot['CIII_EW'],98), bins=20, kde=True, ax=ax1, color='orange')
        plt.axvline(abk.iloc[row_index]['CIII_EW'],lw=5,ls='--')
        ax1.set_title('CIII EW Histogram')
        ax1.set_xlabel('CIII EW')
        ax1.set_ylabel('Frequency')
        st.pyplot(fig1)

    with col4:
        st.subheader("Mg II EW")
        fig3, ax3 = plt.subplots()
        sns.histplot(Clip_Data(tot['MgII_EW'],98), bins=20, kde=True, ax=ax3, color='red')
        plt.axvline(abk.iloc[row_index]['MgII_EW'],lw=5,ls='--')
        ax3.set_title('MgII EW Histogram')
        ax3.set_xlabel('MgII EW')
        ax3.set_ylabel('Frequency')
        st.pyplot(fig3)



