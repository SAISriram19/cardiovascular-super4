import streamlit as st
import numpy as np
import pandas as pd
import cv2
import pickle
import os
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import resample

# Medical information database
MEDICAL_INFO = {
    "Myocardial Infarction": {
        "Description": "A heart attack occurs when the flow of blood to the heart is severely reduced or blocked. The blockage is usually due to a buildup of fat, cholesterol and other substances in the heart (coronary) arteries.",
        "Symptoms": ["Chest pain or discomfort", "Shortness of breath", "Pain or discomfort in the jaw, neck, back, arm, or shoulder", "Nausea", "Lightheadedness", "Cold sweat"],
        "Medications": ["Aspirin", "Thrombolytics", "Anti-platelet agents", "Beta blockers", "ACE inhibitors", "Statins"],
        "Management": ["Immediate medical attention", "Cardiac rehabilitation", "Lifestyle changes (diet, exercise, smoking cessation)", "Regular follow-up with cardiologist"]
    },
    "MI": {
        "Description": "Myocardial Infarction (MI), commonly known as a heart attack, is caused by reduced blood flow to a part of the heart, causing heart muscle damage.",
        "Symptoms": ["Chest pain or pressure", "Pain spreading to shoulder, arm, back, neck or jaw", "Nausea", "Heartburn", "Fatigue", "Shortness of breath"],
        "Medications": ["Aspirin", "Clopidogrel", "Beta-blockers", "ACE inhibitors", "Statin therapy", "Nitroglycerin"],
        "Management": ["Emergency treatment required", "Coronary angioplasty or bypass surgery if needed", "Cardiac rehabilitation", "Lifestyle modifications"]
    },
    "Abnormal": {
        "Description": "An abnormal ECG reading can indicate various heart conditions including arrhythmias, heart disease, or other cardiac abnormalities that may require further investigation.",
        "Symptoms": ["May be asymptomatic", "Palpitations", "Dizziness", "Fainting", "Shortness of breath", "Chest discomfort"],
        "Medications": ["Varies based on specific condition", "Anti-arrhythmics", "Beta-blockers", "Calcium channel blockers", "Blood thinners if needed"],
        "Management": ["Further diagnostic testing required", "Consultation with cardiologist", "Regular monitoring", "Lifestyle modifications"]
    },
    "Normal": {
        "Description": "A normal ECG indicates that the electrical activity of the heart appears within normal limits. No signs of significant cardiac abnormalities were detected.",
        "Symptoms": ["No concerning symptoms"],
        "Medications": ["None required"],
        "Management": ["Maintain heart-healthy lifestyle", "Regular check-ups", "Routine screening as recommended"]
    },
    "History of MI": {
        "Description": "This indicates a previous myocardial infarction (heart attack). Patients with a history of MI are at increased risk for future cardiac events and require careful monitoring and management.",
        "Symptoms": ["May be asymptomatic", "Angina (chest pain)", "Shortness of breath", "Fatigue"],
        "Medications": ["Beta-blockers", "ACE inhibitors", "Statins", "Aspirin", "P2Y12 inhibitors"],
        "Management": ["Regular cardiac monitoring", "Cardiac rehabilitation", "Lifestyle modifications", "Control of risk factors (hypertension, diabetes, cholesterol)"]
    },
    "Atrial Premature": {
        "Description": "Premature atrial contractions (PACs) are extra heartbeats that begin in one of the heart's two upper chambers (atria). These extra beats disrupt the regular heart rhythm.",
        "Symptoms": ["Palpitations", "Skipped heartbeat", "Fluttering in chest", "Occasional dizziness"],
        "Medications": ["Beta-blockers", "Calcium channel blockers", "Anti-arrhythmics if severe"],
        "Management": ["Reduce caffeine and alcohol intake", "Manage stress", "Adequate sleep", "Regular exercise"]
    },
    "PVC": {
        "Description": "Premature Ventricular Contractions (PVCs) are extra heartbeats that begin in one of the heart's two lower pumping chambers (ventricles). These extra beats disrupt the regular heart rhythm.",
        "Symptoms": ["Palpitations", "Skipped heartbeat", "Fluttering in chest", "Dizziness", "Shortness of breath"],
        "Medications": ["Beta-blockers", "Calcium channel blockers", "Anti-arrhythmic drugs if severe"],
        "Management": ["Reduce caffeine and alcohol", "Quit smoking", "Manage stress", "Treat underlying conditions"]
    },
    "ST-T Change": {
        "Description": "ST-T wave changes on an ECG can indicate various conditions including ischemia, electrolyte imbalances, or other cardiac abnormalities. Further evaluation is needed to determine the exact cause.",
        "Symptoms": ["May be asymptomatic", "Chest pain", "Shortness of breath", "Palpitations", "Fatigue"],
        "Medications": ["Depends on underlying cause", "May include anti-anginal medications", "Beta-blockers", "Calcium channel blockers"],
        "Management": ["Cardiology consultation", "Further diagnostic testing", "Risk factor modification", "Regular follow-up"]
    },
    "Conduction": {
        "Description": "Conduction abnormalities refer to problems with the electrical system that controls the heartbeat. This can include various types of heart block or bundle branch blocks.",
        "Symptoms": ["May be asymptomatic", "Dizziness", "Fainting", "Fatigue", "Shortness of breath", "Palpitations"],
        "Medications": ["Depends on specific condition", "May include atropine", "Isoproterenol", "Temporary pacing if severe"],
        "Management": ["Cardiology evaluation", "Possible pacemaker for severe cases", "Regular monitoring", "Treat underlying conditions"]
    },
    "Hypertrophy": {
        "Description": "Cardiac hypertrophy refers to the thickening of the heart muscle, often in response to high blood pressure or heart valve disease. The heart works harder to pump blood, causing the muscle to thicken.",
        "Symptoms": ["Shortness of breath", "Chest pain", "Fatigue", "Dizziness", "Fainting", "Palpitations"],
        "Medications": ["ACE inhibitors", "ARBs", "Beta-blockers", "Calcium channel blockers", "Diuretics"],
        "Management": ["Blood pressure control", "Salt restriction", "Regular exercise", "Weight management", "Regular cardiac monitoring"]
    }
}

def display_medical_info(condition):
    """Display medical information for the predicted condition"""
    if condition not in MEDICAL_INFO:
        st.warning(f"No medical information available for: {condition}")
        return
    
    info = MEDICAL_INFO[condition]
    
    st.subheader(f"Medical Information: {condition}")
    
    # Create a table with the medical information
    st.markdown("""
    <style>
    .medical-table {
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
        font-size: 0.9em;
        border-radius: 5px 5px 0 0;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
    }
    .medical-table thead tr {
        background-color: #009879;
        color: #ffffff;
        text-align: left;
        font-weight: bold;
    }
    .medical-table th,
    .medical-table td {
        padding: 12px 15px;
        border: 1px solid #dddddd;
    }
    .medical-table tbody tr {
        border-bottom: 1px solid #dddddd;
    }
    .medical-table tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    .medical-table tbody tr:last-of-type {
        border-bottom: 2px solid #009879;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Description
    st.markdown("#### Description")
    st.write(info["Description"])
    
    # Symptoms
    st.markdown("#### Symptoms")
    symptoms_html = "<ul>"
    for symptom in info["Symptoms"]:
        symptoms_html += f"<li>{symptom}</li>"
    symptoms_html += "</ul>"
    st.markdown(symptoms_html, unsafe_allow_html=True)
    
    # Medications and Management in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Medications")
        meds_html = "<ul>"
        for med in info["Medications"]:
            meds_html += f"<li>{med}</li>"
        meds_html += "</ul>"
        st.markdown(meds_html, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Management")
        mgmt_html = "<ul>"
        for item in info["Management"]:
            mgmt_html += f"<li>{item}</li>"
        mgmt_html += "</ul>"
        st.markdown(mgmt_html, unsafe_allow_html=True)
    
    # Disclaimer
    st.warning("⚠️ This information is for educational purposes only and not a substitute for professional medical advice. Always consult with a healthcare provider for diagnosis and treatment.")
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from skimage.io import imread
from skimage import color, measure
from skimage.filters import threshold_otsu, gaussian
from skimage.transform import resize
from natsort import natsorted

st.set_page_config(page_title="ECG Analysis System", layout="wide", page_icon="❤️")
st.title("12-Lead ECG Analysis System")

# Custom CSS for better spacing
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin: 5px 0;
    }
    .stExpander .streamlit-expanderHeader {
        font-size: 1.2rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

class ECG:
    def getImage(self, image):
        image = imread(image)
        return image

    def DividingLeads(self, image):
        """
        This Funciton Divides the Ecg image into 12 Leads.
        Bipolar limb leads(Leads1,2,3). Augmented unipolar limb leads(aVR,aVF,aVL).
        Unipolar (+) chest leads(V1,V2,V3,V4,V5,V6)
        return : List containing all 12 leads divided
        """
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Define lead regions with validation
        leads = []
        regions = [
            (300, 600, 150, 643),  # Lead 1
            (300, 600, 646, 1135), # Lead aVR
            (300, 600, 1140, 1625),# Lead V1
            (300, 600, 1630, 2125),# Lead V4
            (600, 900, 150, 643),  # Lead 2
            (600, 900, 646, 1135), # Lead aVL
            (600, 900, 1140, 1625),# Lead V2
            (600, 900, 1630, 2125),# Lead V5
            (900, 1200, 150, 643), # Lead 3
            (900, 1200, 646, 1135),# Lead aVF
            (900, 1200, 1140, 1625),# Lead V3
            (900, 1200, 1630, 2125) # Lead V6
        ]
        
        for y1, y2, x1, x2 in regions:
            # Validate coordinates
            y1 = max(0, min(y1, height))
            y2 = max(0, min(y2, height))
            x1 = max(0, min(x1, width))
            x2 = max(0, min(x2, width))
            
            # Extract lead region
            lead = image[y1:y2, x1:x2]
            if lead.size > 0:  # Only add if we got a valid region
                leads.append(lead)
            else:
                leads.append(np.zeros((300, 475)))  # Add empty region if invalid
        
        return leads

    def preprocess_lead(self, lead_img):
        """Preprocess a single lead image"""
        try:
            # Convert to grayscale and validate
            grayscale = color.rgb2gray(lead_img)
            if np.isnan(grayscale).any() or np.isinf(grayscale).any():
                return None
                
            # Apply Gaussian blur
            blurred_image = gaussian(grayscale, sigma=1.0)
            
            # Get threshold and create binary image
            if np.max(blurred_image) > 0:
                global_thresh = threshold_otsu(blurred_image)
                binary_global = blurred_image < global_thresh
            else:
                return None
                
            # Resize to standard dimensions
            binary_global = resize(binary_global, (300, 450))
            return binary_global
            
        except Exception as e:
            st.error(f"Error in preprocessing: {str(e)}")
            return None

    def extract_signal(self, binary_lead):
        """Extract 1D signal from preprocessed binary lead"""
        try:
            # Find contours
            contours = measure.find_contours(binary_lead, 0.8)
            
            # If no contours found, return zeros
            if not contours:
                return None
                
            # Get the largest contour (most likely the signal)
            contours_shape = sorted([(i, x.shape[0]) for i, x in enumerate(contours)], 
                                 key=lambda x: x[1], reverse=True)
            
            if not contours_shape:
                return None
                
            # Get the largest contour
            largest_contour = contours[contours_shape[0][0]]
            
            # Resize to 33 points (66 features: x,y coordinates)
            test = resize(largest_contour, (33, 2))
            
            # Scale the data
            scaler = MinMaxScaler()
            fit_transform_data = scaler.fit_transform(test)
            
            # Return both X and Y coordinates
            return fit_transform_data.flatten()  # Flatten to 1D array
            
        except Exception as e:
            st.error(f"Error in signal extraction: {str(e)}")
            return None

    def process_all_leads(self, leads):
        """Process all 12 leads and return features"""
        all_features = []
        
        for i, lead in enumerate(leads):
            # Preprocess lead
            binary_lead = self.preprocess_lead(lead)
            if binary_lead is None:
                st.warning(f"Could not process Lead {i+1}, using zeros")
                all_features.append(np.zeros(66))  # 33 points * 2 (x,y)
                continue
                
            # Extract signal
            features = self.extract_signal(binary_lead)
            if features is None or len(features) != 66:
                st.warning(f"Could not extract signal from Lead {i+1}, using zeros")
                all_features.append(np.zeros(66))
            else:
                all_features.append(features)
        
        # Combine all features (12 leads * 66 features = 792 features)
        combined = np.concatenate(all_features)
        
        # Ensure we have exactly 401 features (as expected by the model)
        if len(combined) > 401:
            combined = combined[:401]
        elif len(combined) < 401:
            # Pad with zeros if needed
            combined = np.pad(combined, (0, 401 - len(combined)), 'constant')
            
        return combined.reshape(1, -1)  # Return as 2D array for the model

    def predict_model2(self, signal, model_path, scaler_path, labels):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Scale the signal
        scaled = scaler.transform(signal.reshape(1, -1))
        
        # Load and predict with the model
        model = tf.keras.models.load_model(model_path)
        
        # Reshape for CNN input if needed
        if len(model.input_shape) == 3:  # CNN expects 3D input
            scaled = scaled.reshape(1, 187, 1)
            
        pred = model.predict(scaled)[0]
        return dict(zip(labels, pred))

# Sidebar for file upload
with st.sidebar:
    st.header("Upload ECG")
    uploaded_file = st.file_uploader("Choose a 12-lead ECG image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    ecg = ECG()
    image = ecg.getImage(uploaded_file)
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(image, caption="Uploaded ECG", use_column_width=True)
    
    with col2:
        st.subheader("ECG Analysis Results")
        with st.spinner('Analyzing ECG...'):
            # Get all leads
            leads = ecg.DividingLeads(image)
            
            # Process all leads and get combined features
            combined_features = ecg.process_all_leads(leads)
            
            # Clean up any existing CSV files
            import os
            if os.path.exists('combined_features.csv'):
                os.remove('combined_features.csv')
            
            # Save the combined features for model 1
            np.savetxt('combined_features.csv', combined_features, delimiter=',')
            
            # Load PCA and Model 1
            pca = joblib.load("C:/Users/saisr/Downloads/Cardiovascular-Detection-using-ECG-images-main/testing/models/PCA_ECG.pkl")
            model1 = joblib.load("C:/Users/saisr/Downloads/Cardiovascular-Detection-using-ECG-images-main/testing/models/Heart_Disease_Prediction.pkl")
            
            # Apply PCA and predict with Model 1
            reduced_features = pca.transform(combined_features)
            
            # Get prediction with probabilities if available
            if hasattr(model1, "predict_proba"):
                proba = model1.predict_proba(reduced_features)[0]
                if len(proba) == 4:
                    labels = ["MI", "Abnormal", "Normal", "History of MI"]
                elif len(proba) == 5:
                    labels = ["Myocardial Infarction", "Abnormal", "Normal", "History of MI", "Other"]
                else:
                    labels = [f"Class {i}" for i in range(len(proba))]
                model1_result = dict(zip(labels, proba))
            else:
                label_idx = model1.predict(reduced_features)[0]
                if label_idx < 4:
                    labels = ["MI", "Abnormal", "Normal", "History of MI"]
                    model1_result = {labels[label_idx]: 1.0}
                else:
                    model1_result = {f"Class {label_idx}": 1.0}
            
            # Get top prediction
            top_pred = max(model1_result.items(), key=lambda x: x[1])
            
            # Display top prediction with color coding
            if "Normal" in top_pred[0]:
                st.success(f"✅ {top_pred[0]}: {top_pred[1]:.2%} confidence")
            else:
                st.error(f"⚠️ {top_pred[0]}: {top_pred[1]:.2%} confidence")
            
            # Display medical information for the top prediction
            display_medical_info(top_pred[0])
            
            # Confidence scores button
            if st.button("View All Confidence Scores"):
                st.dataframe(
                    pd.DataFrame(model1_result.items(), columns=["Condition", "Confidence"])
                    .sort_values("Confidence", ascending=False)
                    .style.format({"Confidence": "{:.2%}"})
                )
    
    # Preprocessing steps expander
    with st.expander("View Preprocessing Steps"):
        st.subheader("ECG Preprocessing")
        
        # Show grayscale conversion
        st.write("### 1. Grayscale Conversion")
        gray_img = color.rgb2gray(image)
        st.image(gray_img, caption="Grayscale ECG", use_column_width=True)
        
        # Show lead division
        st.write("### 2. Lead Division")
        st.write("The ECG is divided into 12 standard leads:")
        
        # Create a grid for lead images
        fig, axs = plt.subplots(4, 3, figsize=(15, 15))
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        
        for i, (lead, name) in enumerate(zip(leads, lead_names)):
            row = i // 3
            col = i % 3
            axs[row, col].imshow(lead)
            axs[row, col].set_title(f"Lead {name}")
            axs[row, col].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show signal extraction for Lead II as an example
        st.write("### 3. Signal Extraction (Lead II Example)")
        lead2_img = leads[4]  # Lead II
        binary_lead = ecg.preprocess_lead(lead2_img)
        
        if binary_lead is not None:
            # Show binary image
            st.write("#### Binary Image")
            st.image(binary_lead, caption="Binary Lead II", use_column_width=True)
            
            # Show extracted signal
            signal = ecg.extract_signal(binary_lead)
            if signal is not None:
                st.write("#### Extracted Signal")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(signal[:187])  # Show first 187 points
                ax.set_title("Extracted ECG Signal (Lead II)")
                ax.set_xlabel("Samples")
                ax.set_ylabel("Amplitude")
                st.pyplot(fig)
    
    # Model details expander
    with st.expander("View Detailed Analysis"):
        st.subheader("Detailed Model Analysis")
        
        # MIT-BIH Model
        st.write("### MIT-BIH Arrhythmia Analysis (Lead II)")
        lead2_img = leads[4]  # Lead II
        binary_lead = ecg.preprocess_lead(lead2_img)
        
        if binary_lead is not None:
            mit_signal = ecg.extract_signal(binary_lead)
            if mit_signal is not None:
                mit_signal_resampled = resample(mit_signal, 187)
                
                # Show signal
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(mit_signal_resampled)
                ax.set_title("MIT-BIH Input Signal (Lead II)")
                st.pyplot(fig)
                
                # Show predictions
                mit_result = ecg.predict_model2(
                    mit_signal_resampled, 
                    "mitbih_arrhythmia_model.keras", 
                    "mitbih_arrhythmia_scaler.pkl",
                    ["Normal", "Atrial Premature", "PVC", "Fusion Ventricular", "Fusion Paced"]
                )
                st.dataframe(
                    pd.DataFrame(mit_result.items(), columns=["Rhythm", "Confidence"])
                    .sort_values("Confidence", ascending=False)
                    .style.format({"Confidence": "{:.2%}"})
                )
        
        # PTB-XL Model
        st.write("### PTB-XL Analysis (Leads V1 & V5)")
        
        # Lead V1
        st.write("#### Lead V1")
        v1_binary = ecg.preprocess_lead(leads[2])
        if v1_binary is not None:
            v1_signal = ecg.extract_signal(v1_binary)
            if v1_signal is not None:
                v1_signal_resampled = resample(v1_signal, 187)
                
                # Show signal
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.plot(v1_signal_resampled)
                ax.set_title("PTB-XL Input Signal (Lead V1)")
                st.pyplot(fig)
                
                # Show predictions
                v1_result = ecg.predict_model2(
                    v1_signal_resampled, 
                    "ptb_xl_model.keras", 
                    "ptb_xl_scaler.pkl",
                    ["Normal", "MI", "ST-T Change", "Conduction", "Hypertrophy", "Other"]
                )
                st.dataframe(
                    pd.DataFrame(v1_result.items(), columns=["Condition", "Confidence"])
                    .sort_values("Confidence", ascending=False)
                    .style.format({"Confidence": "{:.2%}"})
                )
        
        # Lead V5
        st.write("#### Lead V5")
        v5_binary = ecg.preprocess_lead(leads[7])
        if v5_binary is not None:
            v5_signal = ecg.extract_signal(v5_binary)
            if v5_signal is not None:
                v5_signal_resampled = resample(v5_signal, 187)
                
                # Show signal
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.plot(v5_signal_resampled)
                ax.set_title("PTB-XL Input Signal (Lead V5)")
                st.pyplot(fig)
                
                # Show predictions
                v5_result = ecg.predict_model2(
                    v5_signal_resampled,
                    "ptb_xl_model.keras",
                    "ptb_xl_scaler.pkl",
                    ["Normal", "MI", "ST-T Change", "Conduction", "Hypertrophy", "Other"]
                )
                st.dataframe(
                    pd.DataFrame(v5_result.items(), columns=["Condition", "Confidence"])
                    .sort_values("Confidence", ascending=False)
                    .style.format({"Confidence": "{:.2%}"})
                )
    
    # Add some space at the bottom
    st.markdown("---")
    st.caption("ECG Analysis System v1.0 | For research use only")

