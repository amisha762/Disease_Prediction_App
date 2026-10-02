import streamlit as st
from PIL import Image
import pandas as pd
import pickle
from fpdf import FPDF
from datetime import datetime
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "Training.csv" / "Training.csv"
MODEL_PATH = BASE_DIR / "disease prediction model.pkl"
ENCODER_PATH = BASE_DIR / "encode01r.pkl"

HOME_IMAGE = BASE_DIR / "d.png"
FLU_IMAGE = BASE_DIR / "flu.png"
DIABETES_IMAGE = BASE_DIR / "diabete.png"
COLD_IMAGE = BASE_DIR / "cold.png"
ASTHMA_IMAGE = BASE_DIR / "asthma.png"


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# SIMPLE CSS
# =========================================================

st.markdown(
    """
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #eef8ff,
        #f8fcff,
        #eaf7f4
    );
}

section[data-testid="stSidebar"] {
    background-color: #075866;
}

section[data-testid="stSidebar"] * {
    color: white;
}

.stButton > button {
    border-radius: 10px;
    font-weight: 600;
}

div[data-testid="stCheckbox"] {
    background-color: rgba(255,255,255,0.75);
    border-radius: 8px;
    padding: 5px 8px;
    margin-bottom: 4px;
}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "diagnose_version" not in st.session_state:
    st.session_state.diagnose_version = 0

if "selected_symptoms" not in st.session_state:
    st.session_state.selected_symptoms = []

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None


# =========================================================
# PDF FUNCTION
# =========================================================

def generate_pdf(symptoms, prediction, confidence=None):

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=12)

    pdf.cell(
        200,
        10,
        txt="Disease Prediction Report",
        ln=True,
        align="C"
    )

    pdf.cell(
        200,
        10,
        txt="User: User",
        ln=True
    )

    pdf.cell(
        200,
        10,
        txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ln=True
    )

    symptom_text = ", ".join(
        [
            s.replace("_", " ").title()
            for s in symptoms
        ]
    )

    pdf.multi_cell(
        0,
        10,
        txt=f"Symptoms: {symptom_text}"
    )

    pdf.cell(
        200,
        10,
        txt=f"Prediction: {prediction}",
        ln=True
    )

    if confidence is not None:

        pdf.cell(
            200,
            10,
            txt=f"Confidence: {confidence:.2f}%",
            ln=True
        )

    filename = "prediction_report.pdf"

    pdf.output(filename)

    return filename


# =========================================================
# LOAD DATASET
# =========================================================

try:

    df = pd.read_csv(DATA_PATH)

    symptoms_list = df.columns[:-1].tolist()

except Exception as e:

    st.error(f"❌ Dataset loading failed: {e}")

    symptoms_list = []


# =========================================================
# LOAD MODEL
# =========================================================

try:

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(ENCODER_PATH, "rb") as file:
        encoder = pickle.load(file)

except Exception as e:

    st.error(f"❌ Model loading failed: {e}")

    model = None
    encoder = None


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🩺 Disease Prediction")

st.sidebar.caption(
    "AI Powered Healthcare System"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "🧭 Navigation",
    [
        "🏠 Home",
        "🩺 Diagnose",
        "📚 Disease Info",
        "👨‍⚕️ Doctors",
        "🚀 Future"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Select symptoms and use the trained "
    "machine learning model to get a "
    "possible disease prediction."
)


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    st.title("🩺 Disease Prediction System")

    st.write(
        "Welcome to your AI-powered medical companion."
    )

    st.info(
        "Select symptoms and use the trained "
        "machine learning model to predict "
        "a possible disease."
    )


    st.markdown("---")


    # -----------------------------------------------------
    # WHAT YOU CAN DO + IMAGE
    # -----------------------------------------------------

    col1, col2 = st.columns(
        [1.5, 1],
        gap="large"
    )


    # -----------------------------------------------------
    # LEFT SIDE - WHAT YOU CAN DO
    # -----------------------------------------------------

    with col1:

        st.subheader("✨ What you can do")

        st.write(
            "🩺 **Diagnose** — "
            "Select or search symptoms and predict "
            "a possible disease."
        )

        st.write(
            "📚 **Disease Info** — "
            "Learn about common diseases and "
            "their symptoms."
        )

        st.write(
            "👨‍⚕️ **Doctors** — "
            "Explore healthcare consultation resources."
        )

        st.write(
            "🚀 **Future** — "
            "See possible future improvements."
        )


    # -----------------------------------------------------
    # RIGHT SIDE - IMAGE
    # -----------------------------------------------------

    with col2:

        try:

            image = Image.open(HOME_IMAGE)

            st.image(
                image,
                caption="Stay Healthy 💙",
                use_container_width=True
            )

        except Exception:

            st.warning(
                "Home image not found."
            )


    st.markdown("---")


    # -----------------------------------------------------
    # QUICK START
    # -----------------------------------------------------

    st.success(
        "💡 Go to the Diagnose section "
        "to start prediction."
    )


# =========================================================
# DIAGNOSE
# =========================================================

elif page == "🩺 Diagnose":

    st.title("🧠 Diagnose Your Condition")

    st.write(
        "Search for symptoms or select them "
        "from the list below."
    )

    st.info(
        "🔎 Type in the search box. "
        "You do not need to press Enter."
    )


    # =====================================================
    # SEARCH BOX
    # =====================================================

    st.subheader("🔎 Search Symptom")


    search_key = (
        f"search_{st.session_state.diagnose_version}"
    )


    selected_from_search = st.selectbox(

        "Search or select a symptom",

        options=[""] + symptoms_list,

        format_func=lambda x:
            "🔍 Type a symptom..."
            if x == ""
            else x.replace("_", " ").title(),

        key=search_key

    )


    # =====================================================
    # ADD SEARCHED SYMPTOM
    # =====================================================

    if selected_from_search != "":

        if (
            selected_from_search
            not in st.session_state.selected_symptoms
        ):

            st.session_state.selected_symptoms.append(
                selected_from_search
            )

        # Create a fresh search box
        st.session_state.diagnose_version += 1

        st.rerun()


    # =====================================================
    # SELECTED SYMPTOMS
    # =====================================================

    st.markdown("---")

    st.subheader("✅ Selected Symptoms")


    if st.session_state.selected_symptoms:

        for symptom in st.session_state.selected_symptoms:

            st.write(
                "☑️ "
                + symptom.replace("_", " ").title()
            )

    else:

        st.caption(
            "No symptoms selected yet."
        )


    # =====================================================
    # ALL SYMPTOMS
    # =====================================================

    st.markdown("---")

    st.subheader("📋 All Symptoms")

    st.caption(
        "You can also select symptoms directly "
        "from the three-column list."
    )


    cols = st.columns(3)


    for i, symptom in enumerate(symptoms_list):

        with cols[i % 3]:

            checkbox_key = (
                f"symptom_"
                f"{st.session_state.diagnose_version}_"
                f"{symptom}"
            )


            is_selected = (
                symptom
                in st.session_state.selected_symptoms
            )


            checked = st.checkbox(

                symptom
                .replace("_", " ")
                .title(),

                value=is_selected,

                key=checkbox_key

            )


            # Add symptom
            if checked:

                if (
                    symptom
                    not in st.session_state.selected_symptoms
                ):

                    st.session_state.selected_symptoms.append(
                        symptom
                    )


            # Remove symptom
            else:

                if (
                    symptom
                    in st.session_state.selected_symptoms
                ):

                    st.session_state.selected_symptoms.remove(
                        symptom
                    )


    # =====================================================
    # SELECTED COUNT
    # =====================================================

    st.markdown("---")

    selected_symptoms = (
        st.session_state.selected_symptoms
    )


    if selected_symptoms:

        st.success(
            f"✅ {len(selected_symptoms)} "
            f"symptom(s) selected"
        )

    else:

        st.warning(
            "Please select at least one symptom."
        )


    # =====================================================
    # PREDICT BUTTON
    # =====================================================

    if st.button(
        "🔍 Predict Disease",
        use_container_width=True
    ):

        if not selected_symptoms:

            st.warning(
                "⚠️ Please select at least "
                "one symptom."
            )

        elif model is not None and encoder is not None:

            # ---------------------------------------------
            # CREATE INPUT VECTOR
            # ---------------------------------------------

            input_vector = [

                1
                if symptom in selected_symptoms
                else 0

                for symptom in symptoms_list

            ]


            # ---------------------------------------------
            # PREDICTION
            # ---------------------------------------------

            pred_encoded = model.predict(
                [input_vector]
            )[0]


            prediction = encoder.inverse_transform(
                [pred_encoded]
            )[0]


            # ---------------------------------------------
            # CONFIDENCE
            # ---------------------------------------------

            confidence = None


            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = model.predict_proba(
                    [input_vector]
                )[0]

                confidence = (
                    probabilities.max()
                    * 100
                )


            # ---------------------------------------------
            # SAVE RESULT
            # ---------------------------------------------

            st.session_state.prediction = prediction

            st.session_state.confidence = confidence


            # Force screen refresh
            st.rerun()


    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    if st.session_state.prediction is not None:

        st.markdown("---")

        st.subheader("✅ Prediction Result")

        st.success(
            f"Predicted Disease: "
            f"{st.session_state.prediction}"
        )


        # -------------------------------------------------
        # SYMPTOMS USED
        # -------------------------------------------------

        st.subheader(
            "📋 Symptoms Used for Prediction"
        )


        for symptom in selected_symptoms:

            st.write(
                "• "
                + symptom.replace(
                    "_",
                    " "
                ).title()
            )


        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        if (
            st.session_state.confidence
            is not None
        ):

            confidence = (
                st.session_state.confidence
            )

            st.subheader(
                "📊 Prediction Confidence"
            )

            st.progress(
                min(
                    int(confidence),
                    100
                )
            )

            st.write(
                f"**{confidence:.2f}%**"
            )


        # =================================================
        # BUTTONS
        # =================================================

        col1, col2 = st.columns(2)


        # -------------------------------------------------
        # PDF
        # -------------------------------------------------

        with col1:

            if st.button(
                "📄 Generate PDF Report",
                use_container_width=True
            ):

                pdf_file = generate_pdf(

                    selected_symptoms,

                    st.session_state.prediction,

                    st.session_state.confidence

                )


                with open(
                    pdf_file,
                    "rb"
                ) as f:

                    st.download_button(

                        "⬇️ Download Report",

                        f,

                        file_name=pdf_file,

                        mime="application/pdf",

                        use_container_width=True

                    )


        # -------------------------------------------------
        # PREDICT AGAIN
        # -------------------------------------------------

        with col2:

            if st.button(
                "🔄 Predict Again",
                use_container_width=True
            ):

                # Clear symptoms
                st.session_state.selected_symptoms = []


                # Clear prediction
                st.session_state.prediction = None


                # Clear confidence
                st.session_state.confidence = None


                # New widget keys
                st.session_state.diagnose_version += 1


                st.rerun()


# =========================================================
# DISEASE INFO
# =========================================================

elif page == "📚 Disease Info":

    st.title("📚 Disease Information")

    st.write(
        "Learn about some common diseases, "
        "their symptoms and general treatment information."
    )

    diseases = [

        (
            "🦠 Flu (Influenza)",
            "Fever, cough, chills, fatigue.",
            "Rest, hydration and appropriate medical care.",
            FLU_IMAGE
        ),

        (
            "🍬 Diabetes",
            "Excessive thirst, fatigue and weight changes.",
            "Medical supervision, monitoring and treatment.",
            DIABETES_IMAGE
        ),

        (
            "🤧 Common Cold",
            "Runny nose, sneezing and sore throat.",
            "Rest, fluids and appropriate medication.",
            COLD_IMAGE
        ),

        (
            "🫁 Asthma",
            "Wheezing, shortness of breath and chest tightness.",
            "Medical supervision and prescribed inhalers.",
            ASTHMA_IMAGE
        )

    ]


    for (
        title,
        symptoms,
        treatment,
        image_path
    ) in diseases:

        st.markdown("---")

        col1, col2 = st.columns(
            [1, 3]
        )


        with col1:

            try:

                image = Image.open(
                    image_path
                )

                st.image(
                    image,
                    width=220
                )

            except Exception:

                st.warning(
                    "Image not found."
                )


        with col2:

            st.subheader(title)

            st.write(
                f"**🩺 Symptoms:** {symptoms}"
            )

            st.write(
                f"**💊 General Information:** {treatment}"
            )


# =========================================================
# DOCTORS
# =========================================================

elif page == "👨‍⚕️ Doctors":

    st.title("👨‍⚕️ Doctor Resources")

    st.write(
        "These are external healthcare resources "
        "for consultation and medical services."
    )


    st.markdown("---")

    st.subheader("🌐 Practo")

    st.write(
        "Book appointments and explore "
        "online consultation services."
    )

    st.link_button(
        "Visit Practo",
        "https://www.practo.com"
    )


    st.markdown("---")

    st.subheader("💊 Mfine")

    st.write(
        "Online healthcare consultation services."
    )

    st.link_button(
        "Visit Mfine",
        "https://www.mfine.co"
    )


    st.markdown("---")

    st.subheader("🏥 Apollo 24x7")

    st.write(
        "Healthcare consultation and medical services."
    )

    st.link_button(
        "Visit Apollo 24x7",
        "https://www.apollo247.com"
    )


    st.markdown("---")

    st.subheader("📞 CallHealth")

    st.write(
        "Healthcare consultation and diagnostic services."
    )

    st.link_button(
        "Visit CallHealth",
        "https://www.callhealth.com"
    )


# =========================================================
# FUTURE
# =========================================================

elif page == "🚀 Future":

    st.title("🚀 Future Improvements")

    st.write(
        "Possible future improvements for the system:"
    )


    st.info(
        "🩺 Smart Wearable Sync\n\n"
        "Connect smartwatch or fitness-band data."
    )


    st.info(
        "🗣️ Voice Input\n\n"
        "Allow users to provide symptoms using voice."
    )


    st.info(
        "📈 Health Progress Dashboard\n\n"
        "Track health information over time."
    )


    st.info(
        "📱 Mobile App Integration\n\n"
        "Access the system through a mobile application."
    )


    st.info(
        "🤖 AI Health Assistant\n\n"
        "Provide additional AI-based healthcare assistance."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🩺 Disease Prediction System | "
    "AI-powered disease prediction using Machine Learning"
)

st.caption(
    "⚠️ This application provides a possible prediction "
    "and should not replace professional medical advice."
)