# =====================================================
# XAI RECOMMENDATION ENGINE
# SHAP + LIME DRIVEN
# =====================================================

import numpy as np

# =====================================================
# MAIN ENGINE
# =====================================================

def generate_recommendations(

    user_input,

    shap_data,

    lime_df
):

    recommendations = []

    # =================================================
    # BUILD LIME MAP
    # =================================================

    lime_map = {}

    for _, row in lime_df.iterrows():

        lime_map[
            row["Feature"]
        ] = row["Contribution"]

    # =================================================
    # FEATURE KNOWLEDGE BASE
    # =================================================

    feature_info = {

        "BS": {

            "observation":
                "Blood sugar significantly influenced the prediction.",

            "clinical_risk":
                "Potential gestational diabetes complications.",

            "recommendation":
                "Monitor glucose levels regularly and follow a physician-approved dietary plan."
        },

        "SystolicBP": {

            "observation":
                "Systolic blood pressure contributed to maternal risk.",

            "clinical_risk":
                "Possible hypertensive disorder during pregnancy.",

            "recommendation":
                "Monitor blood pressure frequently and consult an obstetric specialist."
        },

        "DiastolicBP": {

            "observation":
                "Diastolic blood pressure affected the prediction.",

            "clinical_risk":
                "Elevated risk of preeclampsia.",

            "recommendation":
                "Regular maternal monitoring and cardiovascular assessment recommended."
        },

        "BMI": {

            "observation":
                "BMI contributed to the model prediction.",

            "clinical_risk":
                "Higher likelihood of pregnancy-related complications.",

            "recommendation":
                "Follow physician-guided nutrition and physical activity recommendations."
        },

        "BodyTemp": {

            "observation":
                "Body temperature influenced the prediction outcome.",

            "clinical_risk":
                "Potential infection or inflammatory response.",

            "recommendation":
                "Monitor temperature and seek medical evaluation if symptoms persist."
        },

        "HeartRate": {

            "observation":
                "Heart rate contributed to maternal risk assessment.",

            "clinical_risk":
                "Possible cardiovascular strain or stress response.",

            "recommendation":
                "Regular cardiovascular monitoring and hydration assessment advised."
        },

        "Age": {

            "observation":
                "Maternal age influenced prediction confidence.",

            "clinical_risk":
                "Advanced maternal age may increase pregnancy risks.",

            "recommendation":
                "Maintain regular prenatal checkups and follow specialist guidance."
        },

        "PreviousComplications": {

            "observation":
                "History of previous complications influenced the prediction.",

            "clinical_risk":
                "Increased probability of recurring maternal complications.",

            "recommendation":
                "Enhanced antenatal monitoring is recommended."
        },

        "PreexistingDiabetes": {

            "observation":
                "Preexisting diabetes contributed to risk estimation.",

            "clinical_risk":
                "Higher risk of maternal and fetal complications.",

            "recommendation":
                "Strict glucose monitoring and specialist consultation advised."
        }
    }

    # =================================================
    # PROCESS FEATURES
    # =================================================

    for i, feature in enumerate(
        user_input.keys()
    ):

        if feature not in feature_info:

            continue

        # =============================================
        # SHAP VALUE
        # =============================================

        try:

            shap_value = float(
                shap_data[i]
            )

        except:

            shap_value = 0.0

        # =============================================
        # LIME VALUE
        # =============================================

        lime_value = 0.0

        for lime_feature in lime_map:

            if feature.lower() in lime_feature.lower():

                try:

                    lime_value = float(
                        lime_map[lime_feature]
                    )

                except:

                    lime_value = 0.0

                break

        # =============================================
        # COMBINED XAI SCORE
        # =============================================

        combined_score = (

            abs(shap_value) * 0.7 +

            abs(lime_value) * 0.3

        )

        # =============================================
        # SKIP TINY CONTRIBUTIONS
        # =============================================

        if combined_score < 0.02:

            continue

        # =============================================
        # SEVERITY
        # =============================================

        if combined_score >= 0.25:

            severity = "Critical"

        elif combined_score >= 0.15:

            severity = "High"

        elif combined_score >= 0.05:

            severity = "Moderate"

        else:

            severity = "Low"

        # =============================================
        # BUILD RECOMMENDATION
        # =============================================

        recommendations.append({

            "feature":
                feature,

            "value":
                user_input[feature],

            "severity":
                severity,

            "shap_value":
                round(shap_value, 4),

            "lime_value":
                round(lime_value, 4),

            "combined_score":
                round(combined_score, 4),

            "observation":
                feature_info[feature]["observation"],

            "clinical_risk":
                feature_info[feature]["clinical_risk"],

            "recommendation":
                feature_info[feature]["recommendation"]
        })

    # =================================================
    # FALLBACK
    # =================================================

    if len(recommendations) == 0:

        recommendations.append({

            "feature":
                "General Health",

            "value":
                "-",

            "severity":
                "Low",

            "shap_value":
                0,

            "lime_value":
                0,

            "combined_score":
                0,

            "observation":
                "No major risk-driving features detected.",

            "clinical_risk":
                "Overall maternal indicators appear stable.",

            "recommendation":
                "Continue routine antenatal care, healthy nutrition, hydration and physician follow-up."
        })

    # =================================================
    # SORT BY IMPORTANCE
    # =================================================

    recommendations = sorted(

        recommendations,

        key=lambda x:
            x["combined_score"],

        reverse=True
    )

    return recommendations