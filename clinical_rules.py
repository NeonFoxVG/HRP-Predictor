# =====================================================
# ADVANCED CLINICAL RULE BASE
# =====================================================

clinical_rules = {

    # =================================================
    # SYSTOLIC BLOOD PRESSURE
    # =================================================

    "SystolicBP": [

        {
            "condition": lambda x: x >= 160,
            "severity": "Critical",
            "message": (
                "Severely elevated systolic blood pressure detected."
            ),
            "recommendation": (
                "Immediate maternal evaluation required."
            ),
            "emergency": True
        },

        {
            "condition": lambda x: x >= 140,
            "severity": "High",
            "message": (
                "Elevated systolic blood pressure observed."
            ),
            "recommendation": (
                "Frequent BP monitoring recommended."
            ),
            "emergency": False
        }
    ],

    # =================================================
    # DIASTOLIC PRESSURE
    # =================================================

    "Diastolic": [

        {
            "condition": lambda x: x >= 110,
            "severity": "Critical",
            "message": (
                "Severely elevated diastolic pressure detected."
            ),
            "recommendation": (
                "Urgent physician consultation required."
            ),
            "emergency": True
        },

        {
            "condition": lambda x: x >= 90,
            "severity": "High",
            "message": (
                "High diastolic pressure observed."
            ),
            "recommendation": (
                "Regular BP monitoring advised."
            ),
            "emergency": False
        }
    ],

    # =================================================
    # BLOOD SUGAR
    # =================================================

    "BS": [

        {
            "condition": lambda x: x > 200,
            "severity": "Critical",
            "message": (
                "Critically elevated blood sugar detected."
            ),
            "recommendation": (
                "Immediate glucose management required."
            ),
            "emergency": True
        },

        {
            "condition": lambda x: x > 140,
            "severity": "High",
            "message": (
                "Elevated blood sugar levels observed."
            ),
            "recommendation": (
                "Gestational diabetes screening recommended."
            ),
            "emergency": False
        }
    ],

    # =================================================
    # BMI
    # =================================================

    "BMI": [

        {
            "condition": lambda x: x > 30,
            "severity": "High",
            "message": (
                "Obesity-related maternal risk detected."
            ),
            "recommendation": (
                "Physician-guided nutrition and exercise recommended."
            ),
            "emergency": False
        },

        {
            "condition": lambda x: x >= 25,
            "severity": "Moderate",
            "message": (
                "Elevated BMI may increase pregnancy complications."
            ),
            "recommendation": (
                "Monitor diet and physical activity."
            ),
            "emergency": False
        }
    ],

    # =================================================
    # HEART RATE
    # =================================================

    "HeartRate": [

        {
            "condition": lambda x: x > 120,
            "severity": "High",
            "message": (
                "Abnormally elevated heart rate detected."
            ),
            "recommendation": (
                "Cardiovascular assessment recommended."
            ),
            "emergency": False
        },

        {
            "condition": lambda x: x > 100,
            "severity": "Moderate",
            "message": (
                "Elevated heart rate observed."
            ),
            "recommendation": (
                "Regular monitoring advised."
            ),
            "emergency": False
        }
    ],

    # =================================================
    # BODY TEMPERATURE
    # =================================================

    "BodyTemp": [

        {
            "condition": lambda x: x > 101,
            "severity": "High",
            "message": (
                "High fever-related maternal risk detected."
            ),
            "recommendation": (
                "Immediate infection screening recommended."
            ),
            "emergency": False
        },

        {
            "condition": lambda x: x > 99,
            "severity": "Moderate",
            "message": (
                "Elevated body temperature observed."
            ),
            "recommendation": (
                "Monitor temperature regularly."
            ),
            "emergency": False
        }
    ],

    # =================================================
    # PREEXISTING DIABETES
    # =================================================

    "PreexistingDiabetes": [

        {
            "condition": lambda x: x >= 1,
            "severity": "High",
            "message": (
                "Preexisting diabetes increases maternal risk."
            ),
            "recommendation": (
                "Continuous diabetic monitoring recommended."
            ),
            "emergency": False
        }
    ],

    # =================================================
    # GESTATIONAL DIABETES
    # =================================================

    "GestationalDiabetes": [

        {
            "condition": lambda x: x >= 1,
            "severity": "Moderate",
            "message": (
                "Gestational diabetes risk factor detected."
            ),
            "recommendation": (
                "Frequent glucose monitoring advised."
            ),
            "emergency": False
        }
    ],

    # =================================================
    # PREVIOUS COMPLICATIONS
    # =================================================

    "PreviousComplications": [

        {
            "condition": lambda x: x >= 1,
            "severity": "High",
            "message": (
                "History of previous pregnancy complications detected."
            ),
            "recommendation": (
                "Specialized maternal monitoring recommended."
            ),
            "emergency": False
        }
    ]
}