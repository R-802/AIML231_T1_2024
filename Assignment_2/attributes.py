# This dictionary was generated using GPT3.5 and Attribute_descriptions.txt. 

attribute_dict = {
    "Att1": {
        "Description": "Status of existing checking account",
        "Values": {
            "A11": "<0 DM",
            "A12": "0 <= ... < 200 DM",
            "A13": ">= 200 DM",
            "A14": "No checking account",
        },
    },
    "Att2": {
        "Description": "Duration in month",
    },
    "Att3": {
        "Description": "Credit history",
        "Values": {
            "A30": "no credits taken/ all credits paid back duly",
            "A31": "all credits at this bank paid back duly",
            "A32": "existing credits paid back duly till now",
            "A33": "delay in paying off in the past",
            "A34": "critical account/other credits existing (not at this bank)",
        },
    },
    "Att4": {
        "Description": "Purpose",
        "Values": {
            "A40": "car (new)",
            "A41": "car (used)",
            "A42": "furniture/equipment",
            "A43": "radio/television",
            "A44": "domestic appliances",
            "A45": "repairs",
            "A46": "education",
            "A47": "(vacation - does not exist?)",
            "A48": "retraining",
            "A49": "business",
            "A410": "others",
        },
    },
    "Att5": {
        "Description": "Credit amount",
    },
    "Att6": {
        "Description": "Savings account/bonds",
        "Values": {
            "A61": "< 100 DM",
            "A62": "100 <= ... < 500 DM",
            "A63": "500 <= ... < 1000 DM",
            "A64": ">= 1000 DM",
            "A65": "No savings",
        },
    },
    "Att7": {
        "Description": "Present employment since",
        "Values": {
            "A71": "unemployed",
            "A72": "< 1 year",
            "A73": "1 <= ... < 4 years",
            "A74": "4 <= ... < 7 years",
            "A75": ">= 7 years",
        },
    },
    "Att8": {
        "Description": "Installment rate in percentage of disposable income",
    },
    "Att9": {
        "Description": "Personal status and sex",
        "Values": {
            "A91": "male: divorced/separated",
            "A92": "female: divorced/separated/married",
            "A93": "male: single",
            "A94": "male: married/widowed",
            "A95": "female: single",
        },
    },
    "Att10": {
        "Description": "Other debtors / guarantors",
        "Values": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    },
    "Att11": {
        "Description": "Present residence since",
    },
    "Att12": {
        "Description": "Property",
        "Values": {
            "A121": "real estate",
            "A122": "if not A121: building society savings agreement/life insurance",
            "A123": "if not A121/A122: car or other, not in att6",
            "A124": "no property",
        },
    },
    "Att13": {
        "Description": "Age in years",
    },
    "Att14": {
        "Description": "Other installment plans",
        "Values": {"A141": "bank", "A142": "stores", "A143": "none"},
    },
    "Att15": {
        "Description": "Housing",
        "Values": {"A151": "rent", "A152": "own", "A153": "for free"},
    },
    "Att16": {
        "Description": "Number of existing credits at this bank",
    },
    "Att17": {
        "Description": "Job",
        "Values": {
            "A171": "unemployed/ unskilled - non-resident",
            "A172": "unskilled - resident",
            "A173": "skilled employee / official",
            "A174": "management/self-employed/highly qualified employee/officer",
        },
    },
    "Att18": {
        "Description": "Number of people being liable to provide maintenance for",
    },
    "Att19": {
        "Description": "Telephone",
        "Values": {"A191": "none", "A192": "yes, registered under the customer's name"},
    },
    "Att20": {
        "Description": "Foreign worker",
        "Values": {"A201": "yes", "A202": "no"},
    },
    "Output": {"Description": "Credit risk", "Values": {"1": "Good", "2": "Bad"}},
}
