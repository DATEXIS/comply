HAZARD_LEVEL = ["Low", "Medium", "High"]

# (untreated loss, diagnosis reward, treatment reward
HAZARD_REWARD = [(-1, 5, 10), (-3, 10, 20), (-5, 20, 50)]

PROCEDURE_TYPE = ["Examination", "Treatment"]

PROBABILITY = {
    "low": 0.2,
    "medium": 0.5,
    "high": 0.8,
    "always": 1.0
}

ONSET = {
    "initial": 0,
    "short-term": 3,
    "long-term": 6 
}

STATE_INITIAL_COMPLAINT = [
    "The patient complains about {}.",
    "The chief complaint is {}.",
    "Patient suffers from {}.",
    "Patient presents with {}.",
    "The patient came here for {}."
]

STATE_EXAM_NOSYMPTOM = [
    "{} could not detect additional symptoms.",
    "{} did not uncover new symptoms.",
    "After applying {} we could not see any new symptoms.",
    "No new symptoms observed after applying {}."
]

# TODO: make it possible to switch symptoms and procedure for more possible templates
STATE_EXAM_SYMPTOM = [
    "{} detected {}.",
    "Applied {}. Patient presented with {}.",
    "{} showed {}.",
    "{} was able to demonstrate {}.",
    "Examination {} discovered {}."
]

#TODO: same as above
STATE_TREAT_SYMPTOM = [
    "{} managed to improve the patient's {}.",
    "{} alleviated the {}.",
    "{} cured the patient's {}.",
    "{} treated the {}."
]

STATE_TREAT_NOSYMPTOM = [
    "Patient did not improve after {}.",
    "{} was unsucessful.",
    "{} did not lead to an improvement in the patient.",
    "{} did not improve the patient's symptoms.",
]

STATE_DETERIORATING = [
    
]