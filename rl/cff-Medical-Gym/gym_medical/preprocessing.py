from typing import List, Tuple, Dict
import glob
import xmltodict
import os
import csv
import json
from .constants import PROCEDURE_TYPE, HAZARD_LEVEL, PROBABILITY, ONSET
from .medical import Procedure, Disease, Symptom 


# TODO: symptoms with different procedures depending on disease vs aggregating
# TODO: diseases with not properly split symptoms (e.g. Womb uterus cancer)
def process_csv(path: str, max_diseases: int = None) -> Dict:
    
    procedures = {}
    symptoms = {}
    diseases = {}
    necessary_procedures = {}
    hazard_level = ["Low", "Mid", "High"] # labeled differently than HAZARD_LEVEL
    cleanup_dicts = {
        "symptoms": json.load(open(os.path.join(path, "symptoms_cleanup.json"), "r")),
        "diagnostics": json.load(open(os.path.join(path, "diagnostics_cleanup.json"), "r")),
        "treatments": json.load(open(os.path.join(path, "treatments_cleanup.json"), "r"))
    }
    
    with open(os.path.join(path, "data.csv"), "r", newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        
        for row in reader:
            disease, _, _, symptom, severity, probability, onset, diagnostics, treatments, is_main, *_ = row
            disease = disease.lower()
            
            if symptom == "": #for diseases where the disease itself is the main symptom
                symptom = disease + " discovered"
                is_main = "yes"
                probability = "always"
                onset = "initial"
            
            if symptom in cleanup_dicts["symptoms"]:
                symptom = cleanup_dicts["symptoms"][symptom]
            symptom = symptom.lower()
            
            severity    = severity or "Low"
            probability = probability or "always"
            onset       =  onset or "initial"
            
            is_main         = is_main == "yes"  
            temp_procedures = {"Examination": [], "Treatment": []}
            
            
            
            for ptype, ps in {"Examination": diagnostics, "Treatment": treatments}.items():
                
                if len(ps) == 0: #some symptoms dont have treatments
                    ps_list = []
                elif ps[0] == '{': #if consists of multiple
                    ps_list = json.loads(ps.replace("\'", "\""))['text']
                else:
                    ps_list = [ps]

                for p in ps_list:
                    if p in cleanup_dicts["diagnostics"]:
                        p = cleanup_dicts["diagnostics"][p]
                    elif p in cleanup_dicts["treatments"]:
                        p = cleanup_dicts["treatments"][p]
                    p = p.lower()

                    if not p in procedures.keys():
                        procedures[p] = Procedure(p, p, "", PROCEDURE_TYPE.index(ptype))
                    temp_procedures[ptype].append(procedures[p])


            if not symptom in symptoms:
                symptoms[symptom] = Symptom(symptom,
                                            symptom, 
                                            "", 
                                            hazard_level.index(severity), 
                                            temp_procedures["Examination"].copy(), 
                                            temp_procedures["Treatment"].copy(), 
                                            is_main)
            else:
                symptoms[symptom].add_examinations(temp_procedures["Examination"])
                symptoms[symptom].add_treatments(temp_procedures["Treatment"])
            
            dsymptom = {"symptom": symptoms[symptom], 
                        "probability": PROBABILITY[probability], 
                        "firstDay": ONSET[onset.lower()]}
            if not disease in diseases:
                if (max_diseases == None or len(diseases) < max_diseases):
                    diseases[disease] = Disease(disease, 
                                                disease,
                                                "",  
                                                3, 
                                                {dsymptom["symptom"].id: dsymptom}, 
                                                dsymptom["symptom"] if is_main else None, 
                                                temp_procedures["Examination"].copy(), 
                                                temp_procedures["Treatment"].copy())
            else:
                diseases[disease].add_symptom(dsymptom)

        diseases  = {dname: disease for dname, disease in diseases.items() if len(disease.symptoms) > 1}
        
        if max_diseases is not None:
            necessary_procedures = {}
            for disease in diseases.values():
                for p in disease.treatments + disease.examinations:
                    necessary_procedures[p.id] = p
        
        # with open(os.path.join(path, "actions_csv.txt"), "w") as f:
        #     f.write("\n".join([p.id for p in procedures.values()]))
    return diseases, symptoms, necessary_procedures if max_diseases is not None else procedures


def process_phospital(path: str, max_diseases: int = None) -> Dict:
    localizations = {}
    procedures = {}
    symptoms = {}
    diseases = {}
    necessary_procedures = {}

    localizations_xml = glob.glob(os.path.join(path, "Localization/*"))
    for f in localizations_xml:
        parse = xmltodict.parse(open(f, "rb"))
        for loc in parse["Database"]["GameDBStringTable"]["LocalizedStrings"]["GameDBLocalizedString"]:
            localizations[loc["LocID"]] = loc["Text"]


    exam_xml = os.path.join(path, "Procedures/Examinations.xml")
    treatment_xml = [

        os.path.join(path, "Procedures/Surgery.xml"),
        os.path.join(path, "Procedures/Treatments.xml"),
        os.path.join(path, "Procedures/TreatmentsHospitalization.xml"),
    ]

    exams = xmltodict.parse(open(exam_xml, "rb"))
    for e in exams["Database"]["GameDBExamination"]:
        procedures[e["@ID"]] = Procedure(
            e["@ID"],
            localizations[e["@ID"]],
            localizations[e["AbbreviationLocID"]],
            PROCEDURE_TYPE.index("Examination")
        )        
    for xml in treatment_xml:
        treatments = xmltodict.parse(open(xml, "rb"))
        for t in treatments["Database"]["GameDBTreatment"]:
            procedures[t["@ID"]] = Procedure(
                t["@ID"],
                localizations[t["@ID"]],
                localizations[t["AbbreviationLocID"]],
                PROCEDURE_TYPE.index("Treatment")
            )

    # with open(os.path.join(path, "actions.txt"), "w") as f:
    #     for p in procedures.keys():
    #         f.write(p + "\n")

    symptom_xml = glob.glob(os.path.join(path, "Symptoms*"))
    for xml in symptom_xml:
        symps = xmltodict.parse(open(xml, "rb"))
        for s in symps["Database"]["GameDBSymptom"]:
            es = s["Examinations"]["ExaminationRef"]
            es = [procedures[es]] if isinstance(es, str) else [procedures[e] for e in es]
            
            ts = [] if not "Treatments" in s else s["Treatments"]["TreatmentRef"]
            ts = [procedures[ts]] if isinstance(ts, str) else [procedures[t] for t in ts]

            symptoms[s["@ID"]] = Symptom(
                s["@ID"],
                localizations[s["@ID"]],
                localizations[s["DescriptionLocID"]],
                HAZARD_LEVEL.index(s["Hazard"]),
                es,
                ts,
                s["IsMainSymptom"] == "true"
            )

    disease_xml = glob.glob(os.path.join(path, "Diagnoses/Diagnoses*"))
    disease_xml = [xmltodict.parse(open(f, "rb")) for f in disease_xml]
    for xml in disease_xml:
        for d in xml["Database"]["GameDBMedicalCondition"]:
            if max_diseases is not None and len(diseases.keys()) >= max_diseases:
                break

            dsymptoms = {}
            main_symptom = None
            for s in d["Symptoms"]["GameDBSymptomRules"]:
                o = symptoms[s["GameDBSymptomRef"]]
                if o.is_main:
                    main_symptom = o
                dsymptoms[o.id] = {
                    "symptom": o,
                    "probability": 1.0 if "ProbabilityPercent" not in s else int(s["ProbabilityPercent"]) / 100,
                    "firstDay": 0 if "DayOfFirstprobability" not in s else int(s["DayOfFirstprobability"])
                }
                if max_diseases is not None:
                    for e in o.examinations:
                        necessary_procedures[e.id] = e
                    for t in o.treatments:
                        necessary_procedures[t.id] = t

            es = d["Examinations"]["ExaminationRef"]
            es = [procedures[es]] if isinstance(es, str) else [procedures[e] for e in es]

            ts = [] if not "Treatments" in d else d["Treatments"]["TreatmentRef"]
            ts = [procedures[ts]] if isinstance(ts, str) else [procedures[t] for t in ts]

            # For now ignore procedures that work directly on the disease, and not on a particular symptom
            # if max_diseases is not None:
            #     for e in es:
            #         necessary_procedures[e.id] = e
            #     for t in ts:
            #         necessary_procedures[t.id] = t    

            diseases[d["@ID"]] = Disease(
                d["@ID"],
                localizations[d["@ID"]],
                localizations[d["AbbreviationLocID"]],
                d["Duration"],
                dsymptoms,
                main_symptom,
                es,
                ts,
            )

    return diseases, symptoms, necessary_procedures if max_diseases is not None else procedures
