from typing import List, Dict
from .constants import PROCEDURE_TYPE

class HospitalObject():
    def __init__(self, id: str, name: str, desc: str):
        self.id = id
        self.name = name
        self.description = desc

    def __eq__(self, other):
        return type(self) == type(other) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return f"{self.id}: {self.name}"


class Procedure(HospitalObject):
    def __init__(self, id: str, name: str, desc: str, ptype: int):
        super().__init__(id, name, desc)
        self.type = ptype

    def __str__(self):
        return super(Procedure, self).__str__() + f" | Type: {PROCEDURE_TYPE[self.type]}"


class Symptom(HospitalObject):
    def __init__(self, id: str, name: str, desc: str, hazard: int, examinations: List[Procedure], treatments: List[Procedure], is_main: bool):
        super().__init__(id, name, desc)
        self.hazard = hazard
        self.examinations = examinations
        self.treatments = treatments
        self.is_main = is_main

    def add_examinations(self, exams: List):
        exam_ids = [e.id for e in self.examinations]
        self.examinations += [e for e in exams if e.id not in exam_ids]
        
    def add_treatments(self, treatments: List):
        treatment_ids = [t.id for t in self.treatments]
        self.treatments += [t for t in treatments if t.id not in treatment_ids]
        
    def __str__(self):
        return super(Symptom, self).__str__() + f" | IsMain: {self.is_main} | Examinations: {[str(e) for e in self.examinations]} | Treatments: {[ str(t) for t in self.treatments]} | Hazard: {self.hazard}"


class Disease(HospitalObject):
    def __init__(self, id: str, name: str, desc: str, duration: int, symptoms: Dict, main_symptom: Dict, examinations: List[Procedure], treatments: List[Procedure]):
        super().__init__(id, name, desc)
        self.duration = duration
        self.symptoms = symptoms
        self.main_symptom = main_symptom
        self.treatments = treatments
        self.examinations = examinations

    def add_symptom(self, dsymptom: Dict):
        symp_ids  = [symp["symptom"].id for symp in self.symptoms.values()]
        
        if dsymptom["symptom"].id not in symp_ids:
            self.symptoms[dsymptom["symptom"].id] = dsymptom
            
            p_ids = {"Examination": [e.id for e in self.examinations], "Treatment": [t.id for t in self.treatments]}
            new_procedures = {"Examination": [], "Treatment": []}
            
            if dsymptom["symptom"].is_main:
                self.main_symptom = dsymptom["symptom"] 
                
            for ptype, ps in {"Examination": dsymptom["symptom"].examinations, "Treatment": dsymptom["symptom"].treatments}.items():
                for p in ps:
                    if p.id not in p_ids[ptype]:
                        new_procedures[ptype].append(p)
            
            self.treatments += new_procedures["Treatment"]
            self.examinations += new_procedures["Examination"]
            
    def __str__(self):
        return super(Disease, self).__str__() + f" | Symptoms: {self.symptoms} | Examinations: {self.examinations} | Treatments: {self.treatments}"
