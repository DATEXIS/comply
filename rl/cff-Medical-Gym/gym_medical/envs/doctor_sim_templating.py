import gymnasium
import random
import numpy as np
import os
from collections import deque
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
from gymnasium.spaces import Space
from transformers import AutoTokenizer
from typing import Tuple, List
from ..preprocessing import process_phospital, process_csv, Disease, Procedure
from ..constants import *
from .doctor_sim import DoctorSim, Patient
import random

class DoctorSimTemplating(DoctorSim):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_episode_steps=100, observation_length=512, tokenizer="bert-base-uncased", data_path="../../data/project_hospital", max_diseases = None, fruitfly = False, is_csv = True):
        DoctorSim.__init__(self, max_episode_steps, observation_length, tokenizer, data_path, max_diseases, fruitfly)
        
        # save previous untokenized observation to extend it in each step
        self.obs = ""
        
    def _step(self, procedure: Procedure):
        done = False
        hazard, new_symptoms, cured_main, discovered_main = self._patient.apply_procedure(procedure)
        new_symptom_names = [s.name for s in new_symptoms]
        
        if self._patient.total_hazard <= 0:
            self._patient.total_hazard = 0
            done = True

        r_index = procedure.type + 1
        procedure_reward = sum([HAZARD_REWARD[s.hazard][r_index]
                                      for s in new_symptoms])
        if procedure_reward > 0:
            reward = procedure_reward
        else:    
            reward = hazard 


        if discovered_main :
            reward = 100
        if cured_main:
            reward = 1000
            done = True
            
        new_obs = ""
        if procedure.type == PROCEDURE_TYPE.index("Examination"):
            if new_symptoms:
                new_obs = random.choice(STATE_EXAM_SYMPTOM).format(procedure.name, ','.join(new_symptom_names))
            else:
                new_obs = random.choice(STATE_EXAM_NOSYMPTOM).format(procedure.name)
        else:
            if new_symptoms:
                new_obs = random.choice(STATE_TREAT_SYMPTOM).format(procedure.name, ','.join(new_symptom_names))
            else:
                new_obs = random.choice(STATE_TREAT_NOSYMPTOM).format(procedure.name)
        
        self.obs += new_obs
        
        return self.tokenize(self.obs), reward, done
    
    def step(self, action: int):
        if self._patient is None:
            raise ValueError("Environment needs to be reset first")

        procedure = self.procedures[self.procedures_to_actions[action]]
        obs, r, done = self._step(procedure)
        

        return obs, r, done, {"disease_target": self._disease_target}
    
    def reset(self):
        self._disease_target = random.choice(range(len(self.diseases.values())))
        self._patient = Patient(list(self.diseases.values())[self._disease_target])
        self.procedure_reward = 0
        
        self.obs = random.choice(STATE_INITIAL_COMPLAINT).format(self._patient.discovered_symptoms[0].name)

        return self.tokenize(self.obs)
