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
from ..constants import PROCEDURE_TYPE, HAZARD_LEVEL, HAZARD_REWARD
from .doctor_sim import DoctorSim, Patient

class DoctorSimMultiBinary(DoctorSim):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_episode_steps=100, observation_length=512, tokenizer="bert-base-uncased", data_path="../../data/project_hospital", max_diseases = None, fruitfly=False, window_size=1, is_csv = True):
        DoctorSim.__init__(self, max_episode_steps, observation_length, tokenizer, data_path, max_diseases, fruitfly)

        self.window_size = window_size
        
        self.symptomlist = sorted(self.symptoms.keys())
        
        self.obs_shape = (self.window_size, len(self.symptomlist) + len(self.procedures_to_actions))
        self.observation_space  = gymnasium.spaces.MultiBinary(self.obs_shape)
        self.prev_obs = deque([list(l) for l in np.zeros(self.obs_shape)], maxlen = self.window_size)
        

    def tokenize(self, obs: str) -> List[int]:
        discovered_ids = [s.id for s in self._patient.discovered_symptoms]
        procedure_ids = [p.id for p in self._patient.applied_procedures]
        symptom_multibinary = [1 if s in discovered_ids else 0 for s in self.symptomlist]
        procedure_multibinary = [1 if p in procedure_ids else 0 for p in self.procedures_to_actions]
        self.prev_obs.append(symptom_multibinary + procedure_multibinary)
        
        tokenized = np.array(self.prev_obs)
        #print(f"###########SHAPE: {tokenized.shape}")
        return tokenized
    
    def reset(self):
        self._disease_target = random.choice(range(len(self.diseases.values())))
        self._patient = Patient(list(self.diseases.values())[self._disease_target])
        self.procedure_reward = 0

        return self.tokenize(self._patient.get_representation())
