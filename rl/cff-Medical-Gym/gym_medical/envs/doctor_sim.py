import random
import numpy as np
import os
import gymnasium
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
from gymnasium.spaces import Space
from transformers import AutoTokenizer
from typing import Tuple, List
from ..preprocessing import process_phospital, process_csv, Disease, Procedure
from ..constants import PROCEDURE_TYPE, HAZARD_LEVEL, HAZARD_REWARD


class Patient:
    def __init__(self, disease: Disease):
        self.disease = disease

        self.symptoms = []
        self.future_symptoms = {}
        for s in self.disease.symptoms.values():
            if random.random() < s["probability"] and not s["symptom"].is_main:
                if s["firstDay"] == 0:
                    self.symptoms.append(s["symptom"])
                else:
                    if s["firstDay"] not in self.future_symptoms:
                        self.future_symptoms[s["firstDay"]] = [s["symptom"]]
                    else:
                        self.future_symptoms[s["firstDay"]].append(
                            s["symptom"]
                        )
        if self.symptoms == []:
            # print(self.disease.name)
            # print(self.disease.symptoms)
            self.symptoms.append(
                random.choice(
                    [
                        s
                        for s in self.disease.symptoms.values()
                        if not s["symptom"].is_main
                    ]
                )["symptom"]
            )
        self.discovered_symptoms = [random.choice(self.symptoms)]
        self.symptoms += [self.disease.main_symptom]
        self.treated_symptoms = []
        self.applied_procedures = []
        self.total_hazard = 200
        self.steps_alive = 0

    def apply_procedure(self, procedure: Procedure) -> Tuple[int, List, bool]:
        new = []
        cured_main = False
        discovered_main = False
        self.applied_procedures += [procedure]
        # check if we reveal a new symptom
        if procedure.type == PROCEDURE_TYPE.index("Examination"):

            undiscovered = set(self.symptoms) - set(self.discovered_symptoms)
            for symptom in undiscovered:
                if procedure in symptom.examinations:
                    new += [symptom]
                    if symptom.is_main:
                        discovered_main = True
                    if (
                        procedure.id == "EXM_INTERVIEW"
                    ):  # nerf interview action to only uncover one symptom at a time
                        break
            self.discovered_symptoms += new
        # check if we treat a revealed symptom
        else:
            for symptom in set(self.discovered_symptoms) - set(
                self.treated_symptoms
            ):
                if procedure in symptom.treatments:
                    new += [symptom]
                    if symptom.is_main:
                        cured_main = True
            self.treated_symptoms += new

        untreated = set(self.symptoms) - set(self.treated_symptoms)
        hazard = sum([HAZARD_REWARD[s.hazard][0] for s in untreated])
        self.total_hazard += hazard

        self.steps_alive += 1
        if self.steps_alive in self.future_symptoms:
            self.symptoms += self.future_symptoms[self.steps_alive]

        return hazard, new, cured_main, discovered_main

    def get_representation(self):

        # if self.disease.main_symptom in self.discovered_symptoms:
        #     return "Yes"
        # else:
        #     return "No"

        output = "The patient has the following symptoms: "
        output += ", ".join([s.name for s in self.discovered_symptoms])
        output += ". These symptoms have been treated already: "
        output += ", ".join([s.name for s in self.treated_symptoms])
        output += ". These procedures have been applied: "
        output += ", ".join([s.name for s in self.applied_procedures])
        return output

    def __str__(self):
        return (
            f"""--------------------\n"""
            f"""Patient\n"""
            f"""Disease: {self.disease.name}\n"""
            f"""Total Symptoms: {", ".join([s["symptom"].name for s in self.disease.symptoms.values()])}\n"""
            f"""Actual Symptoms: {", ".join([s.name for s in self.symptoms])}\n"""
            f"""--------------------\n"""
            f"""Discovered Symptoms: {", ".join([s.name for s in self.discovered_symptoms])}\n"""
            f"""Treated Symptoms: {", ".join([s.name for s in self.treated_symptoms])}\n"""
            f"""Applied Procedures: {", ".join([p.name for p in self.applied_procedures])}\n"""
            f"""Hazard: {self.total_hazard}\n"""
            f"""--------------------"""
        )


class TextSpace(Space):
    def __init__(self, length, vocab_size):
        assert length > 0
        assert vocab_size > 0
        self.length = length
        self.vocab_size = vocab_size
        super(TextSpace, self).__init__((1, length), np.int64)

    def sample(self):
        return [
            random.randint(0, self.vocab_size - 1) for _ in range(self.length)
        ]

    def contains(self, x):
        if isinstance(x, list) or isinstance(x, np.ndarray):
            for i in x:
                if not isinstance(i, int) or (i < 0 or i > self.vocab_size):
                    print("All words must be in integers and in vocabulary")
                    return False
        else:
            print(
                "TextSpace expected a list or numpy array but got "
                + str(type(x))
            )
            return False
        return True

    def __repr__(self):
        return (
            f"TextSpace: obs length: {self.length}, Vocab: {self.vocab_size}"
        )

    def __eq__(self, other):
        return (
            isinstance(other, TextSpace)
            and self.length == other.length
            and self.vocab_size == other.length
        )


class DoctorSim(gymnasium.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        max_episode_steps=100,
        observation_length=512,
        tokenizer="bert-base-uncased",
        data_path="../../data/project_hospital",
        max_diseases=None,
        fruitfly=False,
        # complex_ff=False,
        is_csv=True,
    ):

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer, use_fast=True
        )
        self.fruitfly = fruitfly  # Change the way the tokenization happens in the case of the Fruitfly model
        # self.complex_ff = complex_ff# Change the way the tokenization happens in the case of the Fruitfly model

        if is_csv:
            self.diseases, self.symptoms, self.procedures = process_csv(
                data_path, max_diseases=max_diseases
            )
            self.procedures_to_actions = (
                open(os.path.join(data_path, "actions_csv.txt"))
                .read()
                .splitlines()
            )
        else:
            self.diseases, self.symptoms, self.procedures = process_phospital(
                data_path, max_diseases=max_diseases
            )
            self.procedures_to_actions = (
                open(os.path.join(data_path, "actions.txt"))
                .read()
                .splitlines()
            )

        self.procedures_to_actions = [
            a
            for a in self.procedures_to_actions
            if a in self.procedures.keys()
        ]

        print("----------------------")
        print(len(self.procedures_to_actions))
        print(len(self.procedures.keys()))
        print("----------------------")

        self.observation_length = observation_length
        self.observation_space = TextSpace(
            observation_length, len(self.tokenizer)
        )

        self.action_space = spaces.Discrete(len(self.procedures.keys()))
        self.procedure_reward = 0

        self._patient = None
        self._disease_target = -1

    def get_patient(self):
        return self._patient

    def _step(self, procedure: Procedure):
        done = False
        hazard, new_symptoms, cured_main, discovered_main = (
            self._patient.apply_procedure(procedure)
        )

        if self._patient.total_hazard <= 0:
            self._patient.total_hazard = 0
            done = True

        r_index = procedure.type + 1
        procedure_reward = sum(
            [HAZARD_REWARD[s.hazard][r_index] for s in new_symptoms]
        )
        if procedure_reward > 0:
            reward = procedure_reward
        else:
            reward = hazard

        if discovered_main:
            reward = 100
        if cured_main:
            reward = 1000
            done = True

        return self._patient.get_representation(), reward, done

    def step(self, action: int):
        if self._patient is None:
            raise ValueError("Environment needs to be reset first")

        obs, r, done = self._step(
            self.procedures[self.procedures_to_actions[action]]
        )

        return (
            self.tokenize(obs),
            r,
            done,
            False, # Truncated
            {"disease_target": self._disease_target},
            
        )

    def reset(self, seed=None, options=None):
        self._disease_target = random.choice(
            range(len(self.diseases.values()))
        )
        self._patient = Patient(
            list(self.diseases.values())[self._disease_target]
        )
        self.procedure_reward = 0

        return self.tokenize(self._patient.get_representation()) , {}

    def tokenize(self, obs: str) -> List[int]:
        if self.fruitfly:
            ids = self.tokenizer.encode(
                obs,
                padding="max_length",
                truncation=True,
                max_length=self.observation_length,
                add_special_tokens=False,
            )
            # positions = np.arange(len(ids))
            # if self.complex_ff:
            #     return ids, positions
            return ids
        else:
            return self.tokenizer.encode(
                obs,
                padding="max_length",
                truncation=True,
                max_length=self.observation_length,
            )

    def render_html(self, mode="human"):
        view = (
            str(self._patient)
            + "\n"
            + f"Procedure Reward: {self.procedure_reward}"
        )
        return view.replace("\n", "<br>")

    def render(self, mode="human"):
        print(str(self._patient))
        print(f"Procedure Reward: {self.procedure_reward}")

    def close(self): ...
