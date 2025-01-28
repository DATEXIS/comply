import gym
import sys

from tabulate import tabulate
env = gym.make("gym_medical:doctorsimmultibinary-v0", data_path="data/project_hospital/", max_diseases = int(sys.argv[1]) if len(sys.argv) > 1 else None)
print(f"Diseases({len(env.diseases)})")
for i, d in env.diseases.items():
    print(d.name)
print(".........................")
print(f"Procedures({len(env.procedures)})")
for i, p in env.procedures.items():
    print(p.name)
print("--------------------------------------")

while 1==1:
    obs = env.reset()
    assert env.observation_space.contains(obs)
    print(obs)
    print(obs.shape)
    #print(f"Obs: {env.tokenizer.decode(obs, skip_special_tokens=True)}")
    done = False
    while not done:
        env.render()
        a = -3
        while a < 0:
            a = int(input("Enter Action Nr.: "))
            if a == -1:
                d = env.get_patient().disease
                table = []
                for s in d.symptoms.values():
                    s = s["symptom"]
                    symptom = s.name
                    examinations = ", ".join(f"{e.name}({env.procedures_to_actions.index(e.id)})" for e in s.examinations)
                    treatments = ", ".join(f"{t.name}({env.procedures_to_actions.index(t.id)})" for t in s.treatments)
                    is_main = s.is_main
                    table += [[ symptom, examinations, treatments, is_main]]
                for i, p in enumerate(env.procedures_to_actions):
                    print(f"{i}  {p}")
                print(tabulate(table, headers=["Symptom", "Examinations", "Treatments", "Is Main Symptom"]))
        obs, r , done, info = env.step(a)
        assert env.observation_space.contains(obs)
        print(f"Reward: {r} | done: {done}")
        #print(f"Obs: {env.tokenizer.decode(obs, skip_special_tokens=True)}")
        print(obs)
        print(obs.shape)
