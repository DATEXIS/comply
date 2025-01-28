import gym
import sys

from tabulate import tabulate
env = gym.make("gym_medical:doctorsim-v0", data_path="data/csv/", max_diseases = int(sys.argv[1]) if len(sys.argv) > 1 else None, is_csv=True)
# print(f"Diseases({len(env.diseases)})")
# for i, d in env.diseases.items():
#     print(d.name)
# print(".........................")
#print(f"Procedures({len(env.procedures)})")
# for i, p in env.procedures.items():
#     print(p)
# print("--------------------------------------")
# print(f"Symptoms({len(env.symptoms)})")
# for i, s in env.symptoms.items():
#     print(s.id)

print(f"Dataset Statistics")
print(f"Number of Diseases: {len(env.diseases)}")
print(f"Number of Symptoms: {len(env.symptoms)}")
print(".................................")
for p in env.procedures.values():
    if p.type == 1:
        print(p.name)
print(".................................")
print(f"Number of Examinations: {len([ p for p in env.procedures.values() if p.type == 0])}")
print(f"Number of Treatments: {len([ p for p in env.procedures.values() if p.type == 1])}")
spd = [len(d.symptoms) for d in env.diseases.values()]
print(f"Number of Symptoms per Disease: Min: {min(spd)}, Avg.:{sum(spd)/len(spd)} Max:{max(spd)}")
pds = [len(s.examinations) + len(s.treatments) for s in env.symptoms.values()]
print(f"Number of Procedures per Symptom: Min: {min(pds)}, Avg.:{sum(pds)/len(pds)} Max:{max(pds)}")
pdd = [len(d.examinations) + len(d.treatments) for d in env.diseases.values()]
for d in env.diseases.values():
    if len(d.examinations) + len(d.treatments) == max(pdd):
        for p in d.examinations + d.treatments:
            print(p)
print(f"Number of Procedures per Disease: Min: {min(pdd)}, Avg.:{sum(pdd)/len(pdd)} Max:{max(pdd)}")
print(f"Number of Diseases without main symptom: {len([d.id for d in env.diseases.values() if d.main_symptom == None])}")
while 1==1:
    obs = env.reset()
    breakpoint()
    assert env.observation_space.contains(obs)
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
