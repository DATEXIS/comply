from gymnasium.envs.registration import register

register(
    id='doctorsim-v0',
    entry_point='gym_medical.envs:DoctorSim',
)
register(
    id='doctorsimmultibinary-v0',
    entry_point='gym_medical.envs:DoctorSimMultiBinary'
)
register(
    id='doctorsimtemplating-v0',
    entry_point='gym_medical.envs:DoctorSimTemplating'
)
