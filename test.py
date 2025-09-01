from genmol.sampler import Sampler
import time 
start_time = time.time()
sampler = Sampler('./model.ckpt')
num_samples = 20

print('Linker Design')
fragment2 = '[11*]N1CCCC1.[17*]c1ccc2c(c1)OCCO2'
samples = sampler.fragment_linking(fragment2, num_samples, randomness=3)
print(samples)

print('Motif Extension')
fragment = '[17*]c1ccc2c(c1)OCCO2'
samples = sampler.fragment_completion(fragment, num_samples, randomness=1.2, gamma=0.3)
print(samples)

print('Scaffold Decoration')
fragment = '[1*]c1cccc(Nc2ncnc3cc([2*])c([3*])cc23)c1'
samples = sampler.fragment_completion(fragment, num_samples, gamma=0.3)
print(samples)

print('Superstructure Generation')
fragment = 'c1ccc(Oc2ccccc2)cc1'
samples = sampler.fragment_completion(fragment, num_samples, gamma=0.4)
print(samples)

end = time.time()

print(end-start_time)
