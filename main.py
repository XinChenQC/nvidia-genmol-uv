def test():


    '''
    ! Linker Design / Scaffold Morphing = generate a linker fragment that connects given two side chains
    ! Motif Extension = generate molecule with existing motif
    ! Scaffold Decoration = same as Motif Extension but start with larger scaffold 
    ! Superstructure Generation = generate a molecule when a substructure constraint is given
    ! Linker Design (1-step) = generate a linker fragment that connects given two side chains without sequence mixing
    '''
    from genmol.sampler import Sampler
    sampler = Sampler('./model.ckpt')
    num_samples = 20
    fragment = 'COc1ccc2c(c1)nc([nH]2)S(=O)Cc1ncc(c(c1C)OC)C' # SMILES


    print('Doing linker_design')
    fragment2 = '[11*]N1CCCC1.[17*]c1ccc2c(c1)OCCO2' # SMILES
    samples = sampler.fragment_linking(fragment2, num_samples, randomness=3)
    print(samples)

    print('Doing motif_extension')
    fragment = '[17*]c1ccc2c(c1)OCCO2' # SMILES
    samples = sampler.fragment_completion(fragment, num_samples, randomness=1.2, gamma=0.3)
    print(samples)

    print('Doing scaffold_decoration')
    fragment = '[1*]c1cccc(Nc2ncnc3cc([2*])c([3*])cc23)c1' # SMILES
    samples = sampler.fragment_completion(fragment, num_samples, gamma=0.3)
    print(samples)

    print('c1ccc(Oc2ccccc2)cc1')
    samples = sampler.fragment_completion(fragment, num_samples, gamma=0.4)
    print(samples)


if __name__ == "__main__":
    test()
