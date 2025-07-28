def main():
    from genmol.sampler import Sampler
    sampler = Sampler('./model.ckpt')
    num_samples = 2
    samples = sampler.fragment_completion('CCCCCCCCCCCC', num_samples, randomness=1.2, gamma=0.3)
    print(samples)


if __name__ == "__main__":
    main()
