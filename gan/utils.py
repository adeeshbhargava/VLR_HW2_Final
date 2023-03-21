import argparse
import torch
import torchvision
from cleanfid import fid
from matplotlib import pyplot as plt


def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename + ".png")


@torch.no_grad()
def get_fid(gen, dataset_name, dataset_resolution, z_dimension, batch_size, num_gen):
    gen_fn = lambda z: (gen.forward_given_samples(z) / 2 + 0.5) * 255
    score = fid.compute_fid(
        gen=gen_fn,
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        num_gen=num_gen,
        z_dim=z_dimension,
        batch_size=batch_size,
        verbose=True,
        dataset_split="custom",
    )
    return score


@torch.no_grad()
def interpolate_latent_space(gen, path):
    # TODO 1.2: Generate and save out latent space interpolations.
    # Generate 100 samples of 128-dim vectors
    # Do so by linearly interpolating for 10 steps across each of the first two dimensions between -1 and 1.
    # Keep the rest of the z vector for the samples to be some fixed value (e.g. 0).
    # Forward the samples through the generator.
    # Save out an image holding all 100 samples.
    # Use torchvision.utils.save_image to save out the visualization.
    # Use nrow=10 to arrange the samples in a 10x10 grid.
    
    z = torch.zeros(1, 128).repeat(100, 1)
    lin = torch.linspace(-1, 1, 10)
    x, y = torch.meshgrid(lin, lin,indexing='ij')
    z[:, :2] = torch.stack((x, y), -1).reshape(-1, 2)
    out = gen.forward_given_samples(z.cuda())
    out = out*0.5 + 0.5
    torchvision.utils.save_image(out, path, nrow=10)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_amp", action="store_true")
    args = parser.parse_args()
    return args
