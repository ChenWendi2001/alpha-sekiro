import pickle

import numpy as np
from PIL import Image


if __name__ == "__main__":
    names = ["agent-hp-full", "boss-hp-full",
             "agent-ep-full", "boss-ep-full"]
    modes = ["HSV"] * 4
    for name, mode in zip(names, modes):
        image = Image.open(name + ".png").convert(mode)
        pickle.dump(
            np.array(image, np.int16).transpose(2, 0, 1),
            open(name + ".pkl", "wb"))
