# Backward-compatibility shim.
#
# When R3GAN checkpoints (.pkl) were saved, their Generator/Discriminator classes
# were stored under the module path `training.networks` (relative to r3gan/).
# Unpickling those files will look up `training.networks.Generator` in sys.modules.
#
# This shim re-exports Generator and Discriminator from the canonical location
# (models/temporal_organoid_generator/networks.py), so old checkpoints continue
# to load without changes. The same network classes are frozen and reused as
# the image decoder inside Stage 2's temporal virtual organoid construction.
from models.temporal_organoid_generator.networks import Generator, Discriminator  # noqa: F401

__all__ = ['Generator', 'Discriminator']
