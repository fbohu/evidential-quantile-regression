from ._base import DataModule, OutputType
from ._registry import DATASET_REGISTRY
from .bike_sharing import BikeSharingNormalDataModule, BikeSharingPoissonDataModule
from .cifar import Cifar10DataModule, Cifar100DataModule
from .mnist import FashionMnistDataModule, MnistDataModule
from .nyu_depth_v2 import NyuDepthV2DataModule
from .sensorless_drive import SensorlessDriveDataModule
from .uci import WineDataModule, BostonDataModule, ConcreteDataModule, PowerPlantDataModule, YachtDataModule, EnergyEfficiencyDataModule, Kin8nmDataModule, NavalDataModule, ProteinDataModule

__all__ = [
    "BikeSharingNormalDataModule",
    "BikeSharingPoissonDataModule",
    "Cifar100DataModule",
    "Cifar10DataModule",
    "WineDataModule",
    "BostonDataModule",
    "ConcreteDataModule",
    "PowerPlantDataModule",
    "YachtDataModule",
    "EnergyEfficiencyDataModule",
    "Kin8nmDataModule",
    "NavalDataModule",
    "ProteinDataModule",
    "DATASET_REGISTRY",
    "DataModule",
    "FashionMnistDataModule",
    "MnistDataModule",
    "NyuDepthV2DataModule",
    "OutputType",
    "SensorlessDriveDataModule",
]

