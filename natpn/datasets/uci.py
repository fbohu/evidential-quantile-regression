import logging
from typing import Any, Dict, Optional
import pandas as pd
import torch
from lightkit.data import DataLoader
from lightkit.utils import PathType
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import TensorDataset
from torchvision.datasets.utils import download_url  # type: ignore
from ._base import DataModule, OutputType
from ._registry import register
from ._utils import scale_oodom, StandardScaler, tabular_ood_dataset, tabular_train_test_split

logger = logging.getLogger(__name__)


class _UciDataModule(DataModule):
    def __init__(self, root: Optional[PathType] = None, seed: Optional[int] = None):
        """
        Args:
            root: The directory where the dataset can be found or where it should be downloaded to.
            seed: An optional seed which governs how train/test splits are created.
        """
        super().__init__(root, seed)
        self.did_setup = False
        self.did_setup_test = False

        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

    @property
    def output_type(self) -> OutputType:
        return "normal"

    @property
    def input_size(self) -> torch.Size:
        return torch.Size([8])

    def prepare_data(self) -> None:
        # Download concrete dataset
        logger.info("No need to prep'...")
        super().prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "test" and not self.did_setup_test:
            data = pd.read_excel(str(self.root / "energy" / "data.xlsx"))
            X = torch.from_numpy(data.to_numpy()[:, :8]).float()
            self.ood_datasets = {
                "energy": tabular_ood_dataset(
                    self.test_dataset.tensors[0], self.input_scaler.transform(X)
                ),
                "energy_oodom": tabular_ood_dataset(
                    self.test_dataset.tensors[0], self.input_scaler.transform(scale_oodom(X))
                ),
            }

            # Mark done
            self.did_setup_test = True

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=512, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=4096)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=4096)

    def ood_dataloaders(self) -> Dict[str, DataLoader[Any]]:
        return {
            name: DataLoader(dataset, batch_size=4096)
            for name, dataset in self.ood_datasets.items()
        }


from read_data import get_synth_data, load_dataset
@register("wine")
class WineDataModule(_UciDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        if not self.did_setup:
            (X_train, y_train), (X_test, y_test), y_train_mu, y_train_scale = load_dataset('wine')

            X_train = torch.from_numpy(X_train).float()
            y_train = torch.from_numpy(y_train).float().squeeze()
            X_test = torch.from_numpy(X_test).float()
            y_test = torch.from_numpy(y_test).float().squeeze()
            
            (X_train, X_val), (y_train, y_val) = tabular_train_test_split(
                X_train, y_train, train_size=0.8, generator=self.generator
            )
            # Fit transforms
            self.input_scaler.fit(X_train)
            self.output_scaler.fit(y_train)

            # Create datasets
            self.train_dataset = TensorDataset(
                X_train, y_train
            )
            self.val_dataset = TensorDataset(
                X_val, y_val
            )
            self.test_dataset = TensorDataset(
                X_test, y_test
            )
            # Mark done
            self.did_setup = True

        super().setup(stage)

    @property
    def input_size(self) -> torch.Size:
        return torch.Size([11])


@register("boston")
class BostonDataModule(_UciDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        if not self.did_setup:
            (X_train, y_train), (X_test, y_test), y_train_mu, y_train_scale = load_dataset('boston')

            X_train = torch.from_numpy(X_train).float()
            y_train = torch.from_numpy(y_train).float().squeeze()
            X_test = torch.from_numpy(X_test).float()
            y_test = torch.from_numpy(y_test).float().squeeze()
            
            (X_train, X_val), (y_train, y_val) = tabular_train_test_split(
                X_train, y_train, train_size=0.8, generator=self.generator
            )
            # Fit transforms
            self.input_scaler.fit(X_train)
            self.output_scaler.fit(y_train)

            # Create datasets
            self.train_dataset = TensorDataset(
                X_train, y_train
            )
            self.val_dataset = TensorDataset(
                X_val, y_val
            )
            self.test_dataset = TensorDataset(
                X_test, y_test
            )
            # Mark done
            self.did_setup = True
        super().setup(stage)


    @property
    def input_size(self) -> torch.Size:
        return torch.Size([13])


@register("concrete")
class ConcreteDataModule(_UciDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        if not self.did_setup:
            (X_train, y_train), (X_test, y_test), y_train_mu, y_train_scale = load_dataset('concrete')

            X_train = torch.from_numpy(X_train).float()
            y_train = torch.from_numpy(y_train).float().squeeze()
            X_test = torch.from_numpy(X_test).float()
            y_test = torch.from_numpy(y_test).float().squeeze()
            
            (X_train, X_val), (y_train, y_val) = tabular_train_test_split(
                X_train, y_train, train_size=0.8, generator=self.generator
            )
            # Fit transforms
            self.input_scaler.fit(X_train)
            self.output_scaler.fit(y_train)

            # Create datasets
            self.train_dataset = TensorDataset(
                X_train, y_train
            )
            self.val_dataset = TensorDataset(
                X_val, y_val
            )
            self.test_dataset = TensorDataset(
                X_test, y_test
            )
            # Mark done
            self.did_setup = True

        super().setup(stage)

@register("power-plant")
class PowerPlantDataModule(_UciDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        if not self.did_setup:
            (X_train, y_train), (X_test, y_test), y_train_mu, y_train_scale = load_dataset('power-plant')
            X_train = torch.from_numpy(X_train).float()
            y_train = torch.from_numpy(y_train).float().squeeze()
            X_test = torch.from_numpy(X_test).float()
            y_test = torch.from_numpy(y_test).float().squeeze()
            
            (X_train, X_val), (y_train, y_val) = tabular_train_test_split(
                X_train, y_train, train_size=0.8, generator=self.generator
            )
            # Fit transforms
            self.input_scaler.fit(X_train)
            self.output_scaler.fit(y_train)

            # Create datasets
            self.train_dataset = TensorDataset(
                X_train, y_train
            )
            self.val_dataset = TensorDataset(
                X_val, y_val
            )
            self.test_dataset = TensorDataset(
                X_test, y_test
            )
            # Mark done
            self.did_setup = True

        super().setup(stage)

    @property
    def input_size(self) -> torch.Size:
        return torch.Size([4])



@register("yacht")
class YachtDataModule(_UciDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        if not self.did_setup:
            (X_train, y_train), (X_test, y_test), y_train_mu, y_train_scale = load_dataset('yacht')

            X_train = torch.from_numpy(X_train).float()
            y_train = torch.from_numpy(y_train).float().squeeze()
            X_test = torch.from_numpy(X_test).float()
            y_test = torch.from_numpy(y_test).float().squeeze()
            
            (X_train, X_val), (y_train, y_val) = tabular_train_test_split(
                X_train, y_train, train_size=0.8, generator=self.generator
            )
            # Fit transforms
            self.input_scaler.fit(X_train)
            self.output_scaler.fit(y_train)

            # Create datasets
            self.train_dataset = TensorDataset(
                X_train, y_train
            )
            self.val_dataset = TensorDataset(
                X_val, y_val
            )
            self.test_dataset = TensorDataset(
                X_test, y_test
            )
            # Mark done
            self.did_setup = True

        super().setup(stage)

    @property
    def input_size(self) -> torch.Size:
        return torch.Size([6])


@register("energy-efficiency")
class EnergyEfficiencyDataModule(_UciDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        if not self.did_setup:
            (X_train, y_train), (X_test, y_test), y_train_mu, y_train_scale = load_dataset('energy-efficiency')

            X_train = torch.from_numpy(X_train).float()
            y_train = torch.from_numpy(y_train).float().squeeze()
            X_test = torch.from_numpy(X_test).float()
            y_test = torch.from_numpy(y_test).float().squeeze()
            
            (X_train, X_val), (y_train, y_val) = tabular_train_test_split(
                X_train, y_train, train_size=0.8, generator=self.generator
            )
            # Fit transforms
            self.input_scaler.fit(X_train)
            self.output_scaler.fit(y_train)

            # Create datasets
            self.train_dataset = TensorDataset(
                X_train, y_train
            )
            self.val_dataset = TensorDataset(
                X_val, y_val
            )
            self.test_dataset = TensorDataset(
                X_test, y_test
            )
            # Mark done
            self.did_setup = True

        super().setup(stage)


@register("kin8nm")
class Kin8nmDataModule(_UciDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        if not self.did_setup:
            (X_train, y_train), (X_test, y_test), y_train_mu, y_train_scale = load_dataset('kin8nm')

            X_train = torch.from_numpy(X_train).float()
            y_train = torch.from_numpy(y_train).float().squeeze()
            X_test = torch.from_numpy(X_test).float()
            y_test = torch.from_numpy(y_test).float().squeeze()
            
            (X_train, X_val), (y_train, y_val) = tabular_train_test_split(
                X_train, y_train, train_size=0.8, generator=self.generator
            )
            # Fit transforms
            self.input_scaler.fit(X_train)
            self.output_scaler.fit(y_train)

            # Create datasets
            self.train_dataset = TensorDataset(
                X_train, y_train
            )
            self.val_dataset = TensorDataset(
                X_val, y_val
            )
            self.test_dataset = TensorDataset(
                X_test, y_test
            )
            # Mark done
            self.did_setup = True

        super().setup(stage)

    @property
    def input_size(self) -> torch.Size:
        return torch.Size([8])

@register("naval")
class NavalDataModule(_UciDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        if not self.did_setup:
            (X_train, y_train), (X_test, y_test), y_train_mu, y_train_scale = load_dataset('naval')

            X_train = torch.from_numpy(X_train).float()
            y_train = torch.from_numpy(y_train).float().squeeze()
            X_test = torch.from_numpy(X_test).float()
            y_test = torch.from_numpy(y_test).float().squeeze()
            
            (X_train, X_val), (y_train, y_val) = tabular_train_test_split(
                X_train, y_train, train_size=0.8, generator=self.generator
            )
            # Fit transforms
            self.input_scaler.fit(X_train)
            self.output_scaler.fit(y_train)

            # Create datasets
            self.train_dataset = TensorDataset(
                X_train, y_train
            )
            self.val_dataset = TensorDataset(
                X_val, y_val
            )
            self.test_dataset = TensorDataset(
                X_test, y_test
            )
            # Mark done
            self.did_setup = True

        super().setup(stage)

    @property
    def input_size(self) -> torch.Size:
        return torch.Size([16])

@register("protein")
class ProteinDataModule(_UciDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        if not self.did_setup:
            (X_train, y_train), (X_test, y_test), y_train_mu, y_train_scale = load_dataset('protein')

            X_train = torch.from_numpy(X_train).float()
            y_train = torch.from_numpy(y_train).float().squeeze()
            X_test = torch.from_numpy(X_test).float()
            y_test = torch.from_numpy(y_test).float().squeeze()
            
            (X_train, X_val), (y_train, y_val) = tabular_train_test_split(
                X_train, y_train, train_size=0.8, generator=self.generator
            )
            # Fit transforms
            self.input_scaler.fit(X_train)
            self.output_scaler.fit(y_train)

            # Create datasets
            self.train_dataset = TensorDataset(
                X_train, y_train
            )
            self.val_dataset = TensorDataset(
                X_val, y_val
            )
            self.test_dataset = TensorDataset(
                X_test, y_test
            )
            # Mark done
            self.did_setup = True

        super().setup(stage)

    @property
    def input_size(self) -> torch.Size:
        return torch.Size([9])



@register("synth")
class SynthDataModule(_UciDataModule):
    def __init__(self, X_train, y_train, X_test, y_test, root: Optional[PathType] = None, seed: Optional[int] = None):
        """
        Args:
            root: The directory where the dataset can be found or where it should be downloaded to.
            seed: An optional seed which governs how train/test splits are created.
        """
        super().__init__(root, seed)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


    def setup(self, stage: Optional[str] = None) -> None:
        if not self.did_setup:

            X_train = torch.from_numpy(self.X_train).float()
            y_train = torch.from_numpy(self.y_train).float().squeeze()
            X_test = torch.from_numpy(self.X_test).float()
            y_test = torch.from_numpy(self.y_test).float().squeeze()
            
            (X_train, X_val), (y_train, y_val) = tabular_train_test_split(
                X_train, y_train, train_size=0.8, generator=self.generator
            )
            # Fit transforms
            self.input_scaler.fit(X_train)
            self.output_scaler.fit(y_train)

            # Create datasets
            self.train_dataset = TensorDataset(
                X_train, y_train
            )
            self.val_dataset = TensorDataset(
                X_val, y_val
            )
            self.test_dataset = TensorDataset(
                X_test, y_test
            )
            # Mark done
            self.did_setup = True

        super().setup(stage)

    @property
    def input_size(self) -> torch.Size:
        return torch.Size([1])