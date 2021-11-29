#
# https://github.com/ashleve/lightning-hydra-template
#

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    from src import utils
    from src.test import test

    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # job type is train or test
    test(config)


if __name__ == "__main__":
    main()