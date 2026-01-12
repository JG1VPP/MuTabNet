import argparse
from pathlib import Path

from mmengine import Config
from mmengine.dist import master_only
from mmengine.runner import Runner
from mmengine.utils import get_git_hash


class TableRunner(Runner):
    @master_only
    def dump_config(self):
        path = Path(self.log_dir)
        name = Path(self.cfg.filename)

        self.cfg.update(version=get_git_hash())
        self.cfg.dump(path.joinpath(name.name))


def options():
    args = argparse.ArgumentParser()
    args.add_argument("config")
    args.add_argument("--work-dir", required=True)

    return args.parse_args()


def process(config: str, work_dir: str):
    # config
    config = Config.fromfile(config)
    config.update(work_dir=work_dir)

    # runner
    runner = TableRunner.from_cfg(config)

    # train
    runner.train()
    runner.test()


if __name__ == "__main__":
    process(**vars(options()))
