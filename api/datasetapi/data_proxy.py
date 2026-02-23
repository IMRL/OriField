import os
import yaml
import numpy as np

from .registry import Registry, build_from_name
from .config import cfg_from_file

from .dataset import SemanticKitti, Nuscenes, Waymo, Kitti360
from .dataloader import SalsaNextLoader, PerspectiveViewLoader


DATABAG = Registry('Databag')
LOADERBAG = Registry('Loaderbag')


class Databag():
    def __init__(self):
        self.dsname = None
        self.trainset = None
        self.valset = None


@DATABAG.register
class SemanticKittiDatabag(Databag):
    def __init__(self, data_root, panoptic=False):
        print("[SemanticKittiDatabag] init" + "*" * 50)

        super(SemanticKittiDatabag, self).__init__()
        dsname = "kitti"
        data_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset/semantic-kitti.yaml")
        data_path = os.path.join(data_root, "semantic-kitti/sequences")
        trainset = SemanticKitti(
            panoptic=panoptic,
            root=data_path,
            sequences=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
            config_path=data_config_path
        )
        valset = SemanticKitti(
            panoptic=panoptic,
            root=data_path,
            sequences=[8],
            config_path=data_config_path
        )
        # randla_trainset = SemanticKITTI('training', dataset_path=data_path)
        # randla_valset = SemanticKITTI('validation', dataset_path=data_path)
        print("[panoptic: {}]".format(panoptic))
        print("cls_weight: {}".format(trainset.cls_weight))
        print("cls_alpha: {}".format(trainset.cls_alpha))

        self.dsname = dsname
        self.trainset = trainset
        # self.randla_trainset = randla_trainset
        self.valset = valset
        # self.randla_valset = randla_valset


@DATABAG.register
class NuscenesDatabag(Databag):
    def __init__(self, data_root, panoptic=False):
        print("[NuscenesDatabag] init" + "*" * 50)

        super(NuscenesDatabag, self).__init__()
        dsname = "nus"
        data_path = os.path.join(data_root, "nuscenes")
        trainset = Nuscenes(
            panoptic=panoptic,
            root=data_path, version="v1.0-trainval", split="train", has_image=False,
        )
        valset = Nuscenes(
            panoptic=panoptic,
            root=data_path, version="v1.0-trainval", split="val", has_image=False,
        )
        randla_trainset = None
        randla_valset = None
        print("[panoptic: {}]".format(panoptic))
        print("cls_weight: {}".format(trainset.cls_weight))
        print("cls_alpha: {}".format(trainset.cls_alpha))

        self.dsname = dsname
        self.trainset = trainset
        self.randla_trainset = randla_trainset
        self.valset = valset
        self.randla_valset = randla_valset


@DATABAG.register
class WaymoDatabag(Databag):
    def __init__(self, data_root, panoptic=False):
        print("[WaymoDatabag] init" + "*" * 50)

        super(WaymoDatabag, self).__init__()
        dsname = "waymo"
        data_path = os.path.join(data_root, "waymo")
        trainset = Waymo(
            panoptic=panoptic,
            root=data_path, version='v2.0.0', split='train', has_image=False,
        )
        valset = Waymo(
            panoptic=panoptic,
            root=data_path, version='v2.0.0', split='val', has_image=False,
        )
        randla_trainset = None
        randla_valset = None
        print("[panoptic: {}]".format(panoptic))
        print("cls_weight: {}".format(trainset.cls_weight))
        print("cls_alpha: {}".format(trainset.cls_alpha))

        self.dsname = dsname
        self.trainset = trainset
        self.randla_trainset = randla_trainset
        self.valset = valset
        self.randla_valset = randla_valset


class Loaderbag():
    def __init__(self):
        self.train_salsa_loader = None
        self.val_salsa_loader = None
        self.tta_val_salsa_loader = None
        self.input_channels = [0]
        self.meta_size = True


@DATABAG.register
class Kitti360Databag(Databag):
    def __init__(self, data_root, panoptic=False):
        print("[Kitti360Databag] init" + "*" * 50)

        super(Kitti360Databag, self).__init__()
        dsname = "kitti-360"
        data_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset/semantic-kitti.yaml")
        data_path = os.path.join(data_root, "kitti-360/KITTI-360")
        trainset = Kitti360(
            panoptic=panoptic,
            root=data_path,
            config_path=data_config_path,
            split="val",
            has_image=False,
            has_pcd=True,
            has_label=False
        )
        valset = Kitti360(
            panoptic=panoptic,
            root=data_path,
            config_path=data_config_path,
            split="val",
            has_image=False,
            has_pcd=True,
            has_label=False
        )
        randla_trainset = None
        randla_valset = None
        print("[panoptic: {}]".format(panoptic))
        print("cls_weight: {}".format(trainset.cls_weight))
        print("cls_alpha: {}".format(trainset.cls_alpha))

        self.dsname = dsname
        self.trainset = trainset
        self.randla_trainset = randla_trainset
        self.valset = valset
        self.randla_valset = randla_valset


class LoaderbagPoint_RandLA0(Loaderbag):
    def __init__(self, databag):
        super(LoaderbagPoint_RandLA0, self).__init__()
        # L Point
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataloader/single_loader_{}.yaml".format(databag.dsname))
        pmfconfig = yaml.safe_load(open(config_path, "r"))
        self.train_salsa_loader = SalsaNextLoader(
            dataset=databag.trainset,
            config=pmfconfig,
            return_raw=True,
        )
        self.val_salsa_loader = SalsaNextLoader(
            return_uproj=True,
            dataset=databag.valset,
            config=pmfconfig,
            return_raw=True,
            is_train=False)
        self.input_channels = 'RandLA0'


class LoaderbagPoint_ColumnNet(Loaderbag):
    def __init__(self, databag):
        super(LoaderbagPoint_ColumnNet, self).__init__()
        # L Point
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataloader/single_loader_{}.yaml".format(databag.dsname))
        pmfconfig = yaml.safe_load(open(config_path, "r"))
        self.train_salsa_loader = SalsaNextLoader(
            dataset=databag.trainset,
            config=pmfconfig,
            return_mv=True,
        )
        self.val_salsa_loader = SalsaNextLoader(
            return_uproj=True,
            dataset=databag.valset,
            config=pmfconfig,
            return_mv=True,
            is_train=False)
        self.input_channels = 'ColumnNet'


class LoaderbagPoint_RandLA(Loaderbag):
    def __init__(self, databag):
        super(LoaderbagPoint_RandLA, self).__init__()
        # L Point
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataloader/single_loader_{}.yaml".format(databag.dsname))
        pmfconfig = yaml.safe_load(open(config_path, "r"))
        self.train_salsa_loader = SalsaNextLoader(
            dataset=databag.trainset,
            config=pmfconfig,
            return_randla=True,
            randla_dataset=databag.randla_trainset)
        self.val_salsa_loader = SalsaNextLoader(
            dataset=databag.valset,
            config=pmfconfig,
            is_train=False,
            return_randla=True,
            randla_dataset=databag.randla_valset)
        self.input_channels = 'RandLA'
        self.meta_size = False


# 可以用单传感器RV Loader和双传感器RV Loader来加载L，后者由于模态对齐问题不能做数据增强。
@LOADERBAG.register
class LoaderbagRV(Loaderbag):
    def __init__(self, databag, multi_scan=False, multi_scan_align=False, multi_scan_residual=False):
        super(LoaderbagRV, self).__init__()
        # L RV
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataloader/single_loader_{}.yaml".format(databag.dsname))
        pmfconfig = yaml.safe_load(open(config_path, "r"))
        self.train_salsa_loader = SalsaNextLoader(
            multi_scan=multi_scan, multi_scan_align=multi_scan_align, multi_scan_residual=multi_scan_residual,
            dataset=databag.trainset,
            config=pmfconfig, return_mv=True)
        self.val_salsa_loader = SalsaNextLoader(
            multi_scan=multi_scan, multi_scan_align=multi_scan_align, multi_scan_residual=multi_scan_residual,
            dataset=databag.valset,
            config=pmfconfig, return_mv=True,
            is_train=False, return_uproj=True)
        self.tta_val_salsa_loader = SalsaNextLoader(
            multi_scan=multi_scan, multi_scan_align=multi_scan_align, multi_scan_residual=multi_scan_residual,
            dataset=databag.valset,
            config=pmfconfig, return_mv=True,
            is_train=False, return_uproj=True, is_tta=True)
        self.input_channels = [0, 1, 2, 3, 4]  # R X Y Z I


class LoaderbagRV_PCRGB(Loaderbag):
    def __init__(self, databag):
        super(LoaderbagRV_PCRGB, self).__init__()
        # L+RGB RV
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataloader/double_loader_{}.yaml".format(databag.dsname))
        pmfconfig = yaml.safe_load(open(config_path, "r"))
        self.train_salsa_loader = PerspectiveViewLoader(
            dataset=databag.trainset,
            config=pmfconfig,
            is_train=True, pcd_aug=False, img_aug=True, use_padding=True, is_range_view=True)

        self.val_salsa_loader = PerspectiveViewLoader(
            dataset=databag.valset,
            config=pmfconfig,
            is_train=False, use_padding=True, is_range_view=True)
        self.input_channels = [0,1,2,3,4,5,6,7]  # R X Y Z I R G B


class LoaderbagPV_PCRGB(Loaderbag):
    def __init__(self, databag):
        super(LoaderbagPV_PCRGB, self).__init__()
        # L+RGB PV
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataloader/double_loader_{}.yaml".format(databag.dsname))
        pmfconfig = yaml.safe_load(open(config_path, "r"))
        pcd_aug = False
        # pcd_aug = True  # FIXME 模态不对齐
        self.train_salsa_loader = PerspectiveViewLoader(
            dataset=databag.trainset,
            config=pmfconfig,
            is_train=True, pcd_aug=pcd_aug, img_aug=True, use_padding=True)

        self.val_salsa_loader = PerspectiveViewLoader(
            dataset=databag.valset,
            config=pmfconfig,
            is_train=False, use_padding=True)
        # databag.input_channels = [0,1,2,3,4,5,6,7]  # R X Y Z I R G B
        # databag.input_channels = [5,6,7]  # R X Y Z I R G B
        self.input_channels = [0,1,2,3,4]


class DataProxy():
    instance = {}
    datasetcfg = cfg_from_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.yaml"))
    @classmethod
    def getInstance(cls, key, data_root):
        if key not in cls.instance.keys():
            print(f"DataProxy getInstance {key}")
            cfgitem = getattr(cls.datasetcfg, key)
            databagname, loaderbagname = cfgitem.classname.databagname, cfgitem.classname.loaderbagname
            panoptic, multi_scan = cfgitem.mode.panoptic, cfgitem.mode.multi_scan
            databag = build_from_name(DATABAG, databagname, args={"data_root": data_root, "panoptic": panoptic})
            loaderbag = build_from_name(LOADERBAG, loaderbagname, args={"databag": databag, "multi_scan": multi_scan})
            cls.instance[key] = loaderbag
        return cls.instance[key]
