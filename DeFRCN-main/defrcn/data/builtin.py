import os
from .meta_voc import register_meta_voc
from .meta_coco import register_meta_coco
from .builtin_meta import _get_builtin_metadata
from detectron2.data import DatasetCatalog, MetadataCatalog


# -------- COCO -------- #
def register_all_coco(root="datasets"):

    METASPLITS = [
        ("coco14_trainval_all", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k.json"),
        ("coco14_trainval_base", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k.json"),
        ("coco14_test_all", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("coco14_test_base", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("coco14_test_novel", "coco/val2014", "cocosplit/datasplit/5k.json"),
    ]
    for prefix in ["all", "novel"]:
        for shot in [1, 2, 3, 5, 10, 30]:
            for seed in range(10):
                name = "coco14_trainval_{}_{}shot_seed{}".format(prefix, shot, seed)
                METASPLITS.append((name, "coco/trainval2014", ""))

    for name, imgdir, annofile in METASPLITS:
        register_meta_coco(
            name,
            _get_builtin_metadata("coco_fewshot"),
            os.path.join(root, imgdir),
            os.path.join(root, annofile),
        )


# -------- PASCAL VOC -------- #
def register_all_voc(root="datasets"):

    METASPLITS = [
        ("voc_2007_trainval_base1", "VOC2007", "trainval", "base1", 1),
        ("voc_2007_trainval_base2", "VOC2007", "trainval", "base2", 2),
        ("voc_2007_trainval_base3", "VOC2007", "trainval", "base3", 3),
        ("voc_2007_trainval_base10", "VOC2007", "trainval", "base10", 10),
        ("voc_2012_trainval_base1", "VOC2012", "trainval", "base1", 1),
        ("voc_2012_trainval_base2", "VOC2012", "trainval", "base2", 2),
        ("voc_2012_trainval_base3", "VOC2012", "trainval", "base3", 3),
        ("voc_2012_trainval_base10", "VOC2012", "trainval", "base10", 10),
        ("voc_2007_trainval_all1", "VOC2007", "trainval", "base_novel_1", 1),
        ("voc_2007_trainval_all2", "VOC2007", "trainval", "base_novel_2", 2),
        ("voc_2007_trainval_all3", "VOC2007", "trainval", "base_novel_3", 3),
        ("voc_2007_trainval_all10", "VOC2007", "trainval", "base_novel_10", 10),
        ("voc_2012_trainval_all1", "VOC2012", "trainval", "base_novel_1", 1),
        ("voc_2012_trainval_all2", "VOC2012", "trainval", "base_novel_2", 2),
        ("voc_2012_trainval_all3", "VOC2012", "trainval", "base_novel_3", 3),
        ("voc_2012_trainval_all10", "VOC2012", "trainval", "base_novel_10", 10),
        ("voc_2007_test_base1", "VOC2007", "test", "base1", 1),
        ("voc_2007_test_base2", "VOC2007", "test", "base2", 2),
        ("voc_2007_test_base3", "VOC2007", "test", "base3", 3),
        ("voc_2007_test_base9", "VOC2007", "test", "base9", 9),
        ("voc_2007_test_novel1", "VOC2007", "test", "novel1", 1),
        ("voc_2007_test_novel2", "VOC2007", "test", "novel2", 2),
        ("voc_2007_test_novel3", "VOC2007", "test", "novel3", 3),
        ("voc_2007_test_all1", "VOC2007", "test", "base_novel_1", 1),
        ("voc_2007_test_all2", "VOC2007", "test", "base_novel_2", 2),
        ("voc_2007_test_all3", "VOC2007", "test", "base_novel_3", 3),
        ("voc_2007_test_all4", "VOC2007", "test", "base_novel_4", 4),
        ("voc_2007_test_all5", "VOC2007", "test", "base_novel_5", 5),

        ("voc_2007_test_all7", "VOC2007", "test", "base_novel_8", 8),
    ]
    for prefix in ["all", "novel"]:
        for sid in [1,2,3,5,6,7,8,9,10,11,12]:
            for shot in [1, 2, 3, 5, 10]:
                for year in [2007, 2012]:
                    for seed in range(30):
                        seed = "_seed{}".format(seed)
                        name = "voc_{}_trainval_{}{}_{}shot{}".format(
                            year, prefix, sid, shot, seed
                        )
                        dirname = "VOC{}".format(year)
                        img_file = "{}_{}shot_split_{}_trainval".format(
                            prefix, shot, sid
                        )
                        keepclasses = (
                            "base_novel_{}".format(sid)
                            if prefix == "all"
                            else "novel{}".format(sid)
                        )
                        METASPLITS.append(
                            (name, dirname, img_file, keepclasses, sid)
                        )

    for name, dirname, split, keepclasses, sid in METASPLITS:
        year = 2007 if "2007" in name else 2012
        register_meta_voc(
            name,
            _get_builtin_metadata("voc_fewshot"),
            os.path.join(root, dirname),
            split,
            year,
            keepclasses,
            sid,
        )

 
    MetadataCatalog.get(name).evaluator_type = "pascal_voc"

    name = "voc_2007_trainval_all6_30shot_seed0"
    dirname = "VOC2007"
    keepclasses = "base_novel_6"
    register_meta_voc(
        name ,
        _get_builtin_metadata("voc_fewshot"),
        os.path.join(root, dirname),
        6,
        2007,
        keepclasses,
        6,
    )
    name = "voc_2007_trainval_all4_10shot_seed0"
    dirname = "VOC2007"
    keepclasses = "base_novel_4"
    register_meta_voc(
        name ,
        _get_builtin_metadata("voc_fewshot"),
        os.path.join(root, dirname),
        4,
        2007,
        keepclasses,
        4,
    )
    name = "voc_2007_trainval_all4_5shot_seed0"
    dirname = "VOC2007"
    keepclasses = "base_novel_4"
    register_meta_voc(
        name ,
        _get_builtin_metadata("voc_fewshot"),
        os.path.join(root, dirname),
        4,
        2007,
        keepclasses,
        4,
    )

    
    MetadataCatalog.get("voc_2007_test_all1").evaluator_type = "pascal_voc"
    MetadataCatalog.get(name).evaluator_type = "pascal_voc"
    name = "voc_2007_test_all9"
    keepclasses = "base_novel_9"
    dirname = "VOCCustom"
    register_meta_voc(
        name ,
        _get_builtin_metadata("voc_fewshot"),
        os.path.join(root, dirname),
        9,
        year,
        keepclasses,
        9,
    )
    MetadataCatalog.get(name).evaluator_type = "pascal_voc"
    name = "voc_2007_test_all6"
    keepclasses = "base_novel_6"
    dirname = "VOCCustom"
    register_meta_voc(
        name ,
        _get_builtin_metadata("voc_fewshot"),
        os.path.join(root, dirname),
        6,
        year,
        keepclasses,
        6,
    )
    MetadataCatalog.get(name).evaluator_type = "pascal_voc"
    name = "group2_3base"
    keepclasses = "base_novel_8"
    dirname = "VOCCustom"
    register_meta_voc(
        name ,
        _get_builtin_metadata("voc_fewshot"),
        os.path.join(root, dirname),
        8,
        year,
        keepclasses,
        8,
    )
    MetadataCatalog.get(name).evaluator_type = "pascal_voc"
    name = "voc_2007_test_all11"
    keepclasses = "base_novel_11"
    dirname = "VOCCustom"
    register_meta_voc(
        name ,
        _get_builtin_metadata("voc_fewshot"),
        os.path.join(root, dirname),
        11,
        year,
        keepclasses,
        11,
    )
    MetadataCatalog.get(name).evaluator_type = "pascal_voc"
    MetadataCatalog.get(name).evaluator_type = "pascal_voc"
    MetadataCatalog.get("voc_2007_test_all5").evaluator_type = "pascal_voc"
    MetadataCatalog.get("voc_2007_test_all11").evaluator_type = "pascal_voc"
    MetadataCatalog.get("voc_2007_test_all7").evaluator_type = "pascal_voc"
    MetadataCatalog.get("voc_2007_test_all4").evaluator_type = "pascal_voc"
    MetadataCatalog.get("voc_2007_test_base9").evaluator_type = "pascal_voc"
    
register_all_coco()
register_all_voc()